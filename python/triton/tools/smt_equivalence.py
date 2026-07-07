"""SMT-based translation validation for Triton kernels (ideal-real semantics).

This driver checks whether two Triton kernels are functionally equivalent by:
  1. compiling each kernel to TTIR (host-side, no GPU required),
  2. merging them into one module as ``@src`` / ``@tgt``,
  3. running ``triton-opt --convert-triton-to-smt`` to encode them into the SMT
     dialect under ideal-real floating-point semantics,
  4. exporting to SMT-LIB with ``mlir-translate --export-smtlib``, and
  5. deciding the refinement query with Z3.

``unsat`` means no input makes the two kernels' outputs differ, i.e. they are
equivalent. ``sat`` yields a concrete counterexample.

Floating-point values are modeled as ideal reals (not IEEE-754); see the
TritonToSMT pass documentation.
"""
from __future__ import annotations

import glob
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "compile_to_ttir",
    "check_equivalence",
    "check_kernels",
    "EquivalenceResult",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _find_tool(env_var: str, *candidates: str) -> str:
    """Locate a build tool via env override, glob candidates, or PATH."""
    if env_var in os.environ:
        return os.environ[env_var]
    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    # Fall back to PATH lookup on the basename of the first candidate.
    base = os.path.basename(candidates[0]) if candidates else env_var
    found = shutil.which(base)
    if found:
        return found
    raise FileNotFoundError(
        f"could not locate {env_var} (tried {candidates}); set {env_var}")


def default_triton_opt() -> str:
    return _find_tool("TRITON_OPT",
                      str(_REPO_ROOT / "build" / "*" / "bin" / "triton-opt"))


def default_mlir_translate() -> str:
    # llvm-project/ is where scripts/build-llvm-project.sh (which applies the
    # required SMT Real-sort patch) builds by default; .llvm-project/ is a
    # custom checkout layout (see scripts/patches/README.md).
    return _find_tool(
        "MLIR_TRANSLATE",
        str(_REPO_ROOT / "llvm-project" / "build" / "bin" / "mlir-translate"),
        str(_REPO_ROOT / ".llvm-project" / "build" / "bin" / "mlir-translate"))


def default_z3() -> str:
    return _find_tool("Z3", "/opt/homebrew/bin/z3", "/usr/bin/z3", "/usr/local/bin/z3")


def compile_to_ttir(fn, signature: dict, constexprs: Optional[dict] = None,
                    target=None) -> str:
    """Compile a @triton.jit kernel to TTIR text using only the frontend.

    This deliberately stops after the TTIR stage so no GPU backend tool (e.g.
    ptxas) is required; TTIR is all the equivalence checker needs.
    """
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend
    if target is None:
        target = GPUTarget("cuda", 80, 32)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {})
    backend = make_backend(target)
    options = backend.parse_options(dict(src.parse_options()))
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(target, options, codegen_fns, module_map, context)
    stages: dict = {}
    backend.add_stages(stages, options, src.language)
    module = stages["ttir"](module, {})
    # Print without debug locations so the merge step needs no separate
    # `triton-opt --strip-debuginfo` subprocess round-trip.
    return module.str_nodebug()


def _run(cmd, stdin_text: str, timeout: Optional[float] = None) -> str:
    proc = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({' '.join(map(str, cmd))}):\n{proc.stderr}\n{proc.stdout}")
    return proc.stdout


_FUNC_RE = re.compile(r'(tt\.func\s+(?:public\s+|private\s+)?)@(?:\w+|"[^"]*")')


def _skip_balanced(text: str, i: int) -> int:
    """Given text[i] == '{', return the index just past the matching '}'.

    String literals are skipped so braces inside quoted attribute values do not
    confuse the balance.
    """
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise RuntimeError("unbalanced braces in TTIR input")


def _module_body(text: str) -> str:
    """Extract the top-level module's body, handling `module attributes {...} {`."""
    m = re.search(r"\bmodule\b", text)
    if m is None:
        raise RuntimeError("no top-level `module` found in TTIR input")
    try:
        open_idx = text.index("{", m.end())
        # An `attributes {...}` clause precedes the body brace; skip it. Any
        # module attributes are metadata this checker does not depend on, so
        # they are dropped by the merge.
        if "attributes" in text[m.end():open_idx]:
            open_idx = text.index("{", _skip_balanced(text, open_idx))
    except ValueError as e:
        raise RuntimeError("malformed module header in TTIR input") from e
    close_idx = _skip_balanced(text, open_idx)
    return text[open_idx + 1:close_idx - 1].strip()


def _strip_and_extract(ttir: str, new_name: str, triton_opt: str) -> str:
    """Strip debug info and return the single function body renamed to new_name."""
    # compile_to_ttir already prints without locations; only file-based CLI
    # inputs still need the strip subprocess.
    if "loc(" in ttir or "#loc" in ttir:
        ttir = _run([triton_opt, "--strip-debuginfo"], ttir)
    # Reject modules that do not contain exactly one function (e.g. private
    # helpers emitted for noinline kernels), which would corrupt the merge.
    num_funcs = len(re.findall(r"\btt\.func\b", ttir))
    if num_funcs != 1:
        raise RuntimeError(
            f"expected exactly one tt.func per kernel, found {num_funcs}")
    body = _module_body(ttir)
    body, n = _FUNC_RE.subn(rf"\1@{new_name}", body, count=1)
    if n != 1:
        raise RuntimeError(f"could not rename function (matched {n})")
    return body


def build_query_module(src_ttir: str, tgt_ttir: str, triton_opt: str) -> str:
    """Merge two kernels' TTIR into one module as @src and @tgt."""
    s = _strip_and_extract(src_ttir, "src", triton_opt)
    t = _strip_and_extract(tgt_ttir, "tgt", triton_opt)
    return f"module {{\n{s}\n{t}\n}}\n"


@dataclass
class EquivalenceResult:
    verdict: str             # EQUIVALENT | NOT_EQUIVALENT | UNSUPPORTED | UNKNOWN
    addressing_status: str   # z3 result of the addressing-identity scope
    equivalence_status: str  # z3 result of the equivalence scope
    smtlib: str
    z3_output: str

    @property
    def equivalent(self) -> bool:
        return self.verdict == "EQUIVALENT"


def check_equivalence(src_ttir: str, tgt_ttir: str, *, block_size: int = 0,
                      counterexample: bool = False, timeout: float = 60.0,
                      triton_opt: Optional[str] = None,
                      mlir_translate: Optional[str] = None,
                      z3: Optional[str] = None) -> EquivalenceResult:
    """Check equivalence of two kernels given their TTIR text.

    Kernel-dependent failures (unsupported constructs, solver timeouts) become
    verdicts (UNSUPPORTED/UNKNOWN); environment errors (missing, unpatched, or
    pass-less tools) raise RuntimeError with an actionable message.
    """
    triton_opt = triton_opt or default_triton_opt()
    mlir_translate = mlir_translate or default_mlir_translate()
    z3 = z3 or default_z3()

    # A non-finite timeout would raise from subprocess rather than time out.
    if timeout is not None and not math.isfinite(timeout):
        return EquivalenceResult("UNKNOWN", "bad-timeout", "bad-timeout", "",
                                 f"invalid timeout value: {timeout!r}")

    # Building the merged module can fail (e.g. a module that is not exactly one
    # renamable tt.func, including quoted symbols); surface that as UNSUPPORTED
    # rather than raising from the public verdict API.
    try:
        module = build_query_module(src_ttir, tgt_ttir, triton_opt)
    except RuntimeError as e:
        return EquivalenceResult("UNSUPPORTED", "n/a", "n/a", "", str(e))
    pass_opt = "--convert-triton-to-smt"
    if block_size > 0:
        pass_opt = f"--convert-triton-to-smt=block-size={block_size}"
    # The pass fails (signalPassFailure) for inputs outside its validated
    # contract; treat that as UNSUPPORTED rather than a hard error. A triton-opt
    # that does not register the pass at all is an environment error, not a
    # kernel verdict: mapping it to UNSUPPORTED would make every rejection test
    # pass vacuously, so fail loudly instead.
    try:
        smt_mlir = _run([triton_opt, pass_opt], module)
    except RuntimeError as e:
        if "Unknown command line argument" in str(e):
            raise RuntimeError(
                f"{triton_opt} does not register --convert-triton-to-smt; "
                "build triton-opt from a tree containing the TritonToSMT "
                "pass") from e
        return EquivalenceResult("UNSUPPORTED", "n/a", "n/a", "", str(e))
    # The exporter's input is pass output, so a failure here is an environment
    # error (typically an mlir-translate without scripts/patches/
    # mlir-smt-real.patch); raise it with an actionable message.
    try:
        smtlib = _run([mlir_translate, "--export-smtlib"], smt_mlir)
    except RuntimeError as e:
        if "smt.real" in str(e):
            raise RuntimeError(
                f"{mlir_translate} cannot handle the SMT Real sort; apply "
                "scripts/patches/mlir-smt-real.patch (see "
                "scripts/build-llvm-project.sh) and rebuild") from e
        raise

    # Solve first without (get-model): scope 0 = addressing, scope 1 = equivalence.
    # A solver timeout is a documented UNKNOWN, not an exception.
    try:
        z3_out = _run([z3, "-in"], smtlib, timeout=timeout)
    except (subprocess.TimeoutExpired, ValueError, OverflowError):
        # TimeoutExpired = genuine timeout; ValueError/OverflowError = a timeout
        # value subprocess cannot represent. Both are a documented UNKNOWN.
        return EquivalenceResult("UNKNOWN", "timeout", "timeout", smtlib,
                                 f"z3 timed out or timeout unrepresentable ({timeout})")
    # The pass emits exactly two solver scopes (addressing, then equivalence);
    # require exactly two solver results rather than trusting line positions, so
    # an unexpected extra scope cannot be misread as a verdict.
    results = [ln.strip() for ln in z3_out.splitlines()
               if ln.strip() in ("sat", "unsat", "unknown")]
    if len(results) != 2:
        return EquivalenceResult("UNKNOWN", "n/a", "n/a", smtlib, z3_out)
    addr, equiv = results  # scope 0: addressing, scope 1: equivalence
    if addr == "sat":
        verdict = "UNSUPPORTED"  # non-identity addressing is not modeled
    elif addr == "unsat":
        verdict = {"unsat": "EQUIVALENT",
                   "sat": "NOT_EQUIVALENT"}.get(equiv, "UNKNOWN")
    else:
        verdict = "UNKNOWN"

    # Request a counterexample model only once we know the equivalence scope is
    # sat; asking after unsat/unknown makes z3 error with "model not available".
    if counterexample and verdict == "NOT_EQUIVALENT":
        idx = smtlib.rfind("(check-sat)")
        if idx >= 0:
            end = idx + len("(check-sat)")
            model_query = smtlib[:end] + "\n(get-model)" + smtlib[end:]
            try:
                z3_out = _run([z3, "-in"], model_query, timeout=timeout)
            except (subprocess.TimeoutExpired, RuntimeError):
                # Keep the verdict; the model is best-effort. The rerun can
                # nondeterministically flip to unknown, making z3 exit nonzero
                # on (get-model) ("model is not available").
                pass
    return EquivalenceResult(verdict, addr, equiv, smtlib, z3_out)


def check_kernels(src_fn, tgt_fn, signature: dict, *,
                  src_constexprs: Optional[dict] = None,
                  tgt_constexprs: Optional[dict] = None,
                  tgt_signature: Optional[dict] = None,
                  **kwargs) -> EquivalenceResult:
    """Compile two @triton.jit kernels and check their equivalence."""
    src_ttir = compile_to_ttir(src_fn, signature, src_constexprs)
    tgt_ttir = compile_to_ttir(tgt_fn, tgt_signature or signature, tgt_constexprs)
    return check_equivalence(src_ttir, tgt_ttir, **kwargs)


def _main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-ttir", required=True, help="source kernel TTIR file")
    p.add_argument("--tgt-ttir", required=True, help="target kernel TTIR file")
    p.add_argument("--block-size", type=int, default=0)
    p.add_argument("--counterexample", action="store_true")
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args(argv)

    src = Path(args.src_ttir).read_text()
    tgt = Path(args.tgt_ttir).read_text()
    res = check_equivalence(src, tgt, block_size=args.block_size,
                            counterexample=args.counterexample,
                            timeout=args.timeout)
    print(res.verdict)
    if res.verdict != "EQUIVALENT":
        print(res.z3_output)
    return 0 if res.verdict == "EQUIVALENT" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
