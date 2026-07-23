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


def _skip_balanced(text: str, i: int, open: str = "{", close: str = "}") -> int:
    """Given text[i] == open, return the index just past the matching close.

    String literals are skipped so delimiters inside quoted attribute values do
    not confuse the balance.
    """
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c == open:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise RuntimeError(f"unbalanced '{open}{close}' in TTIR input")


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
    # EQUIVALENT | EQUIVALENT_UNDER_RESTRICT | NOT_EQUIVALENT | UNSUPPORTED |
    # UNKNOWN. EQUIVALENT_UNDER_RESTRICT is a memory-model EQUIVALENT produced
    # with assume_restrict=True: the verdict holds only if the pointer
    # arguments are non-aliasing (a precondition TTIR cannot express, so it is
    # disclosed rather than assumed silently). See check_equivalence.
    verdict: str
    # z3 result of scope 0. On the identity-addressing (legacy) path this is
    # the addressing-identity check (sat => non-identity addressing, which
    # triggers the memory-model retry). On the memory-model path scope 0 is a
    # documented always-unsat placeholder -- addressing correctness is part of
    # the block decomposition itself -- kept so the three-scope positional
    # contract is unchanged (see docs/smt-tv/phase-3.md).
    addressing_status: str
    equivalence_status: str  # z3 result of the equivalence scope
    smtlib: str
    z3_output: str
    # z3 result of the UB-domain-equality scope (unsat => the two kernels are
    # UB on exactly the same inputs; sat => an input makes exactly one kernel
    # UB, so they are not mutually refining). Defaults to "n/a" for results
    # produced before the solver stage.
    ub_status: str = "n/a"
    # True iff this result came from the phase-3 block-memory-model encoding
    # (the retry path for kernels the identity-addressing path cannot handle).
    memory_model: bool = False

    @property
    def equivalent(self) -> bool:
        return self.verdict == "EQUIVALENT"


def check_equivalence(src_ttir: str, tgt_ttir: str, *, block_size: int = 0,
                      assume_restrict: bool = False,
                      counterexample: bool = False, timeout: float = 60.0,
                      triton_opt: Optional[str] = None,
                      mlir_translate: Optional[str] = None,
                      z3: Optional[str] = None) -> EquivalenceResult:
    """Check equivalence of two kernels given their TTIR text.

    Kernel-dependent failures (unsupported constructs, solver timeouts) become
    verdicts (UNSUPPORTED/UNKNOWN); environment errors (missing, unpatched, or
    pass-less tools) raise RuntimeError with an actionable message.

    Routing (phase 3): the identity-addressing encoding is tried first. When
    it rejects the kernels, or accepts them but scope 0 proves the addressing
    is not identity, the check is retried once with the block memory model
    (``--convert-triton-to-smt=memory-model=true``), which handles strided,
    multi-dimensional, and masked addressing; ``memory_model`` on the result
    records which encoding produced the verdict. An explicit ``block_size``
    pins the identity-addressing contract and disables the retry (the option
    is a legacy-path cross-check with no memory-model counterpart). The retry
    can double the worst-case runtime (two solver runs of up to ``timeout``).

    ``assume_restrict`` only affects the block-memory-model path: distinct
    pointer arguments are disjoint SMT arrays by construction, which the pass
    otherwise rejects when that disjointness is load-bearing for the verdict
    (TTIR carries no ``restrict`` guarantee). With ``assume_restrict=True`` the
    pass encodes under the disjointness assumption instead of rejecting, and a
    resulting memory-model ``EQUIVALENT`` is relabeled ``EQUIVALENT_UNDER_
    RESTRICT`` -- equivalent provided the pointer args do not alias. It never
    weakens the legacy (identity-addressing) path or a ``block_size``-pinned
    run.
    """
    triton_opt = triton_opt or default_triton_opt()
    mlir_translate = mlir_translate or default_mlir_translate()
    z3 = z3 or default_z3()

    # A non-finite or unrepresentable timeout would raise from subprocess
    # rather than time out. math.isfinite itself raises on values that cannot
    # convert to float (e.g. 10**10000) or are not numbers at all; both are the
    # same documented UNKNOWN, not an exception from the verdict API.
    try:
        bad_timeout = timeout is not None and not math.isfinite(timeout)
    except (TypeError, OverflowError):
        bad_timeout = True
    if bad_timeout:
        try:
            desc = repr(timeout)  # repr(10**10000) exceeds the int-str limit
        except Exception:
            desc = f"<unprintable {type(timeout).__name__}>"
        return EquivalenceResult("UNKNOWN", "bad-timeout", "bad-timeout", "",
                                 f"invalid timeout value: {desc}")

    # Building the merged module can fail (e.g. a module that is not exactly one
    # renamable tt.func, including quoted symbols); surface that as UNSUPPORTED
    # rather than raising from the public verdict API.
    try:
        module = build_query_module(src_ttir, tgt_ttir, triton_opt)
    except RuntimeError as e:
        return EquivalenceResult("UNSUPPORTED", "n/a", "n/a", "", str(e))

    def _pipeline(pass_opt: str, memory_model: bool) -> EquivalenceResult:
        # The pass fails (signalPassFailure) for inputs outside its validated
        # contract; treat that as UNSUPPORTED rather than a hard error. A
        # triton-opt that does not register the pass (or, on the retry, one
        # too old to know the memory-model option) is an environment error,
        # not a kernel verdict: mapping it to UNSUPPORTED would make every
        # rejection test pass vacuously, so fail loudly instead.
        try:
            smt_mlir = _run([triton_opt, pass_opt], module)
        except RuntimeError as e:
            if "Unknown command line argument" in str(e):
                raise RuntimeError(
                    f"{triton_opt} does not register --convert-triton-to-smt; "
                    "build triton-opt from a tree containing the TritonToSMT "
                    "pass") from e
            if memory_model and "no such option" in str(e):
                raise RuntimeError(
                    f"{triton_opt} does not know the memory-model pass "
                    "option; rebuild triton-opt from a tree containing the "
                    "phase-3 TritonToSMT pass") from e
            return EquivalenceResult("UNSUPPORTED", "n/a", "n/a", "", str(e),
                                     memory_model=memory_model)
        # The exporter's input is pass output, so a failure here is an
        # environment error (typically an mlir-translate without
        # scripts/patches/mlir-smt-real.patch); raise it actionably.
        try:
            smtlib = _run([mlir_translate, "--export-smtlib"], smt_mlir)
        except RuntimeError as e:
            if "smt.real" in str(e):
                raise RuntimeError(
                    f"{mlir_translate} cannot handle the SMT Real sort; apply "
                    "scripts/patches/mlir-smt-real.patch (see "
                    "scripts/build-llvm-project.sh) and rebuild") from e
            raise

        # Solve without (get-model): scope 0 = addressing, scope 1 = UB-domain
        # equality, scope 2 = equivalence (the pass always emits all three, in
        # this order, in BOTH encodings). A solver timeout is a documented
        # UNKNOWN, not an exception.
        try:
            z3_out = _run([z3, "-in"], smtlib, timeout=timeout)
        except (subprocess.TimeoutExpired, ValueError, OverflowError):
            # TimeoutExpired = genuine timeout; ValueError/OverflowError = a
            # timeout value subprocess cannot represent. Both are a documented
            # UNKNOWN.
            return EquivalenceResult(
                "UNKNOWN", "timeout", "timeout", smtlib,
                f"z3 timed out or timeout unrepresentable ({timeout})",
                "timeout", memory_model=memory_model)
        # Require exactly three solver results rather than trusting line
        # positions, so an unexpected extra scope cannot be misread as a
        # verdict.
        results = [ln.strip() for ln in z3_out.splitlines()
                   if ln.strip() in ("sat", "unsat", "unknown")]
        if len(results) != 3:
            return EquivalenceResult("UNKNOWN", "n/a", "n/a", smtlib, z3_out,
                                     memory_model=memory_model)
        addr, ub, equiv = results
        if addr == "sat":
            # Non-identity addressing on the legacy path (the memory-model
            # scope 0 is always unsat); the caller retries with the block
            # memory model.
            verdict = "UNSUPPORTED"
        elif ub != "unsat" and addr == "unsat":
            # sat: some input makes exactly one kernel UB (e.g. it stores an
            # oversized-shift result, or accesses out of bounds, where the
            # other is defined), so they do not mutually refine each other.
            verdict = "NOT_EQUIVALENT" if ub == "sat" else "UNKNOWN"
        elif addr == "unsat":
            verdict = {"unsat": "EQUIVALENT",
                       "sat": "NOT_EQUIVALENT"}.get(equiv, "UNKNOWN")
        else:
            verdict = "UNKNOWN"
        return EquivalenceResult(verdict, addr, equiv, smtlib, z3_out, ub,
                                 memory_model=memory_model)

    legacy_opt = "--convert-triton-to-smt"
    if block_size > 0:
        legacy_opt = f"--convert-triton-to-smt=block-size={block_size}"
    res = _pipeline(legacy_opt, memory_model=False)
    # Phase-3 routing: fall back to the block memory model when the
    # identity-addressing encoding rejected the kernels or disproved identity
    # addressing (scope 0 sat). An explicit block_size pins the legacy
    # contract, so no retry. UNKNOWN results are NOT retried: the legacy
    # encoding accepted the kernels and only the solver struggled.
    mm_opt = "--convert-triton-to-smt=memory-model=true"
    if assume_restrict:
        mm_opt += " assume-restrict=true"
    if block_size <= 0 and (res.verdict == "UNSUPPORTED"
                            or res.addressing_status == "sat"):
        res = _pipeline(mm_opt, memory_model=True)
        # assume_restrict over-discloses: ANY memory-model EQUIVALENT under the
        # flag carries the non-aliasing caveat, even for a kernel that was not
        # actually load-bearing. Over-disclosing a precondition never lies, so
        # this stays sound; we deliberately do NOT add a fourth solver scope or
        # a side channel to make it precise (the three-scope positional
        # contract is fixed). Relabel only the memory-model path -- the legacy
        # path is untouched by assume_restrict.
        if assume_restrict and res.verdict == "EQUIVALENT":
            res.verdict = "EQUIVALENT_UNDER_RESTRICT"

    # Request a counterexample model only once we know the equivalence scope is
    # sat; asking after unsat/unknown makes z3 error with "model not available".
    if counterexample and res.verdict == "NOT_EQUIVALENT":
        idx = res.smtlib.rfind("(check-sat)")
        if idx >= 0:
            end = idx + len("(check-sat)")
            model_query = res.smtlib[:end] + "\n(get-model)" + res.smtlib[end:]
            try:
                res.z3_output = _run([z3, "-in"], model_query, timeout=timeout)
            except (subprocess.TimeoutExpired, RuntimeError):
                # Keep the verdict; the model is best-effort. The rerun can
                # nondeterministically flip to unknown, making z3 exit nonzero
                # on (get-model) ("model is not available").
                pass
    return res


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
