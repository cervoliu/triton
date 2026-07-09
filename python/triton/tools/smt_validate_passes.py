"""Validate Triton TTIR transformation passes with the SMT equivalence checker.

Mirrors mlir-tv's methodology (Bang et al., CAV 2022): for every function in a
TTIR corpus, run a semantics-preserving pass and check the original against
the transformed function with ``smt_equivalence.check_equivalence``. Functions
outside the checker's contract come back UNSUPPORTED — that is expected and
counted, never silently skipped. A NOT_EQUIVALENT is always reportable: either
a pass bug or a checker bug, and both matter.

Usage:
  python -m triton.tools.smt_validate_passes [--corpus test/Triton]
      [--passes canonicalize,cse,triton-combine] [--timeout 20] [--json out]
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from triton.tools import smt_equivalence as tv

__all__ = ["split_funcs", "validate_pass", "sweep", "FuncResult"]

_FUNC_START = re.compile(r"\btt\.func\b")

# Default pass set: TTIR-level transformations that must preserve semantics.
DEFAULT_PASSES = ("canonicalize", "cse", "triton-combine",
                  "triton-reorder-broadcast")


def split_funcs(module_text: str) -> list[str]:
    """Extract each top-level `tt.func ... { ... }` from a module's text."""
    try:
        body = tv._module_body(module_text)
    except RuntimeError:
        return []
    funcs = []
    pos = 0
    while True:
        m = _FUNC_START.search(body, pos)
        if m is None:
            break
        try:
            open_idx = body.index("{", m.start())
            end = tv._skip_balanced(body, open_idx)
        except (ValueError, RuntimeError):
            break
        funcs.append(body[m.start():end].strip())
        pos = end
    return funcs


@dataclass
class FuncResult:
    file: str
    func_index: int
    pass_name: str
    verdict: str
    detail: str = ""


def _parse_file(path: Path, triton_opt: str) -> Optional[str]:
    """Parse-normalize a corpus file; None if it does not parse standalone."""
    try:
        return tv._run([triton_opt, "--strip-debuginfo"], path.read_text())
    except (RuntimeError, subprocess.SubprocessError):
        return None  # e.g. expected-error lit files; counted by the caller


def validate_pass(func_text: str, pass_name: str, triton_opt: str,
                  timeout: float) -> tuple[str, str]:
    """Run `pass_name` on one function and check original vs transformed."""
    src_mod = "module {\n" + func_text + "\n}\n"
    try:
        tgt_mod = tv._run([triton_opt, f"--{pass_name}"], src_mod)
    except RuntimeError as e:
        return "PASS_ERROR", str(e).splitlines()[0][:200]
    try:
        res = tv.check_equivalence(src_mod, tgt_mod, timeout=timeout)
    except RuntimeError as e:
        # Environment errors raise by contract; surface them loudly.
        raise
    detail = "" if res.equivalent else res.z3_output[:200]
    return res.verdict, detail


def sweep(corpus: Path, passes: tuple[str, ...], timeout: float,
          triton_opt: Optional[str] = None) -> list[FuncResult]:
    triton_opt = triton_opt or tv.default_triton_opt()
    results: list[FuncResult] = []
    for path in sorted(corpus.glob("*.mlir")):
        parsed = _parse_file(path, triton_opt)
        if parsed is None:
            results.append(FuncResult(path.name, -1, "*", "PARSE_SKIP"))
            continue
        funcs = split_funcs(parsed)
        for idx, fn in enumerate(funcs):
            for pass_name in passes:
                verdict, detail = validate_pass(fn, pass_name, triton_opt,
                                                timeout)
                results.append(
                    FuncResult(path.name, idx, pass_name, verdict, detail))
    return results


def _main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="test/Triton")
    p.add_argument("--passes", default=",".join(DEFAULT_PASSES))
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--json", help="write full per-function results here")
    args = p.parse_args(argv)

    results = sweep(Path(args.corpus), tuple(args.passes.split(",")),
                    args.timeout)
    counts = Counter(r.verdict for r in results)
    print(f"corpus: {args.corpus}  functions x passes: {len(results)}")
    for verdict, n in counts.most_common():
        print(f"  {verdict:15s} {n}")
    bad = [r for r in results if r.verdict == "NOT_EQUIVALENT"]
    for r in bad:
        print(f"NOT_EQUIVALENT: {r.file}#{r.func_index} --{r.pass_name}")
    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in results], indent=1))
    # NOT_EQUIVALENT means a pass bug or a checker bug; fail the sweep.
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main())
