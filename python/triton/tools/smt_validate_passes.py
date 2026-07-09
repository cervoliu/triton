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

__all__ = ["split_funcs", "validate_pass", "sweep", "FuncResult",
           "SplitResult", "SweepReport"]

_FUNC_START = re.compile(r"\btt\.func\b")

# Default pass set: TTIR-level transformations that must preserve semantics.
DEFAULT_PASSES = ("canonicalize", "cse", "triton-combine",
                  "triton-reorder-broadcast")


@dataclass
class SplitResult:
    """Structured function extraction: every `tt.func` in the module is
    accounted for as exactly one definition, declaration, or error."""
    defs: list
    decls: list
    errors: list


def _split_structured(module_text: str) -> SplitResult:
    """Extract top-level `tt.func`s, parsing the header precisely.

    A `tt.func` header may contain brace-delimited dictionaries BEFORE the
    body: per-argument attributes inside the signature parens, result
    attributes inside `-> (...)`, and a function `attributes {...}` clause.
    External declarations have no body at all. Taking the first `{` after
    `tt.func` as the body (the naive approach) truncates such functions or
    swallows the next definition -- so the header is walked delimiter by
    delimiter, and anything unparseable is recorded as an error, never
    silently dropped.
    """
    res = SplitResult([], [], [])
    try:
        body = tv._module_body(module_text)
    except RuntimeError as e:
        res.errors.append(f"module: {e}")
        return res
    starts = [m.start() for m in _FUNC_START.finditer(body)]
    for idx, start in enumerate(starts):
        limit = starts[idx + 1] if idx + 1 < len(starts) else len(body)
        try:
            # Signature parens (may contain per-arg attribute dicts).
            lp = body.index("(", start, limit)
            j = tv._skip_balanced(body, lp, "(", ")")
            # Optional results: `-> type` or `-> (type {attrs}, ...)`.
            k = j
            while k < limit and body[k].isspace():
                k += 1
            if body.startswith("->", k):
                k += 2
                while k < limit and body[k].isspace():
                    k += 1
                if k < limit and body[k] == "(":
                    k = tv._skip_balanced(body, k, "(", ")")
                # else: a bare result type; contains no braces, scan on.
            # Optional `attributes {...}` clause, then the body brace.
            seg = body[k:limit]
            m_attr = re.search(r"\battributes\b", seg)
            brace = seg.find("{")
            if m_attr is not None and (brace == -1 or m_attr.start() < brace):
                ab = body.index("{", k + m_attr.end(), limit)
                k = tv._skip_balanced(body, ab)
                seg = body[k:limit]
                brace = seg.find("{")
            if brace == -1:
                res.decls.append(body[start:limit].strip())
            else:
                end = tv._skip_balanced(body, k + brace)
                res.defs.append(body[start:end].strip())
        except (ValueError, RuntimeError) as e:
            res.errors.append(
                f"tt.func at offset {start}: {e}: "
                f"{body[start:start + 80]!r}")
    return res


def split_funcs(module_text: str) -> list[str]:
    """Extract each top-level `tt.func` DEFINITION from a module's text."""
    return _split_structured(module_text).defs


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


@dataclass
class SweepReport:
    """Sweep results with each accounting unit kept separate.

    `attempts` holds exactly one record per (definition, pass) pair -- the
    only unit in which verdict counts are meaningful. File- and
    function-level accounting are reported alongside, never mixed in.
    """
    attempts: list  # list[FuncResult], one per definition x pass
    files_total: int = 0
    files_parse_skipped: int = 0
    parse_skipped: list = None  # file names
    definitions: int = 0
    declarations: int = 0
    split_errors: list = None  # "file: detail" strings; harness-integrity

    def __post_init__(self):
        self.parse_skipped = self.parse_skipped or []
        self.split_errors = self.split_errors or []


def sweep(corpus: Path, passes: tuple[str, ...], timeout: float,
          triton_opt: Optional[str] = None) -> SweepReport:
    triton_opt = triton_opt or tv.default_triton_opt()
    rep = SweepReport(attempts=[])
    for path in sorted(corpus.glob("*.mlir")):
        rep.files_total += 1
        parsed = _parse_file(path, triton_opt)
        if parsed is None:
            rep.files_parse_skipped += 1
            rep.parse_skipped.append(path.name)
            continue
        split = _split_structured(parsed)
        rep.definitions += len(split.defs)
        rep.declarations += len(split.decls)
        rep.split_errors.extend(f"{path.name}: {e}" for e in split.errors)
        for idx, fn in enumerate(split.defs):
            for pass_name in passes:
                verdict, detail = validate_pass(fn, pass_name, triton_opt,
                                                timeout)
                rep.attempts.append(
                    FuncResult(path.name, idx, pass_name, verdict, detail))
    return rep


def _main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="test/Triton")
    p.add_argument("--passes", default=",".join(DEFAULT_PASSES))
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--json", help="write full per-function results here")
    args = p.parse_args(argv)

    passes = tuple(args.passes.split(","))
    rep = sweep(Path(args.corpus), passes, args.timeout)
    print(f"corpus: {args.corpus}")
    print(f"files: {rep.files_total} total, "
          f"{rep.files_parse_skipped} parse-skipped "
          f"({', '.join(rep.parse_skipped) or 'none'})")
    print(f"functions: {rep.definitions} definitions, "
          f"{rep.declarations} declarations, "
          f"{len(rep.split_errors)} split errors")
    counts = Counter(r.verdict for r in rep.attempts)
    print(f"attempts (definitions x {len(passes)} passes): "
          f"{len(rep.attempts)}")
    for verdict, n in counts.most_common():
        print(f"  {verdict:15s} {n}")
    bad = [r for r in rep.attempts if r.verdict == "NOT_EQUIVALENT"]
    for r in bad:
        print(f"NOT_EQUIVALENT: {r.file}#{r.func_index} --{r.pass_name}")
    for e in rep.split_errors:
        print(f"SPLIT_ERROR: {e}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "files_total": rep.files_total,
            "parse_skipped": rep.parse_skipped,
            "definitions": rep.definitions,
            "declarations": rep.declarations,
            "split_errors": rep.split_errors,
            "attempts": [asdict(r) for r in rep.attempts],
        }, indent=1))
    # NOT_EQUIVALENT means a pass bug or a checker bug; a split error means
    # the harness itself failed to account for a function. Both fail loudly.
    return 1 if (bad or rep.split_errors) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
