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


_FUNC_HEADER = re.compile(r"tt\.func(\s+(public|private|nested))?\s+@")


def _top_level_func_starts(body: str):
    """Find `tt.func` operation starts, skipping strings and brace groups.

    A raw regex scan is not enough: `tt.func` can appear as a dialect
    attribute NAME (`attributes {tt.func = false}`), inside a quoted symbol
    or string attribute, or in a nested region -- all of which live inside
    braces or quotes at this level. Every accepted start must also look like
    a function header (`tt.func [visibility] @...`); anything else is a
    fatal discovery error rather than a silently mis-split candidate.
    """
    def token_at(i, tok):
        if not body.startswith(tok, i):
            return False
        if i > 0 and (body[i - 1].isalnum() or body[i - 1] in "._$"):
            return False
        j = i + len(tok)
        return j >= len(body) or not (body[j].isalnum() or body[j] in "_$")

    def skip_ws(k):
        while k < len(body) and body[k].isspace():
            k += 1
        return k

    def skip_module_header(k):
        """From just past a `module` token, return the index just past the
        module's body `{` (descended into), parsing the header as grammar:
        optional @symbol (bare or quoted -- a module NAMED @attributes must
        not be confused with an attributes clause), optional `attributes`
        keyword + dict, then the region brace."""
        k = skip_ws(k)
        if k < len(body) and body[k] == "@":
            k += 1
            if k < len(body) and body[k] == '"':
                j = k + 1
                while j < len(body) and body[j] != '"':
                    j += 2 if body[j] == "\\" else 1
                k = j + 1
            else:
                while k < len(body) and (body[k].isalnum() or
                                         body[k] in "._$-"):
                    k += 1
            k = skip_ws(k)
        if token_at(k, "attributes"):
            k = skip_ws(k + len("attributes"))
            if k >= len(body) or body[k] != "{":
                raise RuntimeError("module `attributes` not followed by '{'")
            k = skip_ws(tv._skip_balanced(body, k))
        if k >= len(body) or body[k] != "{":
            raise RuntimeError("module header not followed by a region")
        return k + 1  # descend into the module body

    starts, errors = [], []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == '"':
            j = i + 1
            while j < n and body[j] != '"':
                j += 2 if body[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            try:
                i = tv._skip_balanced(body, i)
            except RuntimeError as e:
                errors.append(f"discovery: {e}")
                return starts, errors
            continue
        if token_at(i, "module"):
            try:
                i = skip_module_header(i + len("module"))
            except RuntimeError as e:
                errors.append(f"discovery: module at offset {i}: {e}")
                i += len("module")
            continue
        if token_at(i, "tt.func"):
            if _FUNC_HEADER.match(body, i):
                starts.append(i)
            else:
                errors.append(
                    f"discovery: top-level 'tt.func' at offset {i} is not a "
                    f"function header: {body[i:i + 60]!r}")
            i += len("tt.func")
            continue
        i += 1
    return starts, errors


def _find_body_brace(body: str, start: int, limit: int):
    """Locate the opening brace of a tt.func's operation region.

    Walks the header from just past `tt.func`, tracking every delimiter kind:
    quoted strings (symbol names, string attrs), nested `<...>`/`(...)`/`[...]`
    (types and signatures -- a `{` inside any of these belongs to a type or
    argument attribute, e.g. `#ttg.slice<{dim = 0, ...}>` or
    `%p: !tt.ptr<i32> {tt.divisibility = 16}`), `->` (whose `>` is not a
    closing angle), and a top-level `attributes {...}` clause (skipped). The
    first `{` at zero nesting depth that is not an attributes clause opens the
    region. Returns None for a bodyless external declaration; raises
    RuntimeError on malformed input.
    """
    i = start
    depth = 0
    while i < limit:
        c = body[i]
        if c == '"':
            j = i + 1
            while j < limit and body[j] != '"':
                j += 2 if body[j] == "\\" else 1
            if j >= limit:
                raise RuntimeError("unterminated string in tt.func header")
            i = j + 1
            continue
        if body.startswith("->", i):
            i += 2
            continue
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
            if depth < 0:
                raise RuntimeError("unbalanced delimiter in tt.func header")
        elif c == "{":
            if depth > 0:
                # Attribute dict inside a type or signature; skip it whole.
                i = tv._skip_balanced(body, i)
                continue
            if body[:i].rstrip().endswith("attributes"):
                i = tv._skip_balanced(body, i)
                continue
            return i
        i += 1
    if depth != 0:
        raise RuntimeError("unbalanced delimiter in tt.func header")
    return None


def _split_structured(module_text: str) -> SplitResult:
    """Extract top-level `tt.func`s, parsing the header precisely.

    A `tt.func` header may contain brace-delimited dictionaries BEFORE the
    body: per-argument attributes, result attributes, inline attributes in
    dialect types (`#ttg.slice<{...}>`), and a function `attributes {...}`
    clause; external declarations have no body at all. `_find_body_brace`
    walks the header tracking every delimiter kind, and anything unparseable
    is recorded as an error, never silently dropped.
    """
    res = SplitResult([], [], [])
    try:
        body = tv._module_body(module_text)
    except RuntimeError as e:
        res.errors.append(f"module: {e}")
        return res
    starts, disc_errors = _top_level_func_starts(body)
    res.errors.extend(disc_errors)
    for idx, start in enumerate(starts):
        limit = starts[idx + 1] if idx + 1 < len(starts) else len(body)
        try:
            open_idx = _find_body_brace(body, starts[idx], limit)
            if open_idx is None:
                res.decls.append(body[start:limit].strip())
            else:
                end = tv._skip_balanced(body, open_idx)
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
    # Only an atomic registered-pass name is accepted: pipeline grammar
    # (parens, commas, braces, dots, whitespace) could nest an anchor that
    # matches nothing in the single-function candidate -- e.g.
    # builtin.module(canonicalize) or func.func(canonicalize) -- making an
    # identity run look like a validated EQUIVALENT.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", pass_name):
        raise RuntimeError(
            f"{pass_name!r} is not an atomic pass name (pipeline syntax is "
            "not accepted; the harness schedules builtin.module(<pass>))")
    src_mod = "module {\n" + func_text + "\n}\n"
    # Run through --pass-pipeline rather than --{name}: a recognized driver
    # OPTION (e.g. allow-unregistered-dialect) is a valid flag but schedules
    # no transformation, so it would certify an identity run as EQUIVALENT.
    # Pipeline syntax only admits registered passes, and rejects anything
    # else before touching the input -- a configuration error, not a
    # per-function result.
    pipeline = f"--pass-pipeline=builtin.module({pass_name})"
    try:
        tgt_mod = tv._run([triton_opt, pipeline], src_mod)
    except RuntimeError as e:
        if ("does not refer to a registered pass" in str(e)
                or "failed to parse pass pipeline" in str(e)
                or "Unknown command line argument" in str(e)):
            raise RuntimeError(
                f"{pass_name!r} is not a registered pass") from e
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

    passes = tuple(p.strip() for p in args.passes.split(","))
    if not passes or any(not p for p in passes):
        print("error: empty pass name in --passes")
        return 2
    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f"error: corpus {corpus} is not a directory")
        return 2
    if not any(corpus.glob("*.mlir")):
        print(f"error: corpus {corpus} contains no .mlir files")
        return 2
    rep = sweep(corpus, passes, args.timeout)
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
    # Exit nonzero on anything that makes the measurement untrustworthy:
    # NOT_EQUIVALENT (pass bug or checker bug), split errors (a function the
    # harness failed to account for), a zero-work sweep, or a pass that never
    # ran successfully on any definition (all PASS_ERROR => likely
    # misconfiguration rather than coverage).
    if bad or rep.split_errors:
        return 1
    if not rep.attempts:
        print("error: sweep performed no function-pass attempts")
        return 2
    for pass_name in passes:
        per = [r for r in rep.attempts if r.pass_name == pass_name]
        if per and all(r.verdict == "PASS_ERROR" for r in per):
            print(f"error: --{pass_name} failed on every definition; "
                  "suspected misconfiguration")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
