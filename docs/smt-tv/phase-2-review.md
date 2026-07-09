# Review: Phase 2 milestone 2 shifts, integer ext/trunc, and well-definedness

## Verdict

**Approved.** Round 4 found no remaining soundness hole, crash path, or fail-open verdict within the supported translation-validation contract.

This review was static-only as requested. No commands were run and no files were modified.

## Progress since the previous review

Commit `973531529` closes the sole round-3 finding, `[P2] Oversized integer timeout escapes before an EquivalenceResult`, in `python/triton/tools/smt_equivalence.py::check_equivalence`.

Both failure stages now degrade safely to `UNKNOWN` with a bad-timeout diagnostic:

- Conversion or validation of an unrepresentably large integer no longer leaks `OverflowError`.
- Diagnostic construction no longer calls an unsafe decimal `repr` on an integer exceeding Python’s configured digit limit.

The parametrized regressions `test_unrepresentable_timeout_is_unknown[huge-int]` and `test_unrepresentable_timeout_is_unknown[str]` in `python/test/unit/tools/test_smt_equivalence.py` pin both paths.

The earlier protections also remain intact:

- Raw `i1`/SMT-Bool values cannot be coerced into byte-vector stores.
- Non-ranked shaped types are rejected before encoding.
- Dynamic-shaped stores are rejected.
- `index` and zero-width `i0` types, including tensor element types, are rejected up front.

## Findings

No findings.

## Validation summary

The static audit rechecked the following soundness boundaries:

- The shift encodings in `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp:520-552` use an unsigned `amount < bitwidth` well-definedness condition. This correctly rejects amounts with negative-looking bit patterns and amounts equal to or greater than the width.
- `arith.shli` overflow flags and right-shift `exact` semantics are explicitly rejected rather than silently discarded.
- Shift well-definedness is collected per tensor lane, including the reduction encoding paths at `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp:657-666` and `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp:835-850`.
- The `extui`, `extsi`, and `trunci` encodings at `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp:578-626` implement zero extension, sign extension, and low-bit truncation respectively, including the explicit SMT-Bool handling required for `i1`.
- Odd, nonzero integer widths do not alter the bit-vector semantics of shifts or intermediate extension/truncation operations.
- `arith.extui` `nneg` and `arith.trunci` overflow flags are rejected, preventing poison-producing casts from being treated as ordinary bit-only casts.
- The well-definedness query at `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp:1123-1138` requires the conjunction of source and target side conditions. Unsupported or poison-producing cases therefore cannot reach the value-equivalence query as silently defined executions.
- The driver requires exactly the expected three solver results and cannot return `EQUIVALENT` unless both address equivalence and well-definedness are proved with `unsat`. Invalid timeout values now return `UNKNOWN`.
- Potential attacks involving dynamic tensor construction, sub-byte stores, `i1` memory interpretation, and timeout scaling did not yield a concrete accepted path or semantics mismatch grounded in the current implementation. None met the threshold for a `PLAUSIBLE-UNREPRODUCED` finding, so no reproducer is included.
- Caller-supplied runtime evidence reports `scripts/smt-verify.sh` green with 58 pytest cases and 4 lit tests.

