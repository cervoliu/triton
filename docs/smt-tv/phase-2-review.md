# Review: Phase 2 milestone 2 shifts, integer ext/trunc, and well-definedness

## Verdict

**Changes requested.** Shift and cast semantics remain sound, but a type-correct oversized integer timeout can still escape the public driver API as an uncaught exception.

## Progress since the previous review

Commit `fa48b2494` closes both round-2 type-validation holes. Scalar `index` values and zero-width integer values are now rejected before encoding; `smtSortFor` also fails closed for `i0`. The regressions `test_reject_index_constant` and `test_reject_zero_width_integer` pin these cases.

The earlier fixes remain intact: raw Bool-to-bit-vector stores are rejected, unsupported non-ranked shaped types fail before width queries, shift poison conditions are discharged through the dedicated well-definedness scope, and legal `extui`, `extsi`, and `trunci` operations preserve their fixed-width semantics.

## Findings

### [P2] Oversized integer timeout escapes before an `EquivalenceResult`

**References:** `python/triton/tools/smt_equivalence.py:97-103`, `python/triton/tools/smt_equivalence.py:216-219`, `python/triton/tools/smt_equivalence.py:261-268`.

**Location:** `check_equivalence` calls `math.isfinite(timeout)` before entering the exception handler that converts unrepresentable subprocess timeout values into `UNKNOWN`.

**Soundness claim at risk:** the public verdict API should return an `EquivalenceResult` for invalid or unrepresentable solver timeouts rather than raising. This is a robustness failure, not a false-equivalence result.

**Reproduction status: PLAUSIBLE-UNREPRODUCED.** Python accepts integers where a `float` parameter is expected, but converting an arbitrarily large integer for `math.isfinite` raises `OverflowError`. The following valid self-query should fail before any subprocess is launched:

```sh
PYTHONPATH=python .venv/bin/python - <<'PY'
from triton.tools import smt_equivalence as tv

kernel = r"""
module {
  tt.func public @k(%o: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i32
    %i = tt.get_program_id x : i32
    %p = tt.addptr %o, %i : !tt.ptr<i32>, i32
    tt.store %p, %zero : !tt.ptr<i32>
    tt.return
  }
}
"""

result = tv.check_equivalence(
    kernel,
    kernel,
    timeout=10**10000,
    triton_opt="build/cmake.macosx-26.0-arm64-cpython-3.12/bin/triton-opt",
    mlir_translate=".llvm-project/build/bin/mlir-translate",
    z3="z3",
)
print(result.verdict)
PY
```

Expected failure:

```text
OverflowError: int too large to convert to float
```

The later solver invocation catches `OverflowError`, but this earlier validation does not.

**Suggested fix:** wrap timeout normalization and `math.isfinite` in `try/except (TypeError, OverflowError)` and return the existing `UNKNOWN`/`bad-timeout` result. Add a regression using `timeout=10**10000`; optionally cover a nonnumeric value as well.

## Validation summary

- Performed a static audit only, as required for this round.
- The shift bound uses unsigned `amount < width`; `shli` overflow flags, right-shift `exact`, and `i1` shifts fail closed.
- Zero extension, sign extension, odd-width casts, and low-bit truncation—including the `i1` cases—match the Arith and SMT bit-vector semantics.
- Per-lane shifts accumulate per-lane well-definedness conditions. The driver requires exactly three solver results and cannot return `EQUIVALENT` unless addressing and well-definedness are both `unsat`.
- No type-gate bypass, false `EQUIVALENT`, WD vacuity, signedness mismatch, or cast-width error was identified.
- Caller-provided validation reports 56 passing pytest cases in `python/test/unit/tools/test_smt_equivalence.py` and four passing conversion lit tests through `scripts/smt-verify.sh`.

