# Review: TTIR-to-SMT soundness, equivalence-driver robustness, and SMT Real integration

## Verdict

**Changes requested.** A volatile memory access can still be erased and reported `EQUIVALENT`, and two accepted API/input paths abort instead of returning a controlled rejection or verdict.

## Progress since the previous review

Both previously reported false-equivalence cases are fixed. Chained `tt.addptr` operations are now rejected before 32-bit offset accumulation can hide a `2^32` displacement, and a load whose element sort differs from the underlying array after an `i1`/`i8` pointer bitcast is now rejected. The corresponding driver regressions pass. The driver also retains the fail-closed quoted-symbol handling and exact-two-solver-result check, while the Real exporter rejects the prior newline/SMT-command payload. The focused Python suite now has 24 passing tests, the two Triton conversion lit tests pass, and all 16 existing MLIR SMT dialect/export tests pass.

## Findings

### [P1] Reject volatile loads instead of proving away an observable access

**References:** `phase-1.md:79-91`, `include/triton/Dialect/Triton/IR/TritonOps.td:214-237`, `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp:563-588`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:287-310`, `python/triton/tools/smt_equivalence.py:189-196`.

I reproduced this with two single-block functions of identical `(!tt.ptr<i32>, !tt.ptr<i32>) -> ()` signature. `@src` computes `a + program_id(0)` and performs an unused `tt.load` carrying `{isVolatile = true}`; `@tgt` omits that load. Both then store integer zero to `out + program_id(0)`. The exact `triton-opt --convert-triton-to-smt | mlir-translate --export-smtlib | z3 -in` pipeline printed `unsat` for both scopes, so the driver classifies the pair as `EQUIVALENT`.

The source performs an observable volatile global-memory read and the target does not; NVIDIA lowering explicitly carries the attribute to the PTX `volatile` qualifier. The SMT load handler never inspects `getIsVolatile()`, and because the loaded SSA value is dead the read disappears completely from the exported assertion. The current model has no memory-access trace, so volatile loads must be rejected rather than treated as ordinary array selects.

### [P2] Reject static store shapes whose element count overflows `int64_t`

**References:** `phase-1.md:79-98`, `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp:281-301`, `.llvm-project/src/mlir/lib/IR/BuiltinTypeInterfaces.cpp:72-92`.

I reproduced this with verifier-valid, identical `@src` and `@tgt` functions that splat a scalar value and output pointer to `tensor<4294967296x4294967296xf32>` and perform their single store. Plain `triton-opt` parses and verifies the module successfully. Running `triton-opt --convert-triton-to-smt` exits with signal 6 and reports `Assertion failed: (num.has_value() && "integer overflow in element count computation"), function getNumElements`.

The pass rejects dynamic store shapes but calls `RankedTensorType::getNumElements()` unconditionally for every static shape. In the assert-enabled build that API aborts when the dimension product does not fit `int64_t`; in a non-assert build the fallback multiplication can overflow. Use `tryGetNumElements()` and fail the pass with a diagnostic when no count is representable.

### [P2] Validate solver timeouts before passing them to `subprocess.run`

**References:** `python/triton/tools/smt_equivalence.py:97-103`, `python/triton/tools/smt_equivalence.py:147-181`, `python/triton/tools/smt_equivalence.py:223-237`.

I reproduced this through the public API with an identical, valid single-store i8 TTIR self-query. With `timeout=60.0` it returns `EQUIVALENT`; with `timeout=float("nan")` it raises `ValueError: cannot convert float NaN to integer` instead of returning an `EquivalenceResult`. I also reproduced uncaught `OverflowError` for `float("inf")` and for the finite value `10_000_000_000.0`. The CLI accepts the non-finite forms because `--timeout` uses an unrestricted `type=float` parser, and the solver call catches only `subprocess.TimeoutExpired`. Validate that the timeout is finite and representable, or convert these subprocess timeout-conversion failures to `UNKNOWN`.

## Validation summary

- `make PYTHON=.venv/bin/python` completed successfully; a subsequent `ninja -C build/cmake.macosx-26.0-arm64-cpython-3.12 triton-opt` reported no work.
- `PYTHONPATH=python .venv/bin/pytest -s --tb=short python/test/unit/tools/test_smt_equivalence.py`: 24 passed.
- Triton lit tests `triton_to_smt.mlir` and `triton_to_smt_casts.mlir`: 2 passed.
- MLIR `Dialect/SMT` and `Target/SMTLIB` suites: 16 passed.
- The prior chained-`tt.addptr` and raw i8-through-i1-bitcast reproducers now fail the pass and return `UNSUPPORTED`; the prior Real-constant command-injection payload is rejected by `mlir-translate`.
- The volatile-load reproducer returned `unsat`/`unsat`; the overflowing static tensor shape produced a `triton-opt` SIGABRT; invalid public-API timeout values raised the exceptions described above.
- No other false `EQUIVALENT` result or crash was reproduced within the stated ideal-real, fixed-width-integer, and identity-addressing boundaries.
