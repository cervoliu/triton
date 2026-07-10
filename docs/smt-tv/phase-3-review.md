# Review: Phase 3 — block memory model and driver retry routing

## Verdict

**Changes requested.** A non-positive `max-lanes` value silently disables the mandatory tractability guard, allowing the encoder’s quadratic race construction to hang or exhaust memory.

## Progress since the previous review

Commit `a897b506f` adds the phase-3 block-memory encoding while preserving the legacy phase-1/2 path:

- Pointer arguments become distinct, fully initialized SMT arrays with shared symbolic non-negative sizes.
- `tt.addptr` offsets are sign-extended from i32 and accumulated in bv64, avoiding the legacy chained-offset wraparound.
- Static tensors through rank 2 are concretely materialized in row-major order.
- Active OOB accesses, poison reaching memory, and intra-store different-value collisions contribute to the exact UB predicate.
- Loads observe prior stores, and multiple stores fold sequentially into each function’s private final state.
- Memory-derived addressing is rejected through taint propagation, while memory-dependent masks and stored values remain supported.
- Scope 0 remains as the documented constant-false placeholder; scopes 1 and 2 respectively compare UB domains and final block states.
- The driver preserves the three-result positional contract and retries the memory model only after legacy rejection or a satisfiable identity-addressing scope.
- Tests were added for strided 2-D addressing, masking, reductions, chained pointers, races, sequential memory, and the documented rejection boundaries.

The main default-path encoding appears internally consistent under the implementation notes’ deliberate assumptions. No static false-`EQUIVALENT` path was identified in block decomposition, per-block final-state comparison, OOB masking, store folding, or memory-dependence propagation.

## Findings

### [P2] Reject non-positive `max-lanes` instead of disabling the tractability guard

**References:** `include/triton/Conversion/TritonToSMT/Passes.td:69-72`, `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp:1456-1471`, `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp:1848-1860`, `python/triton/tools/smt_equivalence.py:104-110`, `python/triton/tools/smt_equivalence.py:267-281`, `python/test/unit/tools/test_smt_equivalence.py:1888-1894`.

Static status: `PLAUSIBLE-UNREPRODUCED`; execution is unavailable in this sandbox.

The option is documented as the maximum materialized extent product, and phase 3 requires every oversized tensor to be rejected. The implementation instead checks the limit only when `maxLanes > 0`:

```cpp
if (maxLanes > 0 && *ne > maxLanes)
  return err(...);
```

Consequently, both `max-lanes=0` and negative values disable the cap entirely. This is not documented as an opt-out and contradicts the phase’s mandatory tractability boundary. It is particularly dangerous because every store subsequently enumerates all lane pairs when constructing race conditions. The driver supplies no timeout to the `triton-opt` subprocess; its timeout applies only to Z3 after encoding completes.

Caller reproducer:

```sh
OPT=build/<actual-build-dir>/bin/triton-opt

"$OPT" \
  --convert-triton-to-smt='memory-model=true max-lanes=1' \
  test/Conversion/triton_to_smt_memory_model.mlir

"$OPT" \
  --convert-triton-to-smt='memory-model=true max-lanes=0' \
  test/Conversion/triton_to_smt_memory_model.mlir
```

The existing module has extent product 4. The first command should reject it, while the current condition allows the second command to encode it. That establishes the guard bypass without requiring a resource-exhaustion run. A 257×257 variant then enters roughly 2.18 billion lane-pair checks per store instead of rejecting immediately, before SMT solving or its timeout begins.

Reject `maxLanes <= 0` before constructing any scope and add direct lit regressions for zero and negative values. The current oversized-extent pytest covers only the default positive limit and does not exercise this option boundary.

## Validation summary

This was a static review of commit `a897b506f`; no build, test, solver, reproducer, or file write was possible.

Statically inspected:

- The complete phase-3 specification and implementation notes.
- The prior phase-2 review.
- Pass option definitions and both encoder paths.
- Block declaration, offset accumulation, OOB predicates, taint propagation, masking, race construction, sequential stores, reductions, and final-state comparison.
- Driver fallback routing, result-count validation, verdict decoding, and environment-error handling.
- The phase-3 pytest and lit additions.

Before applying a fix, the caller should run the reproducer above. Afterward, add zero/negative `max-lanes` regressions and run `scripts/smt-verify.sh` so skips cannot masquerade as passing soundness coverage.
