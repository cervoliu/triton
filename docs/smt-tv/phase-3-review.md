# Review: Phase 3 — block memory model and driver retry routing, round 2

## Verdict

**Approved.** The confirmed round-1 tractability-guard bypass is fixed and no remaining blocking soundness or crash issue was found within phase 3's documented contract.

## Progress since the previous review

Commit `30d74931a` closes the round-1 finding:

- `memory-model` mode rejects `max-lanes <= 0` before constructing any solver scope.
- The per-tensor check now applies the positive cap unconditionally; zero and negative values cannot mean "unlimited."
- The pass option documentation explicitly states that the cap is mandatory and must be positive.
- The diagnostic formats the `int64_t` value through `Twine`, avoiding the prior character-overload output.
- Lit regressions cover `max-lanes=0` and `max-lanes=-4`.
- Pytest independently invokes the pass for both values and checks for the expected rejection diagnostic.

The caller reproduced the original defect before fixing it: `max-lanes=1` rejected with status 1, while `max-lanes=0` encoded with status 0. After the fix, both non-positive cases reject.

The rest of the phase-3 implementation remains unchanged: bv64 block offsets, symbolic block sizes, rank-2 lane materialization, OOB and intra-store-race UB, sequential stores, memory-dependence tainting, three fixed solver scopes, and legacy-to-memory-model retry routing.

(Round 1 requested changes for the `max-lanes` guard bypass — a non-positive
value silently disabled the mandatory extent-product tractability bound ahead
of the quadratic race construction. The fix and its reproducer are pinned by
the lit BADCAP runs in `test/Conversion/triton_to_smt_memory_model.mlir` and
`test_reject_nonpositive_max_lanes` in
`python/test/unit/tools/test_smt_equivalence.py`.)

## Findings

No findings.

## Validation summary

Caller-supplied verification reports `scripts/smt-verify.sh` green:

- Lit: 5/5 passing.
- Pytest: 99/99 passing.
- No skips.

This round statically re-reviewed the combined `a897b506f + 30d74931a` state. No command, build, solver, or reproducer was executed inside the review sandbox.

The static audit covered:

- Up-front option validation and all tensor extent/rank guards.
- Shared source/target arrays, sizes, scalar inputs, and program IDs.
- Signed bv64 OOB predicates and exact mask gating.
- Poison propagation at loads and stores.
- Intra-store collision predicates and their structural elisions.
- Sequential store folding and load-after-store visibility.
- Per-block whole-array final-state comparison under shared UB domains.
- Memory-derived-address taint propagation through accepted operations.
- Rejection of unsupported pointer decomposition, shapes, atomics, `tt.dot`, and control flow.
- Driver retry conditions, three-result validation, and verdict precedence.

The implementation notes' deliberate boundaries—fully initialized argument blocks, intra-store race policy, program-ordered multiple stores, disjoint pointer arguments, per-program refinement, and the preserved legacy encoding—remain consistently implemented and do not create an identified false-`EQUIVALENT` path within the stated model.
