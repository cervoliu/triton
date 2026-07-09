# Review: Phase 2 milestones 2–4 — milestone-4 sweep harness round 5

## Verdict

**Approved.** No blocking soundness or measurement-integrity issue remains within milestone 4’s documented exploratory/default-sweep scope.

## Progress since the previous review

Commit `0a8557441` closes both round-4 findings:

- Nested-module headers are parsed grammatically: an optional bare or quoted symbol is consumed before an exact `attributes` keyword is recognized.
- Malformed module headers produce fatal discovery errors.
- Regressions cover `module @attributes`, `module @"attributes"`, and `module @attributes attributes {...}`.
- Pass entries must match the complete ASCII leaf-token expression `^[A-Za-z0-9][A-Za-z0-9-]*$`.
- Nested pipelines, option syntax, commas, braces, dots, whitespace, and driver flags cannot occupy the leaf-pass slot.
- Pipeline-grammar attacks now produce configuration errors.

The default sweeps remain stable at 596 attempts, 40 `EQUIVALENT`, zero `NOT_EQUIVALENT`, and zero split/discovery errors, with exit status zero.

Milestones 2 and 3 remain approved because this commit does not change the encoder or refinement driver semantics.

Caller-supplied evidence reports `scripts/smt-verify.sh` green with 77 pytest cases and 4 lit tests. No reproducer was executed during this static review.

## Findings

No findings.

## Validation summary

### Milestone 2

The previously approved conclusions remain unchanged:

- Shift poison uses the unsigned width bound.
- Unsupported shift and cast poison-generating flags are rejected.
- `extui`, `extsi`, and `trunci` retain fixed-width semantics, including `i1`.
- The raw-`i1` store, shaped-type, `index`, `i0`, and timeout failures remain fixed.

### Milestone 3

The previously approved conclusions remain unchanged:

- Poison is tracked per scalar or tensor lane.
- `select` suppresses poison only in its unselected arm.
- Masks gate pointer and stored-value poison exactly at memory sinks.
- Scope 1 compares UB domains; scope 2 compares outputs only where neither program is UB.
- Identical UB domains are accepted under bidirectional refinement.

### Milestone 4

The structural-accounting path now fails closed for the reviewed grammar:

- Strings and attribute dictionaries cannot create false function starts.
- Ordinary and symbol-bearing nested modules are traversed.
- `attributes` is recognized by grammar position rather than a text suffix.
- Invalid module headers produce discovery errors.
- Per-file reconciliation caught the earlier 22-definition regression and the restored aggregate is pinned by corpus regressions.

The pass interface is now appropriately constrained:

- Default passes are explicitly listed in `python/triton/tools/smt_validate_passes.py:29-31`.
- Pass names receive a whole-string ASCII validation before pipeline construction at `python/triton/tools/smt_validate_passes.py:242-250`.
- Only the validated atomic token is embedded in the fixed module pipeline at `python/triton/tools/smt_validate_passes.py:251-260`.
- Invalid pipeline names produce configuration failure.
- Even if diagnostic wording changes and an invalid name degrades to `PASS_ERROR`, the per-pass all-`PASS_ERROR` guard at `python/triton/tools/smt_validate_passes.py:379-384` prevents a successful sweep.
- The two Triton-specific defaults are registered as module transformations in `include/triton/Dialect/Triton/Transforms/Passes.td:6` and `:25`.

Aggregation remains fail-closed for conclusive evidence:

- Any observed `NOT_EQUIVALENT` fails the sweep.
- Split or discovery errors fail the sweep.
- Missing input, empty pass lists, zero attempts, and all-`PASS_ERROR` configurations fail.
- `PASS_ERROR`, `UNSUPPORTED`, and `UNKNOWN` cannot overwrite an observed counterexample.

The documentation’s empirical claim is appropriately bounded:

- There were 596 attempted function-pass pairs.
- Only the 40 `EQUIVALENT` pairs are semantically decided.
- Zero semantic counterexamples were observed.
- The other 556 outcomes remain explicitly inconclusive.
- Module-alias-induced `PASS_ERROR` is documented as a coverage limitation rather than evidence of pass correctness.
- The 40/40 result for the four `triton_to_smt*.mlir` files is supported by per-function reconciliation.

Two residual boundaries do not warrant blocking this exploratory milestone:

- The structural scanner is not a complete MLIR parser for arbitrary modules nested beneath unrelated region-bearing operations. No such case was tied to the reviewed default corpus or its claims.
- A custom registered atomic analysis or no-op pass may legitimately produce identity output. The published result concerns the four reviewed default transformations and does not claim that every custom pass rewrites every input.

No candidate met the threshold for a repository-grounded `PLAUSIBLE-UNREPRODUCED` finding.

