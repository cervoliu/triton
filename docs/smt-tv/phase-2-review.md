# Review: Phase 2 milestone 3 — exact poison tracking and bidirectional refinement

## Verdict

**Approved.** No soundness hole, crash path, or fail-open verdict was identified within milestone 3's supported validation contract.

This review was static-only as requested. No commands were run and no files were modified.

## Progress since the previous review

Commit `a01fcc042` replaces the milestone-2 blanket well-definedness requirement with exact per-value poison tracking in `lib/Conversion/TritonToSMT/TritonToSMTPass.cpp`.

Each scalar or tensor lane is represented by an `EncodedValue` carrying both its SMT value and poison predicate. Program equivalence is now checked as equality of deterministic partial functions:

1. Scope 1 searches for an input where source and target UB differ.
2. Scope 2 searches for an observable output difference where neither program reaches UB.

The driver in `python/triton/tools/smt_equivalence.py::check_equivalence` treats a satisfiable UB-difference query as `NOT_EQUIVALENT`.

Caller-supplied evidence reports `scripts/smt-verify.sh` green with 62 pytest cases and 4 lit tests.

## Findings

No findings.

## Validation summary

### Poison propagation

The supported operations preserve poison at the same scalar or tensor-lane granularity as their values:

- Constants and `tt.make_range` begin non-poison.
- Strict arithmetic, comparisons, casts, and pointer arithmetic disjoin the corresponding operands' poison.
- Shape transformations duplicate or permute poison in the same way they transform payload values.
- Integer extension and truncation preserve their input poison, including the special SMT-Bool representation of `i1`.
- Shifts add the unsigned `amount >= bitwidth` poison condition to operand poison.
- Unsupported poison-generating flags remain rejection cases rather than being silently encoded as unflagged operations.

The `arith.select` rule is exact:

```text
p(result) = p(condition) OR ite(condition, p(true), p(false))
```

Consequently, poison in an unselected arm is suppressed, while a poisoned condition poisons the result independently of the condition's arbitrary SMT payload. Scalar conditions selecting tensor values must apply this same condition and condition-poison predicate to every result lane.

Reduction encoding carries each lane's complete `EncodedValue` through the combiner region and `tt.reduce.return`. Poison is not pre-aggregated across unrelated rows or results, so a poisoned lane cannot contaminate an independent reduction result.

### Memory sinks and mask gating

The load/store rules distinguish poison that reaches an active memory operation from poison in an inactive operand:

- An unmasked load reaches UB exactly when its pointer is poison.
- An active masked-load lane reaches UB when its pointer is poison.
- A poison mask reaches UB regardless of the mask payload.
- An inactive masked-load lane ignores pointer poison and takes the selected `other` value's poison.
- A successfully loaded active value is non-poison.
- An unmasked store reaches UB when its pointer or stored value is poison.
- A masked store gates pointer and stored-value poison on the mask value, while mask poison remains unconditional UB.
- `tt.addptr` propagates base-pointer and offset poison lane by lane; the resulting pointer poison is consumed only if that pointer reaches an active load or store.

These distinctions prevent both under-approximation and over-approximation. Under-approximation could hide a real UB-domain difference; over-approximation could exclude a genuinely defined input and thereby hide a later observable output difference.

### Refinement queries

Let D_P(x) = ¬UB_P(x). The two scopes establish bidirectional refinement:

- `UB_src != UB_tgt` is satisfiable exactly when the programs have different defined-input domains.
- `!UB_src && !UB_tgt && outputsDiffer` is satisfiable exactly when their observations differ on the common defined domain.

Treating programs as equivalent on inputs where both reach UB is defensible: neither refinement direction constrains behavior outside its source program's defined domain. The programs need not reach UB at the same instruction or for the same poison cause; only equality of their aggregate defined domains is semantically relevant.

The driver result mapping is sound:

- UB-difference `sat` → `NOT_EQUIVALENT`.
- Output-difference `sat` after UB-domain equality → `NOT_EQUIVALENT`.
- Both scopes `unsat` → `EQUIVALENT`.
- Any required `unknown`, malformed result, timeout, launch failure, or rejected translation → `UNKNOWN` or a clean validation error, never `EQUIVALENT`.

No attack considered during the static audit met the threshold for a repository-grounded `PLAUSIBLE-UNREPRODUCED` finding, so no reproducer is included.
