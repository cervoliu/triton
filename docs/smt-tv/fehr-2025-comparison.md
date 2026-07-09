# Comparison: Fehr et al. 2025, "First-Class Verification Dialects for MLIR"

(PLDI 2025, PACMPL vol. 9, art. 206; local PDF under `~/dev/paper/`.)

The paper defines MLIR *semantic dialects* — `poison`, `effect`, `ub_effect`,
`mem_effect`, `memory`, `transfer` — plus per-theory SMT dialects, and treats
a program dialect's semantics as a lowering into them, refined stepwise down
to SMT-LIB. Three tools sit on top (translation validation, peephole-rewrite
verification, transfer-function verification); their TV found 5 real
poison-introduction miscompilations in upstream MLIR canonicalize (all in
`arith.select` rewrites — Table 2, p. 16).

## Facts that matter for this project

- **Same SMT foundation.** The paper's low-level smt dialects were contributed
  upstream as the single MLIR `smt` dialect (p. 6 fn. 3) — the dialect this
  project already targets. The high-level dialects are xDSL/Python research
  prototypes (p. 9), not upstream and not C++.
- **No floats anywhere** (p. 15): no Real or FloatingPoint theory. Nothing to
  adopt for our ideal-real design, and their work cannot replace
  `scripts/patches/mlir-smt-real.patch`. The realistic patch exit is
  upstreaming our Real sort; the paper demonstrates that contribution channel.
- **Poison semantics agree with ours.** Per-value tracking with the exact
  select rule `p(c) ∨ ite(c, p(t), p(f))` — the rule whose violation caused
  all 5 of their MLIR bugs. They encode poison as SMT datatype pairs and then
  need a datatype-elimination pass for a 24.6–59.5% solver speedup (§6.4);
  our separate-predicate encoding is already that optimized form.
- **Documented divergence:** in their (Lee et al./Alive2-style) memory model,
  bytes can hold poison — storing poison is not UB (Fig. 3, p. 7). We treat
  poison reaching any memory operation as UB: a deliberate strengthening that
  avoids poison-in-memory entirely, sound for equivalence because both
  kernels get the same rule. Revisit only if we validate UB-*reducing*
  transformations (directional refinement).
- **Their memory model** (ptr = (block_id, 64-bit offset), OOB/dead-block UB,
  poison-initialized alloc) is strictly more general than our
  per-pointer-argument arrays + identity-addressing contract. Worth lifting
  (as semantics, not as their dialect stack) when we tackle aliasing,
  atomics, or shared-memory allocs.

## Absorb / skip

1. **Absorb: upstream the Real-sort patch** to MLIR's smt dialect.
2. **Absorb: simplify the emitted smt module before export** (canonicalize /
   CSE / folding). They report ~2× query-size reduction with some queries
   discharged without a solver (§4.4) — for us a pass-pipeline addition.
3. **Absorb: TV test generation** (§6.1): bounded-exhaustive tiny functions
   over boundary constants {−1, 0, 1, MAX, MIN} plus random ≤N-op functions,
   driven through the passes `smt_validate_passes.py` already sweeps. Proven
   recipe, proven bug class (select poison-introduction).
4. **Later: mem_effect-style pointer/UB semantics** once the
   identity-addressing contract is outgrown.
5. **Skip: restructuring TritonToSMT into their multi-level dialect
   architecture.** Its payoff is N-dialects × M-tools semantic reuse; we have
   one source dialect and one tool, and the high-level dialects are unported
   research artifacts. A rewrite without payoff.
6. **Skip: bitwidth-independent semantics** (§5.2.5) — undecidable fragment,
   incomplete in practice, and our kernels have fixed widths.
