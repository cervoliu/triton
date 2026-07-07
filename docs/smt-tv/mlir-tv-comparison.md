# Our SMT encoder vs. mlir-tv (Bang et al., CAV 2022)

A comparison of this project's Triton→SMT translation validator
(`lib/Conversion/TritonToSMT`) against the reference tool **mlir-tv**
("SMT-Based Translation Validation for Machine Learning Compiler", Bang,
Nam, Chun, Jhoo & Lee, CAV 2022; code at <https://github.com/aqjune/mlir-tv>).
The goal is to see what we can *absorb* from a mature, published implementation.

## TL;DR

mlir-tv and our pass are both SMT translation validators, but they sit at
opposite ends of a precision/scope tradeoff.

- **mlir-tv** is a *general* MLIR validator (tosa / linalg / tensor / memref,
  ~99 ops, reductions, matmul, loops) built around an **abstract,
  over-approximating** floating-point encoding (minimal-width bit-vectors +
  uninterpreted functions) with an abstraction-refinement loop. It never misses
  a bug but may raise false alarms, which it refines away.
- **Ours** is a *narrow, deliberately-rejecting* validator for straight-line
  **elementwise** Triton, with an **idealized real-number** FP model. Anything
  outside its contract is rejected (`UNSUPPORTED`), never silently reported
  equivalent.

The single most important difference is the float encoding, and it is where
most of the leverage lives.

## Axis-by-axis

### 1. Floating point — the core divergence

| | mlir-tv | ours |
|---|---|---|
| Model | Abstract **bit-vector** of minimal width (`⌈log₂(#distinct floats)⌉`), sign+magnitude, with UFs for add/mul/div/exp | **Ideal `Real`** (SMT-LIB reals) |
| Special values | NaN/±Inf/±0/±MAX reserved bit-patterns, IEEE special-case `ite` cascades | none — finite reals only; non-finite constants **rejected** |
| Commutativity | structural: `fp_add(x,y) & fp_add(y,x)` (bitwise-AND both operand orders of a UF) | free — real `+`/`*` are already commutative/associative |
| Associativity | opt-in via `--associative` / a hash-multiset precondition scheme (their headline contribution) | free but **unconditional** (reals *are* associative) |
| Soundness claim | over-approximation → no missed bugs; false alarms possible, then refined | a *different idealized* semantics — neither over- nor under- w.r.t. IEEE |

This is the crux. mlir-tv's FP encoding is a sound over-approximation of the
IEEE properties it models, refined to remove false alarms. Ours treats floats
as exact reals, which means:

- We get associativity / commutativity / distributivity **for free** — exactly
  the class of reorderings ML compilers perform (mlir-tv's `--associative` mode
  exists precisely to grant this). For those transforms our checker is *simpler
  and stronger*.
- But we silently accept some rewrites IEEE forbids (`x+0.0→x` at `x=-0.0` —
  the exact bug they reported to LLVM), and we cannot model NaN / Inf / rounding
  at all — we reject them instead.

**Leverage:** if we ever want IEEE-faithfulness, their minimal-width abstract-BV
encoding (not full FPA bit-blasting) is the proven-practical middle ground —
**13.6× faster** than concrete IEEE-754 in their evaluation, and shrinking the
bit-width gave a further 2.2×. This aligns with our roadmap note that
FpSan-style error tracking is the future direction.

### 2. Integers

Nearly identical and mutually validating: **both use fixed-width bit-vectors
with wrapping arithmetic.** We chose this over unbounded `Int` after a review
found unsound rewrites; the paper independently made the same call. Two
divergences:

- **Shifts:** mlir-tv encodes `shl`/`ashr`/`lshr` and flags shift-amount ≥ width
  as UB (`st.wellDefined(op, amnt.ult(bw))`). We currently *reject* shifts. Their
  approach is a ready-made recipe for supporting them soundly.
- **Integer div/rem:** they *also* don't support `divsi`/`divui`/`remsi`/`remui`
  (0 matches in their `encode.cpp`) — the same conclusion we reached (total SMT
  bv-div is unsound without a poison model). Independent corroboration.

### 3. Memory

- **mlir-tv:** a full Alive2-style block model — per-element-type block families
  that never alias across types, block ids, liveness / writability / initialized
  arrays, layout maps with affine inverse functions, UB on
  OOB/uninitialized/dead access.
- **ours:** one SMT array per pointer argument, bv32 offset, a `fresh` flag that
  rejects chained `addptr`, identity addressing proven by a dedicated solver
  scope.

Ours is far narrower but adequate for elementwise Triton. The leverage here is
conceptual: their **local vs non-local blocks (only non-local checked at
refinement)** and **UB-as-a-per-op-boolean** (not a poison-per-value lattice)
are the two ideas worth adopting if we grow beyond single-store kernels.

### 4. Reductions / dot

mlir-tv's entire Section 5 is about this: UF `sum`/`dot`, hash-based multiset
equality for permutation-invariance, a `SUM_MUL` decomposition of dot, and
unroll-for-constant-length. **We have none of it** — we are elementwise-only.
This is our biggest *capability gap*.

**Key insight for us:** their hardest contribution (hash-based reduction
associativity) exists *only because their floats are abstract UFs*. Because we
model floats as **ideal reals**, addition/multiplication are already
associative and commutative — so a reduction over a **static** Triton block
extent can be encoded by literally **unrolling** it into a fold
`(((x₀⊕x₁)⊕x₂)…)`, and two different reduction orders are provably equal by
plain real arithmetic. **We skip the expensive machinery entirely.** This is
what makes phase-2 reductions tractable (see `phase-2.md`).

### 5. Refinement relation

- **mlir-tv:** a proper 3-query refinement `src ⊑ tgt` (UB, return values,
  memory), NaN-aware float equality, target may be *more* defined. Directional.
- **ours:** symmetric equivalence (`distinct` on final array state) plus a
  separate addressing-identity scope.

Their directional refinement is more correct for validating real compiler
passes (a pass may legitimately *reduce* UB). Ours assumes both sides equally
defined — fine within our reject-everything-unsafe contract, but a limitation to
note.

## What we can leverage — ranked

1. **Reductions (highest value).** Adopt the *idea* of reduction-as-fold, but —
   thanks to ideal reals — **without** the hash/multiset machinery. Unroll
   `tt.reduce` over the static extent. Lifts our elementwise-only ceiling.
   *(Phase-2 first milestone.)*
2. **A precision ladder for floats.** Their `DEFAULT → USE_SUM_ONLY →
   UNROLL_TO_ADD` / `FULLY_ABS → SUM_MUL` refinement is a template for an
   optional stricter mode later, without abandoning reals as the fast default.
3. **Shift + UB-flag recipe.** Their `wellDefined(op, cond)` per-op boolean
   conjunction is a clean, lightweight way to encode shifts (and eventually
   div/rem) soundly — replacing our "reject" with "encode + UB side-condition,"
   no full poison lattice needed.
4. **NaN-aware + directional refinement**, if/when we move toward
   IEEE-faithfulness.
5. **Validation-methodology reuse.** They ran against MLIR's own 2,467 unit-test
   function pairs and found real spec bugs. Triton's lowering-pass test suite is
   the analogous target for us.

**What NOT to copy:** their abstract minimal-width BV float encoding itself, *as
long as* our target transforms are algebraic reorderings — ideal reals give us
associativity for free and are simpler and strictly faster for that class. Their
encoding only pays off when you must reason about NaN/Inf/rounding/signed-zero,
which we currently reject by design.

## Reference

- Bang, Nam, Chun, Jhoo, Lee. *SMT-Based Translation Validation for Machine
  Learning Compiler.* CAV 2022, LNCS 13372, pp. 386–407.
  <https://doi.org/10.1007/978-3-031-13188-2_19>
- Code: <https://github.com/aqjune/mlir-tv> (analyzed at local checkout
  `~/dev/mlir-tv`). Key files: `src/abstractops.{h,cpp}` (FP abstraction),
  `src/encode.cpp` (per-op encoding), `src/memory.{h,cpp}` (block model),
  `src/vcgen.cpp` (refinement + abstraction-refinement loop).
