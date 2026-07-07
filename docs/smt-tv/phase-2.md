# Phase 2 — Absorb / leverage mlir-tv

**Goal.** Systematically pull the useful ideas from mlir-tv (Bang et al., CAV
2022; see `mlir-tv-comparison.md`) into our Triton→SMT validator, expanding what
we can validate *without* giving up the "sound-by-rejection, ideal-real,
fixed-width-integer" posture from phase 1 (`phase-1.md`).

## Ranked backlog

1. **Reductions** — `tt.reduce` with add/max/min combiners (later `tt.dot`).
   *Highest value: the single biggest capability mlir-tv has that we lack.*
   **← first milestone (below).**
2. **Shifts + integer `ext`/`trunc`** via a per-op **well-definedness flag**
   (adopt mlir-tv's `wellDefined(op, cond)` — UB as a boolean conjunct — instead
   of rejecting outright). Unblocks the common quantization pack/unpack idiom.
3. **Directional refinement** (`src ⊑ tgt`) + NaN-aware compare — only when/if we
   move toward IEEE-faithfulness (a separate, larger direction).
4. **Validation-methodology reuse** — run the validator across Triton's own
   lowering-pass tests, mirroring how mlir-tv validated MLIR's 2,467 unit-test
   pairs and found real spec bugs.

## First milestone — 1-D `tt.reduce` (add/max/min)

### Why reductions are tractable for us

mlir-tv's hardest contribution — the hash-based multiset encoding for reduction
associativity — exists *only because its floats are abstract uninterpreted
functions*. **We model floats as ideal reals**, so `+`/`*` are already
associative and commutative. A reduction over a **static** block extent `N` can
therefore be encoded by literally unrolling it into a fold
`(((x₀ ⊕ x₁) ⊕ x₂) … ⊕ x_{N-1})`, and two different fold orders are discharged by
Z3 as plain real arithmetic. No multiset / hash / permutation machinery.

### The architectural forcing function

Phase 1 represents an entire `tensor<Nxf32>` by **one** SMT term: its value at
the single symbolic lane `i`. A reduction *collapses* the reduced axis — the
output depends on **all N** lanes — so one term cannot express the fold. The
fix: when a `tt.reduce` is present, materialize the reduced tensor as **N
distinct SMT terms** (one array-select per lane), then fold.

`tt.reduce` on a rank-1, axis-0 input yields a **scalar** per program
(`tl.sum(x)` → `out[pid] = ⊕_k x[k]`), so the store stays scalar and phase 1's
scope-0 identity check and writer decomposition (`pid = i`, `block = 1`) are
reused unchanged.

### Encoding changes (`lib/Conversion/TritonToSMT/TritonToSMTPass.cpp`)

- Extend `EncodedValue` with `SmallVector<Value> lanes` + `isVector()`. The
  existing scalar path is preserved byte-for-byte (keeps all phase-1 tests
  green); a parallel **vector path** is entered only when the block contains a
  `triton::ReduceOp`.
- Factor each op's scalar core out of the `TypeSwitch` into a reusable helper so
  the vector path can apply it per lane (avoids duplicating ~20 cases and
  generalizes to `tt.dot` later).
- Vector rules: `make_range` → `lanes[k] = start+k`; `splat` → broadcast;
  `addptr(vec,vec)` → per-lane offset; `load(vec)` → per-lane `array.select`;
  elementwise arith/math → per-lane reuse. `ReduceOp` → fold lanes with the
  combiner into a scalar (reuse `realBin` for add; the `ite(RealCmp ge/le,…)`
  idiom for max/min); `ReduceReturnOp` → no-op. Post-reduce scalar math +
  `tt.store` reuse the existing scalar switch. This supports multiple reduces and
  post-reduce math (`sum(a)+sum(b)`, `2*sum(x)`).
- Classify the combiner with `ReduceOp::getSingleCombiner()`
  (`lib/Dialect/Triton/IR/Ops.cpp`) — the single 2-in/1-out op of a canonical
  `{combine; reduce.return}` region (handling the commutatively-reversed mapping),
  else `nullptr`.

### Sound-by-rejection — reject loudly (never a silent EQUIVALENT)

- `getSingleCombiner() == nullptr` (multi-statement region, non-canonical, or the
  fused arg-max/min 2-operand form).
- combiner not in {`addf`, `maxnumf`, `maximumf`, `minnumf`, `minimumf`}
  (optionally `addi`).
- `axis != 0`, input rank ≠ 1, non-static or mismatched extent `N`.
- **masked load feeding a reduce** — masked-out lanes need the combiner's
  identity as `other`: `0.0` works for add but max/min would need ±inf, which is
  non-representable as a real (and already rejected). *Follow-up:* allow masks
  only when the combiner is add and `other == 0.0`.
- a tensor store coexisting with a reduce; `tt.scan` / `tt.dot` / rank>1
  broadcast. All existing per-lane rejections still apply.

### Why `tt.dot` is deferred

`tt.dot` produces a 2-D accumulator (`out[m,n] = ⊕_k a[m,k]·b[k,n]`), needing
full M·N lane materialization *and* a block-store contract — a separate, larger
increment. Once the lane-vector machinery exists, dot is the same real-arithmetic
fold; only the 2-D store is new.

### Tests (mirror `python/test/unit/tools/test_smt_equivalence.py`)

- **EQUIVALENT:** `sum(a+b) ≡ sum(a)+sum(b)`; `2*sum(x) ≡ sum(x+x)`; left-fold ≡
  tree reassociation (inline TTIR); `max(max(a),max(b)) ≡ max(concat)`.
- **NOT_EQUIVALENT:** `sum` vs `max`; `sum(x)` vs `sum(x)+1`; N=4 vs a 3-lane
  dropped-term reduce.
- **UNSUPPORTED:** `axis=1`/rank>1; fused arg-max reduce; masked-reduce;
  `tt.scan`; `tt.dot`; differing `N` between src and tgt.

## Working method

Implement an increment → run the Codex reviewer (`scripts/smt-review.sh`, or
`/codex:rescue` for a targeted soundness pass) → fix findings → repeat until
`VERDICT: APPROVED`. Pin each fix with an adversarial regression test, matching
the phase-1 hardening pattern.
