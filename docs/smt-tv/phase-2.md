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
3. **Directional refinement** (`src ⊑ tgt`). Treat equivalence as bidirectional refinement. 
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
  `tt.scan`; `tt.dot`; a combiner region with extra (e.g. side-effecting) ops;
  a `tt.reshape allow_reorder` whose result is used by anything other than the
  reduce.

A *within-function* extent mismatch (a `tt.make_range` whose extent differs from
the reduced `N`) is rejected. Differing `N` *between* src and tgt is not a
soundness hole and is **not** rejected: the two functions select different input
addresses, so the equivalence query is sound and simply returns NOT_EQUIVALENT.

## Second milestone — shifts + integer ext/trunc via well-definedness (DONE)

Implemented (commits `728ad6b0d`..`973531529`, review-approved after 4 Codex
rounds; report in `phase-2-review.md`):

- `shli`/`shrui`/`shrsi` → total `smt.bv.shl/lshr/ashr` plus a collected
  well-definedness condition `amt <u width` per shift (per lane in reduction
  context). Poison flags (`nsw`/`nuw`/`exact`) are rejected.
- The pass now always emits **three** solver scopes in fixed order:
  0 addressing, 1 well-definedness (assert ¬(∧ conditions); trivially unsat
  when none), 2 equivalence. The driver requires exactly three results and
  never reports EQUIVALENT unless scopes 0 and 1 are both unsat
  (`EquivalenceResult.wd_status`).
- A constant or masked shift amount (`x & 31`) proves WD for all inputs and
  validates; an unbounded amount was UNSUPPORTED — even for syntactically
  identical kernels. *(Superseded by milestone 3: refinement semantics now
  accept identical UB domains.)*
- `extui` generalized beyond `i1` (zext via concat), `extsi` added (sext via
  ite+concat; `i1` → 0/−1), `trunci` added (extract; to-`i1` takes the low bit
  into `smt.bool`).
- Review hardening: the Bool→bit-vector *store* coercion was removed (NVIDIA
  sign-extends `i1` stores to `0xff`, so "true as 0x01" into a byte-observable
  buffer could prove a false EQUIVALENT); verifier-valid `vector<...>`,
  `index`, and `i0` types are rejected up front instead of crashing; an
  unrepresentable driver timeout degrades to UNKNOWN.

## Third milestone — equivalence as bidirectional refinement (DONE)

Backlog item 3, implemented on top of milestone 2's scope machinery:

- **Exact per-value poison tracking.** `EncodedValue` carries an optional
  poison predicate (and per-lane predicates in reduction mode). Ops propagate
  poison exactly: any-operand-poison for all pure elementwise ops except
  `select` (`p(c) ∨ ite(c, p(t), p(f))`); shifts add `amt ≥u width`; loaded
  memory values are never poison; a masked-out lane takes `other`'s poison.
- **UB = poison reaching memory.** A load/store contributes
  `p(mask) ∨ (mask ∧ (p(addr) ∨ p(value)))` to the function's UB predicate.
  Masked-out poison is *not* UB — this exactness is load-bearing: a coarse
  global well-definedness conjunction cannot distinguish dead poison from
  stored poison and would report a false EQUIVALENT for the dead-vs-stored
  pair (pinned by `test_dead_vs_stored_poison_not_equivalent`).
- **Query.** Scope 1 now asserts `UB_src ≠ UB_tgt` (unsat ⇒ identical UB
  domains); scope 2 asserts outputs differ on an input where neither side is
  UB. Together: bidirectional refinement. Identical-UB kernels (same
  unbounded shift) are EQUIVALENT — matching mlir-tv — and a kernel that is
  UB where the other is defined is NOT_EQUIVALENT
  (`EquivalenceResult.ub_status = "sat"`). Kernels without poison sources
  reduce to the previous query exactly.

## Fourth milestone — validation-methodology reuse (DONE)

Backlog item 4: `python/triton/tools/smt_validate_passes.py` sweeps a TTIR
corpus, runs each function through semantics-preserving passes
(`canonicalize`, `cse`, `triton-combine`, `triton-reorder-broadcast`), and
checks original vs transformed with the equivalence driver — mirroring how
mlir-tv validated MLIR's unit-test pairs. Out-of-contract functions are
counted as UNSUPPORTED (never silently skipped); any NOT_EQUIVALENT fails the
sweep (a pass bug or a checker bug — both reportable).

First sweep results (2026-07-09):

| corpus | funcs×passes | EQUIVALENT | UNSUPPORTED | PASS_ERROR | PARSE_SKIP | NOT_EQUIVALENT |
|---|---|---|---|---|---|---|
| `test/Triton` | 285 | 0 | 260 | 20 | 5 | 0 |
| `test/Conversion` | 330 | 32 | 88 | 196 | 14 | 0 |

`test/Triton` is dominated by out-of-contract constructs (`scf.for`,
non-void lit-snippet functions, `tt.dot`, tensor descriptors), matching
mlir-tv's partial-coverage experience; the in-contract kernels in
`test/Conversion/triton_to_smt*.mlir` all validate EQUIVALENT under every
pass. No pass bug surfaced. Growing the EQUIVALENT column is exactly the
phase-3 coverage agenda (loops, `tt.dot`, stride-general addressing).

## Working method

Implement an increment → run the Codex reviewer via the plugin
(`/codex:rescue` with the brief in `docs/smt-tv/review-prompt.md`) → fix
findings → repeat until `VERDICT: APPROVED`. Pin each fix with an adversarial
regression test, matching the phase-1 hardening pattern.
