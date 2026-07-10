# Phase 3 — a real memory model (absorb mlir-tv's block model)

> **Status: COMPLETE — Codex review APPROVED (round 2)**; the `memory-model`
> pass option + driver retry routing landed, see `phase-3-review.md`.
> Sections below are the original plan; every place the implementation
> deliberately diverges is recorded in
> [Implementation notes](#implementation-notes-resolved-decisions) at the end
> — read them together.

**Goal.** Retire the single restriction that rejects nearly every real Triton
kernel: **identity-only addressing**. Replace the phase-1/2 "one array per
pointer arg, scope-0 proves `pid = i`" contract with a proper **block memory
model** absorbed from mlir-tv (Bang et al., CAV 2022; `mlir-tv-comparison.md`
§3), so we can validate **strided, multi-dimensional, masked** addressing —
2-D elementwise, broadcast, transpose, boundary tiles — while keeping the
sound-by-rejection, ideal-real, fixed-width-integer posture.

This phase is **memory only**. `tt.dot` and `scf.for` remain deferred to phase 4
(they sit on top of the N-D lane materialization this phase introduces, but add
a matmul fold and control-flow unrolling respectively — separate increments).

## Why the memory model is the real ceiling

Phase 2 already broke the pure-elementwise ceiling (reductions), yet the tool
still rejects almost every real kernel — not because of the ops, but because of
how they *address*. Today: one `Array<bv32 → sort(pointee)>` per pointer arg,
`fresh` flag rejects chained `addptr`, and **scope 0 proves the store index is
the identity** (`pid = i`, block = 1). Real Triton addresses affinely:

```
offs_m = pid_m*BM + tl.arange(0, BM)          # [BM]
offs_n = pid_n*BN + tl.arange(0, BN)          # [BN]
p = X + offs_m[:, None]*stride_xm + offs_n[None, :]*stride_xn   # [BM, BN]
x = tl.load(p, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
```

Identity addressing rejects all of it. Lifting that restriction — not adding
ops — is what makes the validator *useful*.

## What we absorb from mlir-tv, and what we skip

mlir-tv's memory is a full Alive2-style block model: per-element-type block
families that never alias across types, block ids, liveness / writability /
initialized arrays, affine layout maps with inverse functions, and **UB on
OOB / uninitialized / dead access**. We take the subset Triton needs and skip
the rest — recording each skip so it reads as a deliberate modeling choice, not
a gap.

**Absorb:**

- **Block ids + typed non-aliasing.** Each `!tt.ptr<T>` argument becomes a
  distinct *block*: `(id, elementType, Array<bv32 → sort(T)>, size, init)`.
  Distinct args ⇒ distinct ids ⇒ never alias — this **formalizes** phase-1's
  "pointer args assumed disjoint" (Triton's `restrict` semantics) instead of
  leaving it an implicit assumption. Blocks of different element type never
  alias (mlir-tv's typed families); this is consistent with our existing
  byte-backed-`i1` handling.
- **Affine layout maps.** Per lane, decompose the `tt.addptr` offset into a
  symbolic **affine** `bv32` expression over `(pid, arg-scalars, lane indices)`:
  `off = Σ stride_d · idx_d + base`. This is the layout map. The store-index
  equivalence that scope 0 proved by asserting identity is now proved
  *symbolically over the affine offset* — the restriction disappears.
- **OOB / uninitialized / dead access = UB.** An active-mask access with
  `off ≥u size ∨ off <u 0`, or a *read of an uninitialized* offset, contributes
  to the function's UB predicate. This plugs **directly** into phase-2's exact
  poison / "UB = poison reaching memory" framework (milestone 3): OOB and
  uninit are just two more UB sources folded into scope-1 UB-domain equality. A
  masked-out lane performs no access and takes `other` (no UB) — the same
  exactness that pins `test_dead_vs_stored_poison_not_equivalent`.

**Skip (Triton doesn't need it — reject if it ever appears):**

- **Dynamic alloc / free, local vs non-local blocks, liveness.** TTIR kernels
  don't `malloc`/`free`; every pointer-arg block is live for the whole kernel.
  No local blocks ⇒ no local/non-local refinement split.
- **Writability tracking**, unless a `tt.ptr` carries a read-only/`const`
  attribute — then a store to it is rejected up front (not modeled as UB).
- **Affine *inverse* functions.** mlir-tv needs them for its lambda-tensor
  abstraction; we materialize lanes concretely, so we only need the forward
  map. (This is the per-element vs lambda divergence from `mlir-tv-comparison.md`
  §on tensor encoding.)

## The architectural forcing function — N-D lane materialization

Phase 1 represents `tensor<Nxf32>` by one term at symbolic lane `i`. Phase 2's
reduction path added a 1-D `SmallVector<Value> lanes`. Strided N-D addressing
needs each lane to carry a **concrete symbolic offset** (so we can decompose it
affinely and bounds-check it), so:

- Generalize `EncodedValue`'s `lanes` to an N-D materialization with a static
  shape (`SmallVector<int64_t> shape`, row-major lane order). Validate through
  **rank ≤ 2** this phase (matmul tiles, layernorm/softmax rows); reject higher
  rank. The 1-D reduction path is the rank-1 special case. Keep the lane
  structures **rank-generic** so the rank ceiling is a *single guard*, not an
  assumption threaded through the code — lifting it in phase 4 is then one
  check plus tests, not a rewrite.
- Each pointer-typed lane carries `(blockId, offsetExpr)` where `offsetExpr` is
  the affine `bv32` value, not just a select result. `tt.addptr(base, off)`
  adds the per-lane offset to the base's affine expr; `tt.make_range`,
  `tt.splat`, `tt.expand_dims`/`[:, None]`, `tt.broadcast`, `arith.muli` (by a
  loop-invariant stride), `arith.addi` build the affine expr. `tt.load`/
  `tt.store` consume `(blockId, offsetExpr)` per lane.
- Preserve the scalar and 1-D reduction paths byte-for-byte (keeps phase-1/2
  tests green); the N-D path is entered only when a load/store addresses a
  rank ≥ 2 tensor or a non-identity affine offset is detected.

## Scope contract change (touch the pass AND the driver together)

CLAUDE.md's invariant: the pass emits three scopes in fixed order (0 addressing,
1 UB-domain, 2 equivalence) and the driver decodes positionally. Under the block
model, **addressing correctness stops being a separate identity proof** and
becomes part of the memory-state refinement:

- **Scope 0 (addressing) is repurposed, not removed** — keep the three-scope
  positional contract intact *for this phase*. Its new job: assert **addressing
  well-formedness / block-decomposition consistency** — every access decomposed
  to a known block with an affine offset, and any *provable* intra-block
  store/store or load/store overlap between src and tgt that the refinement
  query relies on is consistent. When the affine decomposition is total and
  blocks are disjoint by construction, scope 0 is trivially unsat (like scope 1
  with no WD conditions today). Keeping the slot avoids entangling a
  driver-contract break with the memory-model rewrite — two independent sources
  of regression that should not land together.
- **Scope 1 (UB-domain)** gains OOB + uninitialized-read UB sources, folded into
  the existing per-value poison predicate. `UB_src ≠ UB_tgt` unsat ⇒ identical
  UB domains (unchanged query shape).
- **Scope 2 (equivalence)** generalizes from `distinct` on a single final array
  to **per-block, per-in-bounds-offset** final-state equality, on inputs where
  neither side is UB. Input blocks start fully initialized with symbolic values;
  output blocks start uninitialized (reads before writes ⇒ UB via scope 1).

**Open question to resolve early (do not defer).** Whether scope 0 carries real
content or is a vestigial always-unsat placeholder hinges on where the one
genuinely query-shaped non-refinement fact lives: an **intra-kernel write-write
hazard** — two active lanes of the *same* kernel writing the same
`(block, offset)` with different values (a data race). The working decision is
that a race is **UB and belongs in scope 1**, alongside OOB and uninit — which
leaves scope 0 as a documented trivially-unsat slot. Confirm this against the
first multi-store 2-D kernel: if races cannot be expressed cleanly in the UB
predicate, they become scope 0's content instead. Either way, resolve it in the
first increment; do not carry an undecided placeholder through the review loop
(phase-2 milestone 4 showed harness/measurement ambiguity draws as many rounds
as encoder logic).

Driver: `EquivalenceResult` still requires exactly three results; document that
scope 0's *meaning* changed (comment + `docs/smt-tv/README.md`). Collapsing to
two scopes (if scope 0 lands as vestigial) is a **separate, explicit** phase-4
change — never silently.

## Sound-by-rejection contract (reject loudly — never a silent EQUIVALENT)

- **Non-affine / data-dependent offsets** — gather/scatter, indirect
  `tt.load` (pointer loaded from memory), an `addptr` offset that isn't an
  affine function of `(pid, arg-scalars, lane indices)`.
- **Dynamic strides** that aren't loop-invariant kernel scalars (a stride read
  from memory, or computed from a data-dependent value).
- **Chained `addptr`** that can't be folded into a single affine expression
  (the `fresh` flag's job, generalized).
- **Rank > 2**, non-static tensor extent, mismatched shapes within a function.
- **Extent-product blow-up** — a *distinct* guard from the rank cap, targeting a
  different failure mode. Rank bounds *implementation complexity* (broadcast /
  transpose logic); the extent product `Π extents` bounds *encoder + solver
  tractability*, since N-D materialization concretely unrolls every lane
  (`64×64` = 4096 terms; `1024×1024` = a million). Reject when `Π extents`
  exceeds a threshold **regardless of rank** — a rank-2 `1024×1024` block must
  be rejected loudly (→ UNSUPPORTED with a reason), never left to hang or OOM
  the encoder. "No silent caps."
- **Pointer aliasing we can't discharge** — two blocks we cannot prove disjoint
  (should not arise for distinct args, but reject rather than assume if a
  bitcast/reinterpret muddies block identity).
- **Stores to a read-only pointer**, atomics (`tt.atomic_*`), `tt.dot`,
  `scf.*`, `tt.scan`, tensor-of-pointers we can't decompose per lane.

Every rejection gets an adversarial test; every soundness-relevant *acceptance*
(e.g. masked-OOB-guard ≡ in-bounds) gets a regression test.

## Tests (mirror `python/test/unit/tools/test_smt_equivalence.py`)

- **EQUIVALENT:** 2-D contiguous vs `stride`-parametrized addressing that
  computes the same offsets; broadcast-add `x[:, None] + y[None, :]` order
  invariance; a masked boundary tile ≡ itself under `canonicalize`; row-wise
  reduction addressing (reuses phase-2 fold) over a 2-D load.
- **NOT_EQUIVALENT:** swapped strides (`stride_m` ↔ `stride_n`, i.e. a
  transpose) storing to the same block; off-by-one base offset; a mask with the
  wrong bound (`< M` vs `<= M`) that changes the stored set.
- **UB / refinement:** an **unmasked** tile that can exceed `size` is UB where a
  correctly-masked version is defined ⇒ NOT_EQUIVALENT (`ub_status = "sat"`);
  two kernels masked to the *same* bound have identical UB domains ⇒ EQUIVALENT
  even when both would OOB unmasked; reading an output block before writing it
  is UB.
- **UNSUPPORTED:** gather (`tl.load(ptr_tensor)`), data-dependent stride, rank-3
  block, an oversized block whose extent product exceeds the tractability
  threshold, atomic add, store to a `const` pointer, `tt.dot`.

## Why `tt.dot` / `scf.for` stay in phase 4

Once N-D lane materialization + block memory exist, `tt.dot` is the same
real-arithmetic fold as a reduction (`out[m,n] = ⊕_k a[m,k]·b[k,n]`) plus a 2-D
block store — mechanically small but a distinct increment. `scf.for` needs a
**static trip-count** contract (bounded unrolling; reject dynamic/data-dependent
bounds) and loop-carried accumulator threading — the hardest piece, and the one
most likely to need its own review cycle. Keeping phase 3 to the memory model
keeps each Codex review tractable.

## Working method

Same as phase 2: implement an increment → run the Codex reviewer via the plugin
(`/codex:rescue` with `docs/smt-tv/review-prompt.md`, naming "phase 3") → fix
findings → repeat until `VERDICT: APPROVED`, pinning each fix with an
adversarial regression test. Expect the block model and the scope-0 repurpose to
draw heavy adversarial attention (aliasing, uninit-read UB exactness, affine
decomposition soundness) — budget review rounds accordingly.

## Implementation notes (resolved decisions)

What landed, and where it deliberately diverges from the plan above. Every
divergence is a modeling decision, not a gap.

- **Two encodings, driver-routed.** The identity-addressing (phase-1/2)
  encoding is preserved byte-for-byte and tried first. When the legacy pass
  rejects the kernels, or accepts them but scope 0 disproves identity
  addressing, the driver retries once with
  `--convert-triton-to-smt=memory-model=true`; `EquivalenceResult.memory_model`
  records which encoding produced the verdict. An explicit `block_size` pins
  the legacy contract and disables the retry. Both encodings emit the same
  three scopes in the same order, so the positional driver contract is
  untouched.

- **Scope 0 resolved: vestigial always-unsat placeholder.** The open question
  is settled the way the plan leaned: an intra-store write-write race is UB
  and lives in scope 1. Under the block model every access either decomposes
  to a known block with a modeled per-lane offset or is rejected, and distinct
  blocks are disjoint SMT arrays by construction — so scope 0 asserts a
  constant `false` (trivially unsat). Collapsing to two scopes remains an
  explicit phase-4 change.

- **64-bit block offsets; chained `tt.addptr` is supported, not rejected.**
  The frontend lowers all multi-term addressing (`X + a[:, None] + b[None,
  :]`) as *chained* addptr, so rejecting chains would reject nearly every real
  2-D kernel. The phase-2 chain rejection existed because a 32-bit offset
  accumulation wraps where real 64-bit pointer arithmetic does not; the block
  model removes the unsoundness instead of the feature: every `tt.addptr`
  sign-extends its i32 offset into a **bv64 block offset** and accumulates
  there, exactly matching real pointer arithmetic. The old wrap trap (two
  chained `2^31-1`-ish offsets summing to `idx + 2^32`) is pinned as a
  NOT_EQUIVALENT regression (`test_chained_addptr_wrap_not_equivalent`).

- **OOB is UB, via signed 64-bit bounds.** Each pointer argument's block is an
  `Array<bv64 → sort(T)>` plus a symbolic `bv64` element count asserted
  non-negative in each scope's prologue. An access by an *active* lane at
  offset `off` is UB iff `off <s 0 ∨ off >=s size` — with sign-extended exact
  offsets this is precisely the out-of-allocation condition. Masked-out lanes
  perform no access (no UB) and loads take `other` (still required — a masked
  load without `other` stays rejected).

- **Uninitialized-read UB is deliberately NOT absorbed.** All blocks start
  fully initialized with symbolic contents. Two reasons. (1) Fidelity: CUDA
  global memory pointed to by a kernel argument always holds *some* value;
  reading it is defined-but-unknown, unlike mlir-tv's memrefs. (2) Soundness
  of the alternative: TTIR cannot distinguish input from output pointers, and
  declaring written-blocks-start-uninitialized would make every in-place
  read-modify-write kernel UB-everywhere — two such kernels would then have
  equal (total) UB domains and a vacuous equivalence scope, i.e. a false
  EQUIVALENT for kernels computing different things. The planned
  "reading an output block before writing it is UB" test is replaced by
  sequential-semantics tests: loads read the *current* block contents, so a
  load after a store to the same block sees the stored value
  (`test_load_after_store_sees_stored_value`), and multiple stores fold in
  program order (later store wins; no single-store restriction in this mode).

- **Races: intra-store, different-value overlap only.** Two active lanes of
  one `tt.store` writing *different* values to the same `(block, offset)` is
  UB; equal-value overlaps are deterministic regardless of lane order and are
  not UB. Overlaps *across* two stores are program-order-defined (later wins),
  not races. Race terms are O(lanes²) per store; pairs whose offsets are
  structurally must-differ (`base + c1` vs `base + c2`, looking through the
  sign-extension) or whose value terms coincide are elided — eliding only
  provably-false terms, so always sound. Pinned by
  `test_racy_store_vs_race_free_not_equivalent` /
  `test_same_race_both_sides_equivalent`.

- **"Affine" implemented as memory-independence, which subsumes it.** The
  plan's affine decomposition existed to reject data-dependent addressing;
  with concretely materialized lanes, lane indices are constants and any
  *memory-independent* offset expression over `(pid, arg scalars, lane
  constants)` built from the supported ops is encoded exactly — affine or not.
  The sound boundary is data-dependence, enforced as a taint: load results are
  marked memory-derived, the mark propagates through every op, and `tt.addptr`
  rejects a tainted offset (gather/scatter, strides loaded from memory,
  indirect pointers). Masks and stored values MAY be memory-dependent: the
  initial block contents are shared symbolic inputs, so predicates over them
  stay exact. (A 1-D identity-addressed gather remains accepted by the
  *legacy* path, as in phases 1–2, where it is an exact nested select with no
  OOB semantics.)

- **Per-program refinement.** Memory-model scopes declare `pid0..pid2` as free
  bv32 symbols shared by both kernels (`tt.get_program_id` axes 0–2): proving
  every program instance equivalent proves the launch equivalent. As in phases
  1–2, cross-program interference (grids whose programs race with each other)
  is outside the model; both kernels are encoded under the same
  interference-free assumption.

- **Tractability bounds.** Rank ≤ 2 (single guard, rank-generic lane
  structures underneath) and an extent-product cap (`max-lanes`, default 256)
  — a distinct guard, as planned. Both reject loudly with the offending
  numbers in the message; nothing is silently truncated.

- **Read-only pointers.** TTIR carries no read-only/`const` attribute on
  `!tt.ptr` arguments today, so there is nothing to check; the planned
  store-to-const rejection is moot until such an attribute exists.

- **Newly validated (previously UNSUPPORTED).** Non-identity addressing
  (shifted stores), rank-2 tensors and axis reductions over them, masked
  loads feeding reductions, multiple stores per kernel, chained addptr,
  kernels with extraneous `tt.make_range` extents. Each flip is pinned by an
  updated regression test asserting the new verdict AND `memory_model=True`
  (proving the legacy guard still rejected first).
