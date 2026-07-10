# SMT-based translation validation for Triton

An operation-wise SMT encoding of Triton IR, added as a standalone lowering pass
parallel to the existing ones, that checks functional equivalence of two Triton
kernels. It automates what `formalize-demo` did by hand. Floating-point is
modeled as **ideal reals**; integers as **faithful fixed-width bit-vectors**.

## Build dependency

The pass depends on the SMT `Real` sort added to MLIR by
`scripts/patches/mlir-smt-real.patch`. That patch is applied by
`scripts/build-llvm-project.sh` (the from-source LLVM path this project uses via
`LLVM_SYSPATH`); the default prebuilt-LLVM download path does **not** carry it
and will not compile the pass. This is an accepted build-integration dependency
(the tree pins the patched from-source LLVM), not a runtime soundness property.

## Components

- **Part A — MLIR `smt` dialect `Real` sort** (in the vendored LLVM). Adds
  `RealType` and `smt.real.{constant,add,mul,sub,div,neg,cmp}` plus SMT-LIB
  export. Carried as a tracked patch `scripts/patches/mlir-smt-real.patch`,
  applied on top of the pinned LLVM by `scripts/build-llvm-project.sh`.
- **Part B — `lib/Conversion/TritonToSMT`** (`triton-opt --convert-triton-to-smt`).
  Encodes two `tt.func`s into the `smt` dialect and emits two solver scopes.
  *(Superseded in phase 2: the pass now always emits THREE scopes —
  0 addressing, 1 UB-domain equality, 2 equivalence — see `phase-2.md`
  milestone 3.)*
- **Part C — driver `python/triton/tools/smt_equivalence.py`**. Compiles kernels
  to TTIR (frontend only, no GPU/`ptxas`), merges them, runs the pass →
  `mlir-translate --export-smtlib` → Z3, and returns a verdict.

## Semantic treatments (how each construct is modeled)

**Value domains**

- **Floating-point → ideal reals** (`!smt.real`): exact real arithmetic;
  `math.fma(a,b,c)` = `a*b+c`. IEEE-754 is *not* modeled — no rounding, NaN,
  ±inf, signed zero, or subnormals. FP constants must be finite and are emitted
  as exact dyadic rationals.
- **Integers → fixed-width bit-vectors** (`!smt.bv<N>` at each value's true
  width): wrapping arithmetic; `arith.cmpi` lowers to signed/unsigned `bv.cmp`
  per predicate (`slt`≠`ult`); `subi` = `x + bvneg(y)`. Faithful to machine
  i8/i16/i32/… overflow and signedness.
- **`i1` → SMT bool**: `i1` constants and comparison results are booleans.
- **Pointers → (memory array, bit-vector offset)**: arrays are
  `Array<bv32 → sort(pointee)>`, the range sort being `Real` (float pointee),
  `bv<N>` (integer), or `bool` (`i1`). A masked load is
  `ite(mask, select(arr, addr), other)`; addresses are 32-bit.

**Operations encoded** (one rule per op): `tt.func`/`tt.return`,
`tt.get_program_id` (axis 0), `arith.constant` (scalar/splat int/float/i1),
`tt.make_range`, `tt.splat`, `arith.addi/subi/muli/andi/ori/xori`,
`arith.cmpi`, `tt.addptr`, `tt.load`,
`arith.mulf/addf/subf/negf/divf`, `math.fma`, `math.absf`, `arith.cmpf`,
`arith.select`, `arith.maxnumf/maximumf/minnumf/minimumf`, `tt.store`;
bool-to-storage idioms: `arith.extui`/`arith.uitofp` from `i1`, and
pointer-to-pointer `tt.bitcast` with a bit-vector→Bool coercion at the *store*
boundary (byte-backed `i1` buffers: 0 = false, nonzero = true). The reverse
direction — storing a raw `i1` value into a byte-observable `iN` buffer — is
rejected: real lowerings disagree on the written byte (NVIDIA sign-extends to
`0xff`). Reinterpreting *loads* through a bitcast are likewise rejected.
The bitwise ops `andi/ori/xori` dispatch on the operand sort: bit-vector ops
for `iN`, boolean `smt.and/or/xor` for `i1` masks. `arith.divf` uses SMT-LIB
real division, which is total but unspecified at zero denominators — both
functions see the same division function, so equivalence stays sound (and
`a/b` vs `a*(1/b)` is correctly *not* equivalent).

**Refinement query** — two `smt.solver` scopes, exported to SMT-LIB and solved
by Z3, over a symbolic output index `i`:

- **Scope 0 — addressing identity.** Writer decomposition is
  `pid = i / block`, `lane = i % block` (`bvudiv`/`bvurem`). Asserts
  `storeOffset(pid,lane) ≠ i` for either function. `unsat` ⇒ identity addressing
  is proven; `sat` ⇒ non-identity ⇒ verdict **UNSUPPORTED**.
- **Scope 1 — equivalence.** For every output array, computes the final state
  `ite(mask, storedValue, initialArray[i])` and asserts the two functions can
  differ. `unsat` ⇒ **EQUIVALENT**; `sat` ⇒ **NOT_EQUIVALENT** (Z3 model = a
  concrete counterexample).

**Verdicts:** `EQUIVALENT` / `NOT_EQUIVALENT` / `UNSUPPORTED` / `UNKNOWN`
(solver `unknown`/timeout).

## Boundaries

**Soundness posture.** The checker **never reports `EQUIVALENT` for inputs
outside its validated contract** — it rejects them (`UNSUPPORTED`) or fails the
pass with a diagnostic. It does not aim to *prove non-equivalence* for
unsupported inputs; it declines to answer.

**Accepted contract**

- straight-line, single-block `tt.func` (no control flow);
- exactly one `tt.store`, to a single output pointer argument;
- canonical identity addressing, verified by scope 0 (store offset == index);
- 32-bit pointer offsets / indexing; `program_id` axis 0 only;
- source and target share an identical full signature;
- finite floating-point constants.

**Rejected loudly** (pass error or `UNSUPPORTED` — never a silent "equivalent")

- non-identity / shifted / transposed / gathered addressing;
- more than one store, or multiple output buffers;
- functions with non-void results (return values are not compared);
- external (empty-body) or multi-block functions; dynamic-shaped stores, or
  static store shapes whose element count overflows `int64`;
- volatile loads (observable memory accesses are not modeled);
- chained `tt.addptr` (32-bit offset accumulation can diverge from real 64-bit
  pointer arithmetic; a single in-range addptr is fine);
- raw loads that reinterpret a buffer's element sort through a pointer bitcast
  (the `i1`↔`i8` idiom is sound only on the store side);
- modules containing any op other than `tt.func` (e.g. an injected `smt.solver`
  that could otherwise forge a solver result);
- `i1` integer arithmetic (`addi`/`subi`/`muli` on booleans);
- arith ops carrying `nsw`/`nuw`/`nneg` poison flags;
- pointer-valued `arith.select`;
- pointer bitcasts that change the pointee representation or byte size — only a
  same-byte-size integer reinterpretation (the `i1`↔`i8` byte-backed-bool idiom)
  is kept; float-format and element-size changes are rejected;
- integer division / remainder (`divsi`/`divui`/`remsi`/`remui`) — their UB
  (division by zero, signed overflow) is not modeled. (Shifts were rejected in
  phase 1 for the same reason; phase-2 milestone 2 now models them with a
  solver-discharged well-definedness condition — see `phase-2.md`.)
  so they are rejected rather than encoded as total bit-vector ops;
- masked loads without a fallback `other` (masked-out lanes are undefined);
- `tt.make_range` whose extent disagrees with the store's lane count;
- signature mismatch; unsupported ops; float constants that are non-finite or
  wider than binary64;
- width- or domain-changing casts (`extsi`/`trunci`/`extf`/`truncf`/
  `sitofp`/…) other than the `i1`-source `extui`/`uitofp` above;
  non-32-bit indexing;
- a kernel module that does not contain exactly one function (driver).

**Out of scope** (not yet encoded)

- `tt.dot` and rank>1 / non-axis-0 reductions (1-D axis-0 `tt.reduce` with an
  add/max/min combiner is supported as of phase 2 — see `phase-2.md`);
- control flow: `scf.for`/`if`/`while`, loops;
- atomics; multi-dimensional grids (`program_id` axis 1/2);
- integer pack/unpack (blocked on `ext`/`trunc`) — the common quantization idiom.

**Modeling assumptions** (deliberate idealizations, not verified)

- floats behave as mathematical reals (the point of the tool — IEEE effects are
  invisible, so rewrites unsafe only under IEEE are treated as equivalent);
- distinct pointer arguments address disjoint, valid buffers — no aliasing,
  out-of-bounds, undefined behavior, or pointer provenance;
- loads observe the entry memory state; a single, race-free kernel launch.

## Files

- **Triton (tracked):** `bin/RegisterTritonDialects.h`,
  `{include,lib}/Conversion/CMakeLists.txt`, `scripts/build-llvm-project.sh`
  (modified); `{include/triton,lib}/Conversion/TritonToSMT/`,
  `python/triton/tools/smt_equivalence.py`,
  `python/test/unit/tools/test_smt_equivalence.py`,
  `test/Conversion/triton_to_smt.mlir`, `test/Conversion/triton_to_smt_casts.mlir`,
  `scripts/patches/mlir-smt-real.patch` (+ `scripts/patches/README.md`) (new).
- **Vendored LLVM (via the tracked patch):**
  `mlir/include/mlir/Dialect/SMT/IR/{SMT.td,SMTTypes.td,SMTVisitors.h,
  SMTRealOps.td}`, `mlir/lib/Dialect/SMT/IR/SMTTypes.cpp`,
  `mlir/lib/Target/SMTLIB/ExportSMTLIB.cpp`.

## Validation

- **Equivalences:** FMA `affine_naive` vs BLOCK=128 `affine_optimized`;
  `x−y ≡ x+(−y)`; `2*x ≡ x+x`; relu `where(a>0,a,0) ≡ maximum(a,0)`;
  naive vs BLOCK-vectorized add; `ptr<i32>` `a+1 ≡ 1+a`; `|a| ≡ where(a>0,a,−a)`;
  `(a+a)/b ≡ (2a)/b`; bool store `a==b ≡ ~(a!=b)` (extui/bitcast idiom);
  `(a==b).to(f32) ≡ where(a==b,1,0)` → all **EQUIVALENT**.
- **Counterexample:** `a*b+c` vs `a*b` → **NOT_EQUIVALENT** (model: `i=0,n=1,C=1`).
- **Counterexamples:** `a*b+c` vs `a*b` → **NOT_EQUIVALENT** (`i=0,n=1,C=1`);
  i8 `(x+1)>x` overflow, differing output destination, `a/b` vs `a*(1/b)` →
  **NOT_EQUIVALENT**.
- **Adversarial (hardened over a 3-round Codex↔Claude review loop):** no input
  produces a false `EQUIVALENT` or crashes the pass. Rejected (`UNSUPPORTED`):
  shifted / chained / non-identity addressing; multiple stores or output
  buffers; signature mismatch; non-finite or wider-than-f64 constants; `i1`
  integer arithmetic; `nsw`/`nuw`/`nneg` poison flags; pointer-valued `select`;
  representation-changing pointer bitcasts; raw reinterpreting loads; volatile
  loads; external / multi-block functions; dynamic or int64-overflowing store
  shapes; injected `smt.solver` scopes; quoted symbols; non-finite solver
  timeouts; and `smt.real.constant` SMT-LIB injection.
- **Suites:** `pytest python/test/unit/tools/test_smt_equivalence.py` → 26 passed;
  `test/Conversion/triton_to_smt.mlir` + `triton_to_smt_casts.mlir` → PASS
  (`llvm-lit`); the 16 MLIR SMT dialect/export tests still pass.
- **Real kernels** (`ntops_vs_torch_triton` artifact set, 41 operators): 17
  torch-inductor kernels prove **EQUIVALENT** in self-validation (abs, add,
  bitwise_and/not/or, clamp, div, eq, ge, gt, le, lt, mul, ne, neg, relu,
  sub); every other kernel is rejected with a specific diagnostic — no
  crashes, no silent verdicts. Cross-framework (ninetoothed vs inductor)
  demo via instance specialization: see `smt-encoding-tutorial.md`.

## Roadmap

ideal reals + faithful bit-vector integers →
1-D `tt.reduce` add/max/min (phase 2, current) →
`tt.dot` and multi-dim reductions →
integer `ext`/`trunc` for quantization pack/unpack →
multi-output / relaxed store contract →
loops →
FpSan-style floating-point-error domain (reusing the same per-op dispatch and
driver).
