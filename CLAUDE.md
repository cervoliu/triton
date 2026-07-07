# Triton (SMT translation-validation fork)

This checkout's active work is an SMT-based formal equivalence checker for
Triton kernels: the `--convert-triton-to-smt` pass
(`lib/Conversion/TritonToSMT/`), the driver
(`python/triton/tools/smt_equivalence.py`), and a `Real` sort added to MLIR's
`smt` dialect via `scripts/patches/mlir-smt-real.patch`. Docs index:
`docs/smt-tv/README.md`.

## Environment

- Python: use `.venv/bin/python` (plain `python` is not on PATH).
- Triton build dir: `build/cmake.*/` (ninja). Rebuild the pass with
  `ninja -C build/cmake.*/ triton-opt`.
- Two LLVM layouts exist:
  - `.llvm-project/{src,build}` — custom checkout at the pinned rev
    (`cmake/llvm-info.json`) with the SMT patch applied as working-tree
    changes. This is where patch iteration happens.
  - `llvm-project/` — created by `scripts/build-llvm-project.sh`, which
    hard-resets to the pinned rev and applies `scripts/patches/*.patch`.
    Never edit it in place; `git clean`/`reset --hard` runs on every build.
- Tool discovery honors `TRITON_OPT`, `MLIR_TRANSLATE`, `Z3` env vars.

## Verify

Run `scripts/smt-verify.sh` after any change to the pass, driver, or patch.
It builds triton-opt, probes the environment (pass registered, mlir-translate
patched, z3 present), then runs the lit tests and the pytest suite, failing
loudly on skips. Do not trust a green pytest run alone: the suite skips
without tools, and skipped soundness tests look identical to passing ones.

- lit only: `.llvm-project/build/bin/llvm-lit -sv build/cmake.*/test/Conversion --filter triton_to_smt`
- pytest only: `cd python && ../.venv/bin/python -m pytest test/unit/tools/test_smt_equivalence.py`

## Rules

- **Soundness first:** the pass must REJECT (`signalPassFailure`) anything
  outside its validated contract rather than approximate. Never trade a
  rejection for a silently-wrong EQUIVALENT. Every soundness fix gets an
  adversarial regression test in `test_smt_equivalence.py`.
- Modeling decisions are deliberate, not bugs: floats are ideal reals (not
  IEEE-754), integers are fixed-width bit-vectors, pointer args are assumed
  disjoint, byte-backed `i1` buffers use nonzero=true. See
  `docs/smt-tv/phase-1.md` ("Modeling assumptions") before "fixing" them.
- Elementwise op semantics live in ONE place: `encodeElementwise` in
  `TritonToSMTPass.cpp`. Do not reintroduce per-context copies.
- Never edit `.llvm-project/src` SMT files without regenerating the patch:
  `scripts/regenerate-smt-patch.sh` (then rebuild mlir-translate and commit
  the patch). The patch is the source of truth for clean builds.
- Driver error contract: kernel-dependent failures become verdicts
  (UNSUPPORTED/UNKNOWN); environment errors raise RuntimeError.
- Do not weaken the checker's equivalence criteria for cross-framework demos;
  such workarounds live in scripts outside the repo.
