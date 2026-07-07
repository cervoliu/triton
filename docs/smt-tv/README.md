# SMT-based translation validation for Triton — dev docs

Internal design notes for the Triton→SMT translation-validation tool
(`--convert-triton-to-smt` pass in `lib/Conversion/TritonToSMT`, driver
`python/triton/tools/smt_equivalence.py`, plus a `Real`-sort patch to the
vendored MLIR `smt` dialect under `scripts/patches/`).

These are internal `.md` design notes — deliberately kept out of Triton's
Sphinx documentation tree (`docs/*.rst`, `docs/conf.py`).

## Index

- [`phase-1.md`](phase-1.md) — the phase-1 design: an elementwise encoder,
  floats as ideal reals, integers as fixed-width bit-vectors, sound-by-rejection.
  The accepted contract, boundaries, and validation status.
- [`phase-1-review.md`](phase-1-review.md) — the external (Codex) review that
  hardened phase 1 (findings + verdicts from the review loop).
- [`smt-encoding-tutorial.md`](smt-encoding-tutorial.md) — a worked example
  walking one real kernel pair through the entire pipeline.
- [`mlir-tv-comparison.md`](mlir-tv-comparison.md) — comparison against the
  reference tool mlir-tv (Bang et al., CAV 2022) and what we can absorb from it.
- [`phase-2.md`](phase-2.md) — the phase-2 goal (absorb/leverage mlir-tv) and its
  first milestone: reductions.

## Review harness

The Codex review loop lives in `scripts/`: `scripts/smt-review.sh` (runs the
external reviewer) and `scripts/smt-review-prompt.md` (its prompt).

## Verification & patch tooling

- `scripts/smt-verify.sh` — one-command verification: builds triton-opt,
  probes the environment (pass registered, mlir-translate patched, z3
  present), then runs the lit tests and pytest suite, failing loudly on skips.
- `scripts/regenerate-smt-patch.sh` — regenerates
  `scripts/patches/mlir-smt-real.patch` from an LLVM checkout and verifies it
  applies cleanly to the pinned revision. Run after any edit to the SMT
  dialect files in `.llvm-project/src`.
