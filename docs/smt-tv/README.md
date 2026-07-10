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
  walking one real kernel pair through the entire pipeline (written against
  the phase-1 two-scope query; see the supersession note at its top).
- [`mlir-tv-comparison.md`](mlir-tv-comparison.md) — comparison against the
  reference tool mlir-tv (Bang et al., CAV 2022) and what we can absorb from it.
- [`fehr-2025-comparison.md`](fehr-2025-comparison.md) — comparison against
  Fehr et al.'s first-class verification dialects (PLDI 2025): shared smt
  foundation, poison-semantics agreement, absorb/skip decisions.
- [`phase-2.md`](phase-2.md) — phase 2, COMPLETE and review-approved: all four
  backlog milestones — reductions, shifts/ext/trunc via solver-discharged
  well-definedness, equivalence as bidirectional refinement with exact poison
  tracking (the current three-scope query semantics), and the pass-validation
  sweep harness.
- [`phase-2-review.md`](phase-2-review.md) — the approved Codex review of
  phase-2 milestones 2–4 (latest round; earlier rounds' findings are pinned
  as regression tests).

## Review harness

Reviews are delivered to the external Codex reviewer through the Codex plugin
(`/codex:rescue`, or the codex-rescue subagent) using the brief in
[`review-prompt.md`](review-prompt.md). The loop: implement an increment →
send the brief (naming the phase under review) → fix findings → repeat until
`VERDICT: APPROVED`, pinning each fix with an adversarial regression test.
The reviewer's sandbox denies exec and file writes, so it reviews statically,
names reproducers for the caller to run, and emits the review markdown for
transcription into `<phase>-review.md`. Project context reaches it via
`AGENTS.md` → `CLAUDE.md`.

## Verification & patch tooling

- `scripts/smt-verify.sh` — one-command verification: builds triton-opt,
  probes the environment (pass registered, mlir-translate patched, z3
  present), then runs the lit tests and pytest suite, failing loudly on skips.
- `scripts/regenerate-smt-patch.sh` — regenerates
  `scripts/patches/mlir-smt-real.patch` from an LLVM checkout and verifies it
  applies cleanly to the pinned revision. Run after any edit to the SMT
  dialect files in `.llvm-project/src`.
- `python/triton/tools/smt_validate_passes.py` — mlir-tv-style pass-validation
  sweep: splits a TTIR corpus into functions, runs each through
  semantics-preserving passes (atomic pass names only), and checks original
  vs transformed with the driver. NOT_EQUIVALENT or any accounting error
  fails the sweep.
