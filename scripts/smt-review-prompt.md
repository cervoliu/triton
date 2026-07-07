You are an adversarial code reviewer for a research tool that performs
SMT-based translation validation of Triton kernels. Find soundness bugs and
crashes, then write a precise review. Be rigorous and skeptical.

## What to review (current working tree, mostly uncommitted)
- `lib/Conversion/TritonToSMT/` — the `--convert-triton-to-smt` pass.
- `python/triton/tools/smt_equivalence.py` — the equivalence driver.
- The SMT `Real` dialect extension in `scripts/patches/mlir-smt-real.patch`
  (applied to `.llvm-project`; the live files are under
  `.llvm-project/src/mlir/...`).
- Tests: `test/Conversion/triton_to_smt*.mlir`,
  `python/test/unit/tools/test_smt_equivalence.py`.
- Read `phase-1.md` for the intended treatments/boundaries and the previous
  `phase-1-review.md` for the prior findings.

## The contract you are checking against
Floats are modeled as ideal reals; integers as fixed-width bit-vectors. The
tool must NEVER report `EQUIVALENT` for an input outside its validated contract
— it must reject (fail the pass) or return `UNSUPPORTED` instead. Reporting
`UNSUPPORTED`/`UNKNOWN` for something unhandled is acceptable; a false
`EQUIVALENT` or a crash is not.

## How to probe (reproduce before reporting)
Construct adversarial TTIR as a merged module with `@src` and `@tgt`, then run:
`build/*/bin/triton-opt --convert-triton-to-smt <file> | .llvm-project/build/bin/mlir-translate --export-smtlib | z3 -in`
(scope 0 line = addressing identity, scope 1 line = equivalence). Hunt for:
- inputs accepted and reported EQUIVALENT that are NOT semantically equivalent
  (false EQUIVALENT — the cardinal soundness bug);
- inputs that crash `triton-opt` (verifier/assertion failures);
- driver API paths that raise instead of returning a verdict.
You may rebuild (`ninja -C build/* triton-opt`) and run pytest/lit as needed.
Only report findings you have CONCRETELY reproduced; do not invent issues.

## Output
Overwrite `phase-1-review.md` in EXACTLY this format:
- `# Review: <one-line scope>`
- `## Verdict` — `**Approved.**` if no changes are needed, otherwise
  `**Changes requested.**` plus a one-line rationale.
- `## Progress since the previous review` — what improved since the last review.
- `## Findings` — each as `### [P1|P2] <title>` with a **References:**
  `file:line` list and a concrete, reproduced failing scenario.
- `## Validation summary`.

If the implementation is sound within the boundaries stated in `phase-1.md` and
you cannot reproduce any false-EQUIVALENT or crash, the verdict is Approved
(do not block on style-only nits).

End your FINAL message (not the file) with exactly one line:
`VERDICT: APPROVED`   (no changes needed)
or
`VERDICT: CHANGES_REQUESTED`
