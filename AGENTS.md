# Working on Triton

## Build and Testing Guidelines
- Before running tests for native/compiler changes, run `make` in the triton directory to rebuild triton. DO NOT RUN `make` if you only changed Python code or code in `python/triton_kernels`.
- For compiler changes, add tests in `python/test/` (pytest) or test (lit). Keep GPU-only tests in `python/test/unit/` or `python/test/gluon/`, name them `test_<feature>_<condition>`, and avoid creating new test files unless requested.
- Run pytest with `-s --tb=short`. Run a single test with `pytest file.py::test_name`.
- The build dir is given by `BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
- Run lit from the build dir:  `cd BUILD_DIR; ninja triton-opt; lit -v test/<path>.mlir` (example: `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir`).
- Lit tests can be run locally (no GPU required).
- Compiler crashes sometimes print an MLIR reproducer (external_resources / mlir_reproducer). Save the full MLIR + {-# ... #-} metadata to `/tmp/<file>.mlir`, then run `triton-opt /tmp/<file>.mlir --run-reproducer` to reproduce locally.

## SMT translation-validation work (this fork)

For any work on the SMT equivalence checker (`lib/Conversion/TritonToSMT/`,
`python/triton/tools/smt_equivalence.py`, `scripts/patches/`, `docs/smt-tv/`),
read `CLAUDE.md` at the repo root FIRST — it is the authoritative project
instruction file for this fork: environment facts (use `.venv/bin/python`;
the two LLVM layouts; `TRITON_OPT`/`MLIR_TRANSLATE`/`Z3` overrides), the
verification entrypoint (`scripts/smt-verify.sh` — do NOT use `make` for this
work), and the non-negotiable soundness rules. The review brief for
adversarial soundness reviews is `docs/smt-tv/review-prompt.md`.
