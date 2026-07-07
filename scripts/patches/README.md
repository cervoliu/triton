# MLIR patches applied on top of the pinned LLVM

`scripts/build-llvm-project.sh` clones LLVM, hard-resets it to the revision in
`cmake/llvm-info.json`, and then applies every `*.patch` in this directory with
`git apply` before configuring. Because the hard reset wipes local edits, these
patches are the source of truth for any Triton-local changes to the vendored
LLVM/MLIR tree.

## `mlir-smt-real.patch`

Adds a `Real` sort (SMT-LIB Reals theory) to the MLIR `smt` dialect and its
SMT-LIB exporter: `RealType` plus `smt.real.{constant,add,mul,sub,div,neg,cmp}`,
wired through `SMTVisitors.h` and `ExportSMTLIB.cpp`. This is required by the
`--convert-triton-to-smt` translation-validation pass, which models
floating-point values as ideal reals. Upstreaming this to LLVM would let us drop
the patch.

The patch paths are relative to the llvm-project repository root (`mlir/...`).

### Applying manually

If your LLVM checkout is not managed by `build-llvm-project.sh` (e.g. a custom
`.llvm-project/src` layout), apply it by hand from the LLVM repo root and rebuild
the affected targets:

```sh
git -C <llvm-project-root> apply <triton>/scripts/patches/mlir-smt-real.patch
ninja -C <llvm-build> mlir-translate mlir-opt
```
