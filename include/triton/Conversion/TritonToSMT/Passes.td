#ifndef TRITON_CONVERSION_TRITONTOSMT_PASSES
#define TRITON_CONVERSION_TRITONTOSMT_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToSMT : Pass<"convert-triton-to-smt", "mlir::ModuleOp"> {
  let summary = "Encode Triton (TTIR) functions into the SMT dialect for "
                "translation validation";
  let description = [{
    This pass symbolically encodes straight-line, elementwise Triton IR
    functions into the `smt` dialect under ideal-real floating-point semantics
    (floating-point values are modeled as mathematical reals, not IEEE-754).

    Given two functions with matching signatures (a source and a target), it
    emits three `smt.solver` scopes, in fixed order, each declaring the shared
    symbolic inputs and encoding both functions at a symbolic output index `i`:

    * scope 0 (addressing): asserts a store offset can differ from `i`;
      `unsat` proves identity addressing.
    * scope 1 (well-definedness): asserts the negation of the conjunction of
      all per-op well-definedness conditions (e.g. shift amount < width);
      `unsat` proves no input reaches a poison case, so the total SMT
      encodings are faithful. Trivially `unsat` when no such ops occur.
    * scope 2 (equivalence): asserts the two functions' final output arrays
      can differ; `unsat` means the functions are equivalent.

    Exporting the result to SMT-LIB (`mlir-translate --export-smtlib`) and
    running an SMT solver yields one sat/unsat/unknown line per scope.

    The encoding is operation-wise: every supported Triton/arith/math operation
    has its own local encoding rule. It currently targets straight-line,
    elementwise kernels and 1-D reductions (no loops or atomics).
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::smt::SMTDialect"];

  let options = [
    Option<"srcFunc", "src", "std::string", /*default*/"\"\"",
           "name of the source function (default: first function in module)">,
    Option<"tgtFunc", "tgt", "std::string", /*default*/"\"\"",
           "name of the target function (default: second function in module)">,
    Option<"blockSize", "block-size", "int64_t", /*default*/"0",
           "block size; 0 infers it from tt.make_range (scalar kernels use 1)">,
  ];
}

#endif // TRITON_CONVERSION_TRITONTOSMT_PASSES
