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
    * scope 1 (UB-domain equality): asserts the two functions' UB predicates
      differ on some input; `unsat` proves both are UB on exactly the same
      inputs. Poison (e.g. an oversized shift amount) is tracked exactly per
      value and counts as UB only when it reaches a memory operation.
      Trivially `unsat` when neither function can be UB.
    * scope 2 (equivalence): asserts the final output arrays can differ on an
      input where neither function is UB; `unsat` means the functions are
      equivalent in the sense of bidirectional refinement.

    Exporting the result to SMT-LIB (`mlir-translate --export-smtlib`) and
    running an SMT solver yields one sat/unsat/unknown line per scope.

    The encoding is operation-wise: every supported Triton/arith/math operation
    has its own local encoding rule. It currently targets straight-line,
    elementwise kernels and 1-D reductions (no loops or atomics).

    With `memory-model` set (phase 3), the identity-addressing restriction is
    replaced by a block memory model: every pointer argument becomes a
    distinct block (an SMT array plus a symbolic element count), tensors are
    materialized as concrete lanes with a static shape (rank <= 2, extent
    product bounded by `max-lanes`), and per-lane symbolic offsets replace the
    identity proof. Out-of-bounds accesses by an active lane, poison reaching
    a memory op, and two active lanes of one store writing different values
    to the same offset (an intra-store race) all fold into the UB predicate
    checked by scope 1. The same three scopes are emitted in the same order;
    scope 0 carries a trivially-unsat placeholder assertion because block
    decomposition is total by construction (anything else is rejected).
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
           "block size; 0 infers it from tt.make_range (scalar kernels use 1); "
           "only meaningful for the identity-addressing path (rejected when "
           "combined with memory-model)">,
    Option<"memoryModel", "memory-model", "bool", /*default*/"false",
           "use the phase-3 block memory model: one block per pointer "
           "argument with a symbolic element count, tensors materialized as "
           "concrete lanes (rank <= 2), out-of-bounds and intra-store-race "
           "accesses folded into the UB predicate; scope 0 becomes a "
           "documented always-unsat placeholder">,
    Option<"maxLanes", "max-lanes", "int64_t", /*default*/"256",
           "memory-model mode: maximum tensor extent product materialized as "
           "concrete lanes; larger tensors are rejected loudly (encoder and "
           "solver tractability bound, independent of the rank <= 2 cap). "
           "Must be positive: the guard is mandatory and a non-positive "
           "value is rejected rather than treated as unlimited">,
  ];
}

#endif // TRITON_CONVERSION_TRITONTOSMT_PASSES
