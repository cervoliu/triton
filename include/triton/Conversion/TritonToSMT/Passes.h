#ifndef TRITON_CONVERSION_TRITONTOSMT_PASSES_H
#define TRITON_CONVERSION_TRITONTOSMT_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#include "triton/Conversion/TritonToSMT/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToSMT/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_CONVERSION_TRITONTOSMT_PASSES_H
