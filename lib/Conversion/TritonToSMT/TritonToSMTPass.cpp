//===- TritonToSMTPass.cpp - Encode Triton IR into the SMT dialect --------===//
//
// Operation-wise encoding of straight-line, elementwise Triton (TTIR) functions
// into the `smt` dialect for SMT-based translation validation.
//
// Semantics:
//   * floating-point values -> ideal reals (`!smt.real`), NOT IEEE-754;
//   * integer values        -> fixed-width bit-vectors (`!smt.bv<N>`) with
//                              wrapping arithmetic and signed/unsigned compares;
//   * `i1` values            -> `!smt.bool`;
//   * pointers               -> (memory array, bit-vector offset), the array's
//                              range sort derived from the pointee type.
//
// The pass targets a deliberately narrow, *validated* contract and REJECTS
// anything outside it (rather than silently reporting equivalence):
//   * exactly one `tt.store` per function, to a single output pointer argument;
//   * canonical identity addressing (offset == linear index `i`), which is
//     verified by a dedicated `smt.check` scope;
//   * 32-bit indexing, `program_id` axis 0, matching full signatures.
//
// It emits three solver scopes, in fixed order: scope 0 checks that addressing
// is identity (unsat => identity holds); scope 1 checks well-definedness
// (unsat => no input reaches a poison case, e.g. an oversized shift, so the
// total SMT encodings are faithful); scope 2 asserts the two functions' output
// arrays can differ (unsat => equivalent). See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "triton/Conversion/TritonToSMT/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cmath>

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONTOSMT
#include "triton/Conversion/TritonToSMT/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

// Width of the address/index domain. Triton elementwise kernels index with i32;
// kernels using a different index width are rejected.
static constexpr unsigned kAddrWidth = 32;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Render a finite floating-point value as an exact SMT-LIB real term. Returns
// std::nullopt for non-finite (inf/NaN) values, which have no real meaning.
static std::optional<std::string> formatReal(APFloat value) {
  if (!value.isFinite())
    return std::nullopt;
  if (value.isZero())
    return std::string("0.0");

  bool losesInfo = false;
  value.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  if (losesInfo)
    return std::nullopt; // wider than binary64: cannot be represented exactly
  double d = value.convertToDouble();
  bool neg = std::signbit(d);
  d = std::fabs(d);

  int exp = 0;
  double m = std::frexp(d, &exp); // d == m * 2^exp, m in [0.5, 1)
  int64_t mant = std::llround(std::ldexp(m, 53));
  int e = exp - 53; // d == mant * 2^e
  while (mant != 0 && (mant & 1) == 0) {
    mant >>= 1;
    ++e;
  }

  std::string body;
  if (e >= 0) {
    APInt val(64 + e + 1, static_cast<uint64_t>(mant));
    val = val.shl(static_cast<unsigned>(e));
    SmallString<64> s;
    val.toString(s, 10, /*Signed=*/false);
    body = (s + ".0").str();
  } else {
    unsigned k = static_cast<unsigned>(-e);
    APInt num(k + 64, static_cast<uint64_t>(mant));
    APInt den = APInt(k + 2, 1).shl(k);
    SmallString<64> ns, ds;
    num.toString(ns, 10, false);
    den.toString(ds, 10, false);
    body = ("(/ " + ns + ".0 " + ds + ".0)").str();
  }
  return neg ? ("(- " + body + ")") : body;
}

// Map an MLIR scalar type to its SMT sort. Returns null for unsupported types.
static Type smtSortFor(Type t, MLIRContext *ctx) {
  if (isa<FloatType>(t))
    return smt::RealType::get(ctx);
  if (auto it = dyn_cast<IntegerType>(t)) {
    if (it.getWidth() == 1)
      return smt::BoolType::get(ctx);
    return smt::BitVectorType::get(ctx, it.getWidth());
  }
  return nullptr;
}

static void fillCheckRegions(OpBuilder &builder, Location loc,
                             smt::CheckOp check) {
  for (Region *region : {&check.getSatRegion(), &check.getUnknownRegion(),
                         &check.getUnsatRegion()}) {
    OpBuilder::InsertionGuard guard(builder);
    Block *block = builder.createBlock(region);
    builder.setInsertionPointToStart(block);
    smt::YieldOp::create(builder, loc, ValueRange{});
  }
}

// An encoded Triton SSA value: a plain SMT value, a pointer (memory array +
// bit-vector offset), or -- in reduction mode -- a per-lane *vector* of either
// (see `lanes`). The array symbol identifies which output buffer is meant.
struct EncodedValue {
  Value value;  // plain scalar value (null for pointers / vectors)
  Value array;  // pointer: the memory array symbol
  Value offset; // scalar pointer: the bit-vector element offset
  bool fresh = false; // pointer: true iff no tt.addptr has been applied yet
  // Reduction mode: a tensor materialized as N per-lane terms. For a vector
  // *value* these are the lane SMT values (array == null); for a vector
  // *pointer* they are the per-lane bit-vector offsets (array != null).
  SmallVector<Value> lanes;
  bool isPtr() const { return array != nullptr; }
  bool isVector() const { return !lanes.empty(); }
  // Value at lane k, broadcasting a scalar operand.
  Value laneVal(unsigned k) const { return isVector() ? lanes[k] : value; }
  Value laneOff(unsigned k) const { return isVector() ? lanes[k] : offset; }
  static EncodedValue plain(Value v) { return {v, nullptr, nullptr, false, {}}; }
  static EncodedValue ptr(Value a, Value o, bool fresh = false) {
    return {nullptr, a, o, fresh, {}};
  }
  static EncodedValue vec(SmallVector<Value> ls) {
    EncodedValue e;
    e.lanes = std::move(ls);
    return e;
  }
  static EncodedValue vecPtr(Value a, SmallVector<Value> offs) {
    EncodedValue e;
    e.array = a;
    e.lanes = std::move(offs);
    return e;
  }
};

// A single store's memory effect.
struct StoreInfo {
  Value array;  // output memory array symbol
  Value offset; // bit-vector address written
  Value value;  // stored value (Real/BV/Bool)
  Value mask;   // predicate (null => always)
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertTritonToSMT
    : public mlir::triton::impl::ConvertTritonToSMTBase<ConvertTritonToSMT> {
  using ConvertTritonToSMTBase::ConvertTritonToSMTBase;

  void runOnOperation() override;

  // Declare the shared symbolic inputs for one signature at the current
  // insertion point. Returns false (and emits an error) on an unsupported type.
  bool declareInputs(OpBuilder &b, Location loc, ArrayRef<Type> argTypes,
                     Value &outIndex, SmallVectorImpl<EncodedValue> &outArgs);

  // Encode `func` at the symbolic index `iVal` using `sharedArgs`. Appends the
  // function's stores to `stores` and its per-op well-definedness conditions
  // (Bool terms that must hold for the encoding to be faithful, e.g. shift
  // amount < width) to `wdConds`. Returns false (and emits an error) if the
  // function uses an unsupported construct.
  bool encodeFunction(OpBuilder &b, triton::FuncOp func,
                      ArrayRef<EncodedValue> sharedArgs, Value iVal,
                      SmallVectorImpl<StoreInfo> &stores,
                      SmallVectorImpl<Value> &wdConds);
};

bool ConvertTritonToSMT::declareInputs(OpBuilder &b, Location loc,
                                       ArrayRef<Type> argTypes, Value &outIndex,
                                       SmallVectorImpl<EncodedValue> &outArgs) {
  MLIRContext *ctx = &getContext();
  auto addrTy = smt::BitVectorType::get(ctx, kAddrWidth);

  outIndex = smt::DeclareFunOp::create(b, loc, addrTy, b.getStringAttr("i"))
                 .getResult();
  Value zeroOff =
      smt::BVConstantOp::create(b, loc, APInt(kAddrWidth, 0)).getResult();

  for (auto [idx, argTy] : llvm::enumerate(argTypes)) {
    std::string name = ("arg" + Twine(idx)).str();
    if (auto ptrTy = dyn_cast<triton::PointerType>(argTy)) {
      Type range = smtSortFor(ptrTy.getPointeeType(), ctx);
      if (!range) {
        getOperation().emitError("TritonToSMT: unsupported pointee type ")
            << ptrTy.getPointeeType();
        return false;
      }
      auto arrTy = smt::ArrayType::get(ctx, addrTy, range);
      Value arr =
          smt::DeclareFunOp::create(b, loc, arrTy, b.getStringAttr(name))
              .getResult();
      outArgs.push_back(EncodedValue::ptr(arr, zeroOff, /*fresh=*/true));
    } else if (Type sort = smtSortFor(argTy, ctx)) {
      Value v = smt::DeclareFunOp::create(b, loc, sort, b.getStringAttr(name))
                    .getResult();
      outArgs.push_back(EncodedValue::plain(v));
    } else {
      getOperation().emitError("TritonToSMT: unsupported argument type ")
          << argTy;
      return false;
    }
  }
  return true;
}

bool ConvertTritonToSMT::encodeFunction(OpBuilder &b, triton::FuncOp func,
                                        ArrayRef<EncodedValue> sharedArgs,
                                        Value iVal,
                                        SmallVectorImpl<StoreInfo> &stores,
                                        SmallVectorImpl<Value> &wdConds) {
  MLIRContext *ctx = &getContext();
  Location loc = func.getLoc();
  auto R = smt::RealType::get(ctx);
  auto Bool = smt::BoolType::get(ctx);

  // Bit-vector / real / bool builder helpers.
  auto bvc = [&](const APInt &v) {
    return smt::BVConstantOp::create(b, loc, v).getResult();
  };
  auto addrConst = [&](uint64_t v) { return bvc(APInt(kAddrWidth, v)); };
  auto bvcmp = [&](smt::BVCmpPredicate p, Value x, Value y) {
    return smt::BVCmpOp::create(b, loc, p, x, y).getResult();
  };
  auto sel = [&](Value arr, Value idx, Type range) {
    return smt::ArraySelectOp::create(b, loc, range, arr, idx).getResult();
  };
  auto ite = [&](Value c, Value t, Value e) {
    return smt::IteOp::create(b, loc, t.getType(), c, t, e).getResult();
  };
  auto beq = [&](Value x, Value y) {
    return smt::EqOp::create(b, loc, x, y).getResult();
  };
  auto bnot = [&](Value x) { return smt::NotOp::create(b, loc, x).getResult(); };
  auto band = [&](Value x, Value y) {
    return smt::AndOp::create(b, loc, x, y).getResult();
  };
  auto realBin = [&](Value x, Value y, bool mul) -> Value {
    if (mul)
      return smt::RealMulOp::create(b, loc, R, ValueRange{x, y}).getResult();
    return smt::RealAddOp::create(b, loc, R, ValueRange{x, y}).getResult();
  };
  auto realConst = [&](StringRef s) {
    return smt::RealConstantOp::create(b, loc, R, b.getStringAttr(s))
        .getResult();
  };
  // Reconcile a stored value with the buffer's declared SMT sort at a store
  // through a pointer bitcast. Only the bit-vector -> Bool direction (the
  // frontend's extui+bitcast idiom writing a byte into an i1-declared buffer,
  // read back as zero=false / nonzero=true) is modeled. The reverse direction
  // -- storing a raw i1 value into a byte-observable iN buffer -- is NOT
  // modeled: real lowerings disagree on the written byte (NVIDIA sign-extends
  // i1 to 0xff; "store true as 0x01" would prove a kernel equivalent to one
  // storing literal 1, which real memory distinguishes). Returns null for
  // any other sort mismatch, which the caller rejects.
  auto coerce = [&](Value v, Type sort) -> Value {
    if (v.getType() == sort)
      return v;
    if (auto bvTy = dyn_cast<smt::BitVectorType>(v.getType()))
      if (isa<smt::BoolType>(sort))
        return bnot(beq(v, bvc(APInt(bvTy.getWidth(), 0))));
    return nullptr;
  };
  auto elemOf = [](Type t) -> Type {
    if (auto rt = dyn_cast<RankedTensorType>(t))
      return rt.getElementType();
    return t;
  };

  // Reject external declarations (empty body) and multi-block functions before
  // touching the entry block.
  if (!func.getBody().hasOneBlock()) {
    func.emitError("TritonToSMT: only single-block function bodies are "
                   "supported (external or multi-block functions rejected)");
    return false;
  }
  Block &entry = func.getBody().front();

  // The lane count (block size) is the number of elements the store writes; we
  // derive it from the store's value shape rather than from a tt.make_range,
  // which may be unrelated to (or inconsistent with) the store.
  int64_t block = 1;
  triton::StoreOp storeOp;
  for (Operation &op : entry)
    if (auto st = dyn_cast<triton::StoreOp>(op))
      storeOp = st;
  if (storeOp)
    if (auto vt = dyn_cast<RankedTensorType>(storeOp.getValue().getType())) {
      if (!vt.hasStaticShape()) {
        storeOp.emitError("TritonToSMT: dynamic-shaped store is not supported");
        return false;
      }
      std::optional<int64_t> ne = ShapedType::tryGetNumElements(vt.getShape());
      if (!ne) {
        storeOp.emitError(
            "TritonToSMT: store element count overflows int64 and is not "
            "supported");
        return false;
      }
      block = *ne;
    }
  if (blockSize > 0 && blockSize != block) {
    func.emitError("TritonToSMT: block-size option (")
        << blockSize << ") disagrees with the store lane count (" << block
        << ")";
    return false;
  }

  // Reduction mode: a tt.reduce collapses a block axis, so its input must be
  // materialized as N per-lane terms rather than the single symbolic-lane term
  // the elementwise path uses. Because floats are ideal reals, folding the N
  // lanes in any order is provably equivalent -- no permutation/multiset
  // machinery (as mlir-tv needs) is required. Detect it and derive the reduced
  // extent N. See docs/smt-tv/phase-2.md.
  bool reductionMode = false;
  int64_t reducedN = 0;
  for (Operation &op : entry)
    if (auto red = dyn_cast<triton::ReduceOp>(op)) {
      if (red->getNumOperands() != 1 || red->getNumResults() != 1) {
        red.emitError("TritonToSMT: only single-input/single-result tt.reduce "
                      "is supported (fused arg-reduce is rejected)");
        return false;
      }
      auto srcTy = dyn_cast<RankedTensorType>(red->getOperand(0).getType());
      if (red.getAxis() != 0 || !srcTy || srcTy.getRank() != 1 ||
          !srcTy.hasStaticShape()) {
        red.emitError("TritonToSMT: only axis-0 reduction of a static rank-1 "
                      "tensor is supported");
        return false;
      }
      int64_t n = srcTy.getShape()[0];
      if (reductionMode && n != reducedN) {
        red.emitError("TritonToSMT: all reductions in a function must share a "
                      "single reduced extent");
        return false;
      }
      reductionMode = true;
      reducedN = n;
    }
  // A reduce collapses to a scalar-per-program store, so the store lane count
  // must be 1; a tensor store coexisting with a reduce is not modeled.
  if (reductionMode && block != 1) {
    func.emitError("TritonToSMT: a tensor store coexisting with a reduction is "
                   "not supported (expected a scalar store)");
    return false;
  }

  // Every tt.make_range must span exactly the reduced extent (reduction mode)
  // or the store's lane count (elementwise mode); otherwise the writer/lane
  // decomposition would misattribute lanes.
  int64_t rangeExtent = reductionMode ? reducedN : block;
  for (Operation &op : entry)
    if (auto range = dyn_cast<triton::MakeRangeOp>(op)) {
      int64_t ext = int64_t(range.getEnd()) - int64_t(range.getStart());
      if (ext != rangeExtent) {
        range.emitError("TritonToSMT: tt.make_range extent (")
            << ext << ") does not match the expected extent (" << rangeExtent
            << ")";
        return false;
      }
    }

  // Writer decomposition for output index i: pid = i / block, lane = i % block.
  Value pid = iVal, lane = addrConst(0);
  if (block != 1) {
    Value blk = addrConst(block);
    pid = smt::BVUDivOp::create(b, loc, iVal, blk).getResult();
    lane = smt::BVURemOp::create(b, loc, iVal, blk).getResult();
  }

  DenseMap<Value, EncodedValue> env;
  for (auto [arg, shared] : llvm::zip(entry.getArguments(), sharedArgs))
    env[arg] = shared;
  auto V = [&](Value v) { return env[v].value; };

  auto err = [&](Operation *op, const Twine &what) {
    op->emitError("TritonToSMT: ") << what;
    return false;
  };

  auto rcmp = [&](smt::IntPredicate p, Value x, Value y) {
    return smt::RealCmpOp::create(b, loc, Bool, p, x, y).getResult();
  };

  // The single source of truth for pure elementwise op semantics, shared by
  // the scalar path, the per-lane (reduction) path, and the reduce combiner.
  // Operands are fetched through `A`, which reads the plain value in scalar
  // context and the lane value (broadcasting scalars) in vector context.
  // Returns the encoded value; a null result with a non-empty `why` is a
  // rejection (unsupported flag/operand sort), and a null result with an empty
  // `why` means `op` is not a pure elementwise op at all.
  auto encodeElementwise = [&](Operation *op, llvm::function_ref<Value(Value)> A,
                               std::string &why) -> Value {
    return llvm::TypeSwitch<Operation *, Value>(op)
        .Case<arith::AddFOp>([&](arith::AddFOp o) {
          return realBin(A(o.getLhs()), A(o.getRhs()), /*mul=*/false);
        })
        .Case<arith::SubFOp>([&](arith::SubFOp o) {
          return smt::RealSubOp::create(b, loc, R, A(o.getLhs()), A(o.getRhs()))
              .getResult();
        })
        .Case<arith::MulFOp>([&](arith::MulFOp o) {
          return realBin(A(o.getLhs()), A(o.getRhs()), /*mul=*/true);
        })
        .Case<arith::NegFOp>([&](arith::NegFOp o) {
          return smt::RealNegOp::create(b, loc, R, A(o.getOperand()))
              .getResult();
        })
        .Case<arith::DivFOp>([&](arith::DivFOp o) {
          // SMT-LIB real division is total but unspecified at zero
          // denominators; both functions see the same division function, so
          // the equivalence query stays sound.
          return smt::RealDivOp::create(b, loc, R, A(o.getLhs()), A(o.getRhs()))
              .getResult();
        })
        .Case<arith::MaxNumFOp, arith::MaximumFOp>([&](auto o) -> Value {
          Value x = A(o.getLhs()), y = A(o.getRhs());
          return ite(rcmp(smt::IntPredicate::ge, x, y), x, y);
        })
        .Case<arith::MinNumFOp, arith::MinimumFOp>([&](auto o) -> Value {
          Value x = A(o.getLhs()), y = A(o.getRhs());
          return ite(rcmp(smt::IntPredicate::le, x, y), x, y);
        })
        .Case<math::AbsFOp>([&](math::AbsFOp o) {
          Value x = A(o.getOperand());
          Value isNeg = rcmp(smt::IntPredicate::lt, x, realConst("0.0"));
          Value negX = smt::RealNegOp::create(b, loc, R, x).getResult();
          return ite(isNeg, negX, x);
        })
        .Case<math::FmaOp>([&](math::FmaOp o) {
          Value prod =
              realBin(A(o.getOperand(0)), A(o.getOperand(1)), /*mul=*/true);
          return realBin(prod, A(o.getOperand(2)), /*mul=*/false);
        })
        .Case<arith::CmpFOp>([&](arith::CmpFOp o) -> Value {
          Value x = A(o.getLhs()), y = A(o.getRhs());
          using P = arith::CmpFPredicate;
          using I = smt::IntPredicate;
          switch (o.getPredicate()) {
          case P::OEQ: case P::UEQ: return beq(x, y);
          case P::ONE: case P::UNE: return bnot(beq(x, y));
          case P::OGT: case P::UGT: return rcmp(I::gt, x, y);
          case P::OGE: case P::UGE: return rcmp(I::ge, x, y);
          case P::OLT: case P::ULT: return rcmp(I::lt, x, y);
          case P::OLE: case P::ULE: return rcmp(I::le, x, y);
          default:
            why = "unsupported cmpf predicate";
            return Value();
          }
        })
        // Integer (bit-vector) arithmetic. `i1` operands are SMT booleans, not
        // bit-vectors, so reject them (mapping to smt.bv.* would crash); and
        // nsw/nuw overflow flags introduce poison we do not model.
        .Case<arith::AddIOp, arith::SubIOp, arith::MulIOp>([&](auto o) -> Value {
          if (isa<smt::BoolType>(A(o.getLhs()).getType())) {
            why = "i1 integer arithmetic is not supported";
            return Value();
          }
          if (o.getOverflowFlags() != arith::IntegerOverflowFlags::none) {
            why = "arith overflow flags (nsw/nuw) are not modeled";
            return Value();
          }
          Value x = A(o.getLhs()), y = A(o.getRhs());
          if (isa<arith::MulIOp>(o.getOperation()))
            return smt::BVMulOp::create(b, loc, x, y).getResult();
          // The smt dialect has no bvsub; use x + (-y).
          if (isa<arith::SubIOp>(o.getOperation()))
            y = smt::BVNegOp::create(b, loc, y).getResult();
          return smt::BVAddOp::create(b, loc, x, y).getResult();
        })
        // `i1` values are SMT booleans, wider integers are bit-vectors, so the
        // bitwise ops dispatch on the encoded operand sort.
        .Case<arith::AndIOp>([&](arith::AndIOp o) -> Value {
          Value x = A(o.getLhs()), y = A(o.getRhs());
          if (isa<smt::BoolType>(x.getType()))
            return band(x, y);
          return smt::BVAndOp::create(b, loc, x, y).getResult();
        })
        .Case<arith::OrIOp>([&](arith::OrIOp o) -> Value {
          Value x = A(o.getLhs()), y = A(o.getRhs());
          if (isa<smt::BoolType>(x.getType()))
            return smt::OrOp::create(b, loc, x, y).getResult();
          return smt::BVOrOp::create(b, loc, x, y).getResult();
        })
        .Case<arith::XOrIOp>([&](arith::XOrIOp o) -> Value {
          Value x = A(o.getLhs()), y = A(o.getRhs());
          if (isa<smt::BoolType>(x.getType()))
            return smt::XOrOp::create(b, loc, x, y).getResult();
          return smt::BVXOrOp::create(b, loc, x, y).getResult();
        })
        // Shifts. An oversized shift amount (>= width) is poison in arith but
        // defined for the total SMT bv shifts, so each shift contributes a
        // well-definedness condition `amt < width` (mlir-tv's wellDefined(op,
        // cond) idea). A dedicated solver scope proves the conjunction holds
        // for ALL inputs; if it cannot, the driver reports UNSUPPORTED -- the
        // total encoding is only trusted when no input can reach the poison
        // case.
        .Case<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>(
            [&](auto o) -> Value {
              if constexpr (std::is_same_v<decltype(o), arith::ShLIOp>) {
                if (o.getOverflowFlags() != arith::IntegerOverflowFlags::none) {
                  why = "arith overflow flags (nsw/nuw) are not modeled";
                  return Value();
                }
              } else {
                if (o.getIsExact()) {
                  why = "shr exact flag introduces poison we do not model";
                  return Value();
                }
              }
              Value x = A(o.getLhs()), y = A(o.getRhs());
              if (isa<smt::BoolType>(x.getType())) {
                why = "i1 integer arithmetic is not supported";
                return Value();
              }
              unsigned w = cast<smt::BitVectorType>(x.getType()).getWidth();
              wdConds.push_back(
                  bvcmp(smt::BVCmpPredicate::ult, y, bvc(APInt(w, w))));
              if (isa<arith::ShLIOp>(o.getOperation()))
                return smt::BVShlOp::create(b, loc, x, y).getResult();
              if (isa<arith::ShRUIOp>(o.getOperation()))
                return smt::BVLShrOp::create(b, loc, x, y).getResult();
              return smt::BVAShrOp::create(b, loc, x, y).getResult();
            })
        // Integer division and remainder are intentionally NOT encoded. SMT
        // bit-vector div/rem are total (e.g. bvudiv by zero is all-ones)
        // whereas the arith ops are undefined there; unlike shifts, their
        // well-definedness (nonzero divisor, no INT_MIN/-1) is rarely provable
        // for all inputs, so they fall to Default and are rejected until a
        // use case appears.
        .Case<arith::CmpIOp>([&](arith::CmpIOp o) -> Value {
          Value x = A(o.getLhs()), y = A(o.getRhs());
          using P = arith::CmpIPredicate;
          using B = smt::BVCmpPredicate;
          switch (o.getPredicate()) {
          case P::eq: return beq(x, y);
          case P::ne: return bnot(beq(x, y));
          case P::slt: return bvcmp(B::slt, x, y);
          case P::sle: return bvcmp(B::sle, x, y);
          case P::sgt: return bvcmp(B::sgt, x, y);
          case P::sge: return bvcmp(B::sge, x, y);
          case P::ult: return bvcmp(B::ult, x, y);
          case P::ule: return bvcmp(B::ule, x, y);
          case P::ugt: return bvcmp(B::ugt, x, y);
          case P::uge: return bvcmp(B::uge, x, y);
          }
          llvm_unreachable("covered switch");
        })
        // Integer width casts. i1 sources/results map to/from smt.bool; wider
        // widths are pure bit-vector zext/sext/truncate, which are total (no
        // well-definedness condition needed) once their poison flags are
        // rejected.
        .Case<arith::ExtUIOp>([&](arith::ExtUIOp o) -> Value {
          if (o.getNonNeg()) {
            why = "extui nneg flag introduces poison we do not model";
            return Value();
          }
          unsigned dw = elemOf(o.getType()).getIntOrFloatBitWidth();
          if (elemOf(o.getIn().getType()).isInteger(1))
            return ite(A(o.getIn()), bvc(APInt(dw, 1)), bvc(APInt(dw, 0)));
          Value x = A(o.getIn());
          unsigned sw = cast<smt::BitVectorType>(x.getType()).getWidth();
          // zext = concat(0^{dw-sw}, x)
          return smt::ConcatOp::create(b, loc, bvc(APInt(dw - sw, 0)), x)
              .getResult();
        })
        .Case<arith::ExtSIOp>([&](arith::ExtSIOp o) -> Value {
          unsigned dw = elemOf(o.getType()).getIntOrFloatBitWidth();
          if (elemOf(o.getIn().getType()).isInteger(1))
            return ite(A(o.getIn()), bvc(APInt::getAllOnes(dw)),
                       bvc(APInt(dw, 0)));
          Value x = A(o.getIn());
          unsigned sw = cast<smt::BitVectorType>(x.getType()).getWidth();
          // sext = ite(x <s 0, concat(1^{dw-sw}, x), concat(0^{dw-sw}, x))
          Value isNeg = bvcmp(smt::BVCmpPredicate::slt, x, bvc(APInt(sw, 0)));
          Value hi = ite(isNeg, bvc(APInt::getAllOnes(dw - sw)),
                         bvc(APInt(dw - sw, 0)));
          return smt::ConcatOp::create(b, loc, hi, x).getResult();
        })
        .Case<arith::TruncIOp>([&](arith::TruncIOp o) -> Value {
          if (o.getOverflowFlags() != arith::IntegerOverflowFlags::none) {
            why = "arith overflow flags (nsw/nuw) are not modeled";
            return Value();
          }
          Value x = A(o.getIn());
          unsigned dw = elemOf(o.getType()).getIntOrFloatBitWidth();
          if (dw == 1) {
            // trunc to i1 takes the lowest bit; the result sort is smt.bool.
            auto bit0Ty = smt::BitVectorType::get(op->getContext(), 1);
            Value bit0 =
                smt::ExtractOp::create(b, loc, bit0Ty, /*lowBit=*/0, x)
                    .getResult();
            return beq(bit0, bvc(APInt(1, 1)));
          }
          auto dstTy = smt::BitVectorType::get(op->getContext(), dw);
          return smt::ExtractOp::create(b, loc, dstTy, /*lowBit=*/0, x)
              .getResult();
        })
        .Case<arith::UIToFPOp>([&](arith::UIToFPOp o) -> Value {
          if (o.getNonNeg()) {
            why = "uitofp nneg flag introduces poison we do not model";
            return Value();
          }
          if (!elemOf(o.getIn().getType()).isInteger(1)) {
            why = "only i1 sources are supported for uitofp";
            return Value();
          }
          return ite(A(o.getIn()), realConst("1.0"), realConst("0.0"));
        })
        .Case<arith::SelectOp>([&](arith::SelectOp o) -> Value {
          // A pointer-valued select has no plain/lane SMT value to fetch;
          // reject it (the writer model expects one statically-known output
          // array).
          auto isPtrVal = [&](Value v) {
            auto it = env.find(v);
            return it != env.end() && it->second.isPtr();
          };
          if (isPtrVal(o.getTrueValue()) || isPtrVal(o.getFalseValue())) {
            why = "pointer-valued select is not supported";
            return Value();
          }
          return ite(A(o.getCondition()), A(o.getTrueValue()),
                     A(o.getFalseValue()));
        })
        .Default([&](Operation *) { return Value(); });
  };

  // Reduction (vector) mode: encode one elementwise op at lane `k`,
  // broadcasting scalar operands, under exactly the scalar rules.
  auto encodeVectorLane = [&](Operation *op, unsigned k,
                              std::string &why) -> Value {
    auto A = [&](Value v) -> Value {
      auto it = env.find(v);
      return it == env.end() ? Value() : it->second.laneVal(k);
    };
    return encodeElementwise(op, A, why);
  };

  for (Operation &opRef : entry) {
    Operation *op = &opRef;

    // Only scalars and ranked tensors are modeled. Any other shaped type
    // (e.g. vector<4xi32>, which is verifier-valid for arith ops) would flow
    // through elemOf() unstripped and crash width queries downstream; reject
    // it up front.
    auto badType = [](Type t) {
      return isa<ShapedType>(t) && !isa<RankedTensorType>(t);
    };
    for (Type t : op->getOperandTypes())
      if (badType(t))
        return err(op, "unsupported shaped operand type (only ranked tensors "
                       "are modeled)");
    for (Type t : op->getResultTypes())
      if (badType(t))
        return err(op, "unsupported shaped result type (only ranked tensors "
                       "are modeled)");

    // Reduction mode: intercept the ops that must be materialized per-lane or
    // folded, before the scalar elementwise TypeSwitch. Everything scalar
    // (program_id, splat, constants, post-reduce arithmetic, scalar addptr, the
    // scalar store) still falls through to the switch below.
    if (reductionMode) {
      unsigned N = (unsigned)reducedN;
      // make_range -> a concrete lane vector [start, start+N).
      if (auto o = dyn_cast<triton::MakeRangeOp>(op)) {
        SmallVector<Value> ls;
        ls.reserve(N);
        for (unsigned k = 0; k < N; ++k)
          ls.push_back(addrConst((uint64_t)o.getStart() + k));
        env[o.getResult()] = EncodedValue::vec(std::move(ls));
        continue;
      }
      // tt.reduce.return is consumed while reading the combiner.
      if (isa<triton::ReduceReturnOp>(op))
        continue;
      // tt.reduce -> fold the N input lanes with the combiner into a scalar.
      if (auto o = dyn_cast<triton::ReduceOp>(op)) {
        Operation *comb = o.getSingleCombiner();
        if (!comb)
          return err(o, "unsupported tt.reduce combiner region (non-canonical "
                        "or fused arg-reduce)");
        // getSingleCombiner() only checks the *yielded* op; it tolerates extra
        // ops in the region (e.g. a side-effecting/volatile load) which this
        // pass would silently drop. Require the region to be exactly
        // {combiner, reduce.return}.
        Block &combBlock = o.getCombineOp().front();
        if (combBlock.getOperations().size() != 2)
          return err(o, "tt.reduce combiner region must contain only the "
                        "combining op and the terminator");
        if (!isa<arith::AddFOp, arith::MaxNumFOp, arith::MaximumFOp,
                 arith::MinNumFOp, arith::MinimumFOp>(comb))
          return err(o, "unsupported tt.reduce combiner '" +
                            comb->getName().getStringRef() +
                            "' (supported: addf, max/minnumf, max/minimumf)");
        EncodedValue in = env[o->getOperand(0)];
        if (!in.isVector() || in.lanes.size() != N)
          return err(o, "reduction input is not a materialized lane vector");
        // Float (ideal-real) reductions only in this milestone.
        if (!isa<smt::RealType>(in.lanes.front().getType()))
          return err(o, "only floating-point (real) reductions are supported");
        // The combiner op's semantics come from the shared elementwise
        // encoder, with the region's block arguments bound to (acc, lane).
        // Folding the lanes in a fixed linear order is only valid because the
        // whitelist above admits combiners that are associative and
        // commutative over ideal reals.
        auto combine = [&](Value acc, Value x) -> Value {
          auto A = [&](Value v) -> Value {
            if (v == combBlock.getArgument(0))
              return acc;
            if (v == combBlock.getArgument(1))
              return x;
            return Value();
          };
          std::string why;
          return encodeElementwise(comb, A, why);
        };
        Value acc = in.lanes[0];
        for (unsigned k = 1; k < N; ++k) {
          acc = combine(acc, in.lanes[k]);
          if (!acc)
            return err(o, "failed to encode tt.reduce combiner");
        }
        env[o->getResult(0)] = EncodedValue::plain(acc);
        continue;
      }
      // addptr with a per-lane offset (or vector base) -> vector pointer.
      if (auto o = dyn_cast<triton::AddPtrOp>(op)) {
        EncodedValue base = env[o.getPtr()];
        EncodedValue off = env[o.getOffset()];
        if (base.isVector() || off.isVector()) {
          Type offElem = o.getOffset().getType();
          if (auto st = dyn_cast<RankedTensorType>(offElem))
            offElem = st.getElementType();
          if (!offElem.isInteger(kAddrWidth))
            return err(o, "only 32-bit pointer offsets are supported");
          if (!base.isPtr() || !base.fresh)
            return err(o, "vector tt.addptr requires a fresh base pointer "
                          "(chained addptr is not supported)");
          SmallVector<Value> offs;
          offs.reserve(N);
          for (unsigned k = 0; k < N; ++k)
            offs.push_back(smt::BVAddOp::create(b, loc, base.laneOff(k),
                                                off.laneVal(k))
                               .getResult());
          env[o.getResult()] = EncodedValue::vecPtr(base.array, std::move(offs));
          continue;
        }
        // scalar addptr -> fall through to the existing case.
      }
      // load through a per-lane vector pointer -> a vector of selects.
      if (auto o = dyn_cast<triton::LoadOp>(op)) {
        EncodedValue p = env[o.getPtr()];
        if (p.isVector()) {
          if (o.getIsVolatile())
            return err(o, "volatile loads are observable and not modeled");
          if (o.getMask())
            return err(o, "masked load feeding a reduction is not supported "
                          "(masked-out lanes need the combiner identity)");
          Type range = cast<smt::ArrayType>(p.array.getType()).getRangeType();
          Type want = smtSortFor(elemOf(o.getType()), ctx);
          if (!want)
            return err(o, "unsupported loaded element type");
          if (want != range)
            return err(o, "load reinterprets the buffer's element sort via a "
                          "pointer bitcast; not supported");
          SmallVector<Value> ls;
          ls.reserve(N);
          for (unsigned k = 0; k < N; ++k)
            ls.push_back(sel(p.array, p.lanes[k], range));
          env[o.getResult()] = EncodedValue::vec(std::move(ls));
          continue;
        }
        // scalar load -> fall through.
      }
      // tt.reshape feeding a reduction. tl.sum/tl.max emit an identity
      // `tt.reshape allow_reorder` (N -> N) whose *only* user is the tt.reduce.
      // The lanes pass through unchanged: order is irrelevant because the
      // combiner is associative+commutative over ideal reals (and allow_reorder
      // permits any order anyway). We require the sole user to be a tt.reduce:
      // `allow_reorder` leaves the element order unspecified, so if the result
      // fed lane-wise arithmetic the pairing would be nondeterministic and the
      // identity pass-through could hide a real difference. Any other use (or a
      // non-identity shape) falls through and is rejected below.
      if (auto o = dyn_cast<triton::ReshapeOp>(op)) {
        EncodedValue in = env[o.getSrc()];
        auto st = dyn_cast<RankedTensorType>(o.getType());
        bool soleReduceUser =
            o.getResult().hasOneUse() &&
            isa<triton::ReduceOp>(*o.getResult().getUsers().begin());
        if (in.isVector() && soleReduceUser && st && st.getRank() == 1 &&
            st.hasStaticShape() && st.getShape()[0] == (int64_t)N) {
          env[o.getResult()] = in;
          continue;
        }
        // else fall through -> rejected as an unsupported vector op.
      }
      // Any other op with a vector operand is elementwise: encode per lane.
      bool anyVec = false;
      for (Value operand : op->getOperands()) {
        auto it = env.find(operand);
        if (it != env.end() && it->second.isVector()) {
          anyVec = true;
          break;
        }
      }
      if (anyVec) {
        SmallVector<Value> ls(N);
        for (unsigned k = 0; k < N; ++k) {
          std::string why;
          Value r = encodeVectorLane(op, k, why);
          if (!r)
            return err(op, why.empty()
                               ? ("operation '" + op->getName().getStringRef() +
                                  "' is not supported in reduction (vector) "
                                  "context")
                                     .str()
                               : why);
          ls[k] = r;
        }
        env[op->getResult(0)] = EncodedValue::vec(std::move(ls));
        continue;
      }
    }

    // Pure elementwise ops are encoded by the shared encoder (scalar context:
    // operands are the plain values). A rejection there is final; a "not
    // elementwise" miss falls through to the structural cases below.
    {
      std::string why;
      Value r = encodeElementwise(op, V, why);
      if (r) {
        env[op->getResult(0)] = EncodedValue::plain(r);
        continue;
      }
      if (!why.empty())
        return err(op, why);
    }

    bool ok =
        llvm::TypeSwitch<Operation *, bool>(op)
            .Case<triton::FuncOp, triton::ReturnOp>([&](auto) { return true; })
            .Case<triton::GetProgramIdOp>([&](triton::GetProgramIdOp o) {
              if (o.getAxisAsInt() != 0)
                return err(o, "only program_id axis 0 is supported");
              env[o.getResult()] = EncodedValue::plain(pid);
              return true;
            })
            .Case<triton::MakeRangeOp>([&](triton::MakeRangeOp o) {
              Value v = lane;
              if (o.getStart() != 0)
                v = smt::BVAddOp::create(b, loc, addrConst(o.getStart()), lane)
                        .getResult();
              env[o.getResult()] = EncodedValue::plain(v);
              return true;
            })
            .Case<arith::ConstantOp>([&](arith::ConstantOp o) {
              Attribute a = o.getValue();
              if (auto fa = dyn_cast<FloatAttr>(a)) {
                auto s = formatReal(fa.getValue());
                if (!s)
                  return err(o, "unsupported float constant (non-finite or "
                                "exceeds f64 precision)");
                env[o.getResult()] = EncodedValue::plain(
                    smt::RealConstantOp::create(b, loc, R, b.getStringAttr(*s))
                        .getResult());
              } else if (auto ia = dyn_cast<IntegerAttr>(a)) {
                if (ia.getType().getIntOrFloatBitWidth() == 1)
                  env[o.getResult()] = EncodedValue::plain(
                      smt::BoolConstantOp::create(
                          b, loc, Bool, b.getBoolAttr(ia.getValue() != 0))
                          .getResult());
                else
                  env[o.getResult()] = EncodedValue::plain(bvc(ia.getValue()));
              } else if (auto dea = dyn_cast<DenseElementsAttr>(a)) {
                if (!dea.isSplat())
                  return err(o, "non-splat dense constant is unsupported");
                Type et = dea.getElementType();
                if (isa<FloatType>(et)) {
                  auto s = formatReal(dea.getSplatValue<APFloat>());
                  if (!s)
                    return err(o, "unsupported float constant (non-finite or "
                                "exceeds f64 precision)");
                  env[o.getResult()] = EncodedValue::plain(
                      smt::RealConstantOp::create(b, loc, R, b.getStringAttr(*s))
                          .getResult());
                } else if (auto it = dyn_cast<IntegerType>(et)) {
                  if (it.getWidth() == 1)
                    env[o.getResult()] = EncodedValue::plain(
                        smt::BoolConstantOp::create(
                            b, loc, Bool,
                            b.getBoolAttr(!dea.getSplatValue<APInt>().isZero()))
                            .getResult());
                  else
                    env[o.getResult()] = EncodedValue::plain(
                        bvc(dea.getSplatValue<APInt>()));
                } else
                  return err(o, "unsupported constant element type");
              } else
                return err(o, "unsupported constant");
              return true;
            })
            .Case<triton::SplatOp>([&](triton::SplatOp o) {
              env[o.getResult()] = env[o.getSrc()];
              return true;
            })
            // Pointers / memory.
            .Case<triton::BitcastOp>([&](triton::BitcastOp o) {
              auto srcP =
                  dyn_cast<triton::PointerType>(elemOf(o.getSrc().getType()));
              auto dstP = dyn_cast<triton::PointerType>(elemOf(o.getType()));
              if (!srcP || !dstP)
                return err(o, "only pointer-to-pointer bitcast is supported");
              Type se = srcP.getPointeeType(), de = dstP.getPointeeType();
              // Only integer<->integer reinterpretations that keep the addressing
              // granularity (byte size) are sound in this value-based model:
              // float bitcasts reinterpret bits (e.g. f16 vs bf16 map to the same
              // real yet differ in memory) and size changes alter the element
              // stride. The i1<->i8 byte-backed-bool idiom is preserved.
              auto intBytes = [](Type t) -> int {
                auto it = dyn_cast<IntegerType>(t);
                return it ? int((it.getWidth() + 7) / 8) : -1;
              };
              if (se != de && (intBytes(se) < 0 || intBytes(de) < 0 ||
                               intBytes(se) != intBytes(de)))
                return err(o, "unsupported pointer bitcast: only a same-byte-size "
                              "integer pointee reinterpretation is modeled");
              env[o.getResult()] = env[o.getSrc()];
              return true;
            })
            .Case<triton::AddPtrOp>([&](triton::AddPtrOp o) {
              // Offsets must be 32-bit (the address domain width).
              Type offElem = o.getOffset().getType();
              if (auto st = dyn_cast<RankedTensorType>(offElem))
                offElem = st.getElementType();
              if (!offElem.isInteger(kAddrWidth))
                return err(o, "only 32-bit pointer offsets are supported");
              EncodedValue base = env[o.getPtr()];
              // A single addptr with a valid (non-negative, in-range) index is
              // sound at 32 bits, but chaining addptr accumulates offsets that
              // real 64-bit pointer arithmetic would not wrap; reject chains.
              if (!base.fresh)
                return err(o, "chained tt.addptr is not supported (32-bit offset "
                              "accumulation can differ from 64-bit pointer "
                              "arithmetic)");
              env[o.getResult()] = EncodedValue::ptr(
                  base.array,
                  smt::BVAddOp::create(b, loc, base.offset, V(o.getOffset()))
                      .getResult(),
                  /*fresh=*/false);
              return true;
            })
            .Case<triton::LoadOp>([&](triton::LoadOp o) {
              // A volatile load is an observable side effect; since the model
              // has no memory-access trace, a dead volatile read would vanish.
              if (o.getIsVolatile())
                return err(o, "volatile loads are observable and not modeled");
              EncodedValue p = env[o.getPtr()];
              Type range = cast<smt::ArrayType>(p.array.getType()).getRangeType();
              Type want = smtSortFor(elemOf(o.getType()), ctx);
              if (!want)
                return err(o, "unsupported loaded element type");
              // Reading a buffer at a different sort than it was declared (a raw
              // load through a pointer bitcast, e.g. an i1/byte-backed buffer
              // read as i8) cannot be modeled faithfully: the ideal value drops
              // the raw byte, collapsing distinct bytes. Reject it -- the
              // i1<->i8 bitcast idiom is only sound on the store side.
              if (want != range)
                return err(o, "load reinterprets the buffer's element sort via a "
                              "pointer bitcast; not supported");
              Value loaded = sel(p.array, p.offset, range);
              // A masked load without `other` leaves masked-out lanes undefined
              // in Triton; ignoring the mask would be unsound, so require the
              // fallback and model it explicitly.
              if (o.getMask()) {
                if (!o.getOther())
                  return err(o, "masked load without a fallback `other` is "
                                "undefined and not supported");
                loaded = ite(V(o.getMask()), loaded, V(o.getOther()));
              }
              env[o.getResult()] = EncodedValue::plain(loaded);
              return true;
            })
            .Case<triton::StoreOp>([&](triton::StoreOp o) {
              EncodedValue p = env[o.getPtr()];
              // Store through a pointer bitcast: coerce the value to the
              // buffer's declared sort so the final-state comparison is typed
              // by the buffer (an i8 written into an i1 buffer means
              // "nonzero").
              Type range = cast<smt::ArrayType>(p.array.getType()).getRangeType();
              Value v = coerce(V(o.getValue()), range);
              if (!v)
                return err(o, "stored value sort is incompatible with the "
                              "buffer sort");
              stores.push_back(
                  {p.array, p.offset, v, o.getMask() ? V(o.getMask()) : Value()});
              return true;
            })
            .Default([&](Operation *o) {
              return err(o, "unsupported operation '" +
                                o->getName().getStringRef() + "'");
            });
    if (!ok)
      return false;
  }

  if (stores.size() != 1)
    return err(func, "exactly one store is required (found " +
                         Twine(stores.size()) + ")");
  return true;
}

void ConvertTritonToSMT::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  Location loc = module.getLoc();

  // The module must contain only tt.func ops. A stray smt.solver (or any other
  // op) would otherwise be preserved and exported, forging extra solver results
  // that the driver could misread as an EQUIVALENT verdict.
  for (Operation &op : *module.getBody())
    if (!isa<triton::FuncOp>(op)) {
      module.emitError("TritonToSMT: module must contain only tt.func ops (found '")
          << op.getName().getStringRef() << "')";
      return signalPassFailure();
    }

  // Select the source and target functions.
  auto funcs = llvm::to_vector(module.getOps<triton::FuncOp>());
  auto byName = [&](StringRef n) -> triton::FuncOp {
    for (auto f : funcs)
      if (f.getName() == n)
        return f;
    return {};
  };
  triton::FuncOp src, tgt;
  if (!srcFunc.empty() || !tgtFunc.empty()) {
    src = byName(srcFunc);
    tgt = byName(tgtFunc);
  } else if (funcs.size() == 2) {
    src = funcs[0];
    tgt = funcs[1];
  }
  if (!src || !tgt) {
    module.emitError("TritonToSMT: expected exactly two functions (or use "
                     "--src=/--tgt=)");
    return signalPassFailure();
  }
  // Require identical full signatures (types, not just arity).
  if (src.getFunctionType() != tgt.getFunctionType()) {
    module.emitError("TritonToSMT: source and target signatures differ");
    return signalPassFailure();
  }
  // Only the memory effect (stores) is compared; a non-void result would be an
  // observable output that is never encoded, so reject it.
  if (src.getFunctionType().getNumResults() != 0) {
    module.emitError("TritonToSMT: functions with non-void results are "
                     "unsupported (return values are not compared)");
    return signalPassFailure();
  }

  auto argTypes = llvm::to_vector(src.getArgumentTypes());
  OpBuilder builder(ctx);

  // Build one solver scope, returning the terminating check op's builder state.
  // `mode`: 0 = addressing-identity check, 1 = well-definedness check,
  // 2 = equivalence check. All three scopes are always emitted, in this order,
  // so the driver's scope-position contract stays fixed.
  auto buildScope = [&](int mode) -> bool {
    builder.setInsertionPointToEnd(module.getBody());
    auto solver = smt::SolverOp::create(builder, loc, TypeRange{}, ValueRange{});
    Block *body = builder.createBlock(&solver.getBodyRegion());
    builder.setInsertionPointToStart(body);
    smt::SetLogicOp::create(builder, loc, builder.getStringAttr("ALL"));

    Value iVal;
    SmallVector<EncodedValue> shared;
    if (!declareInputs(builder, loc, argTypes, iVal, shared))
      return false;

    SmallVector<StoreInfo> storesS, storesT;
    SmallVector<Value> wdConds;
    if (!encodeFunction(builder, src, shared, iVal, storesS, wdConds) ||
        !encodeFunction(builder, tgt, shared, iVal, storesT, wdConds))
      return false;
    const StoreInfo &s = storesS.front(), &t = storesT.front();

    Value assertion;
    if (mode == 0) {
      // Addressing must be identity: storeOffset == i. Assert the negation; if
      // unsat, identity holds for both functions.
      Value ds = smt::DistinctOp::create(builder, loc, s.offset, iVal).getResult();
      Value dt = smt::DistinctOp::create(builder, loc, t.offset, iVal).getResult();
      assertion = smt::OrOp::create(builder, loc, ds, dt).getResult();
    } else if (mode == 1) {
      // Well-definedness: assert the negation of the conjunction of all
      // collected conditions (both functions). unsat => no input can reach a
      // poison case (e.g. an oversized shift), so the total SMT encodings are
      // faithful. With no conditions the scope is trivially unsat.
      if (wdConds.empty()) {
        assertion = smt::BoolConstantOp::create(
                        builder, loc, smt::BoolType::get(ctx),
                        builder.getBoolAttr(false))
                        .getResult();
      } else {
        Value conj = wdConds.front();
        for (size_t k = 1; k < wdConds.size(); ++k)
          conj = smt::AndOp::create(builder, loc, conj, wdConds[k]).getResult();
        assertion = smt::NotOp::create(builder, loc, conj).getResult();
      }
    } else {
      // Compare the final state of every written output array at index i.
      auto finalOf = [&](const StoreInfo &st, Value arr) -> Value {
        Type range = cast<smt::ArrayType>(arr.getType()).getRangeType();
        Value init =
            smt::ArraySelectOp::create(builder, loc, range, arr, iVal).getResult();
        if (st.array != arr)
          return init; // this function does not write `arr`
        Value stored = st.value;
        if (st.mask)
          stored = smt::IteOp::create(builder, loc, range, st.mask, st.value, init)
                       .getResult();
        return stored;
      };
      SmallVector<Value> arrays{s.array};
      if (t.array != s.array)
        arrays.push_back(t.array);
      SmallVector<Value> diffs;
      for (Value arr : arrays) {
        diffs.push_back(smt::DistinctOp::create(builder, loc, finalOf(s, arr),
                                                finalOf(t, arr))
                            .getResult());
      }
      assertion = diffs.front();
      for (size_t k = 1; k < diffs.size(); ++k)
        assertion =
            smt::OrOp::create(builder, loc, assertion, diffs[k]).getResult();
    }
    smt::AssertOp::create(builder, loc, assertion);

    auto check = smt::CheckOp::create(builder, loc, TypeRange{});
    fillCheckRegions(builder, loc, check);
    builder.setInsertionPointToEnd(body);
    smt::YieldOp::create(builder, loc, ValueRange{});
    return true;
  };

  if (!buildScope(/*addressing=*/0) || !buildScope(/*well-definedness=*/1) ||
      !buildScope(/*equivalence=*/2))
    return signalPassFailure();

  // Remove all non-SMT ops so the module can be exported with mlir-translate.
  SmallVector<Operation *> toErase;
  for (Operation &op : *module.getBody())
    if (!isa<smt::SolverOp>(op))
      toErase.push_back(&op);
  for (Operation *op : toErase)
    op->erase();
}

} // namespace
