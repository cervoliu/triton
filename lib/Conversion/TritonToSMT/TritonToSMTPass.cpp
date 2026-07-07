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
// It emits two solver scopes: scope 0 checks that addressing is identity (unsat
// => identity holds); scope 1 asserts the two functions' output arrays can
// differ (unsat => equivalent). See Passes.td.
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

// An encoded Triton SSA value: a plain SMT value, or a pointer (memory array +
// bit-vector offset). The array symbol identifies which output buffer is meant.
struct EncodedValue {
  Value value;  // plain value (null for pointers)
  Value array;  // pointer: the memory array symbol
  Value offset; // pointer: the bit-vector element offset
  bool fresh = false; // pointer: true iff no tt.addptr has been applied yet
  bool isPtr() const { return array != nullptr; }
  static EncodedValue plain(Value v) { return {v, nullptr, nullptr, false}; }
  static EncodedValue ptr(Value a, Value o, bool fresh = false) {
    return {nullptr, a, o, fresh};
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
  // function's stores to `stores`. Returns false (and emits an error) if the
  // function uses an unsupported construct.
  bool encodeFunction(OpBuilder &b, triton::FuncOp func,
                      ArrayRef<EncodedValue> sharedArgs, Value iVal,
                      SmallVectorImpl<StoreInfo> &stores);
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
                                        SmallVectorImpl<StoreInfo> &stores) {
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
  // Reconcile a value with an expected SMT sort at a load/store boundary.
  // Bool<->bit-vector coercions model byte-backed i1 buffers (reading: zero is
  // false, nonzero is true; writing: true is stored as 1). Returns null for
  // incompatible sorts.
  auto coerce = [&](Value v, Type sort) -> Value {
    if (v.getType() == sort)
      return v;
    if (isa<smt::BoolType>(v.getType()))
      if (auto bvTy = dyn_cast<smt::BitVectorType>(sort))
        return ite(v, bvc(APInt(bvTy.getWidth(), 1)),
                   bvc(APInt(bvTy.getWidth(), 0)));
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
  // Every tt.make_range must span exactly the store's lane count, or the writer
  // decomposition (pid = i / block, lane = i % block) would misattribute lanes.
  for (Operation &op : entry)
    if (auto range = dyn_cast<triton::MakeRangeOp>(op)) {
      int64_t ext = int64_t(range.getEnd()) - int64_t(range.getStart());
      if (ext != block) {
        range.emitError("TritonToSMT: tt.make_range extent (")
            << ext << ") does not match the store lane count (" << block << ")";
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

  for (Operation &opRef : entry) {
    Operation *op = &opRef;
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
            // Integer (bit-vector) arithmetic. `i1` operands are SMT booleans,
            // not bit-vectors, so reject them (mapping to smt.bv.* would crash);
            // and nsw/nuw overflow flags introduce poison we do not model.
            .Case<arith::AddIOp>([&](arith::AddIOp o) {
              if (isa<smt::BoolType>(V(o.getLhs()).getType()))
                return err(o, "i1 integer arithmetic is not supported");
              if (o.getOverflowFlags() != arith::IntegerOverflowFlags::none)
                return err(o, "arith overflow flags (nsw/nuw) are not modeled");
              env[o.getResult()] = EncodedValue::plain(
                  smt::BVAddOp::create(b, loc, V(o.getLhs()), V(o.getRhs()))
                      .getResult());
              return true;
            })
            .Case<arith::SubIOp>([&](arith::SubIOp o) {
              if (isa<smt::BoolType>(V(o.getLhs()).getType()))
                return err(o, "i1 integer arithmetic is not supported");
              if (o.getOverflowFlags() != arith::IntegerOverflowFlags::none)
                return err(o, "arith overflow flags (nsw/nuw) are not modeled");
              // The smt dialect has no bvsub; use x + (-y).
              Value negY = smt::BVNegOp::create(b, loc, V(o.getRhs())).getResult();
              env[o.getResult()] = EncodedValue::plain(
                  smt::BVAddOp::create(b, loc, V(o.getLhs()), negY).getResult());
              return true;
            })
            .Case<arith::MulIOp>([&](arith::MulIOp o) {
              if (isa<smt::BoolType>(V(o.getLhs()).getType()))
                return err(o, "i1 integer arithmetic is not supported");
              if (o.getOverflowFlags() != arith::IntegerOverflowFlags::none)
                return err(o, "arith overflow flags (nsw/nuw) are not modeled");
              env[o.getResult()] = EncodedValue::plain(
                  smt::BVMulOp::create(b, loc, V(o.getLhs()), V(o.getRhs()))
                      .getResult());
              return true;
            })
            // `i1` values are SMT booleans, wider integers are bit-vectors, so
            // the bitwise ops dispatch on the encoded operand sort.
            .Case<arith::AndIOp>([&](arith::AndIOp o) {
              Value x = V(o.getLhs()), y = V(o.getRhs());
              Value r;
              if (isa<smt::BoolType>(x.getType()))
                r = band(x, y);
              else
                r = smt::BVAndOp::create(b, loc, x, y).getResult();
              env[o.getResult()] = EncodedValue::plain(r);
              return true;
            })
            .Case<arith::OrIOp>([&](arith::OrIOp o) {
              Value x = V(o.getLhs()), y = V(o.getRhs());
              Value r;
              if (isa<smt::BoolType>(x.getType()))
                r = smt::OrOp::create(b, loc, x, y).getResult();
              else
                r = smt::BVOrOp::create(b, loc, x, y).getResult();
              env[o.getResult()] = EncodedValue::plain(r);
              return true;
            })
            .Case<arith::XOrIOp>([&](arith::XOrIOp o) {
              Value x = V(o.getLhs()), y = V(o.getRhs());
              Value r;
              if (isa<smt::BoolType>(x.getType()))
                r = smt::XOrOp::create(b, loc, x, y).getResult();
              else
                r = smt::BVXOrOp::create(b, loc, x, y).getResult();
              env[o.getResult()] = EncodedValue::plain(r);
              return true;
            })
            // Integer division, remainder, and shifts are intentionally NOT
            // encoded. SMT bit-vector div/rem are total (e.g. bvudiv by zero is
            // all-ones) and bv shifts are defined for oversized amounts, whereas
            // the corresponding arith ops are undefined there. Without a
            // definedness/poison model, encoding them as total operations is
            // unsound (a kernel storing `1/0` would look equivalent to one
            // storing `-1`), so they fall through to Default and are rejected.
            .Case<arith::CmpIOp>([&](arith::CmpIOp o) {
              Value x = V(o.getLhs()), y = V(o.getRhs());
              using P = arith::CmpIPredicate;
              using B = smt::BVCmpPredicate;
              Value r;
              switch (o.getPredicate()) {
              case P::eq: r = beq(x, y); break;
              case P::ne: r = bnot(beq(x, y)); break;
              case P::slt: r = bvcmp(B::slt, x, y); break;
              case P::sle: r = bvcmp(B::sle, x, y); break;
              case P::sgt: r = bvcmp(B::sgt, x, y); break;
              case P::sge: r = bvcmp(B::sge, x, y); break;
              case P::ult: r = bvcmp(B::ult, x, y); break;
              case P::ule: r = bvcmp(B::ule, x, y); break;
              case P::ugt: r = bvcmp(B::ugt, x, y); break;
              case P::uge: r = bvcmp(B::uge, x, y); break;
              }
              env[o.getResult()] = EncodedValue::plain(r);
              return true;
            })
            // Casts. Only the bool-to-storage idioms are encoded: comparison
            // results (i1) widened for storing as bytes or floats. All other
            // width- or domain-changing casts remain rejected.
            .Case<arith::ExtUIOp>([&](arith::ExtUIOp o) {
              if (o.getNonNeg())
                return err(o, "extui nneg flag introduces poison we do not model");
              if (!elemOf(o.getIn().getType()).isInteger(1))
                return err(o, "only i1 sources are supported for extui");
              unsigned w = elemOf(o.getType()).getIntOrFloatBitWidth();
              env[o.getResult()] = EncodedValue::plain(
                  ite(V(o.getIn()), bvc(APInt(w, 1)), bvc(APInt(w, 0))));
              return true;
            })
            .Case<arith::UIToFPOp>([&](arith::UIToFPOp o) {
              if (o.getNonNeg())
                return err(o, "uitofp nneg flag introduces poison we do not model");
              if (!elemOf(o.getIn().getType()).isInteger(1))
                return err(o, "only i1 sources are supported for uitofp");
              env[o.getResult()] = EncodedValue::plain(
                  ite(V(o.getIn()), realConst("1.0"), realConst("0.0")));
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
            // Floating-point (ideal real) arithmetic.
            .Case<arith::MulFOp>([&](arith::MulFOp o) {
              env[o.getResult()] =
                  EncodedValue::plain(realBin(V(o.getLhs()), V(o.getRhs()), true));
              return true;
            })
            .Case<arith::AddFOp>([&](arith::AddFOp o) {
              env[o.getResult()] =
                  EncodedValue::plain(realBin(V(o.getLhs()), V(o.getRhs()), false));
              return true;
            })
            .Case<arith::SubFOp>([&](arith::SubFOp o) {
              env[o.getResult()] = EncodedValue::plain(
                  smt::RealSubOp::create(b, loc, R, V(o.getLhs()), V(o.getRhs()))
                      .getResult());
              return true;
            })
            .Case<arith::NegFOp>([&](arith::NegFOp o) {
              env[o.getResult()] = EncodedValue::plain(
                  smt::RealNegOp::create(b, loc, R, V(o.getOperand()))
                      .getResult());
              return true;
            })
            .Case<arith::DivFOp>([&](arith::DivFOp o) {
              // SMT-LIB real division is total but unspecified at zero
              // denominators; both functions see the same division function,
              // so the equivalence query stays sound.
              env[o.getResult()] = EncodedValue::plain(
                  smt::RealDivOp::create(b, loc, R, V(o.getLhs()),
                                         V(o.getRhs()))
                      .getResult());
              return true;
            })
            .Case<math::AbsFOp>([&](math::AbsFOp o) {
              Value x = V(o.getOperand());
              Value isNeg = smt::RealCmpOp::create(
                                b, loc, Bool, smt::IntPredicate::lt, x,
                                realConst("0.0"))
                                .getResult();
              Value negX = smt::RealNegOp::create(b, loc, R, x).getResult();
              env[o.getResult()] = EncodedValue::plain(ite(isNeg, negX, x));
              return true;
            })
            .Case<math::FmaOp>([&](math::FmaOp o) {
              Value prod = realBin(V(o.getOperand(0)), V(o.getOperand(1)), true);
              env[o.getResult()] =
                  EncodedValue::plain(realBin(prod, V(o.getOperand(2)), false));
              return true;
            })
            .Case<arith::CmpFOp>([&](arith::CmpFOp o) {
              Value x = V(o.getLhs()), y = V(o.getRhs());
              using P = arith::CmpFPredicate;
              using I = smt::IntPredicate;
              auto rc = [&](I p) {
                return smt::RealCmpOp::create(b, loc, Bool, p, x, y).getResult();
              };
              Value r;
              switch (o.getPredicate()) {
              case P::OEQ: case P::UEQ: r = beq(x, y); break;
              case P::ONE: case P::UNE: r = bnot(beq(x, y)); break;
              case P::OGT: case P::UGT: r = rc(I::gt); break;
              case P::OGE: case P::UGE: r = rc(I::ge); break;
              case P::OLT: case P::ULT: r = rc(I::lt); break;
              case P::OLE: case P::ULE: r = rc(I::le); break;
              default: return err(o, "unsupported cmpf predicate");
              }
              env[o.getResult()] = EncodedValue::plain(r);
              return true;
            })
            .Case<arith::SelectOp>([&](arith::SelectOp o) {
              // Pointer-valued select has a null plain value; dereferencing it
              // via V() would crash. Reject it (the writer model expects one
              // statically-known output array).
              if (env[o.getTrueValue()].isPtr() || env[o.getFalseValue()].isPtr())
                return err(o, "pointer-valued select is not supported");
              env[o.getResult()] = EncodedValue::plain(
                  ite(V(o.getCondition()), V(o.getTrueValue()),
                      V(o.getFalseValue())));
              return true;
            })
            .Case<arith::MaxNumFOp, arith::MaximumFOp>([&](auto o) {
              Value x = V(o.getLhs()), y = V(o.getRhs());
              env[o.getResult()] = EncodedValue::plain(
                  ite(smt::RealCmpOp::create(b, loc, Bool, smt::IntPredicate::ge,
                                             x, y)
                          .getResult(),
                      x, y));
              return true;
            })
            .Case<arith::MinNumFOp, arith::MinimumFOp>([&](auto o) {
              Value x = V(o.getLhs()), y = V(o.getRhs());
              env[o.getResult()] = EncodedValue::plain(
                  ite(smt::RealCmpOp::create(b, loc, Bool, smt::IntPredicate::le,
                                             x, y)
                          .getResult(),
                      x, y));
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
  // `mode`: 0 = addressing-identity check, 1 = equivalence check.
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
    if (!encodeFunction(builder, src, shared, iVal, storesS) ||
        !encodeFunction(builder, tgt, shared, iVal, storesT))
      return false;
    const StoreInfo &s = storesS.front(), &t = storesT.front();

    Value assertion;
    if (mode == 0) {
      // Addressing must be identity: storeOffset == i. Assert the negation; if
      // unsat, identity holds for both functions.
      Value ds = smt::DistinctOp::create(builder, loc, s.offset, iVal).getResult();
      Value dt = smt::DistinctOp::create(builder, loc, t.offset, iVal).getResult();
      assertion = smt::OrOp::create(builder, loc, ds, dt).getResult();
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

  if (!buildScope(/*addressing=*/0) || !buildScope(/*equivalence=*/1))
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
