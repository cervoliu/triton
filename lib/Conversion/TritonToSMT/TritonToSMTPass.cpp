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
// is identity (unsat => identity holds); scope 1 checks that the two
// functions' UB domains coincide (unsat => both are UB on exactly the same
// inputs; poison, e.g. an oversized shift, is tracked exactly per value and
// only becomes UB when it reaches a memory operation); scope 2 asserts the
// output arrays can differ on an input where neither function is UB (unsat =>
// equivalent as bidirectional refinement). See Passes.td.
//
// With the `memory-model` option (phase 3), the identity-addressing contract
// is replaced by a block memory model absorbed from mlir-tv: every pointer
// argument is a distinct block (SMT array + symbolic element count), tensors
// are materialized as concrete lanes with a static shape (rank <= 2, extent
// product bounded by `max-lanes`), and loads/stores consume per-lane symbolic
// offsets. Out-of-bounds active accesses, poison reaching memory, and two
// active lanes of one store writing different values to the same offset (an
// intra-store race) fold into the UB predicate. The same three scopes are
// emitted; scope 0 becomes a documented always-unsat placeholder (block
// decomposition is total by construction). See docs/smt-tv/phase-3.md.
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
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

// Width of the memory-model block offset domain. Real Triton pointer
// arithmetic sign-extends each i32 offset and accumulates in 64-bit address
// space, so chained tt.addptr sums must NOT be modeled modulo 2^32 (two
// chained offsets of 2^31 reach base + 2^32, not base + 0). Every block
// offset in memory-model mode is a bv64.
static constexpr unsigned kPtrWidth = 64;

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

// Map an MLIR scalar type to its SMT sort. Returns null for unsupported types
// (including `index`, whose width is target-defined, and zero-width `i0`,
// which has no valid !smt.bv<0> counterpart).
static Type smtSortFor(Type t, MLIRContext *ctx) {
  if (isa<FloatType>(t))
    return smt::RealType::get(ctx);
  if (auto it = dyn_cast<IntegerType>(t)) {
    if (it.getWidth() == 0)
      return nullptr;
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
  // Poison predicate (smt Bool): true on inputs where this value is poison.
  // Null means provably never poison. Propagated EXACTLY (not merely
  // over-approximated): the bidirectional-refinement query compares the two
  // functions' UB domains for equality, and an over-approximation could hide
  // a domain difference (e.g. dead poison on one side only) behind equal
  // formulas, yielding a false EQUIVALENT.
  Value poison;
  // Reduction mode: a tensor materialized as N per-lane terms. For a vector
  // *value* these are the lane SMT values (array == null); for a vector
  // *pointer* they are the per-lane bit-vector offsets (array != null).
  SmallVector<Value> lanes;
  // Per-lane poison predicates; parallel to `lanes` when non-empty. An empty
  // vector with non-empty `lanes` means no lane is ever poison.
  SmallVector<Value> lanePoison;
  // --- Phase-3 memory-model fields (inert defaults on the legacy paths). ---
  // Static tensor shape, row-major lane order; empty means scalar. A shaped
  // value with empty `lanes` is a splat: every lane reads `value` (or, for a
  // pointer, `offset`), which the laneVal/laneOff broadcasting already does.
  SmallVector<int64_t> shape;
  // Pointer: the block's symbolic element count (memory-model mode). An
  // active access at offset >=u size is UB.
  Value size;
  // True iff this value derives from loaded memory. tt.addptr offsets must be
  // memory-independent (gather/scatter and loaded strides are rejected);
  // masks and stored values may be memory-dependent (their encoding stays
  // exact, since the initial memory contents are themselves shared inputs).
  bool mem = false;
  bool isPtr() const { return array != nullptr; }
  bool isVector() const { return !lanes.empty(); }
  bool isShaped() const { return !shape.empty(); }
  // Value at lane k, broadcasting a scalar operand.
  Value laneVal(unsigned k) const { return isVector() ? lanes[k] : value; }
  Value laneOff(unsigned k) const { return isVector() ? lanes[k] : offset; }
  // Poison at lane k, broadcasting a scalar operand's poison.
  Value lanePsn(unsigned k) const {
    if (!isVector())
      return poison;
    return lanePoison.empty() ? Value() : lanePoison[k];
  }
  static EncodedValue plain(Value v, Value p = nullptr) {
    EncodedValue e;
    e.value = v;
    e.poison = p;
    return e;
  }
  static EncodedValue ptr(Value a, Value o, bool fresh = false) {
    EncodedValue e;
    e.array = a;
    e.offset = o;
    e.fresh = fresh;
    return e;
  }
  static EncodedValue vec(SmallVector<Value> ls, SmallVector<Value> ps = {}) {
    EncodedValue e;
    e.lanes = std::move(ls);
    e.lanePoison = std::move(ps);
    return e;
  }
  static EncodedValue vecPtr(Value a, SmallVector<Value> offs,
                             SmallVector<Value> ps = {}) {
    EncodedValue e;
    e.array = a;
    e.lanes = std::move(offs);
    e.lanePoison = std::move(ps);
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

// Validate a pointer-to-pointer bitcast's pointee reinterpretation. Only
// integer<->integer reinterpretations that keep the addressing granularity
// (byte size) are sound in this value-based model: float bitcasts reinterpret
// bits (e.g. f16 vs bf16 map to the same real yet differ in memory) and size
// changes alter the element stride. The i1<->i8 byte-backed-bool idiom is
// preserved. Returns an error message, or nullptr when the bitcast is modeled.
static const char *checkPtrBitcastPointees(Type se, Type de) {
  auto intBytes = [](Type t) -> int {
    auto it = dyn_cast<IntegerType>(t);
    return it ? int((it.getWidth() + 7) / 8) : -1;
  };
  if (se != de && (intBytes(se) < 0 || intBytes(de) < 0 ||
                   intBytes(se) != intBytes(de)))
    return "unsupported pointer bitcast: only a same-byte-size integer "
           "pointee reinterpretation is modeled";
  return nullptr;
}

// True iff the two bit-vector terms are structurally guaranteed to differ on
// every input: distinct constants, or `base + c1` vs `base + c2` with the
// same base term and distinct constant addends (bvadd wraps, so c1 != c2
// implies the sums differ for every base). Used ONLY to elide intra-store
// race terms that can never fire; returning false is always sound (the term
// is emitted and the solver decides).
static bool provablyDistinctBV(Value a, Value b) {
  auto constOf = [](Value v) -> std::optional<APInt> {
    if (auto c = v.getDefiningOp<smt::BVConstantOp>())
      return c.getValue().getValue();
    return std::nullopt;
  };
  if (a == b)
    return false;
  auto ca = constOf(a), cb = constOf(b);
  if (ca && cb)
    return *ca != *cb;
  auto aa = a.getDefiningOp<smt::BVAddOp>();
  auto ba = b.getDefiningOp<smt::BVAddOp>();
  if (aa && ba) {
    if (aa.getLhs() == ba.getLhs()) {
      auto x = constOf(aa.getRhs()), y = constOf(ba.getRhs());
      if (x && y)
        return *x != *y;
    }
    if (aa.getRhs() == ba.getRhs()) {
      auto x = constOf(aa.getLhs()), y = constOf(ba.getLhs());
      if (x && y)
        return *x != *y;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Shared SMT term builders and the elementwise-op encoder
//===----------------------------------------------------------------------===//

// Builder facade shared by the legacy (symbolic-lane), reduction (1-D lane),
// and memory-model (N-D lane) encoders. Pure elementwise op semantics live in
// encodeElementwise() and NOWHERE else.
struct SMTBuilder {
  OpBuilder &b;
  Location loc;
  smt::RealType R;
  smt::BoolType Bool;

  SMTBuilder(OpBuilder &b, Location loc, MLIRContext *ctx)
      : b(b), loc(loc), R(smt::RealType::get(ctx)),
        Bool(smt::BoolType::get(ctx)) {}

  Value bvc(const APInt &v) {
    return smt::BVConstantOp::create(b, loc, v).getResult();
  }
  Value addrConst(uint64_t v) { return bvc(APInt(kAddrWidth, v)); }
  Value ptrConst(uint64_t v) { return bvc(APInt(kPtrWidth, v)); }
  // Sign-extend a bv32 offset into the 64-bit block address space, matching
  // tt.addptr's real pointer-arithmetic semantics.
  Value sextToPtr(Value x32) {
    Value isNeg = bvcmp(smt::BVCmpPredicate::slt, x32,
                        bvc(APInt(kAddrWidth, 0)));
    Value hi = ite(isNeg, bvc(APInt::getAllOnes(kPtrWidth - kAddrWidth)),
                   bvc(APInt(kPtrWidth - kAddrWidth, 0)));
    return smt::ConcatOp::create(b, loc, hi, x32).getResult();
  }
  Value bvcmp(smt::BVCmpPredicate p, Value x, Value y) {
    return smt::BVCmpOp::create(b, loc, p, x, y).getResult();
  }
  Value sel(Value arr, Value idx, Type range) {
    return smt::ArraySelectOp::create(b, loc, range, arr, idx).getResult();
  }
  Value ite(Value c, Value t, Value e) {
    return smt::IteOp::create(b, loc, t.getType(), c, t, e).getResult();
  }
  Value beq(Value x, Value y) {
    return smt::EqOp::create(b, loc, x, y).getResult();
  }
  Value bnot(Value x) { return smt::NotOp::create(b, loc, x).getResult(); }
  Value band(Value x, Value y) {
    return smt::AndOp::create(b, loc, x, y).getResult();
  }
  Value bor(Value x, Value y) {
    return smt::OrOp::create(b, loc, x, y).getResult();
  }
  Value boolConst(bool v) {
    return smt::BoolConstantOp::create(b, loc, Bool, b.getBoolAttr(v))
        .getResult();
  }
  Value realBin(Value x, Value y, bool mul) {
    if (mul)
      return smt::RealMulOp::create(b, loc, R, ValueRange{x, y}).getResult();
    return smt::RealAddOp::create(b, loc, R, ValueRange{x, y}).getResult();
  }
  Value realConst(StringRef s) {
    return smt::RealConstantOp::create(b, loc, R, b.getStringAttr(s))
        .getResult();
  }
  Value rcmp(smt::IntPredicate p, Value x, Value y) {
    return smt::RealCmpOp::create(b, loc, Bool, p, x, y).getResult();
  }
  // Poison algebra over optional Bool predicates (null == provably-false).
  // Propagation must stay EXACT (see EncodedValue::poison): every rule in
  // encodeElementwise matches the arith/LLVM poison semantics of the op.
  Value orPoison(Value a, Value c) {
    if (!a)
      return c;
    if (!c)
      return a;
    return bor(a, c);
  }
  Value andCond(Value cond, Value p) { // cond ∧ p, null-aware p
    if (!p)
      return nullptr;
    return band(cond, p);
  }
  // Reconcile a stored value with the buffer's declared SMT sort at a store
  // through a pointer bitcast. Only the bit-vector -> Bool direction (the
  // frontend's extui+bitcast idiom writing a byte into an i1-declared buffer,
  // read back as zero=false / nonzero=true) is modeled. The reverse direction
  // -- storing a raw i1 value into a byte-observable iN buffer -- is NOT
  // modeled: real lowerings disagree on the written byte (NVIDIA sign-extends
  // i1 to 0xff; "store true as 0x01" would prove a kernel equivalent to one
  // storing literal 1, which real memory distinguishes). Returns null for
  // any other sort mismatch, which the caller rejects.
  Value coerce(Value v, Type sort) {
    if (v.getType() == sort)
      return v;
    if (auto bvTy = dyn_cast<smt::BitVectorType>(v.getType()))
      if (isa<smt::BoolType>(sort))
        return bnot(beq(v, bvc(APInt(bvTy.getWidth(), 0))));
    return nullptr;
  }
  static Type elemOf(Type t) {
    if (auto rt = dyn_cast<RankedTensorType>(t))
      return rt.getElementType();
    return t;
  }

  // Encode an arith.constant (scalar or splat dense) as one SMT value.
  // Returns null with `why` set for unsupported constants.
  Value encodeConstant(arith::ConstantOp o, std::string &why) {
    Attribute a = o.getValue();
    if (auto fa = dyn_cast<FloatAttr>(a)) {
      auto s = formatReal(fa.getValue());
      if (!s) {
        why = "unsupported float constant (non-finite or exceeds f64 "
              "precision)";
        return Value();
      }
      return realConst(*s);
    }
    if (auto ia = dyn_cast<IntegerAttr>(a)) {
      if (ia.getType().getIntOrFloatBitWidth() == 1)
        return boolConst(ia.getValue() != 0);
      return bvc(ia.getValue());
    }
    if (auto dea = dyn_cast<DenseElementsAttr>(a)) {
      if (!dea.isSplat()) {
        why = "non-splat dense constant is unsupported";
        return Value();
      }
      Type et = dea.getElementType();
      if (isa<FloatType>(et)) {
        auto s = formatReal(dea.getSplatValue<APFloat>());
        if (!s) {
          why = "unsupported float constant (non-finite or exceeds f64 "
                "precision)";
          return Value();
        }
        return realConst(*s);
      }
      if (auto it = dyn_cast<IntegerType>(et)) {
        if (it.getWidth() == 1)
          return boolConst(!dea.getSplatValue<APInt>().isZero());
        return bvc(dea.getSplatValue<APInt>());
      }
      why = "unsupported constant element type";
      return Value();
    }
    why = "unsupported constant";
    return Value();
  }

  // The single source of truth for pure elementwise op semantics, shared by
  // the scalar path, the per-lane (reduction and memory-model) paths, and the
  // reduce combiner. Operands are fetched through `A` (value) and `P` (poison
  // predicate), which read the plain value in scalar context and the lane
  // value (broadcasting scalars) in vector context; `isPtrVal` reports
  // whether an operand is an encoded pointer (for the select rejection). On
  // success sets `outPoison` to the result's exact poison predicate (null ==
  // never). A null result with a non-empty `why` is a rejection (unsupported
  // flag/operand sort); a null result with an empty `why` means `op` is not a
  // pure elementwise op.
  Value encodeElementwise(Operation *op, llvm::function_ref<Value(Value)> A,
                          llvm::function_ref<Value(Value)> P,
                          llvm::function_ref<bool(Value)> isPtrVal,
                          Value &outPoison, std::string &why) {
    // Default: any-operand-poison, which is the exact arith rule for every
    // op below except select (overridden in its case).
    outPoison = nullptr;
    for (Value operand : op->getOperands())
      outPoison = orPoison(outPoison, P(operand));
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
            return bor(x, y);
          return smt::BVOrOp::create(b, loc, x, y).getResult();
        })
        .Case<arith::XOrIOp>([&](arith::XOrIOp o) -> Value {
          Value x = A(o.getLhs()), y = A(o.getRhs());
          if (isa<smt::BoolType>(x.getType()))
            return smt::XOrOp::create(b, loc, x, y).getResult();
          return smt::BVXOrOp::create(b, loc, x, y).getResult();
        })
        // Shifts. An oversized shift amount (>= width) yields poison in arith
        // but a defined value for the total SMT bv shifts, so the result's
        // poison predicate gains `amt >=u width` (mlir-tv's wellDefined(op,
        // cond) idea, tracked per value). Poison only becomes UB -- and only
        // then affects the verdict -- when it reaches a memory operation.
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
              outPoison = orPoison(
                  outPoison,
                  bvcmp(smt::BVCmpPredicate::uge, y, bvc(APInt(w, w))));
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
          if (isPtrVal(o.getTrueValue()) || isPtrVal(o.getFalseValue())) {
            why = "pointer-valued select is not supported";
            return Value();
          }
          // Exact poison rule: select is NOT poison-strict in its unchosen
          // arm -- p(cond) ∨ ite(cond, p(t), p(f)). The default any-operand
          // rule would over-approximate, which the refinement query forbids.
          Value pc = P(o.getCondition());
          Value pt = P(o.getTrueValue()), pf = P(o.getFalseValue());
          Value chosen = nullptr;
          if (pt || pf) {
            auto orFalse = [&](Value p) -> Value {
              return p ? p : boolConst(false);
            };
            chosen = ite(A(o.getCondition()), orFalse(pt), orFalse(pf));
          }
          outPoison = orPoison(pc, chosen);
          return ite(A(o.getCondition()), A(o.getTrueValue()),
                     A(o.getFalseValue()));
        })
        .Default([&](Operation *) { return Value(); });
  }
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
  // function's stores to `stores` and sets `outUB` to the function's UB
  // predicate: a Bool that is true exactly on inputs where poison (tracked
  // per value, e.g. from an oversized shift) reaches a memory operation.
  // Null means the function is UB-free on all inputs. Returns false (and
  // emits an error) if the function uses an unsupported construct.
  bool encodeFunction(OpBuilder &b, triton::FuncOp func,
                      ArrayRef<EncodedValue> sharedArgs, Value iVal,
                      SmallVectorImpl<StoreInfo> &stores, Value &outUB);

  // Memory-model (phase 3) encoder: encode `func` over concrete lanes with
  // per-lane symbolic offsets into per-pointer-arg blocks. `pids` are the
  // three shared program-id symbols; `memFinal` maps each written block's
  // initial array symbol to its final contents (sequential store folding);
  // `outUB` collects poison-reaches-memory, OOB-active-access, and
  // intra-store race conditions. Returns false (emitting an error) for
  // anything outside the validated contract.
  bool encodeFunctionMM(OpBuilder &b, triton::FuncOp func,
                        ArrayRef<EncodedValue> sharedArgs,
                        ArrayRef<Value> pids,
                        llvm::MapVector<Value, Value> &memFinal, Value &outUB);
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
                                        Value &outUB) {
  MLIRContext *ctx = &getContext();
  Location loc = func.getLoc();
  SMTBuilder sb(b, loc, ctx);

  // Builder aliases (shared implementations live in SMTBuilder; the poison
  // algebra and the coerce rule are documented there).
  auto bvc = [&](const APInt &v) { return sb.bvc(v); };
  auto addrConst = [&](uint64_t v) { return sb.addrConst(v); };
  auto sel = [&](Value arr, Value idx, Type range) {
    return sb.sel(arr, idx, range);
  };
  auto ite = [&](Value c, Value t, Value e) { return sb.ite(c, t, e); };
  auto elemOf = [](Type t) { return SMTBuilder::elemOf(t); };
  auto orPoison = [&](Value a, Value c) { return sb.orPoison(a, c); };
  auto andCond = [&](Value cond, Value p) { return sb.andCond(cond, p); };
  // UB conditions accumulated as poison reaches memory operations.
  SmallVector<Value> ubConds;

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

  // Pure elementwise op semantics come from the shared encoder
  // (SMTBuilder::encodeElementwise -- the single source of truth); this
  // wrapper binds the pointer-operand check to this function's env.
  auto isPtrValFn = [&](Value v) {
    auto it = env.find(v);
    return it != env.end() && it->second.isPtr();
  };
  auto encodeElementwise = [&](Operation *op, llvm::function_ref<Value(Value)> A,
                               llvm::function_ref<Value(Value)> P,
                               Value &outPoison, std::string &why) -> Value {
    return sb.encodeElementwise(op, A, P, isPtrValFn, outPoison, why);
  };

  // Reduction (vector) mode: encode one elementwise op at lane `k`,
  // broadcasting scalar operands, under exactly the scalar rules.
  auto encodeVectorLane = [&](Operation *op, unsigned k, Value &outPoison,
                              std::string &why) -> Value {
    auto A = [&](Value v) -> Value {
      auto it = env.find(v);
      return it == env.end() ? Value() : it->second.laneVal(k);
    };
    auto P = [&](Value v) -> Value {
      auto it = env.find(v);
      return it == env.end() ? Value() : it->second.lanePsn(k);
    };
    return encodeElementwise(op, A, P, outPoison, why);
  };

  for (Operation &opRef : entry) {
    Operation *op = &opRef;

    // Only scalars and ranked tensors are modeled, and only integer widths
    // with a valid SMT counterpart. Verifier-valid inputs can otherwise crash
    // width queries downstream: vector<4xi32> flows through elemOf()
    // unstripped, a scalar `index` constant asserts in
    // getIntOrFloatBitWidth(), and `i0` would construct an invalid
    // !smt.bv<0>. Reject them all up front.
    auto badType = [&](Type t) {
      if (isa<ShapedType>(t) && !isa<RankedTensorType>(t))
        return true;
      Type e = elemOf(t);
      if (isa<IndexType>(e))
        return true;
      auto it = dyn_cast<IntegerType>(e);
      return it && it.getWidth() == 0;
    };
    for (Type t : op->getOperandTypes())
      if (badType(t))
        return err(op, "unsupported operand type (only ranked tensors of "
                       "nonzero-width integers, floats, and pointers are "
                       "modeled)");
    for (Type t : op->getResultTypes())
      if (badType(t))
        return err(op, "unsupported result type (only ranked tensors of "
                       "nonzero-width integers, floats, and pointers are "
                       "modeled)");

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
          auto Pnone = [](Value) -> Value { return Value(); };
          std::string why;
          Value ignored;
          return encodeElementwise(comb, A, Pnone, ignored, why);
        };
        Value acc = in.lanes[0];
        for (unsigned k = 1; k < N; ++k) {
          acc = combine(acc, in.lanes[k]);
          if (!acc)
            return err(o, "failed to encode tt.reduce combiner");
        }
        // The whitelisted combiners are poison-strict, so the fold's poison
        // is exactly the disjunction of the lane poisons.
        Value foldPoison;
        for (unsigned k = 0; k < N; ++k)
          foldPoison = orPoison(foldPoison, in.lanePsn(k));
        env[o->getResult(0)] = EncodedValue::plain(acc, foldPoison);
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
          SmallVector<Value> offs, offPs(N);
          offs.reserve(N);
          bool anyPsn = false;
          for (unsigned k = 0; k < N; ++k) {
            offs.push_back(smt::BVAddOp::create(b, loc, base.laneOff(k),
                                                off.laneVal(k))
                               .getResult());
            offPs[k] = off.lanePsn(k); // base is a fresh arg: never poison
            anyPsn |= offPs[k] != nullptr;
          }
          if (!anyPsn)
            offPs.clear();
          env[o.getResult()] =
              EncodedValue::vecPtr(base.array, std::move(offs),
                                   std::move(offPs));
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
          for (unsigned k = 0; k < N; ++k) {
            ls.push_back(sel(p.array, p.lanes[k], range));
            // Loading through a poisoned address is UB (this unmasked load
            // always executes). Loaded values themselves are never poison.
            if (Value psn = p.lanePsn(k))
              ubConds.push_back(psn);
          }
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
        SmallVector<Value> ls(N), ps(N);
        bool anyPoison = false;
        for (unsigned k = 0; k < N; ++k) {
          std::string why;
          Value p;
          Value r = encodeVectorLane(op, k, p, why);
          if (!r)
            return err(op, why.empty()
                               ? ("operation '" + op->getName().getStringRef() +
                                  "' is not supported in reduction (vector) "
                                  "context")
                                     .str()
                               : why);
          ls[k] = r;
          ps[k] = p;
          anyPoison |= p != nullptr;
        }
        if (!anyPoison)
          ps.clear();
        env[op->getResult(0)] = EncodedValue::vec(std::move(ls), std::move(ps));
        continue;
      }
    }

    // Pure elementwise ops are encoded by the shared encoder (scalar context:
    // operands are the plain values). A rejection there is final; a "not
    // elementwise" miss falls through to the structural cases below.
    {
      std::string why;
      Value p;
      auto Psc = [&](Value v) -> Value {
        auto it = env.find(v);
        return it == env.end() ? Value() : it->second.poison;
      };
      Value r = encodeElementwise(op, V, Psc, p, why);
      if (r) {
        env[op->getResult(0)] = EncodedValue::plain(r, p);
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
              std::string why;
              Value v = sb.encodeConstant(o, why);
              if (!v)
                return err(o, why);
              env[o.getResult()] = EncodedValue::plain(v);
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
              // The soundness rule lives in checkPtrBitcastPointees.
              if (const char *msg = checkPtrBitcastPointees(
                      srcP.getPointeeType(), dstP.getPointeeType()))
                return err(o, msg);
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
              EncodedValue r = EncodedValue::ptr(
                  base.array,
                  smt::BVAddOp::create(b, loc, base.offset, V(o.getOffset()))
                      .getResult(),
                  /*fresh=*/false);
              // A pointer built from a poisoned offset is itself poison; it
              // becomes UB at the load/store that dereferences it.
              r.poison = env[o.getOffset()].poison; // fresh base: never poison
              env[o.getResult()] = r;
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
              Value resultPoison;
              // A masked load without `other` leaves masked-out lanes undefined
              // in Triton; ignoring the mask would be unsound, so require the
              // fallback and model it explicitly.
              //
              // UB accounting (exact): a poison mask is UB; a poison address
              // is UB only when the access executes (mask true). Loaded
              // memory values are never poison; a masked-out lane takes
              // `other`'s poison.
              if (o.getMask()) {
                if (!o.getOther())
                  return err(o, "masked load without a fallback `other` is "
                                "undefined and not supported");
                Value mask = V(o.getMask());
                if (Value pm = env[o.getMask()].poison)
                  ubConds.push_back(pm);
                if (Value pa = andCond(mask, p.poison))
                  ubConds.push_back(pa);
                loaded = ite(mask, loaded, V(o.getOther()));
                if (Value po = env[o.getOther()].poison) {
                  resultPoison = ite(mask, sb.boolConst(false), po);
                }
              } else if (p.poison) {
                ubConds.push_back(p.poison);
              }
              env[o.getResult()] = EncodedValue::plain(loaded, resultPoison);
              return true;
            })
            .Case<triton::StoreOp>([&](triton::StoreOp o) {
              EncodedValue p = env[o.getPtr()];
              // Store through a pointer bitcast: coerce the value to the
              // buffer's declared sort so the final-state comparison is typed
              // by the buffer (an i8 written into an i1 buffer means
              // "nonzero").
              Type range = cast<smt::ArrayType>(p.array.getType()).getRangeType();
              Value v = sb.coerce(V(o.getValue()), range);
              if (!v)
                return err(o, "stored value sort is incompatible with the "
                              "buffer sort");
              // UB accounting (exact): a poison mask is UB; a poison address
              // or poison stored value is UB only when the store executes
              // (mask true). Masked-out poison is NOT UB -- that asymmetry is
              // why poison must be tracked per value rather than as a global
              // well-definedness conjunction.
              Value valPoison = env[o.getValue()].poison;
              Value execPoison = orPoison(p.poison, valPoison);
              if (o.getMask()) {
                if (Value pm = env[o.getMask()].poison)
                  ubConds.push_back(pm);
                if (Value pe = andCond(V(o.getMask()), execPoison))
                  ubConds.push_back(pe);
              } else if (execPoison) {
                ubConds.push_back(execPoison);
              }
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
  outUB = nullptr;
  for (Value c : ubConds)
    outUB = orPoison(outUB, c);
  return true;
}

//===----------------------------------------------------------------------===//
// Memory-model (phase 3) encoder
//===----------------------------------------------------------------------===//

bool ConvertTritonToSMT::encodeFunctionMM(
    OpBuilder &b, triton::FuncOp func, ArrayRef<EncodedValue> sharedArgs,
    ArrayRef<Value> pids, llvm::MapVector<Value, Value> &memFinal,
    Value &outUB) {
  MLIRContext *ctx = &getContext();
  Location loc = func.getLoc();
  SMTBuilder sb(b, loc, ctx);
  auto elemOf = [](Type t) { return SMTBuilder::elemOf(t); };

  if (!func.getBody().hasOneBlock()) {
    func.emitError("TritonToSMT: only single-block function bodies are "
                   "supported (external or multi-block functions rejected)");
    return false;
  }
  Block &entry = func.getBody().front();

  DenseMap<Value, EncodedValue> env;
  for (auto [arg, shared] : llvm::zip(entry.getArguments(), sharedArgs))
    env[arg] = shared;

  SmallVector<Value> ubConds;

  auto err = [&](Operation *op, const Twine &what) {
    op->emitError("TritonToSMT: ") << what;
    return false;
  };
  auto isPtrValFn = [&](Value v) {
    auto it = env.find(v);
    return it != env.end() && it->second.isPtr();
  };
  // Current contents of each block, keyed by its initial array symbol. Loads
  // read the CURRENT array, so a load placed after a store to the same block
  // sees the stored values (sequential per-program semantics). As in phases
  // 1-2, cross-program interference is outside the model: a grid whose
  // programs race with each other is UB territory the tool does not reason
  // about, and both kernels are modeled under the same interference-free
  // assumption.
  auto curMem = [&](Value arr) -> Value {
    auto it = memFinal.find(arr);
    return it == memFinal.end() ? arr : it->second;
  };
  auto product = [](ArrayRef<int64_t> shape) {
    int64_t n = 1;
    for (int64_t d : shape)
      n *= d;
    return n;
  };
  // Shared sign-extension terms for i32 offsets entering the 64-bit block
  // address space. The cache keeps equal source terms mapped to one bv64
  // term, and the reverse map lets the race-elision matcher reason about the
  // 32-bit sources (sext is injective).
  DenseMap<Value, Value> sextCache; // i32 term -> bv64 term
  DenseMap<Value, Value> sextSrc;   // bv64 term -> i32 source term
  auto sextOff = [&](Value x32) -> Value {
    auto it = sextCache.find(x32);
    if (it != sextCache.end())
      return it->second;
    Value r = sb.sextToPtr(x32);
    sextCache[x32] = r;
    sextSrc[r] = x32;
    return r;
  };
  // Structural must-differ check on bv64 offsets, looking through the sext
  // wrapper and a shared base addend. Only used to elide race terms that can
  // never fire; returning false is always sound.
  auto provablyDistinctOff = [&](Value a, Value b) -> bool {
    if (a == b)
      return false;
    auto srcOf = [&](Value v) -> Value {
      auto it = sextSrc.find(v);
      return it == sextSrc.end() ? Value() : it->second;
    };
    if (Value s1 = srcOf(a))
      if (Value s2 = srcOf(b))
        return provablyDistinctBV(s1, s2); // sext is injective
    auto aa = a.getDefiningOp<smt::BVAddOp>();
    auto ba = b.getDefiningOp<smt::BVAddOp>();
    if (aa && ba && aa.getLhs() == ba.getLhs()) {
      Value s1 = srcOf(aa.getRhs()), s2 = srcOf(ba.getRhs());
      if (s1 && s2)
        return provablyDistinctBV(s1, s2); // base + sext(x): add is injective
    }
    return provablyDistinctBV(a, b);
  };
  Value zeroPtr; // lazily-built bv64 zero for the OOB predicate
  // An access at block offset `off` (bv64) is out of bounds iff it falls
  // outside [0, size): real pointer arithmetic makes negative offsets land
  // before the block and size is asserted non-negative in the scope prologue,
  // so the signed comparisons are exactly the real OOB condition.
  auto oobOf = [&](Value off, Value size) -> Value {
    if (!zeroPtr)
      zeroPtr = sb.ptrConst(0);
    return sb.bor(sb.bvcmp(smt::BVCmpPredicate::slt, off, zeroPtr),
                  sb.bvcmp(smt::BVCmpPredicate::sge, off, size));
  };
  // Restrict load-bearing gate (soundness). Each pointer-arg block is a
  // DISTINCT SMT array, so distinct pointer args are disjoint by
  // construction -- but TTIR carries no restrict/const attribute, so that
  // disjointness is ASSUMED, not verified. It can only change an observable
  // output when a memory op touches one block AFTER a store already went to
  // a DIFFERENT block (were the two aliased, the write to Y would then be
  // visible at X). We track the set of written blocks and flag any later
  // cross-block access; touching a block after a store to the SAME block is
  // the sequential-visibility behavior deliberately modeled here and is NOT
  // flagged. Unless assume-restrict is set, such a function is rejected --
  // the check fires before the offending op is folded, so no potentially
  // false EQUIVALENT is ever emitted (see docs/smt-tv/phase-3.md). With
  // assume-restrict the bookkeeping is skipped entirely and the encoding is
  // byte-identical to the disjoint-by-construction model.
  DenseSet<Value> storedBlocks;
  auto restrictLoadBearing = [&](Value block) {
    // True iff some previously-written block differs from this access's block.
    return storedBlocks.size() > (storedBlocks.count(block) ? 1u : 0u);
  };
  const char *restrictWhy =
      "disjointness of pointer arguments is load-bearing for this verdict (a "
      "memory op addresses one pointer-arg block after a store to a different "
      "pointer-arg block) and TTIR provides no restrict guarantee; rejected. "
      "Pass assume-restrict=true to obtain a verdict conditional on "
      "non-aliasing";

  for (Operation &opRef : entry) {
    Operation *op = &opRef;

    // Type validation: only scalars and static ranked tensors of rank <= 2
    // whose extent product stays under the max-lanes tractability threshold
    // are materialized; everything else is rejected loudly ("no silent
    // caps"). The rank cap bounds implementation complexity; the extent cap
    // bounds encoder/solver cost, and applies regardless of rank.
    auto checkType = [&](Type t) -> bool {
      if (isa<ShapedType>(t) && !isa<RankedTensorType>(t))
        return err(op,
                   "unsupported shaped type (only ranked tensors are modeled)");
      if (auto rt = dyn_cast<RankedTensorType>(t)) {
        if (!rt.hasStaticShape())
          return err(op, "dynamic tensor shapes are not supported");
        if (rt.getRank() > 2)
          return err(op, "tensors of rank > 2 are not supported in this phase");
        std::optional<int64_t> ne =
            ShapedType::tryGetNumElements(rt.getShape());
        if (!ne || *ne <= 0)
          return err(op, "tensor extent product is zero or overflows int64");
        // maxLanes is validated positive in runOnOperation; a non-positive
        // value must never silently disable this guard (the race construction
        // below it is quadratic in the lane count).
        if (*ne > maxLanes)
          return err(op, "tensor extent product (" + Twine(*ne) +
                             ") exceeds the max-lanes tractability threshold "
                             "(" +
                             Twine(maxLanes) +
                             "); the block is rejected rather than silently "
                             "truncated");
      }
      Type e = elemOf(t);
      if (isa<IndexType>(e))
        return err(op, "index-typed values are not supported");
      if (auto it2 = dyn_cast<IntegerType>(e))
        if (it2.getWidth() == 0)
          return err(op, "zero-width integers are not supported");
      return true;
    };
    for (Type t : op->getOperandTypes())
      if (!checkType(t))
        return false;
    for (Type t : op->getResultTypes())
      if (!checkType(t))
        return false;

    auto resTensorShape = [&]() -> SmallVector<int64_t> {
      if (op->getNumResults() == 1)
        if (auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType()))
          return SmallVector<int64_t>(rt.getShape());
      return {};
    };
    // Defensive lane-count check: every materialized operand must carry
    // exactly product(shape) lanes, or laneVal/laneOff would index out of
    // bounds (a crash, which the contract forbids as much as unsoundness).
    auto lanesConsistent = [&](const EncodedValue &e) {
      return e.lanes.empty() ||
             int64_t(e.lanes.size()) == product(e.shape);
    };

    std::optional<bool> structural =
        llvm::TypeSwitch<Operation *, std::optional<bool>>(op)
            .Case<triton::FuncOp, triton::ReturnOp, triton::ReduceReturnOp>(
                [&](auto) { return true; })
            .Case<triton::GetProgramIdOp>(
                [&](triton::GetProgramIdOp o) -> std::optional<bool> {
                  int axis = o.getAxisAsInt();
                  if (axis < 0 || axis >= int(pids.size()))
                    return err(o, "unsupported program_id axis");
                  env[o.getResult()] = EncodedValue::plain(pids[axis]);
                  return true;
                })
            .Case<triton::MakeRangeOp>(
                [&](triton::MakeRangeOp o) -> std::optional<bool> {
                  SmallVector<int64_t> shape = resTensorShape();
                  int64_t ext = int64_t(o.getEnd()) - int64_t(o.getStart());
                  if (shape.size() != 1 || shape[0] != ext)
                    return err(o, "tt.make_range extent does not match its "
                                  "result shape");
                  EncodedValue e;
                  e.lanes.reserve(ext);
                  for (int64_t k = 0; k < ext; ++k)
                    e.lanes.push_back(
                        sb.addrConst(uint64_t(o.getStart()) + uint64_t(k)));
                  e.shape = std::move(shape);
                  env[o.getResult()] = e;
                  return true;
                })
            .Case<triton::SplatOp>(
                [&](triton::SplatOp o) -> std::optional<bool> {
                  // A splat keeps the scalar term; laneVal/laneOff broadcast.
                  EncodedValue e = env[o.getSrc()];
                  e.shape = resTensorShape();
                  env[o.getResult()] = e;
                  return true;
                })
            .Case<triton::ExpandDimsOp>(
                [&](triton::ExpandDimsOp o) -> std::optional<bool> {
                  // Inserting a 1-extent axis leaves the row-major lane order
                  // unchanged; only the shape is updated.
                  EncodedValue e = env[o.getSrc()];
                  e.shape = resTensorShape();
                  if (!lanesConsistent(e))
                    return err(o, "internal lane/shape mismatch");
                  env[o.getResult()] = e;
                  return true;
                })
            .Case<triton::BroadcastOp>(
                [&](triton::BroadcastOp o) -> std::optional<bool> {
                  EncodedValue in = env[o.getSrc()];
                  SmallVector<int64_t> dst = resTensorShape();
                  auto srcTy = dyn_cast<RankedTensorType>(o.getSrc().getType());
                  if (!srcTy || srcTy.getRank() != int64_t(dst.size()))
                    return err(o, "tt.broadcast must preserve rank");
                  SmallVector<int64_t> srcShape(srcTy.getShape());
                  if (in.lanes.empty()) { // splat: only the shape changes
                    in.shape = std::move(dst);
                    env[o.getResult()] = in;
                    return true;
                  }
                  if (int64_t(in.lanes.size()) != product(srcShape))
                    return err(o, "internal lane/shape mismatch");
                  // Materialized lanes: replicate along the 1-extent axes
                  // (row-major traversal of the destination shape).
                  unsigned rank = dst.size();
                  SmallVector<int64_t> sstr(rank, 1);
                  for (int r = int(rank) - 2; r >= 0; --r)
                    sstr[r] = sstr[r + 1] * srcShape[r + 1];
                  int64_t n = product(dst);
                  EncodedValue out;
                  out.array = in.array;
                  out.size = in.size;
                  out.fresh = in.fresh;
                  out.mem = in.mem;
                  out.poison = in.poison;
                  out.shape = dst;
                  out.lanes.reserve(n);
                  bool anyP = !in.lanePoison.empty();
                  if (anyP)
                    out.lanePoison.reserve(n);
                  SmallVector<int64_t> idx(rank, 0);
                  for (int64_t k = 0; k < n; ++k) {
                    int64_t s = 0;
                    for (unsigned r2 = 0; r2 < rank; ++r2)
                      s += (srcShape[r2] == 1 ? 0 : idx[r2]) * sstr[r2];
                    out.lanes.push_back(in.lanes[s]);
                    if (anyP)
                      out.lanePoison.push_back(in.lanePoison[s]);
                    for (int r2 = int(rank) - 1; r2 >= 0; --r2) {
                      if (++idx[r2] < dst[r2])
                        break;
                      idx[r2] = 0;
                    }
                  }
                  env[o.getResult()] = out;
                  return true;
                })
            .Case<arith::ConstantOp>(
                [&](arith::ConstantOp o) -> std::optional<bool> {
                  std::string why;
                  Value v = sb.encodeConstant(o, why);
                  if (!v)
                    return err(o, why);
                  EncodedValue e = EncodedValue::plain(v);
                  e.shape = resTensorShape(); // dense splats keep their shape
                  env[o.getResult()] = e;
                  return true;
                })
            .Case<triton::BitcastOp>(
                [&](triton::BitcastOp o) -> std::optional<bool> {
                  auto srcP = dyn_cast<triton::PointerType>(
                      elemOf(o.getSrc().getType()));
                  auto dstP =
                      dyn_cast<triton::PointerType>(elemOf(o.getType()));
                  if (!srcP || !dstP)
                    return err(o,
                               "only pointer-to-pointer bitcast is supported");
                  if (const char *msg = checkPtrBitcastPointees(
                          srcP.getPointeeType(), dstP.getPointeeType()))
                    return err(o, msg);
                  env[o.getResult()] = env[o.getSrc()];
                  return true;
                })
            .Case<triton::AddPtrOp>(
                [&](triton::AddPtrOp o) -> std::optional<bool> {
                  Type offElem = elemOf(o.getOffset().getType());
                  if (!offElem.isInteger(kAddrWidth))
                    return err(o, "only 32-bit pointer offsets are supported");
                  EncodedValue base = env[o.getPtr()];
                  EncodedValue off = env[o.getOffset()];
                  if (!base.isPtr())
                    return err(o, "tt.addptr base is not a modeled pointer");
                  // The sound-by-rejection addressing contract: offsets must
                  // be pure functions of (pid, scalar args, lane indices).
                  // A memory-derived offset (gather/scatter, a stride loaded
                  // from memory, an indirect pointer) is rejected.
                  if (base.mem || off.mem)
                    return err(o, "data-dependent addressing: the pointer "
                                  "offset derives from loaded memory "
                                  "(gather/scatter and loaded strides are not "
                                  "supported)");
                  if (!lanesConsistent(base) || !lanesConsistent(off))
                    return err(o, "internal lane/shape mismatch");
                  // Chained tt.addptr is modeled EXACTLY: each link
                  // sign-extends its i32 offset into the 64-bit block offset
                  // and accumulates there, as real pointer arithmetic does
                  // (the legacy path's chain rejection existed because its
                  // 32-bit accumulation would wrap where reality does not).
                  // An argument base sits at offset 0; skip the vacuous add
                  // so per-lane offsets keep the shape the race-elision
                  // matcher recognizes.
                  auto addOff = [&](Value baseOff, Value delta32) -> Value {
                    Value delta = sextOff(delta32);
                    if (auto c = baseOff.getDefiningOp<smt::BVConstantOp>())
                      if (c.getValue().getValue().isZero())
                        return delta;
                    return smt::BVAddOp::create(b, loc, baseOff, delta)
                        .getResult();
                  };
                  EncodedValue r;
                  r.array = base.array;
                  r.size = base.size;
                  r.fresh = false;
                  r.mem = false;
                  r.shape = resTensorShape();
                  bool anyLanes = !base.lanes.empty() || !off.lanes.empty();
                  if (anyLanes) {
                    int64_t n = r.shape.empty() ? 1 : product(r.shape);
                    if (r.shape.empty())
                      return err(o, "internal lane/shape mismatch");
                    SmallVector<Value> ps(n);
                    bool anyP = false;
                    r.lanes.reserve(n);
                    for (int64_t k = 0; k < n; ++k) {
                      r.lanes.push_back(addOff(base.laneOff(unsigned(k)),
                                               off.laneVal(unsigned(k))));
                      // A chained base may carry poison from an earlier
                      // poisoned offset; the result is poison if either
                      // contribution is.
                      ps[k] = sb.orPoison(base.lanePsn(unsigned(k)),
                                          off.lanePsn(unsigned(k)));
                      anyP |= ps[k] != nullptr;
                    }
                    if (anyP)
                      r.lanePoison = std::move(ps);
                  } else {
                    r.offset = addOff(base.offset, off.value);
                    r.poison = sb.orPoison(base.poison, off.poison);
                  }
                  env[o.getResult()] = r;
                  return true;
                })
            .Case<triton::LoadOp>(
                [&](triton::LoadOp o) -> std::optional<bool> {
                  // A volatile load is an observable side effect; a dead
                  // volatile read would vanish from the model.
                  if (o.getIsVolatile())
                    return err(o, "volatile loads are observable and not "
                                  "modeled");
                  EncodedValue p = env[o.getPtr()];
                  if (!p.isPtr() || !p.size)
                    return err(o, "load through an unmodeled pointer");
                  if (!assumeRestrict && restrictLoadBearing(p.array))
                    return err(o, restrictWhy);
                  if (!lanesConsistent(p))
                    return err(o, "internal lane/shape mismatch");
                  Type range =
                      cast<smt::ArrayType>(p.array.getType()).getRangeType();
                  Type want = smtSortFor(elemOf(o.getType()), ctx);
                  if (!want)
                    return err(o, "unsupported loaded element type");
                  if (want != range)
                    return err(o, "load reinterprets the buffer's element "
                                  "sort via a pointer bitcast; not supported");
                  SmallVector<int64_t> shape = resTensorShape();
                  bool scalarLoad = shape.empty();
                  int64_t n = scalarLoad ? 1 : product(shape);
                  if (p.isShaped() && p.shape != shape)
                    return err(o, "pointer/result shapes disagree");
                  EncodedValue mask, other;
                  bool hasMask = o.getMask() != nullptr;
                  if (hasMask) {
                    // A masked load without `other` leaves masked-out lanes
                    // undefined in Triton; require the fallback (as in the
                    // legacy path).
                    if (!o.getOther())
                      return err(o, "masked load without a fallback `other` "
                                    "is undefined and not supported");
                    mask = env[o.getMask()];
                    other = env[o.getOther()];
                    if ((mask.isShaped() && mask.shape != shape) ||
                        (other.isShaped() && other.shape != shape))
                      return err(o, "mask/other shapes disagree with the "
                                    "load");
                    if (!lanesConsistent(mask) || !lanesConsistent(other))
                      return err(o, "internal lane/shape mismatch");
                  }
                  Value cur = curMem(p.array);
                  // All-splat operands yield an all-splat result: encode one
                  // lane and keep the shape.
                  bool splat = !scalarLoad && p.lanes.empty() &&
                               (!hasMask || (mask.lanes.empty() &&
                                             other.lanes.empty()));
                  int64_t iters = (scalarLoad || splat) ? 1 : n;
                  SmallVector<Value> ls(iters), ps(iters);
                  bool anyP = false;
                  for (int64_t k = 0; k < iters; ++k) {
                    Value off = p.laneOff(unsigned(k));
                    Value oob = oobOf(off, p.size);
                    Value addrPsn = p.lanePsn(unsigned(k));
                    // UB accounting (exact): a poison mask is UB; a poison
                    // address or an out-of-bounds access is UB only when the
                    // lane is active. A masked-out lane performs no access
                    // and takes `other`. Loaded values are never poison.
                    if (hasMask) {
                      Value m = mask.laneVal(unsigned(k));
                      if (Value pm = mask.lanePsn(unsigned(k)))
                        ubConds.push_back(pm);
                      if (Value pa = sb.andCond(m, addrPsn))
                        ubConds.push_back(pa);
                      ubConds.push_back(sb.band(m, oob));
                      ls[k] = sb.ite(m, sb.sel(cur, off, range),
                                     other.laneVal(unsigned(k)));
                      if (Value po = other.lanePsn(unsigned(k))) {
                        ps[k] = sb.ite(m, sb.boolConst(false), po);
                        anyP = true;
                      }
                    } else {
                      if (addrPsn)
                        ubConds.push_back(addrPsn);
                      ubConds.push_back(oob);
                      ls[k] = sb.sel(cur, off, range);
                    }
                  }
                  EncodedValue r;
                  if (scalarLoad || splat) {
                    r.value = ls[0];
                    r.poison = anyP ? ps[0] : Value();
                  } else {
                    r.lanes = std::move(ls);
                    if (anyP)
                      r.lanePoison = std::move(ps);
                  }
                  r.shape = std::move(shape);
                  r.mem = true;
                  env[o.getResult()] = r;
                  return true;
                })
            .Case<triton::StoreOp>(
                [&](triton::StoreOp o) -> std::optional<bool> {
                  EncodedValue p = env[o.getPtr()];
                  if (!p.isPtr() || !p.size)
                    return err(o, "store through an unmodeled pointer");
                  if (!assumeRestrict) {
                    if (restrictLoadBearing(p.array))
                      return err(o, restrictWhy);
                    storedBlocks.insert(p.array);
                  }
                  Type range =
                      cast<smt::ArrayType>(p.array.getType()).getRangeType();
                  EncodedValue val = env[o.getValue()];
                  EncodedValue mask;
                  bool hasMask = o.getMask() != nullptr;
                  if (hasMask)
                    mask = env[o.getMask()];
                  SmallVector<int64_t> shape;
                  if (auto rt =
                          dyn_cast<RankedTensorType>(o.getValue().getType()))
                    shape = SmallVector<int64_t>(rt.getShape());
                  int64_t n = shape.empty() ? 1 : product(shape);
                  if ((p.isShaped() && p.shape != shape) ||
                      (val.isShaped() && val.shape != shape) ||
                      (hasMask && mask.isShaped() && mask.shape != shape))
                    return err(o, "pointer/value/mask shapes disagree");
                  if (!lanesConsistent(p) || !lanesConsistent(val) ||
                      (hasMask && !lanesConsistent(mask)))
                    return err(o, "internal lane/shape mismatch");
                  Value cur = curMem(p.array);
                  SmallVector<Value> offs(n), vals(n), acts(n);
                  for (int64_t k = 0; k < n; ++k) {
                    offs[k] = p.laneOff(unsigned(k));
                    Value v = sb.coerce(val.laneVal(unsigned(k)), range);
                    if (!v)
                      return err(o, "stored value sort is incompatible with "
                                    "the buffer sort");
                    vals[k] = v;
                    acts[k] = hasMask ? mask.laneVal(unsigned(k)) : Value();
                    Value oob = oobOf(offs[k], p.size);
                    // UB accounting (exact): a poison mask is UB; a poison
                    // address, a poison stored value, or an out-of-bounds
                    // write is UB only when the lane is active.
                    Value execPoison = sb.orPoison(p.lanePsn(unsigned(k)),
                                                   val.lanePsn(unsigned(k)));
                    if (hasMask) {
                      if (Value pm = mask.lanePsn(unsigned(k)))
                        ubConds.push_back(pm);
                      if (Value pe = sb.andCond(acts[k], execPoison))
                        ubConds.push_back(pe);
                      ubConds.push_back(sb.band(acts[k], oob));
                    } else {
                      if (execPoison)
                        ubConds.push_back(execPoison);
                      ubConds.push_back(oob);
                    }
                  }
                  // Intra-store write-write race: two active lanes of the
                  // same store writing DIFFERENT values to the same offset is
                  // UB (Triton's lanes are parallel; the winner is
                  // unspecified). Equal-value overlaps are deterministic and
                  // not UB. Pairs whose offsets are structurally distinct or
                  // whose value terms coincide can never race and are elided
                  // -- eliding is sound because the elided term is provably
                  // false.
                  for (int64_t k = 0; k < n; ++k)
                    for (int64_t l = k + 1; l < n; ++l) {
                      if (vals[k] == vals[l])
                        continue;
                      if (provablyDistinctOff(offs[k], offs[l]))
                        continue;
                      Value race = sb.band(sb.beq(offs[k], offs[l]),
                                           sb.bnot(sb.beq(vals[k], vals[l])));
                      if (acts[k])
                        race = sb.band(acts[k], race);
                      if (acts[l])
                        race = sb.band(acts[l], race);
                      ubConds.push_back(race);
                    }
                  // Fold the lanes into the block's current array. On
                  // race-free inputs the fold order is irrelevant; racy
                  // inputs are UB and excluded from the equivalence scope.
                  for (int64_t k = 0; k < n; ++k) {
                    Value stored = smt::ArrayStoreOp::create(
                                       b, loc, cur.getType(), cur, offs[k],
                                       vals[k])
                                       .getResult();
                    cur = acts[k] ? sb.ite(acts[k], stored, cur) : stored;
                  }
                  memFinal[p.array] = cur;
                  return true;
                })
            .Case<triton::ReduceOp>(
                [&](triton::ReduceOp o) -> std::optional<bool> {
                  if (o->getNumOperands() != 1 || o->getNumResults() != 1)
                    return err(o, "only single-input/single-result tt.reduce "
                                  "is supported (fused arg-reduce is "
                                  "rejected)");
                  auto srcTy =
                      dyn_cast<RankedTensorType>(o->getOperand(0).getType());
                  if (!srcTy || !srcTy.hasStaticShape() ||
                      srcTy.getRank() < 1 || srcTy.getRank() > 2)
                    return err(o, "only static rank-1/rank-2 reductions are "
                                  "supported");
                  int64_t axis = o.getAxis();
                  if (axis < 0 || axis >= srcTy.getRank())
                    return err(o, "reduction axis out of range");
                  Operation *comb = o.getSingleCombiner();
                  if (!comb)
                    return err(o, "unsupported tt.reduce combiner region "
                                  "(non-canonical or fused arg-reduce)");
                  // Require the region to be exactly {combiner,
                  // reduce.return}; getSingleCombiner tolerates extra ops it
                  // would silently drop.
                  Block &combBlock = o.getCombineOp().front();
                  if (combBlock.getOperations().size() != 2)
                    return err(o, "tt.reduce combiner region must contain "
                                  "only the combining op and the terminator");
                  if (!isa<arith::AddFOp, arith::MaxNumFOp, arith::MaximumFOp,
                           arith::MinNumFOp, arith::MinimumFOp>(comb))
                    return err(o, "unsupported tt.reduce combiner '" +
                                      comb->getName().getStringRef() +
                                      "' (supported: addf, max/minnumf, "
                                      "max/minimumf)");
                  EncodedValue in = env[o->getOperand(0)];
                  SmallVector<int64_t> srcShape(srcTy.getShape());
                  int64_t total = product(srcShape);
                  if (!in.lanes.empty() &&
                      int64_t(in.lanes.size()) != total)
                    return err(o, "internal lane/shape mismatch");
                  Type laneSort;
                  if (!in.lanes.empty())
                    laneSort = in.lanes.front().getType();
                  else if (in.value)
                    laneSort = in.value.getType();
                  // Float (ideal-real) reductions only: the fixed fold order
                  // below is only provably order-independent because the
                  // whitelisted combiners are associative and commutative
                  // over ideal reals.
                  if (!laneSort || !isa<smt::RealType>(laneSort))
                    return err(o, "only floating-point (real) reductions are "
                                  "supported");
                  auto combine = [&](Value acc, Value x) -> Value {
                    auto A = [&](Value v) -> Value {
                      if (v == combBlock.getArgument(0))
                        return acc;
                      if (v == combBlock.getArgument(1))
                        return x;
                      return Value();
                    };
                    auto Pnone = [](Value) -> Value { return Value(); };
                    auto noPtr = [](Value) { return false; };
                    std::string why;
                    Value ignored;
                    return sb.encodeElementwise(comb, A, Pnone, noPtr, ignored,
                                                why);
                  };
                  int64_t redN = srcShape[axis];
                  int64_t outN = total / redN;
                  int64_t s1 = srcShape.size() == 2 ? srcShape[1] : 1;
                  SmallVector<Value> outLanes(outN), outPs(outN);
                  bool anyP = false;
                  for (int64_t j = 0; j < outN; ++j) {
                    Value acc, psn;
                    for (int64_t r = 0; r < redN; ++r) {
                      int64_t lin;
                      if (srcShape.size() == 1)
                        lin = r;
                      else if (axis == 0)
                        lin = r * s1 + j;
                      else
                        lin = j * s1 + r;
                      Value x = in.laneVal(unsigned(lin));
                      acc = (r == 0) ? x : combine(acc, x);
                      if (!acc)
                        return err(o, "failed to encode tt.reduce combiner");
                      // The whitelisted combiners are poison-strict, so the
                      // fold's poison is exactly the lane-poison disjunction.
                      psn = sb.orPoison(psn, in.lanePsn(unsigned(lin)));
                    }
                    outLanes[j] = acc;
                    outPs[j] = psn;
                    anyP |= psn != nullptr;
                  }
                  EncodedValue r;
                  r.mem = in.mem;
                  if (srcTy.getRank() == 1) {
                    r.value = outLanes[0];
                    r.poison = anyP ? outPs[0] : Value();
                  } else {
                    r.shape = {srcShape[1 - axis]};
                    r.lanes = std::move(outLanes);
                    if (anyP)
                      r.lanePoison = std::move(outPs);
                  }
                  env[o->getResult(0)] = r;
                  return true;
                })
            .Case<triton::ReshapeOp>(
                [&](triton::ReshapeOp o) -> std::optional<bool> {
                  // Same narrow idiom as the reduction path: an
                  // identity-extent rank-1 flattening whose ONLY user is a
                  // tt.reduce. Any fold order is admissible there (the reduce
                  // whitelist admits only associative+commutative combiners
                  // over ideal reals, and allow_reorder leaves the order
                  // unspecified anyway); feeding lane-wise arithmetic would
                  // make the pairing nondeterministic and is rejected.
                  EncodedValue in = env[o.getSrc()];
                  auto st = dyn_cast<RankedTensorType>(o.getType());
                  bool soleReduceUser =
                      o.getResult().hasOneUse() &&
                      isa<triton::ReduceOp>(*o.getResult().getUsers().begin());
                  if (soleReduceUser && st && st.getRank() == 1 &&
                      st.hasStaticShape() && in.isShaped() &&
                      lanesConsistent(in) &&
                      st.getShape()[0] == product(in.shape)) {
                    EncodedValue r = in;
                    r.shape = {st.getShape()[0]};
                    env[o.getResult()] = r;
                    return true;
                  }
                  return err(o, "tt.reshape is only supported as an "
                                "identity-extent rank-1 view feeding a "
                                "tt.reduce");
                })
            .Default([](Operation *) { return std::nullopt; });

    if (structural) {
      if (!*structural)
        return false;
      continue;
    }

    // Elementwise fallback: scalar, all-splat, or per-lane vector context.
    SmallVector<int64_t> eshape;
    bool anyLanes = false, memTaint = false;
    for (Value operand : op->getOperands()) {
      auto it = env.find(operand);
      if (it == env.end())
        continue;
      memTaint |= it->second.mem;
      if (it->second.isShaped()) {
        if (eshape.empty())
          eshape = it->second.shape;
        else if (eshape != it->second.shape)
          return err(op, "operand shapes disagree");
        if (!lanesConsistent(it->second))
          return err(op, "internal lane/shape mismatch");
        anyLanes |= !it->second.lanes.empty();
      }
    }
    if (op->getNumResults() != 1)
      return err(op, "unsupported operation '" +
                         op->getName().getStringRef() + "'");
    // Elementwise ops are shape-preserving; a result shape that disagrees
    // with the operands' means this is not an elementwise op.
    if (resTensorShape() != eshape)
      return err(op, "unsupported operation '" +
                         op->getName().getStringRef() +
                         "' (result shape disagrees with its operands)");

    auto rejectMsg = [&](const std::string &why) {
      return why.empty() ? ("operation '" + op->getName().getStringRef() +
                            "' is not supported")
                               .str()
                         : why;
    };
    if (eshape.empty() || !anyLanes) {
      // Scalar (or all-splat) context: encode once; a splat result keeps the
      // shape so downstream broadcasting stays cheap.
      auto A = [&](Value v) -> Value {
        auto it = env.find(v);
        return it == env.end() ? Value() : it->second.value;
      };
      auto P = [&](Value v) -> Value {
        auto it = env.find(v);
        return it == env.end() ? Value() : it->second.poison;
      };
      std::string why;
      Value psn;
      Value r = sb.encodeElementwise(op, A, P, isPtrValFn, psn, why);
      if (!r)
        return err(op, rejectMsg(why));
      EncodedValue e = EncodedValue::plain(r, psn);
      e.shape = eshape;
      e.mem = memTaint;
      env[op->getResult(0)] = e;
      continue;
    }
    int64_t n = product(eshape);
    SmallVector<Value> ls(n), ps(n);
    bool anyP = false;
    for (int64_t k = 0; k < n; ++k) {
      auto A = [&](Value v) -> Value {
        auto it = env.find(v);
        return it == env.end() ? Value() : it->second.laneVal(unsigned(k));
      };
      auto P = [&](Value v) -> Value {
        auto it = env.find(v);
        return it == env.end() ? Value() : it->second.lanePsn(unsigned(k));
      };
      std::string why;
      Value psn;
      Value r = sb.encodeElementwise(op, A, P, isPtrValFn, psn, why);
      if (!r)
        return err(op, rejectMsg(why));
      ls[k] = r;
      ps[k] = psn;
      anyP |= psn != nullptr;
    }
    EncodedValue e =
        EncodedValue::vec(std::move(ls), anyP ? std::move(ps)
                                              : SmallVector<Value>{});
    e.shape = eshape;
    e.mem = memTaint;
    env[op->getResult(0)] = e;
  }

  outUB = nullptr;
  for (Value c : ubConds)
    outUB = sb.orPoison(outUB, c);
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
  // `mode`: 0 = addressing-identity check, 1 = UB-domain-equality check,
  // 2 = equivalence check. All three scopes are always emitted, in this order,
  // so the driver's scope-position contract stays fixed.
  //
  // Equivalence is bidirectional refinement: the two functions must be UB on
  // exactly the same inputs (scope 1: assert UB_src != UB_tgt; unsat =>
  // domains coincide) and produce equal outputs on every input where neither
  // is UB (scope 2). A function's UB predicate is exact -- poison that never
  // reaches a memory operation does not count -- so two kernels that are UB
  // in the same way are EQUIVALENT, and a kernel that is UB where the other
  // is defined is NOT_EQUIVALENT.
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
    Value ubS = nullptr, ubT = nullptr;
    if (!encodeFunction(builder, src, shared, iVal, storesS, ubS) ||
        !encodeFunction(builder, tgt, shared, iVal, storesT, ubT))
      return false;
    const StoreInfo &s = storesS.front(), &t = storesT.front();

    auto boolOrFalse = [&](Value v) -> Value {
      if (v)
        return v;
      return smt::BoolConstantOp::create(builder, loc,
                                         smt::BoolType::get(ctx),
                                         builder.getBoolAttr(false))
          .getResult();
    };

    Value assertion;
    if (mode == 0) {
      // Addressing must be identity: storeOffset == i. Assert the negation; if
      // unsat, identity holds for both functions.
      Value ds = smt::DistinctOp::create(builder, loc, s.offset, iVal).getResult();
      Value dt = smt::DistinctOp::create(builder, loc, t.offset, iVal).getResult();
      assertion = smt::OrOp::create(builder, loc, ds, dt).getResult();
    } else if (mode == 1) {
      // UB-domain equality: assert the two UB predicates differ somewhere.
      // unsat => both functions are UB on exactly the same inputs. Trivially
      // unsat when neither function can be UB.
      assertion = smt::DistinctOp::create(builder, loc, boolOrFalse(ubS),
                                          boolOrFalse(ubT))
                      .getResult();
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
      // Outputs need only agree where neither function is UB (bidirectional
      // refinement); on the -- provably shared, per scope 1 -- UB domain any
      // behavior is allowed.
      for (Value ub : {ubS, ubT})
        if (ub)
          assertion = smt::AndOp::create(
                          builder, loc,
                          smt::NotOp::create(builder, loc, ub).getResult(),
                          assertion)
                          .getResult();
    }
    smt::AssertOp::create(builder, loc, assertion);

    auto check = smt::CheckOp::create(builder, loc, TypeRange{});
    fillCheckRegions(builder, loc, check);
    builder.setInsertionPointToEnd(body);
    smt::YieldOp::create(builder, loc, ValueRange{});
    return true;
  };

  // Memory-model scope builder (phase 3). The same three scopes are emitted
  // in the same order so the driver's positional contract is untouched:
  //   scope 0: a documented, trivially-unsat placeholder. Under the block
  //     model, addressing correctness is no longer a separate identity proof:
  //     every access either decomposes to a known block with a modeled
  //     per-lane offset or is rejected, and distinct blocks are disjoint SMT
  //     arrays by construction, so there is no residual addressing fact to
  //     check. Intra-store write-write races were the one candidate fact and
  //     are UB (scope 1) instead. See docs/smt-tv/phase-3.md.
  //   scope 1: UB-domain equality, now also covering out-of-bounds active
  //     accesses and intra-store races besides poison reaching memory.
  //   scope 2: per-block final-state equality on inputs where neither
  //     function is UB (whole-array equality: unwritten offsets trivially
  //     agree because both sides share the initial array symbols, so this is
  //     exactly per-offset equality over the written blocks).
  auto buildScopeMM = [&](int mode) -> bool {
    builder.setInsertionPointToEnd(module.getBody());
    auto solver =
        smt::SolverOp::create(builder, loc, TypeRange{}, ValueRange{});
    Block *body = builder.createBlock(&solver.getBodyRegion());
    builder.setInsertionPointToStart(body);
    smt::SetLogicOp::create(builder, loc, builder.getStringAttr("ALL"));
    SMTBuilder sb(builder, loc, ctx);

    auto addrTy = smt::BitVectorType::get(ctx, kAddrWidth);
    // One symbolic program instance: the three grid axes are free symbols
    // shared by both functions (per-program refinement -- the same grid
    // launches both kernels, so proving every program id equivalent proves
    // the launch equivalent under the interference-free assumption).
    SmallVector<Value> pids;
    for (unsigned a = 0; a < 3; ++a)
      pids.push_back(smt::DeclareFunOp::create(
                         builder, loc, addrTy,
                         builder.getStringAttr(("pid" + Twine(a)).str()))
                         .getResult());
    auto ptrTyBV = smt::BitVectorType::get(ctx, kPtrWidth);
    Value zeroOff = sb.ptrConst(0);
    SmallVector<EncodedValue> shared;
    for (auto [idx, argTy] : llvm::enumerate(argTypes)) {
      std::string name = ("arg" + Twine(idx)).str();
      if (auto ptrTy = dyn_cast<triton::PointerType>(argTy)) {
        Type range = smtSortFor(ptrTy.getPointeeType(), ctx);
        if (!range) {
          module.emitError("TritonToSMT: unsupported pointee type ")
              << ptrTy.getPointeeType();
          return false;
        }
        // Blocks live in a 64-bit offset space (see kPtrWidth): the array is
        // indexed by bv64 and the element count is a bv64 constrained to be
        // non-negative, so the signed OOB predicate `off < 0 or off >= size`
        // is exactly the real out-of-allocation condition.
        auto arrTy = smt::ArrayType::get(ctx, ptrTyBV, range);
        // The block: fully-initialized symbolic contents plus a symbolic
        // element count. Distinct arguments are distinct SMT arrays, which
        // formalizes the phase-1 "pointer args are disjoint" (restrict)
        // assumption. TTIR carries no restrict/const attribute, so that
        // disjointness is ASSUMED, not verified; the restrict load-bearing
        // gate in encodeFunctionMM rejects any function where it could change
        // the verdict unless assume-restrict is set (see phase-3.md). Memory
        // is modeled as always readable (CUDA global
        // memory holds SOME value); reading an offset the kernel has not
        // written is defined-but-unknown, not UB -- see phase-3.md for why
        // this deliberately diverges from mlir-tv's uninitialized-read UB.
        Value arr = smt::DeclareFunOp::create(builder, loc, arrTy,
                                              builder.getStringAttr(name))
                        .getResult();
        Value size = smt::DeclareFunOp::create(
                         builder, loc, ptrTyBV,
                         builder.getStringAttr(name + "_size"))
                         .getResult();
        smt::AssertOp::create(
            builder, loc,
            sb.bvcmp(smt::BVCmpPredicate::sge, size, zeroOff));
        EncodedValue e = EncodedValue::ptr(arr, zeroOff, /*fresh=*/true);
        e.size = size;
        shared.push_back(e);
      } else if (Type sort = smtSortFor(argTy, ctx)) {
        shared.push_back(EncodedValue::plain(
            smt::DeclareFunOp::create(builder, loc, sort,
                                      builder.getStringAttr(name))
                .getResult()));
      } else {
        module.emitError("TritonToSMT: unsupported argument type ") << argTy;
        return false;
      }
    }

    llvm::MapVector<Value, Value> memS, memT;
    Value ubS = nullptr, ubT = nullptr;
    if (!encodeFunctionMM(builder, src, shared, pids, memS, ubS) ||
        !encodeFunctionMM(builder, tgt, shared, pids, memT, ubT))
      return false;

    auto boolOrFalse = [&](Value v) -> Value {
      return v ? v : sb.boolConst(false);
    };
    Value assertion;
    if (mode == 0) {
      assertion = sb.boolConst(false); // vestigial: see the comment above
    } else if (mode == 1) {
      assertion = smt::DistinctOp::create(builder, loc, boolOrFalse(ubS),
                                          boolOrFalse(ubT))
                      .getResult();
    } else {
      SmallVector<Value> diffs;
      for (const EncodedValue &e : shared) {
        if (!e.isPtr())
          continue;
        Value arr = e.array;
        if (!memS.count(arr) && !memT.count(arr))
          continue; // neither function writes this block
        auto finalOf = [&](llvm::MapVector<Value, Value> &m) {
          auto it = m.find(arr);
          return it == m.end() ? arr : it->second;
        };
        diffs.push_back(smt::DistinctOp::create(builder, loc, finalOf(memS),
                                                finalOf(memT))
                            .getResult());
      }
      // No writes on either side: nothing observable can differ.
      assertion = diffs.empty() ? sb.boolConst(false) : diffs.front();
      for (size_t k = 1; k < diffs.size(); ++k)
        assertion = sb.bor(assertion, diffs[k]);
      // Outputs need only agree where neither function is UB (bidirectional
      // refinement); on the -- provably shared, per scope 1 -- UB domain any
      // behavior is allowed.
      for (Value ub : {ubS, ubT})
        if (ub)
          assertion = sb.band(sb.bnot(ub), assertion);
    }
    smt::AssertOp::create(builder, loc, assertion);

    auto check = smt::CheckOp::create(builder, loc, TypeRange{});
    fillCheckRegions(builder, loc, check);
    builder.setInsertionPointToEnd(body);
    smt::YieldOp::create(builder, loc, ValueRange{});
    return true;
  };

  if (memoryModel && blockSize > 0) {
    module.emitError(
        "TritonToSMT: block-size applies only to the identity-addressing "
        "path; do not combine it with memory-model");
    return signalPassFailure();
  }
  // The tractability cap is mandatory: a non-positive max-lanes must reject
  // loudly rather than silently disable the guard (the intra-store race
  // construction is quadratic in the lane count, and the driver applies no
  // timeout to the encoder subprocess).
  if (memoryModel && maxLanes <= 0) {
    module.emitError("TritonToSMT: max-lanes must be positive (got " +
                     Twine(maxLanes) + "); the extent-product tractability "
                     "guard cannot be disabled");
    return signalPassFailure();
  }
  bool built = memoryModel
                   ? (buildScopeMM(0) && buildScopeMM(1) && buildScopeMM(2))
                   : (buildScope(/*addressing=*/0) &&
                      buildScope(/*ub-domain=*/1) &&
                      buildScope(/*equivalence=*/2));
  if (!built)
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
