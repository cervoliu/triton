"""End-to-end tests for the SMT-based translation-validation driver.

These require the dev build tools (triton-opt, mlir-translate) and z3; the whole
module is skipped if any is unavailable. No GPU is needed (TTIR only).
"""
import pytest

import triton
import triton.language as tl
from triton.tools import smt_equivalence as tv


def _tools_available() -> bool:
    import subprocess
    try:
        triton_opt = tv.default_triton_opt()
        mlir_translate = tv.default_mlir_translate()
        tv.default_z3()
    except Exception:
        return False
    # The binaries existing is not enough: triton-opt must register the pass
    # (a triton-opt built from main would make every rejection test pass
    # vacuously) and mlir-translate must carry the SMT Real-sort patch.
    try:
        r = subprocess.run([triton_opt, "--convert-triton-to-smt"],
                           input="module {}\n", capture_output=True, text=True,
                           timeout=60)
        if "Unknown command line argument" in r.stderr:
            return False
        probe = ('module {\n'
                 '  smt.solver() : () -> () {\n'
                 '    %r = smt.declare_fun "probe" : !smt.real\n'
                 '    smt.yield\n'
                 '  }\n'
                 '}\n')
        r = subprocess.run([mlir_translate, "--export-smtlib"], input=probe,
                           capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _tools_available(),
    reason="requires triton-opt, mlir-translate and z3")

PTR = "*fp32"


@triton.jit
def affine_naive(a_ptr, b_ptr, c_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=0.0)
    c = tl.load(c_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, a * b + c, mask=m)


@triton.jit
def affine_optimized(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m, other=0.0)
    b = tl.load(b_ptr + offs, mask=m, other=0.0)
    c = tl.load(c_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, tl.math.fma(a, b, c), mask=m)


@triton.jit
def affine_drop_c(a_ptr, b_ptr, c_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, a * b, mask=m)


@triton.jit
def sub_direct(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, a - b, mask=m)


@triton.jit
def sub_negadd(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, a + (-b), mask=m)


@triton.jit
def double_mul(a_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, a * 2.0, mask=m)


@triton.jit
def double_add(a_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, a + a, mask=m)


@triton.jit
def relu_where(a_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, tl.where(a > 0.0, a, 0.0), mask=m)


@triton.jit
def relu_max(a_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, tl.maximum(a, 0.0), mask=m)


@triton.jit
def abs_builtin(a_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, tl.abs(a), mask=m)


@triton.jit
def abs_where(a_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, tl.where(a > 0.0, a, -a), mask=m)


@triton.jit
def div_sum_num(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=1.0)
    tl.store(out_ptr + idx, (a + a) / b, mask=m)


@triton.jit
def div_scaled_num(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=1.0)
    tl.store(out_ptr + idx, (2.0 * a) / b, mask=m)


@triton.jit
def div_direct(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=1.0)
    tl.store(out_ptr + idx, a / b, mask=m)


@triton.jit
def div_reciprocal(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=1.0)
    tl.store(out_ptr + idx, a * (1.0 / b), mask=m)


@triton.jit
def eq_direct(a_ptr, b_ptr, out_ptr, n):
    # Compound mask exercises `arith.andi` on i1; storing an i1 tensor
    # exercises the extui + pointer-bitcast idiom the frontend generates.
    idx = tl.program_id(0)
    m = (idx < n) & (idx >= 0)
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, a == b, mask=m)


@triton.jit
def eq_not_ne(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = (idx < n) & (idx >= 0)
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, ~(a != b), mask=m)


@triton.jit
def eq_as_float(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, (a == b).to(tl.float32), mask=m)


@triton.jit
def eq_as_float_where(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, tl.where(a == b, 1.0, 0.0), mask=m)


@triton.jit
def add_naive(a_ptr, b_ptr, out_ptr, n):
    idx = tl.program_id(0)
    m = idx < n
    a = tl.load(a_ptr + idx, mask=m, other=0.0)
    b = tl.load(b_ptr + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, a + b, mask=m)


@triton.jit
def add_vec(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m, other=0.0)
    b = tl.load(b_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=m)


_S2 = {"a_ptr": PTR, "out_ptr": PTR, "n": "i32"}
_S3 = {"a_ptr": PTR, "b_ptr": PTR, "out_ptr": PTR, "n": "i32"}
_S3B = {"a_ptr": PTR, "b_ptr": PTR, "out_ptr": "*i1", "n": "i32"}
_S4 = {"a_ptr": PTR, "b_ptr": PTR, "c_ptr": PTR, "out_ptr": PTR, "n": "i32"}

# (id, src_fn, tgt_fn, src_sig, tgt_sig, tgt_constexprs, expected_equivalent)
EQUIVALENT_CASES = [
    ("fma_naive_vs_optimized", affine_naive, affine_optimized, _S4,
     {**_S4, "BLOCK": "constexpr"}, {"BLOCK": 128}),
    ("sub_rewrite", sub_direct, sub_negadd, _S3, _S3, {}),
    ("double_real_identity", double_mul, double_add, _S2, _S2, {}),
    ("relu_where_vs_max", relu_where, relu_max, _S2, _S2, {}),
    ("broadcast_add", add_naive, add_vec, _S3, {**_S3, "BLOCK": "constexpr"},
     {"BLOCK": 128}),
    ("abs_builtin_vs_where", abs_builtin, abs_where, _S2, _S2, {}),
    ("div_numerator_rewrite", div_sum_num, div_scaled_num, _S3, _S3, {}),
    ("bool_store_eq_vs_not_ne", eq_direct, eq_not_ne, _S3B, _S3B, {}),
    ("bool_to_float_vs_where", eq_as_float, eq_as_float_where, _S3, _S3, {}),
]


@pytest.mark.parametrize("name,src,tgt,src_sig,tgt_sig,tgt_cx",
                         EQUIVALENT_CASES, ids=[c[0] for c in EQUIVALENT_CASES])
def test_equivalent(name, src, tgt, src_sig, tgt_sig, tgt_cx):
    res = tv.check_kernels(src, tgt, src_sig, tgt_signature=tgt_sig,
                           tgt_constexprs=tgt_cx)
    assert res.equivalent, f"{name}: {res.verdict} (z3={res.equivalence_status})"


def test_division_at_zero_is_uninterpreted():
    # SMT-LIB real division is total but unspecified at zero denominators, so
    # a/b and a*(1/b) are NOT equivalent: at b=0 they apply the arbitrary
    # division function to different arguments. This pins down the deliberate
    # semantics of the divf encoding.
    res = tv.check_kernels(div_direct, div_reciprocal, _S3)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict,
                                             res.equivalence_status)


def test_not_equivalent_counterexample():
    # a*b+c  vs  a*b  differ; expect a concrete counterexample.
    res = tv.check_kernels(affine_naive, affine_drop_c, _S4, counterexample=True)
    assert res.verdict == "NOT_EQUIVALENT", res.verdict
    assert res.equivalence_status == "sat"
    assert "define-fun" in res.z3_output  # model was produced


# --- Phase 2, milestone 1: 1-D tt.reduce (add / max / min). Because floats are
#     ideal reals, folding the block lanes in any order is provably equivalent,
#     so reassociations validate without permutation/multiset machinery. ---

@triton.jit
def red_sum_of_add(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ptr + tl.program_id(0), tl.sum(a + b))


@triton.jit
def red_add_of_sums(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ptr + tl.program_id(0), tl.sum(a) + tl.sum(b))


@triton.jit
def red_scaled_sum(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + tl.program_id(0), 2.0 * tl.sum(x))


@triton.jit
def red_sum_scaled(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + tl.program_id(0), tl.sum(x + x))


@triton.jit
def red_sum(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.program_id(0), tl.sum(tl.load(x_ptr + offs)))


@triton.jit
def red_max(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.program_id(0), tl.max(tl.load(x_ptr + offs)))


@triton.jit
def red_sum_plus_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.program_id(0), tl.sum(tl.load(x_ptr + offs)) + 1.0)


@triton.jit
def red_argmax(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + tl.program_id(0), tl.argmax(x, axis=0).to(tl.float32))


@triton.jit
def red_sum_2d(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # A rank-2, axis-1 reduction -- outside the 1-D milestone contract.
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    x = tl.load(x_ptr + rows * BLOCK + cols)
    tl.store(out_ptr + tl.arange(0, BLOCK), tl.sum(x, axis=1))


_SR2 = {"a_ptr": PTR, "b_ptr": PTR, "out_ptr": PTR, "n": "i32",
        "BLOCK": "constexpr"}
_SR1 = {"x_ptr": PTR, "out_ptr": PTR, "n": "i32", "BLOCK": "constexpr"}
_CX = {"BLOCK": 4}

# (id, src_fn, tgt_fn, signature)
REDUCE_EQUIVALENT_CASES = [
    ("sum_of_add_vs_add_of_sums", red_sum_of_add, red_add_of_sums, _SR2),
    ("scaled_sum_vs_sum_scaled", red_scaled_sum, red_sum_scaled, _SR1),
    ("max_is_reflexive", red_max, red_max, _SR1),
]


@pytest.mark.parametrize("name,src,tgt,sig", REDUCE_EQUIVALENT_CASES,
                         ids=[c[0] for c in REDUCE_EQUIVALENT_CASES])
def test_reduce_equivalent(name, src, tgt, sig):
    res = tv.check_kernels(src, tgt, sig, src_constexprs=_CX,
                           tgt_constexprs=_CX)
    assert res.equivalent, f"{name}: {res.verdict} (z3={res.equivalence_status})"


def test_reduce_sum_vs_max_not_equivalent():
    res = tv.check_kernels(red_sum, red_max, _SR1, src_constexprs=_CX,
                           tgt_constexprs=_CX, counterexample=True)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict,
                                             res.equivalence_status)


def test_reduce_sum_vs_sum_plus_one_not_equivalent():
    res = tv.check_kernels(red_sum, red_sum_plus_one, _SR1, src_constexprs=_CX,
                           tgt_constexprs=_CX)
    assert res.verdict == "NOT_EQUIVALENT", res.verdict


def test_reject_fused_arg_reduce():
    # argmax lowers to a two-result tt.reduce; getSingleCombiner() is null.
    res = tv.check_kernels(red_argmax, red_argmax, _SR1, src_constexprs=_CX,
                           tgt_constexprs=_CX)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_rank2_reduce_supported_via_memory_model():
    # The legacy path supports only axis-0 rank-1 reductions; a rank-2 axis-1
    # reduce is rejected there and routed to the phase-3 block memory model,
    # which materializes the 2-D lanes and folds along the axis.
    res = tv.check_kernels(red_sum_2d, red_sum_2d, _SR1, src_constexprs=_CX,
                           tgt_constexprs=_CX)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model, "expected the block-memory-model retry path"


# --- Adversarial regression tests for the soundness review (inline TTIR so the
#     exact patterns are exercised regardless of frontend lowering). ---

def _mod(body: str) -> str:
    return "module {\n" + body + "\n}\n"


# Both kernels store *different* values to out[idx+1] (non-identity addressing).
_SHIFTED_A = _mod("""
  tt.func public @k(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<f32>, i32
    %av = tt.load %pa, %m, %cst : !tt.ptr<f32>
    %ip1 = arith.addi %idx, %c1 : i32
    %po = tt.addptr %o, %ip1 : !tt.ptr<f32>, i32
    tt.store %po, %av, %m : !tt.ptr<f32>
    tt.return
  }""")
_SHIFTED_2A = _mod("""
  tt.func public @k(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<f32>, i32
    %av = tt.load %pa, %m, %cst : !tt.ptr<f32>
    %d = arith.addf %av, %av : f32
    %ip1 = arith.addi %idx, %c1 : i32
    %po = tt.addptr %o, %ip1 : !tt.ptr<f32>, i32
    tt.store %po, %d, %m : !tt.ptr<f32>
    tt.return
  }""")

# i8 (x+1) >s x  vs  always-true: differ at x=127 only under wrapping bit-vectors.
_I8_CMP = _mod("""
  tt.func public @k(%a: !tt.ptr<i8>, %o: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %one = arith.constant 1.000000e+00 : f32
    %zero = arith.constant 0.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i8>, i32
    %av = tt.load %pa, %m, %c0 : !tt.ptr<i8>
    %xp1 = arith.addi %av, %c1 : i8
    %gt = arith.cmpi sgt, %xp1, %av : i8
    %r = arith.select %gt, %one, %zero : f32
    %po = tt.addptr %o, %idx : !tt.ptr<f32>, i32
    tt.store %po, %r, %m : !tt.ptr<f32>
    tt.return
  }""")
_I8_TRUE = _mod("""
  tt.func public @k(%a: !tt.ptr<i8>, %o: !tt.ptr<f32>, %n: i32) {
    %one = arith.constant 1.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %po = tt.addptr %o, %idx : !tt.ptr<f32>, i32
    tt.store %po, %one, %m : !tt.ptr<f32>
    tt.return
  }""")


def test_non_identity_addressing_not_equivalent():
    # Different values to a shifted address must NOT be reported EQUIVALENT.
    # The legacy path disproves identity addressing (scope 0 sat), which now
    # routes to the block memory model; there the shifted store is modeled
    # per-offset and the differing values are caught by the equivalence scope.
    res = tv.check_equivalence(_SHIFTED_A, _SHIFTED_2A)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict,
                                             res.addressing_status)
    assert res.memory_model, "expected the block-memory-model retry path"


def test_non_identity_addressing_equivalent_pair():
    # The same shifted-store kernel against itself is EQUIVALENT under the
    # block memory model -- the phase-3 point: non-identity addressing gets a
    # real verdict instead of UNSUPPORTED.
    res = tv.check_equivalence(_SHIFTED_A, _SHIFTED_A)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model, "expected the block-memory-model retry path"


def test_integer_overflow_is_faithful():
    # (x+1) >s x is false at x=127 under i8 wrapping => NOT equivalent to true.
    res = tv.check_equivalence(_I8_CMP, _I8_TRUE)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.equivalence_status)


# --- Regression tests for the second-round review (must REJECT, never
#     silently report EQUIVALENT). ---

_DIV0 = _mod("""
  tt.func public @k(%o: !tt.ptr<i32>, %n: i32) {
    %z = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %d = arith.divui %one, %z : i32
    %p = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %p, %d, %m : !tt.ptr<i32>
    tt.return
  }""")
_STORE_M1 = _mod("""
  tt.func public @k(%o: !tt.ptr<i32>, %n: i32) {
    %c = arith.constant -1 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %p = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %p, %c, %m : !tt.ptr<i32>
    tt.return
  }""")
_MASK_NO_OTHER = _mod("""
  tt.func public @k(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<f32>, i32
    %v = tt.load %pa, %m : !tt.ptr<f32>
    %p = tt.addptr %o, %idx : !tt.ptr<f32>, i32
    tt.store %p, %v, %m : !tt.ptr<f32>
    tt.return
  }""")
_MASK_OTHER = _mod("""
  tt.func public @k(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<f32>, i32
    %v = tt.load %pa, %m, %cst : !tt.ptr<f32>
    %p = tt.addptr %o, %idx : !tt.ptr<f32>, i32
    tt.store %p, %v, %m : !tt.ptr<f32>
    tt.return
  }""")
_NONVOID = _mod("""
  tt.func public @k(%o: !tt.ptr<f32>, %n: i32) -> f32 {
    %c = arith.constant 1.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %p = tt.addptr %o, %idx : !tt.ptr<f32>, i32
    tt.store %p, %c, %m : !tt.ptr<f32>
    tt.return %c : f32
  }""")
# A 64-lane store with an unused make_range(128): the make_range must match the
# store lane count, else the writer decomposition would be wrong.
_LANE_MISMATCH = _mod("""
  tt.func public @k(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %c64 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %off0 = arith.muli %pid, %c64 : i32
    %r64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %r128 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %so = tt.splat %off0 : i32 -> tensor<64xi32>
    %off = arith.addi %so, %r64 : tensor<64xi32>
    %sm = tt.splat %n : i32 -> tensor<64xi32>
    %m = arith.cmpi slt, %off, %sm : tensor<64xi32>
    %sa = tt.splat %a : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %pa = tt.addptr %sa, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %pa, %m, %cst : tensor<64x!tt.ptr<f32>>
    %soo = tt.splat %o : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %po = tt.addptr %soo, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %po, %v, %m : tensor<64x!tt.ptr<f32>>
    tt.return
  }""")


def test_reject_integer_division_by_zero():
    # bvudiv is total (1/0 = all-ones); without a definedness model this op is
    # rejected rather than reported equivalent to storing -1.
    res = tv.check_equivalence(_DIV0, _STORE_M1)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_masked_load_without_other():
    res = tv.check_equivalence(_MASK_NO_OTHER, _MASK_OTHER)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_non_void_return():
    res = tv.check_equivalence(_NONVOID, _NONVOID)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_inconsistent_lane_count_routes_to_memory_model():
    # The legacy writer decomposition cannot attribute lanes when an unrelated
    # tt.make_range extent disagrees with the store lane count, so it rejects;
    # the block memory model has no writer decomposition (each lane carries
    # its own offset), so the dead 128-lane range is materialized harmlessly
    # and the self-pair is EQUIVALENT.
    res = tv.check_equivalence(_LANE_MISMATCH, _LANE_MISMATCH)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model, "expected the block-memory-model retry path"


# --- Third-round review regressions (reject; never false EQUIVALENT or crash). ---

# Representation-changing pointer bitcast: f16 store vs bf16 stored through a
# bitcast output pointer. Same bits differ; must not be EQUIVALENT.
_F16 = _mod("""
  tt.func public @k(%o: !tt.ptr<f16>, %n: i32) {
    %c = arith.constant 1.000000e+00 : f16
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %p = tt.addptr %o, %idx : !tt.ptr<f16>, i32
    tt.store %p, %c, %m : !tt.ptr<f16>
    tt.return
  }""")
_BF16_VIA_BITCAST = _mod("""
  tt.func public @k(%o: !tt.ptr<f16>, %n: i32) {
    %c = arith.constant 1.000000e+00 : bf16
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %p = tt.addptr %o, %idx : !tt.ptr<f16>, i32
    %pc = tt.bitcast %p : !tt.ptr<f16> -> !tt.ptr<bf16>
    tt.store %pc, %c, %m : !tt.ptr<bf16>
    tt.return
  }""")
# i1 integer arithmetic (previously aborted triton-opt with signal 6).
_I1_ADD = _mod("""
  tt.func public @k(%a: !tt.ptr<i1>, %o: !tt.ptr<i1>, %n: i32) {
    %f = arith.constant false
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i1>, i32
    %v = tt.load %pa, %m, %f : !tt.ptr<i1>
    %r = arith.addi %v, %v : i1
    %p = tt.addptr %o, %idx : !tt.ptr<i1>, i32
    tt.store %p, %r, %m : !tt.ptr<i1>
    tt.return
  }""")
# Pointer-valued select (previously a signal-11 crash).
_PTR_SELECT = _mod("""
  tt.func public @k(%o: !tt.ptr<f32>, %n: i32) {
    %c = arith.constant 1.000000e+00 : f32
    %t = arith.constant true
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %sel = arith.select %t, %o, %o : !tt.ptr<f32>
    %p = tt.addptr %sel, %idx : !tt.ptr<f32>, i32
    tt.store %p, %c, %m : !tt.ptr<f32>
    tt.return
  }""")
# nsw overflow flag introduces poison the wrapping model does not track.
_NSW_ADD = _mod("""
  tt.func public @k(%o: !tt.ptr<i32>, %n: i32) {
    %c = arith.constant 1 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %r = arith.addi %idx, %c overflow<nsw> : i32
    %p = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %p, %r, %m : !tt.ptr<i32>
    tt.return
  }""")
_PLAIN_ADD = _mod("""
  tt.func public @k(%o: !tt.ptr<i32>, %n: i32) {
    %c = arith.constant 1 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %r = arith.addi %idx, %c : i32
    %p = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %p, %r, %m : !tt.ptr<i32>
    tt.return
  }""")
# A module carrying a stray smt.solver scope must not smuggle a forged result.
_INJECTED = _mod("""
  tt.func public @k(%o: !tt.ptr<f32>, %n: i32) {
    %c = arith.constant 1.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %p = tt.addptr %o, %idx : !tt.ptr<f32>, i32
    tt.store %p, %c, %m : !tt.ptr<f32>
    tt.return
  }
  smt.solver() : () -> () {
    %b = smt.declare_fun : !smt.bool
    smt.assert %b
    smt.check sat {} unknown {} unsat {}
  }""")


def test_reject_representation_changing_bitcast():
    res = tv.check_equivalence(_F16, _BF16_VIA_BITCAST)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_i1_arithmetic():
    res = tv.check_equivalence(_I1_ADD, _I1_ADD)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_pointer_select():
    res = tv.check_equivalence(_PTR_SELECT, _PTR_SELECT)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_overflow_flags():
    res = tv.check_equivalence(_NSW_ADD, _PLAIN_ADD)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_injected_solver_scope():
    res = tv.check_equivalence(_INJECTED, _PLAIN_ADD)
    assert res.verdict == "UNSUPPORTED", res.verdict


# --- Fourth-round review regressions. ---

# Chained tt.addptr can wrap at 32 bits but not under real 64-bit pointer math.
_STORE_I8 = _mod("""
  tt.func public @k(%o: !tt.ptr<i8>, %n: i32) {
    %c = arith.constant 1 : i8
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %p = tt.addptr %o, %idx : !tt.ptr<i8>, i32
    tt.store %p, %c, %m : !tt.ptr<i8>
    tt.return
  }""")
_CHAINED_ADDPTR = _mod("""
  tt.func public @k(%o: !tt.ptr<i8>, %n: i32) {
    %c = arith.constant 1 : i8
    %big = arith.constant 2147483647 : i32
    %two = arith.constant 2 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %p0 = tt.addptr %o, %idx : !tt.ptr<i8>, i32
    %p1 = tt.addptr %p0, %big : !tt.ptr<i8>, i32
    %p2 = tt.addptr %p1, %big : !tt.ptr<i8>, i32
    %p3 = tt.addptr %p2, %two : !tt.ptr<i8>, i32
    tt.store %p3, %c, %m : !tt.ptr<i8>
    tt.return
  }""")
# Raw i8 load through an i1->i8 pointer bitcast reinterprets the buffer sort.
_RAW_NE = _mod("""
  tt.func public @k(%a: !tt.ptr<i1>, %o: !tt.ptr<i8>, %n: i32) {
    %z = arith.constant 0 : i8
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i1>, i32
    %pc = tt.bitcast %pa : !tt.ptr<i1> -> !tt.ptr<i8>
    %raw = tt.load %pc, %m, %z : !tt.ptr<i8>
    %nz = arith.cmpi ne, %raw, %z : i8
    %w = arith.extui %nz : i1 to i8
    %po = tt.addptr %o, %idx : !tt.ptr<i8>, i32
    tt.store %po, %w, %m : !tt.ptr<i8>
    tt.return
  }""")
_RAW_EQ = _mod("""
  tt.func public @k(%a: !tt.ptr<i1>, %o: !tt.ptr<i8>, %n: i32) {
    %z = arith.constant 0 : i8
    %one = arith.constant 1 : i8
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i1>, i32
    %pc = tt.bitcast %pa : !tt.ptr<i1> -> !tt.ptr<i8>
    %raw = tt.load %pc, %m, %z : !tt.ptr<i8>
    %eq = arith.cmpi eq, %raw, %one : i8
    %w = arith.extui %eq : i1 to i8
    %po = tt.addptr %o, %idx : !tt.ptr<i8>, i32
    tt.store %po, %w, %m : !tt.ptr<i8>
    tt.return
  }""")


def test_chained_addptr_wrap_not_equivalent():
    # The chained kernel's offsets sum to idx + 2^32 in real (64-bit) pointer
    # arithmetic but would wrap to idx in a 32-bit model. The legacy path
    # rejects chains outright; the block memory model accumulates each
    # sign-extended offset in a 64-bit block offset, so the chain lands
    # 2^32 elements past @src's store and the pair must NOT be EQUIVALENT.
    # (A false EQUIVALENT here is the exact unsoundness the phase-2 rejection
    # guarded against.)
    res = tv.check_equivalence(_STORE_I8, _CHAINED_ADDPTR)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model, "expected the block-memory-model retry path"


def test_chained_addptr_self_equivalent():
    # The same chained kernel against itself is EQUIVALENT: chains are now
    # modeled exactly, not rejected.
    res = tv.check_equivalence(_CHAINED_ADDPTR, _CHAINED_ADDPTR)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


def test_reject_reinterpreting_load_through_bitcast():
    res = tv.check_equivalence(_RAW_NE, _RAW_EQ)
    assert res.verdict == "UNSUPPORTED", res.verdict


# --- Fifth-round review regressions. ---

# A dead volatile load is an observable access that must not be proved away.
_VOLATILE = _mod("""
  tt.func public @k(%a: !tt.ptr<i32>, %o: !tt.ptr<i32>, %n: i32) {
    %z = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i32>, i32
    %v = tt.load %pa {isVolatile = true} : !tt.ptr<i32>
    %po = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %po, %z, %m : !tt.ptr<i32>
    tt.return
  }""")
_STORE_ZERO = _mod("""
  tt.func public @k(%a: !tt.ptr<i32>, %o: !tt.ptr<i32>, %n: i32) {
    %z = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %po = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %po, %z, %m : !tt.ptr<i32>
    tt.return
  }""")


def test_reject_volatile_load():
    res = tv.check_equivalence(_VOLATILE, _STORE_ZERO)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_nonfinite_timeout_is_unknown():
    # A non-finite/unrepresentable timeout must return a verdict, not raise.
    for t in (float("nan"), float("inf"), 1e10):
        res = tv.check_equivalence(_STORE_ZERO, _STORE_ZERO, timeout=t)
        assert res.verdict in ("UNKNOWN", "EQUIVALENT"), (t, res.verdict)


# --- Phase-2 reduction rejections (inline TTIR: exact patterns regardless of
#     frontend lowering). ---

# A rank-1 reduction whose combiner is *product* (arith.mulf) -- not in the
# supported combiner set {addf, max/minnumf, max/minimumf}; must be rejected.
_RED_MUL = _mod("""
  tt.func public @k(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %pid = tt.get_program_id x : i32
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %sp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %pp = tt.addptr %sp, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %pp : tensor<4x!tt.ptr<f32>>
    %s = "tt.reduce"(%v) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %c = arith.mulf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<4xf32>) -> f32
    %po = tt.addptr %o, %pid : !tt.ptr<f32>, i32
    tt.store %po, %s : !tt.ptr<f32>
    tt.return
  }""")

# A masked vector load feeding a reduction: masked-out lanes would need the
# combiner identity as `other`; rejected in this milestone.
_RED_MASKED = _mod("""
  tt.func public @k(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %pid = tt.get_program_id x : i32
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %sp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %pp = tt.addptr %sp, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %mask = arith.constant dense<true> : tensor<4xi1>
    %other = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %v = tt.load %pp, %mask, %other : tensor<4x!tt.ptr<f32>>
    %s = "tt.reduce"(%v) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %c = arith.addf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<4xf32>) -> f32
    %po = tt.addptr %o, %pid : !tt.ptr<f32>, i32
    tt.store %po, %s : !tt.ptr<f32>
    tt.return
  }""")


def test_reject_unsupported_reduce_combiner():
    res = tv.check_equivalence(_RED_MUL, _RED_MUL)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_masked_load_feeding_reduce_via_memory_model():
    # The legacy reduction path rejects a masked load feeding a reduce; the
    # block memory model supports it exactly (masked-out lanes carry `other`
    # into the fold, matching the semantics of the loaded tensor).
    res = tv.check_equivalence(_RED_MASKED, _RED_MASKED)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model, "expected the block-memory-model retry path"


# A reduce combiner region that also performs a side-effecting (volatile) load:
# getSingleCombiner() still returns the addf, but the pass must NOT silently drop
# the observable load. Require the region to be exactly {combine, return}.
_RED_COMBINER_SIDE_EFFECT = _mod("""
  tt.func public @k(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %pid = tt.get_program_id x : i32
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %sp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %pp = tt.addptr %sp, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %pp : tensor<4x!tt.ptr<f32>>
    %s = "tt.reduce"(%v) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %junk = tt.load %x {isVolatile = true} : !tt.ptr<f32>
      %c = arith.addf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<4xf32>) -> f32
    %po = tt.addptr %o, %pid : !tt.ptr<f32>, i32
    tt.store %po, %s : !tt.ptr<f32>
    tt.return
  }""")

# A `tt.reshape allow_reorder` whose result feeds lane-wise arithmetic (not the
# reduce directly): allow_reorder leaves the element order unspecified, so the
# identity pass-through would be unsound. Must be rejected.
_RED_RESHAPE_REORDER_ELEMENTWISE = _mod("""
  tt.func public @k(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %pid = tt.get_program_id x : i32
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %sp = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %pp = tt.addptr %sp, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %pp : tensor<4x!tt.ptr<f32>>
    %rs = tt.reshape %v allow_reorder : tensor<4xf32> -> tensor<4xf32>
    %sq = arith.mulf %rs, %v : tensor<4xf32>
    %s = "tt.reduce"(%sq) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %c = arith.addf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<4xf32>) -> f32
    %po = tt.addptr %o, %pid : !tt.ptr<f32>, i32
    tt.store %po, %s : !tt.ptr<f32>
    tt.return
  }""")


def test_reject_reduce_combiner_with_side_effect():
    res = tv.check_equivalence(_RED_COMBINER_SIDE_EFFECT,
                               _RED_COMBINER_SIDE_EFFECT)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_reshape_reorder_feeding_elementwise():
    res = tv.check_equivalence(_RED_RESHAPE_REORDER_ELEMENTWISE,
                               _RED_RESHAPE_REORDER_ELEMENTWISE)
    assert res.verdict == "UNSUPPORTED", res.verdict


# --- Phase-2 milestone 2: shifts + integer ext/trunc with solver-discharged
#     well-definedness (mlir-tv's wellDefined(op, cond) as a dedicated scope).
#     Inline TTIR: scalar i32 kernels that load x from %a and store an i32. ---

def _i32_kernel(compute: str, result: str) -> str:
    """A scalar kernel: x = a[idx]; <compute>; o[idx] = <result> (all i32)."""
    return _mod(f"""
  tt.func public @k(%a: !tt.ptr<i32>, %o: !tt.ptr<i32>, %n: i32) {{
    %c0 = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i32>, i32
    %x = tt.load %pa, %m, %c0 : !tt.ptr<i32>
    {compute}
    %po = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %po, {result}, %m : !tt.ptr<i32>
    tt.return
  }}""")


def test_shl_vs_mul_equivalent():
    src = _i32_kernel("""%c2 = arith.constant 2 : i32
    %r = arith.shli %x, %c2 : i32""", "%r")
    tgt = _i32_kernel("""%c4 = arith.constant 4 : i32
    %r = arith.muli %x, %c4 : i32""", "%r")
    res = tv.check_equivalence(src, tgt)
    assert res.equivalent, (res.verdict, res.ub_status, res.equivalence_status)
    assert res.ub_status == "unsat", res.ub_status


def test_shl_vs_mul3_not_equivalent():
    src = _i32_kernel("""%c1 = arith.constant 1 : i32
    %r = arith.shli %x, %c1 : i32""", "%r")
    tgt = _i32_kernel("""%c3 = arith.constant 3 : i32
    %r = arith.muli %x, %c3 : i32""", "%r")
    res = tv.check_equivalence(src, tgt)
    assert res.verdict == "NOT_EQUIVALENT", res.verdict


def test_shrsi_vs_shrui_not_equivalent():
    src = _i32_kernel("""%c1 = arith.constant 1 : i32
    %r = arith.shrsi %x, %c1 : i32""", "%r")
    tgt = _i32_kernel("""%c1 = arith.constant 1 : i32
    %r = arith.shrui %x, %c1 : i32""", "%r")
    res = tv.check_equivalence(src, tgt)
    assert res.verdict == "NOT_EQUIVALENT", res.verdict


def test_shift_by_31_boundary_equivalent():
    k = _i32_kernel("""%c31 = arith.constant 31 : i32
    %r = arith.shrui %x, %c31 : i32""", "%r")
    res = tv.check_equivalence(k, k)
    assert res.equivalent, (res.verdict, res.ub_status)


def test_same_ub_domain_shift_by_width_equivalent():
    # amt == width is poison on ALL inputs -- for both kernels identically.
    # Bidirectional refinement: identical UB domains + (vacuously) equal
    # outputs off-domain => EQUIVALENT.
    k = _i32_kernel("""%c32 = arith.constant 32 : i32
    %r = arith.shli %x, %c32 : i32""", "%r")
    res = tv.check_equivalence(k, k)
    assert res.equivalent, (res.verdict, res.ub_status)
    assert res.ub_status == "unsat", res.ub_status


def test_same_unbounded_shift_equivalent():
    # The shift amount is unbounded, so both kernels are UB on the same
    # inputs (x >= 32) and equal elsewhere => EQUIVALENT.
    k = _i32_kernel("""%r = arith.shli %x, %x : i32""", "%r")
    res = tv.check_equivalence(k, k)
    assert res.equivalent, (res.verdict, res.ub_status)
    assert res.ub_status == "unsat", res.ub_status


def test_asymmetric_ub_not_equivalent():
    # src stores an oversized-shift result (UB for x >= 32); tgt stores a
    # defined value everywhere. Their UB domains differ => NOT_EQUIVALENT.
    src = _i32_kernel("""%r = arith.shli %x, %x : i32""", "%r")
    tgt = _i32_kernel("""%c0b = arith.constant 0 : i32
    %r = arith.addi %x, %c0b : i32""", "%r")
    res = tv.check_equivalence(src, tgt)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.ub_status)
    assert res.ub_status == "sat", res.ub_status


def test_dead_poison_is_not_ub():
    # src computes an unbounded shift but never stores it; poison that does
    # not reach memory is not UB, so src == tgt (both store x).
    src = _i32_kernel("""%dead = arith.shli %x, %x : i32
    %c0b = arith.constant 0 : i32
    %r = arith.addi %x, %c0b : i32""", "%r")
    tgt = _i32_kernel("""%r = arith.addi %x, %x : i32
    %r2 = arith.subi %r, %x : i32""", "%r2")
    res = tv.check_equivalence(src, tgt)
    assert res.equivalent, (res.verdict, res.ub_status)
    assert res.ub_status == "unsat", res.ub_status


def test_dead_vs_stored_poison_not_equivalent():
    # Same dead shift on the src side, but tgt STORES the poison; exact
    # poison tracking must distinguish these (a coarse global conjunction
    # would report them equivalent).
    src = _i32_kernel("""%dead = arith.shli %x, %x : i32
    %c0b = arith.constant 0 : i32
    %r = arith.addi %x, %c0b : i32""", "%r")
    tgt = _i32_kernel("""%r = arith.shli %x, %x : i32""", "%r")
    res = tv.check_equivalence(src, tgt)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.ub_status)
    assert res.ub_status == "sat", res.ub_status


def test_mask_gates_poison_store_equivalent():
    # The store mask excludes exactly the poisoned inputs, so neither kernel
    # is ever UB, and under the mask amt < 32 makes amt == (amt & 31).
    src = _i32_kernel("""%c32b = arith.constant 32 : i32
    %pm = arith.cmpi ult, %x, %c32b : i32
    %sh = arith.shli %x, %x : i32
    %r = arith.select %pm, %sh, %x : i32""", "%r")
    tgt = _i32_kernel("""%c32b = arith.constant 32 : i32
    %c31b = arith.constant 31 : i32
    %pm = arith.cmpi ult, %x, %c32b : i32
    %amt = arith.andi %x, %c31b : i32
    %sh = arith.shli %x, %amt : i32
    %r = arith.select %pm, %sh, %x : i32""", "%r")
    res = tv.check_equivalence(src, tgt)
    assert res.equivalent, (res.verdict, res.ub_status)
    assert res.ub_status == "unsat", res.ub_status


def test_masked_shift_amount_equivalent():
    # x >> (x & 31): the mask makes well-definedness provable for all inputs --
    # the point of solver-discharged WD over syntactic rejection.
    k = _i32_kernel("""%c31 = arith.constant 31 : i32
    %amt = arith.andi %x, %c31 : i32
    %r = arith.shrui %x, %amt : i32""", "%r")
    res = tv.check_equivalence(k, k)
    assert res.equivalent, (res.verdict, res.ub_status)
    assert res.ub_status == "unsat", res.ub_status


def test_reject_shl_overflow_flags():
    k = _i32_kernel("""%c1 = arith.constant 1 : i32
    %r = arith.shli %x, %c1 overflow<nsw> : i32""", "%r")
    res = tv.check_equivalence(k, k)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_shr_exact_flag():
    k = _i32_kernel("""%c1 = arith.constant 1 : i32
    %r = arith.shrui %x, %c1 exact : i32""", "%r")
    res = tv.check_equivalence(k, k)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_trunci_overflow_flags():
    k = _i32_kernel("""%r32 = arith.trunci %x overflow<nsw> : i32 to i16
    %r = arith.extui %r32 : i16 to i32""", "%r")
    res = tv.check_equivalence(k, k)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_ext_trunc_roundtrip_equivalent():
    # trunc(zext(x)) == x, via i16.
    src = _i32_kernel("""%w = arith.trunci %x : i32 to i16
    %r = arith.extui %w : i16 to i32""", "%r")
    tgt = _i32_kernel("""%cffff = arith.constant 65535 : i32
    %r = arith.andi %x, %cffff : i32""", "%r")
    res = tv.check_equivalence(src, tgt)
    assert res.equivalent, (res.verdict, res.equivalence_status)


def test_extsi_vs_shift_pair_equivalent():
    # sext_16_32(trunc(x)) == ashr(shl(x, 16), 16): the classic identity.
    src = _i32_kernel("""%w = arith.trunci %x : i32 to i16
    %r = arith.extsi %w : i16 to i32""", "%r")
    tgt = _i32_kernel("""%c16 = arith.constant 16 : i32
    %hi = arith.shli %x, %c16 : i32
    %r = arith.shrsi %hi, %c16 : i32""", "%r")
    res = tv.check_equivalence(src, tgt)
    assert res.equivalent, (res.verdict, res.equivalence_status)


def test_extsi_vs_extui_not_equivalent():
    src = _i32_kernel("""%w = arith.trunci %x : i32 to i16
    %r = arith.extsi %w : i16 to i32""", "%r")
    tgt = _i32_kernel("""%w = arith.trunci %x : i32 to i16
    %r = arith.extui %w : i16 to i32""", "%r")
    res = tv.check_equivalence(src, tgt)
    assert res.verdict == "NOT_EQUIVALENT", res.verdict


def test_trunc_to_i1_vs_lowbit_test_equivalent():
    # trunci to i1 takes the low bit; equivalent to (x & 1) != 0. The stored
    # sort is smt.bool via an i1 output buffer.
    header = """  tt.func public @k(%a: !tt.ptr<i32>, %o: !tt.ptr<i1>, %n: i32) {{
    %c0 = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i32>, i32
    %x = tt.load %pa, %m, %c0 : !tt.ptr<i32>
    {compute}
    %po = tt.addptr %o, %idx : !tt.ptr<i1>, i32
    tt.store %po, %r, %m : !tt.ptr<i1>
    tt.return
  }}"""
    src = _mod(header.format(compute="%r = arith.trunci %x : i32 to i1"))
    tgt = _mod(header.format(compute="""%c1 = arith.constant 1 : i32
    %lo = arith.andi %x, %c1 : i32
    %r = arith.cmpi ne, %lo, %c0 : i32"""))
    res = tv.check_equivalence(src, tgt)
    assert res.equivalent, (res.verdict, res.equivalence_status)


# Shifts inside a reduction (vector) context: WD conditions accumulate per lane.
_RED_SHL_ADDR = _mod("""
  tt.func public @k(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c1 = arith.constant dense<1> : tensor<4xi32>
    %offs = arith.shli %r, %c1 : tensor<4xi32>
    %sp = tt.splat %a : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %pp = tt.addptr %sp, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %pp : tensor<4x!tt.ptr<f32>>
    %s = "tt.reduce"(%v) <{axis = 0 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %c = arith.addf %x, %y : f32
      tt.reduce.return %c : f32
    }) : (tensor<4xf32>) -> f32
    %po = tt.addptr %o, %pid : !tt.ptr<f32>, i32
    tt.store %po, %s : !tt.ptr<f32>
    tt.return
  }""")
_RED_MUL_ADDR = _RED_SHL_ADDR.replace(
    "arith.shli %r, %c1", "arith.muli %r, %c2").replace(
    "%c1 = arith.constant dense<1>", "%c2 = arith.constant dense<2>")


def test_reduce_with_shifted_addressing_equivalent():
    res = tv.check_equivalence(_RED_SHL_ADDR, _RED_MUL_ADDR)
    assert res.equivalent, (res.verdict, res.ub_status,
                            res.equivalence_status)
    assert res.ub_status == "unsat", res.ub_status


# --- Codex review round 1 (phase-2 milestone 2) regression tests. ---

# P1: storing a raw i1 value through a ptr<i8>->ptr<i1> bitcast into a
# byte-observable i8 buffer. Lowerings disagree on the written byte (NVIDIA
# sign-extends i1 to 0xff), so modeling it as 0x01 would prove this kernel
# equivalent to one storing literal 1. Must be rejected.
_STORE_I1_INTO_I8 = _mod("""
  tt.func public @k(%a: !tt.ptr<i32>, %o: !tt.ptr<i8>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i32>, i32
    %x = tt.load %pa, %m, %c0 : !tt.ptr<i32>
    %b = arith.cmpi ne, %x, %c0 : i32
    %ob = tt.bitcast %o : !tt.ptr<i8> -> !tt.ptr<i1>
    %po = tt.addptr %ob, %idx : !tt.ptr<i1>, i32
    tt.store %po, %b, %m : !tt.ptr<i1>
    tt.return
  }""")
_STORE_ONE_INTO_I8 = _mod("""
  tt.func public @k(%a: !tt.ptr<i32>, %o: !tt.ptr<i8>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c0_8 = arith.constant 0 : i8
    %c1_8 = arith.constant 1 : i8
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i32>, i32
    %x = tt.load %pa, %m, %c0 : !tt.ptr<i32>
    %b = arith.cmpi ne, %x, %c0 : i32
    %r = arith.select %b, %c1_8, %c0_8 : i8
    %po = tt.addptr %o, %idx : !tt.ptr<i8>, i32
    tt.store %po, %r, %m : !tt.ptr<i8>
    tt.return
  }""")


def test_reject_i1_store_into_byte_buffer():
    res = tv.check_equivalence(_STORE_I1_INTO_I8, _STORE_ONE_INTO_I8)
    assert res.verdict == "UNSUPPORTED", res.verdict
    # And the trunci-to-i1 variant of the same pattern.
    src = _STORE_I1_INTO_I8.replace("arith.cmpi ne, %x, %c0 : i32",
                                    "arith.trunci %x : i32 to i1")
    res = tv.check_equivalence(src, _STORE_ONE_INTO_I8)
    assert res.verdict == "UNSUPPORTED", res.verdict


# P2: verifier-valid vector<...> types (not ranked tensors) reached
# getIntOrFloatBitWidth() and crashed; they must be rejected gracefully.
_VECTOR_TYPED_CAST = _mod("""
  tt.func public @k(%o: !tt.ptr<i32>, %n: i32) {
    %v = arith.constant dense<7> : vector<4xi8>
    %w = arith.extui %v : vector<4xi8> to vector<4xi32>
    %c0 = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %po = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %po, %c0, %m : !tt.ptr<i32>
    tt.return
  }""")


def test_reject_vector_typed_ops():
    res = tv.check_equivalence(_VECTOR_TYPED_CAST, _VECTOR_TYPED_CAST)
    assert res.verdict == "UNSUPPORTED", res.verdict


# Codex review round 2: verifier-valid `index` constants and zero-width `i0`
# values previously crashed width queries / built invalid !smt.bv<0>.
_INDEX_CONSTANT = _mod("""
  tt.func public @k(%o: !tt.ptr<i32>, %n: i32) {
    %ix = arith.constant 7 : index
    %c0 = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %po = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %po, %c0, %m : !tt.ptr<i32>
    tt.return
  }""")
_I0_VALUE = _mod("""
  tt.func public @k(%o: !tt.ptr<i32>, %n: i32) {
    %z = arith.constant 0 : i0
    %zz = arith.addi %z, %z : i0
    %c0 = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %po = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %po, %c0, %m : !tt.ptr<i32>
    tt.return
  }""")


def test_reject_index_constant():
    res = tv.check_equivalence(_INDEX_CONSTANT, _INDEX_CONSTANT)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_zero_width_integer():
    res = tv.check_equivalence(_I0_VALUE, _I0_VALUE)
    assert res.verdict == "UNSUPPORTED", res.verdict


# Codex review round 3: math.isfinite itself raises on unrepresentable
# timeouts (10**10000) or non-numbers; both must be the documented UNKNOWN.
_TRIVIAL = _mod("""
  tt.func public @k(%o: !tt.ptr<i32>) {
    %z = arith.constant 0 : i32
    %i = tt.get_program_id x : i32
    %p = tt.addptr %o, %i : !tt.ptr<i32>, i32
    tt.store %p, %z : !tt.ptr<i32>
    tt.return
  }""")


@pytest.mark.parametrize("bad", [10**10000, "soon"], ids=["huge-int", "str"])
def test_unrepresentable_timeout_is_unknown(bad):
    res = tv.check_equivalence(_TRIVIAL, _TRIVIAL, timeout=bad)
    assert res.verdict == "UNKNOWN", res.verdict
    assert res.addressing_status == "bad-timeout", res.addressing_status


# --- Phase-2 milestone 4: pass-validation sweep harness smoke tests. ---

def test_pass_validation_smoke():
    from pathlib import Path
    from triton.tools import smt_validate_passes as vp
    corpus = Path(__file__).resolve().parents[4] / "test" / "Conversion"
    text = (corpus / "triton_to_smt.mlir").read_text()
    funcs = vp.split_funcs(text)
    assert len(funcs) == 2, len(funcs)
    verdict, _ = vp.validate_pass(funcs[0], "canonicalize",
                                  tv.default_triton_opt(), timeout=60.0)
    assert verdict == "EQUIVALENT", verdict


def test_pass_validation_rejects_out_of_contract():
    from triton.tools import smt_validate_passes as vp
    # An scf.for kernel is outside the contract; the sweep must count it as
    # UNSUPPORTED rather than skipping or crashing.
    fn = """tt.func public @k(%o: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      scf.yield
    }
    %z = arith.constant 0 : i32
    %pid = tt.get_program_id x : i32
    %p = tt.addptr %o, %pid : !tt.ptr<i32>, i32
    tt.store %p, %z : !tt.ptr<i32>
    tt.return
  }"""
    verdict, _ = vp.validate_pass(fn, "canonicalize", tv.default_triton_opt(),
                                  timeout=60.0)
    assert verdict == "UNSUPPORTED", verdict


# Codex milestone-4 review round 1: split_funcs must parse tt.func headers
# (arg/result/function attribute dicts, external declarations) rather than
# grabbing the first brace, and must account for every tt.func.
def test_split_funcs_header_forms():
    from triton.tools.smt_validate_passes import _split_structured
    ttir = """module {
  tt.func public @first(%p: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    tt.return
  }
  tt.func private @external()
  tt.func public @res() -> (i32 {tt.foo}) attributes {noinline = false} {
    %c = arith.constant 0 : i32
    tt.return %c : i32
  }
  tt.func private @trailing_decl(i32)
}
"""
    r = _split_structured(ttir)
    assert len(r.defs) == 2, r.defs
    assert len(r.decls) == 2, r.decls
    assert not r.errors, r.errors
    assert r.defs[0].startswith("tt.func public @first")
    assert r.defs[0].rstrip().endswith("}") and "tt.return" in r.defs[0]
    assert r.defs[1].startswith("tt.func public @res")
    assert "tt.return %c" in r.defs[1]


def test_split_funcs_reduce_corpus_file():
    from pathlib import Path
    from triton.tools.smt_validate_passes import _split_structured
    corpus = Path(__file__).resolve().parents[4] / "test" / "Conversion"
    r = _split_structured((corpus / "triton_to_smt_reduce.mlir").read_text())
    assert len(r.defs) == 2 and not r.errors, (len(r.defs), r.errors)
    assert all("tt.reduce" in f and "tt.return" in f for f in r.defs)


def test_split_funcs_error_is_recorded_not_dropped():
    from triton.tools.smt_validate_passes import _split_structured
    # Unbalanced body brace: must surface as an error entry, not a silent
    # break that hides subsequent functions.
    r = _split_structured("module {\n  tt.func public @bad() { %c =\n}\n")
    assert r.errors, r


# Codex milestone-4 review round 2: type-level inline attribute dicts in bare
# result types, and zero-work sweeps must not exit green.
def test_split_funcs_inline_type_attribute_result():
    from triton.tools.smt_validate_passes import _split_structured
    ttir = """module {
  tt.func private @f(%a: tensor<4xi32>) -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #b}>> {
    %r = "some.op"(%a) : (tensor<4xi32>) -> tensor<4xi32, #ttg.slice<{dim = 0, parent = #b}>>
    tt.return %r : tensor<4xi32, #ttg.slice<{dim = 0, parent = #b}>>
  }
  tt.func public @g() {
    tt.return
  }
}
"""
    r = _split_structured(ttir)
    assert len(r.defs) == 2 and not r.errors, (len(r.defs), r.errors)
    assert "tt.return %r" in r.defs[0], r.defs[0]
    assert r.defs[1].startswith("tt.func public @g")


def test_split_funcs_corpus_inline_type_attrs():
    from pathlib import Path
    from triton.tools import smt_validate_passes as vp
    root = Path(__file__).resolve().parents[4] / "test" / "Conversion"
    for name in ("reduce_to_llvm.mlir", "cvt_to_llvm.mlir"):
        parsed = vp._parse_file(root / name, tv.default_triton_opt())
        assert parsed is not None, name
        r = vp._split_structured(parsed)
        assert not r.errors, (name, r.errors)
        assert all(d.rstrip().endswith("}") and "tt.return" in d
                   for d in r.defs), name


def test_sweep_zero_work_fails():
    from triton.tools import smt_validate_passes as vp
    assert vp._main(["--corpus", "/nonexistent/smt-corpus",
                     "--passes", "canonicalize"]) != 0
    assert vp._main(["--corpus", "test", "--passes", ""]) != 0


def test_sweep_unknown_pass_is_fatal():
    from triton.tools import smt_validate_passes as vp
    with pytest.raises(RuntimeError, match="not a registered pass"):
        vp.validate_pass("tt.func public @k() { tt.return }",
                         "no-such-pass-xyz", tv.default_triton_opt(), 30.0)


# Codex milestone-4 review round 3: discovery must not be fooled by tt.func
# as an attribute name or inside strings, and driver options must not
# masquerade as passes.
def test_split_funcs_ttfunc_as_attribute_name():
    from triton.tools.smt_validate_passes import _split_structured
    ttir = """module {
  tt.func public @victim(%o: !tt.ptr<i32>) attributes {tt.func = false} {
    %z = arith.constant 0 : i32
    %pid = tt.get_program_id x : i32
    %p = tt.addptr %o, %pid : !tt.ptr<i32>, i32
    tt.store %p, %z : !tt.ptr<i32>
    tt.return
  }
  tt.func public @after() {
    tt.return
  }
}
"""
    r = _split_structured(ttir)
    assert len(r.defs) == 2 and not r.errors and not r.decls, \
        (len(r.defs), len(r.decls), r.errors)
    assert r.defs[0].startswith("tt.func public @victim")
    assert "tt.store" in r.defs[0] and r.defs[0].rstrip().endswith("}")


def test_split_funcs_ttfunc_in_string_attr():
    from triton.tools.smt_validate_passes import _split_structured
    ttir = """module {
  tt.func public @a() attributes {note = "tt.func @fake() {"} {
    tt.return
  }
  tt.func public @b() {
    tt.return
  }
}
"""
    r = _split_structured(ttir)
    assert len(r.defs) == 2 and not r.errors, (len(r.defs), r.errors)


def test_driver_option_is_not_a_pass():
    from triton.tools import smt_validate_passes as vp
    for bogus in ("allow-unregistered-dialect", "mlir-print-ir-after-all"):
        with pytest.raises(RuntimeError, match="not a registered pass"):
            vp.validate_pass("tt.func public @k() { tt.return }", bogus,
                             tv.default_triton_opt(), 30.0)


def test_split_funcs_nested_modules():
    from triton.tools.smt_validate_passes import _split_structured
    # Multi-module lit files parse-normalize into nested modules; discovery
    # must descend into them rather than skipping them as opaque braces.
    ttir = """module {
  module {
    tt.func public @a() { tt.return }
  }
  module attributes {ttg.target = "cuda:80"} {
    tt.func public @b() { tt.return }
  }
}
"""
    r = _split_structured(ttir)
    assert len(r.defs) == 2 and not r.errors, (len(r.defs), r.errors)


# Codex milestone-4 review round 4: module symbols colliding with the
# `attributes` keyword, and pipeline grammar in pass entries.
def test_split_funcs_module_named_attributes():
    from triton.tools.smt_validate_passes import _split_structured
    base = """module {{
  module {header} {{
    tt.func public @victim(%o: !tt.ptr<i32>) {{
      %z = arith.constant 0 : i32
      tt.store %o, %z : !tt.ptr<i32>
      tt.return
    }}
  }}
  tt.func public @anchor() {{
    tt.return
  }}
}}
"""
    for header in ('@attributes', '@"attributes"',
                   '@attributes attributes {ttg.target = "cuda:80"}'):
        r = _split_structured(base.format(header=header))
        assert len(r.defs) == 2 and not r.errors, (header, len(r.defs),
                                                   r.errors)
        assert any("@victim" in d.split("{", 1)[0] for d in r.defs), header


def test_pipeline_grammar_is_not_a_pass():
    from triton.tools import smt_validate_passes as vp
    fn = "tt.func public @k() { tt.return }"
    for bad in ("builtin.module(canonicalize)", "func.func(canonicalize)",
                "canonicalize{max-iterations=1}", "canonicalize,cse",
                "builtin.module()"):
        with pytest.raises(RuntimeError, match="atomic pass name"):
            vp.validate_pass(fn, bad, tv.default_triton_opt(), 30.0)


# =============================================================================
# Phase 3: block memory model (strided / multi-dimensional / masked addressing)
# =============================================================================
# These kernels are outside the identity-addressing contract, so every check
# below exercises the memory-model retry path (asserted via res.memory_model).
# See docs/smt-tv/phase-3.md.


@triton.jit
def tile2d_mul2(x_ptr, o_ptr, M, N, sm, sn, BM: tl.constexpr,
                BN: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    offs = om[:, None] * sm + on[None, :] * sn
    mask = (om[:, None] < M) & (on[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x * 2.0, mask=mask)


@triton.jit
def tile2d_add2_commuted(x_ptr, o_ptr, M, N, sm, sn, BM: tl.constexpr,
                         BN: tl.constexpr):
    # Same addressing with the offset addition commuted, and x+x for x*2.
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    offs = on[None, :] * sn + om[:, None] * sm
    mask = (om[:, None] < M) & (on[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x + x, mask=mask)


@triton.jit
def tile2d_swapped_strides(x_ptr, o_ptr, M, N, sm, sn, BM: tl.constexpr,
                           BN: tl.constexpr):
    # Transposed addressing: sm and sn swapped.
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    offs = om[:, None] * sn + on[None, :] * sm
    mask = (om[:, None] < M) & (on[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x * 2.0, mask=mask)


@triton.jit
def tile2d_off_by_one(x_ptr, o_ptr, M, N, sm, sn, BM: tl.constexpr,
                      BN: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    offs = om[:, None] * sm + on[None, :] * sn
    mask = (om[:, None] < M) & (on[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs + 1, x * 2.0, mask=mask)


@triton.jit
def tile2d_mask_le(x_ptr, o_ptr, M, N, sm, sn, BM: tl.constexpr,
                   BN: tl.constexpr):
    # Wrong bound: <= M instead of < M changes the accessed set.
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    offs = om[:, None] * sm + on[None, :] * sn
    mask = (om[:, None] <= M) & (on[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x * 2.0, mask=mask)


@triton.jit
def tile2d_unmasked(x_ptr, o_ptr, M, N, sm, sn, BM: tl.constexpr,
                    BN: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    offs = om[:, None] * sm + on[None, :] * sn
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x * 2.0)


_S2D = {"x_ptr": PTR, "o_ptr": PTR, "M": "i32", "N": "i32", "sm": "i32",
        "sn": "i32", "BM": "constexpr", "BN": "constexpr"}
_CX2D = {"BM": 4, "BN": 4}


def _check2d(src, tgt, **kw):
    return tv.check_kernels(src, tgt, _S2D, src_constexprs=_CX2D,
                            tgt_constexprs=_CX2D, **kw)


def test_2d_strided_masked_tile_equivalent():
    # Commuted affine offsets + x*2 vs x+x: same offsets, same values.
    res = _check2d(tile2d_mul2, tile2d_add2_commuted)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model, "expected the block-memory-model retry path"
    # Both kernels would be OOB unmasked, but their UB domains coincide.
    assert res.ub_status == "unsat", res.ub_status


def test_2d_swapped_strides_not_equivalent():
    # A transpose is a different memory function; the access sets differ, so
    # the UB (OOB) domains already differ.
    res = _check2d(tile2d_mul2, tile2d_swapped_strides)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


def test_2d_off_by_one_base_not_equivalent():
    res = _check2d(tile2d_mul2, tile2d_off_by_one)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


def test_2d_wrong_mask_bound_not_equivalent():
    res = _check2d(tile2d_mul2, tile2d_mask_le)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


def test_2d_unmasked_vs_masked_not_equivalent():
    # The unmasked tile can access out of bounds where the masked one is
    # defined: distinct UB domains, hence not mutually refining.
    res = _check2d(tile2d_unmasked, tile2d_mul2)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.z3_output)
    assert res.ub_status == "sat", (res.ub_status, res.z3_output)
    assert res.memory_model


def test_2d_masked_tile_self_equivalent():
    res = _check2d(tile2d_mul2, tile2d_mul2)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


@triton.jit
def bcast_add_ab(a_ptr, b_ptr, o_ptr, ldc, BM: tl.constexpr,
                 BN: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    a = tl.load(a_ptr + om)
    b = tl.load(b_ptr + on)
    c = a[:, None] + b[None, :]
    tl.store(o_ptr + om[:, None] * ldc + on[None, :], c)


@triton.jit
def bcast_add_ba(a_ptr, b_ptr, o_ptr, ldc, BM: tl.constexpr,
                 BN: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    a = tl.load(a_ptr + om)
    b = tl.load(b_ptr + on)
    c = b[None, :] + a[:, None]
    tl.store(o_ptr + om[:, None] * ldc + on[None, :], c)


def test_broadcast_add_order_invariance():
    sig = {"a_ptr": PTR, "b_ptr": PTR, "o_ptr": PTR, "ldc": "i32",
           "BM": "constexpr", "BN": "constexpr"}
    res = tv.check_kernels(bcast_add_ab, bcast_add_ba, sig,
                           src_constexprs=_CX2D, tgt_constexprs=_CX2D)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


@triton.jit
def rowsum2d_axis1(x_ptr, o_ptr, s, BM: tl.constexpr, BN: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.arange(0, BN)
    x = tl.load(x_ptr + om[:, None] * s + on[None, :])
    tl.store(o_ptr + om, tl.sum(x, axis=1))


@triton.jit
def rowsum2d_axis1_commuted(x_ptr, o_ptr, s, BM: tl.constexpr,
                            BN: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.arange(0, BN)
    x = tl.load(x_ptr + on[None, :] + om[:, None] * s)
    tl.store(o_ptr + om, tl.sum(x, axis=1))


@triton.jit
def rowsum2d_axis0(x_ptr, o_ptr, s, BM: tl.constexpr, BN: tl.constexpr):
    # Sums the wrong axis; with BM == BN the store set is identical, so only
    # the stored VALUES differ.
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.arange(0, BN)
    x = tl.load(x_ptr + om[:, None] * s + on[None, :])
    tl.store(o_ptr + om, tl.sum(x, axis=0))


_SROW = {"x_ptr": PTR, "o_ptr": PTR, "s": "i32", "BM": "constexpr",
         "BN": "constexpr"}


def test_rowwise_reduction_over_2d_load_equivalent():
    res = tv.check_kernels(rowsum2d_axis1, rowsum2d_axis1_commuted, _SROW,
                           src_constexprs=_CX2D, tgt_constexprs=_CX2D)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


def test_rowsum_wrong_axis_not_equivalent():
    res = tv.check_kernels(rowsum2d_axis1, rowsum2d_axis0, _SROW,
                           src_constexprs=_CX2D, tgt_constexprs=_CX2D)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


# --- Intra-store write-write races are UB (scope 1), pinned with inline TTIR
#     so the exact overlapping-store pattern is exercised. ---

# All four lanes store the (per-lane distinct) loaded values to offset 0: a
# race whenever two loaded values differ.
_RACE_STORE = _mod("""
  tt.func public @k(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %z = arith.constant dense<0> : tensor<4xi32>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %sx = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %px = tt.addptr %sx, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %px : tensor<4x!tt.ptr<f32>>
    %so = tt.splat %o : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %po = tt.addptr %so, %z : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %po, %v : tensor<4x!tt.ptr<f32>>
    tt.return
  }""")
# Same loads and store set, but every lane stores x-x (all lanes provably
# equal over ideal reals): race-free.
_RACE_FREE = _mod("""
  tt.func public @k(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %z = arith.constant dense<0> : tensor<4xi32>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %sx = tt.splat %x : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %px = tt.addptr %sx, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %px : tensor<4x!tt.ptr<f32>>
    %d = arith.subf %v, %v : tensor<4xf32>
    %so = tt.splat %o : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %po = tt.addptr %so, %z : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %po, %d : tensor<4x!tt.ptr<f32>>
    tt.return
  }""")


def test_racy_store_vs_race_free_not_equivalent():
    # src races exactly when two loaded lanes differ; tgt never races. The UB
    # domains differ, so the kernels do not mutually refine.
    res = tv.check_equivalence(_RACE_STORE, _RACE_FREE)
    assert res.verdict == "NOT_EQUIVALENT", (res.verdict, res.z3_output)
    assert res.ub_status == "sat", (res.ub_status, res.z3_output)
    assert res.memory_model


def test_same_race_both_sides_equivalent():
    # Two kernels that race on exactly the same inputs are EQUIVALENT by the
    # bidirectional-refinement design (as with identical unbounded shifts).
    res = tv.check_equivalence(_RACE_STORE, _RACE_STORE)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


# --- Sequential per-program semantics: a load after a store to the same block
#     sees the stored value. ---

_SEQ_RMW = _mod("""
  tt.func public @k(%o: !tt.ptr<f32>, %x: f32) {
    %pid = tt.get_program_id x : i32
    %po = tt.addptr %o, %pid : !tt.ptr<f32>, i32
    tt.store %po, %x : !tt.ptr<f32>
    %y = tt.load %po : !tt.ptr<f32>
    tt.store %po, %y : !tt.ptr<f32>
    tt.return
  }""")
_SEQ_DIRECT = _mod("""
  tt.func public @k(%o: !tt.ptr<f32>, %x: f32) {
    %pid = tt.get_program_id x : i32
    %po = tt.addptr %o, %pid : !tt.ptr<f32>, i32
    tt.store %po, %x : !tt.ptr<f32>
    tt.store %po, %x : !tt.ptr<f32>
    tt.return
  }""")


def test_load_after_store_sees_stored_value():
    # Multiple stores to the SAME block are legal under the block model
    # (folded in program order), and the reload reads the just-stored value,
    # not the initial memory. The legacy path rejects this (two stores), so it
    # routes to the memory model. (Both memory ops stay on block %o, so the
    # restrict load-bearing gate does not fire.)
    res = tv.check_equivalence(_SEQ_RMW, _SEQ_DIRECT)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model


# --- Sound-by-rejection: everything outside the phase-3 contract stays
#     UNSUPPORTED (never a silent EQUIVALENT). ---


@triton.jit
def gather2d_kernel(x_ptr, i_ptr, o_ptr, B: tl.constexpr):
    # A 1-D identity-addressed gather is exactly encodable on the legacy
    # symbolic-lane path (a nested select with no OOB semantics), so to pin
    # the MEMORY-MODEL rejection the gather must be 2-D.
    om = tl.program_id(0) * B + tl.arange(0, B)
    on = tl.arange(0, B)
    offs = om[:, None] * B + on[None, :]
    idx = tl.load(i_ptr + offs)
    v = tl.load(x_ptr + idx)
    tl.store(o_ptr + offs, v)


def test_reject_gather():
    sig = {"x_ptr": PTR, "i_ptr": "*i32", "o_ptr": PTR, "B": "constexpr"}
    cx = {"B": 4}
    res = tv.check_kernels(gather2d_kernel, gather2d_kernel, sig,
                           src_constexprs=cx, tgt_constexprs=cx)
    assert res.verdict == "UNSUPPORTED", res.verdict
    assert "data-dependent addressing" in res.z3_output, res.z3_output


@triton.jit
def dyn_stride_kernel(x_ptr, s_ptr, o_ptr, BLOCK: tl.constexpr):
    s = tl.load(s_ptr)
    offs = tl.arange(0, BLOCK) * s
    v = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, v)


def test_reject_data_dependent_stride():
    sig = {"x_ptr": PTR, "s_ptr": "*i32", "o_ptr": PTR, "BLOCK": "constexpr"}
    cx = {"BLOCK": 4}
    res = tv.check_kernels(dyn_stride_kernel, dyn_stride_kernel, sig,
                           src_constexprs=cx, tgt_constexprs=cx)
    assert res.verdict == "UNSUPPORTED", res.verdict


@triton.jit
def rank3_kernel(x_ptr, o_ptr, B: tl.constexpr):
    a = tl.arange(0, B)
    t = a[:, None, None] * B * B + a[None, :, None] * B + a[None, None, :]
    v = tl.load(x_ptr + t)
    tl.store(o_ptr + t, v)


def test_reject_rank3_block():
    sig = {"x_ptr": PTR, "o_ptr": PTR, "B": "constexpr"}
    cx = {"B": 2}
    res = tv.check_kernels(rank3_kernel, rank3_kernel, sig,
                           src_constexprs=cx, tgt_constexprs=cx)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_reject_oversized_extent_product():
    # 64x64 = 4096 lanes exceeds the max-lanes tractability threshold; the
    # block must be rejected loudly regardless of its (supported) rank.
    res = tv.check_kernels(tile2d_mul2, tile2d_mul2, _S2D,
                           src_constexprs={"BM": 64, "BN": 64},
                           tgt_constexprs={"BM": 64, "BN": 64})
    assert res.verdict == "UNSUPPORTED", res.verdict


@triton.jit
def atomic_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.atomic_add(o_ptr + offs, v)


def test_reject_atomic():
    sig = {"x_ptr": PTR, "o_ptr": PTR, "BLOCK": "constexpr"}
    cx = {"BLOCK": 4}
    res = tv.check_kernels(atomic_kernel, atomic_kernel, sig,
                           src_constexprs=cx, tgt_constexprs=cx)
    assert res.verdict == "UNSUPPORTED", res.verdict


@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, B: tl.constexpr):
    r = tl.arange(0, B)
    offs = r[:, None] * B + r[None, :]
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    c = tl.dot(a, b)
    tl.store(c_ptr + offs, c)


def test_reject_dot():
    sig = {"a_ptr": PTR, "b_ptr": PTR, "c_ptr": PTR, "B": "constexpr"}
    cx = {"B": 16}
    res = tv.check_kernels(dot_kernel, dot_kernel, sig, src_constexprs=cx,
                           tgt_constexprs=cx)
    assert res.verdict == "UNSUPPORTED", res.verdict


def test_block_size_pins_legacy_contract():
    # An explicit block_size opts out of the memory-model retry: non-identity
    # addressing stays UNSUPPORTED rather than silently switching contracts.
    res = tv.check_equivalence(_SHIFTED_A, _SHIFTED_2A, block_size=1)
    assert res.verdict == "UNSUPPORTED", res.verdict
    assert not res.memory_model


def test_reject_nonpositive_max_lanes():
    # Codex phase-3 review round 1: max-lanes <= 0 must reject rather than
    # silently disable the extent-product tractability guard (the intra-store
    # race construction is quadratic in the lane count and the encoder runs
    # without a subprocess timeout).
    import subprocess
    for cap in ("0", "-4"):
        r = subprocess.run(
            [tv.default_triton_opt(),
             f"--convert-triton-to-smt=memory-model=true max-lanes={cap}"],
            input=tv.build_query_module(_RACE_STORE, _RACE_STORE,
                                        tv.default_triton_opt()),
            capture_output=True, text=True, timeout=60)
        assert r.returncode != 0, (cap, r.stderr)
        assert "max-lanes must be positive" in r.stderr, (cap, r.stderr)


# --- Aliasing / restrict soundness gate (block memory model). Distinct
#     pointer-arg blocks are DISTINCT SMT arrays (disjoint by construction), but
#     TTIR carries no restrict guarantee. A and B are EQUIVALENT iff the three
#     pointer args are mutually disjoint: A re-loads `in` AFTER storing `out`
#     (2*t), while B reuses the first load, so an in-place launch (out == in)
#     makes them differ. The block model must NOT silently prove them
#     EQUIVALENT -- that would be a false verdict for the most common aliasing
#     calling convention. See docs/smt-tv/phase-3.md. ---

_ALIAS_A = _mod("""
  tt.func public @k(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %out2: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %two = arith.constant 2.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pin = tt.addptr %in, %idx : !tt.ptr<f32>, i32
    %t = tt.load %pin, %m, %cst : !tt.ptr<f32>
    %d = arith.mulf %t, %two : f32
    %pout = tt.addptr %out, %idx : !tt.ptr<f32>, i32
    tt.store %pout, %d, %m : !tt.ptr<f32>
    %pin2 = tt.addptr %in, %idx : !tt.ptr<f32>, i32
    %u = tt.load %pin2, %m, %cst : !tt.ptr<f32>
    %pout2 = tt.addptr %out2, %idx : !tt.ptr<f32>, i32
    tt.store %pout2, %u, %m : !tt.ptr<f32>
    tt.return
  }""")
_ALIAS_B = _mod("""
  tt.func public @k(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %out2: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %two = arith.constant 2.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pin = tt.addptr %in, %idx : !tt.ptr<f32>, i32
    %t = tt.load %pin, %m, %cst : !tt.ptr<f32>
    %d = arith.mulf %t, %two : f32
    %pout = tt.addptr %out, %idx : !tt.ptr<f32>, i32
    tt.store %pout, %d, %m : !tt.ptr<f32>
    %pout2 = tt.addptr %out2, %idx : !tt.ptr<f32>, i32
    tt.store %pout2, %t, %m : !tt.ptr<f32>
    tt.return
  }""")


def test_aliasing_loadbearing_rejected_by_default():
    # The disjoint-by-construction block arrays would prove a FALSE EQUIVALENT
    # under aliasing, so the memory model must REJECT this load-bearing pair
    # (a memory op addresses one pointer-arg block after a store to a different
    # one) rather than emit a verdict. The legacy path already rejects the two
    # stores, so the memory-model retry is what fires the gate.
    res = tv.check_equivalence(_ALIAS_A, _ALIAS_B)
    assert res.verdict == "UNSUPPORTED", (res.verdict, res.z3_output)
    assert res.verdict != "EQUIVALENT"
    assert res.memory_model, "expected the block-memory-model retry to be attempted"


def test_aliasing_loadbearing_equivalent_under_restrict():
    # assume_restrict encodes under the disjointness (restrict) assumption
    # instead of rejecting; A == B provided the pointer args do not alias, so
    # the honest verdict is EQUIVALENT_UNDER_RESTRICT (true for A/B under
    # disjointness). The over-disclosed caveat never lies.
    res = tv.check_equivalence(_ALIAS_A, _ALIAS_B, assume_restrict=True)
    assert res.verdict == "EQUIVALENT_UNDER_RESTRICT", (res.verdict, res.z3_output)
    assert res.memory_model


def test_load_compute_store_still_equivalent():
    # An ordinary load -> compute -> store kernel is NOT load-bearing (no memory
    # op touches a block after a store to a different block), so the block model
    # still returns a plain EQUIVALENT under default settings -- no spurious
    # rejection and no caveat. Uses the shifted-store self-pair, which routes to
    # the memory model (non-identity addressing) yet is not load-bearing.
    res = tv.check_equivalence(_SHIFTED_A, _SHIFTED_A)
    assert res.verdict == "EQUIVALENT", (res.verdict, res.z3_output)
    assert res.memory_model, "expected the block-memory-model retry path"
