// RUN: triton-opt %s --convert-triton-to-smt | FileCheck %s
// RUN: triton-opt %s --convert-triton-to-smt | mlir-translate --export-smtlib | FileCheck %s --check-prefix=SMTLIB

// Two affine/FMA kernels: @src is scalar `a*b+c`, @tgt is a BLOCK=128 `math.fma`.
// The pass emits two solver scopes: scope 0 verifies identity addressing
// (storeOffset == i), scope 1 asserts the output arrays can differ. Integers
// are bit-vectors; floats are ideal reals.

// Scope 0 (addressing identity):
// CHECK: smt.solver
// CHECK: smt.declare_fun "i" : !smt.bv<32>
// CHECK: smt.declare_fun "arg0" : !smt.array<[!smt.bv<32> -> !smt.real]>
// CHECK: smt.bv.udiv
// CHECK: smt.distinct %{{.*}}, %i : !smt.bv<32>
// CHECK: smt.check
// Scope 1 (equivalence):
// CHECK: smt.solver
// CHECK-DAG: smt.real.mul
// CHECK-DAG: smt.real.add
// CHECK: smt.distinct %{{.*}}, %{{.*}} : !smt.real
// CHECK: smt.check

// SMTLIB: (declare-const i (_ BitVec 32))
// SMTLIB: (declare-const arg0 (Array (_ BitVec 32) Real))
// SMTLIB: (check-sat)

module {
  tt.func public @src(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %a = tt.addptr %a_ptr, %idx : !tt.ptr<f32>, i32
    %a0 = tt.load %a, %m, %cst : !tt.ptr<f32>
    %b = tt.addptr %b_ptr, %idx : !tt.ptr<f32>, i32
    %b0 = tt.load %b, %m, %cst : !tt.ptr<f32>
    %c = tt.addptr %c_ptr, %idx : !tt.ptr<f32>, i32
    %c0 = tt.load %c, %m, %cst : !tt.ptr<f32>
    %p = arith.mulf %a0, %b0 : f32
    %r = arith.addf %p, %c0 : f32
    %o = tt.addptr %out_ptr, %idx : !tt.ptr<f32>, i32
    tt.store %o, %r, %m : !tt.ptr<f32>
    tt.return
  }
  tt.func public @tgt(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %c128 = arith.constant 128 : i32
    %pid = tt.get_program_id x : i32
    %off = arith.muli %pid, %c128 : i32
    %r0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %s0 = tt.splat %off : i32 -> tensor<128xi32>
    %o2 = arith.addi %s0, %r0 : tensor<128xi32>
    %ms = tt.splat %n : i32 -> tensor<128xi32>
    %m3 = arith.cmpi slt, %o2, %ms : tensor<128xi32>
    %sa = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %a4 = tt.addptr %sa, %o2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %a5 = tt.load %a4, %m3, %cst : tensor<128x!tt.ptr<f32>>
    %sb = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %b6 = tt.addptr %sb, %o2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %b7 = tt.load %b6, %m3, %cst : tensor<128x!tt.ptr<f32>>
    %sc = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %c8 = tt.addptr %sc, %o2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %c9 = tt.load %c8, %m3, %cst : tensor<128x!tt.ptr<f32>>
    %res = math.fma %a5, %b7, %c9 : tensor<128xf32>
    %so = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %o1 = tt.addptr %so, %o2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %o1, %res, %m3 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
