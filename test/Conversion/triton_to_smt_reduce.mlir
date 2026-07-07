// RUN: triton-opt %s --convert-triton-to-smt | FileCheck %s
// RUN: triton-opt %s --convert-triton-to-smt | mlir-translate --export-smtlib | FileCheck %s --check-prefix=SMTLIB

// Phase-2 milestone 1: 1-D tt.reduce (add combiner).
// @src computes sum(a + b); @tgt computes sum(a) + sum(b). Because floats are
// ideal reals (associative + commutative addition), the two are EQUIVALENT.
// The reduce input is materialized as N=4 per-lane array selects; the reduce
// is a fold of smt.real.add over those lanes (no permutation/multiset machinery
// -- see docs/smt-tv/phase-2.md).

// Two solver scopes are emitted (addressing identity, then equivalence).
// CHECK: smt.solver
// CHECK: smt.declare_fun "i" : !smt.bv<32>
// The store is scalar-per-program (block == 1): the writer decomposition is the
// identity, so no bv.udiv is needed and the addressing scope checks storeOffset.
// CHECK: smt.distinct %{{.*}}, %i : !smt.bv<32>
// CHECK: smt.check
// CHECK: smt.solver
// Each lane is an array select; the fold is a chain of real.add.
// CHECK-DAG: smt.array.select
// CHECK-DAG: smt.real.add
// CHECK: smt.distinct %{{.*}}, %{{.*}} : !smt.real
// CHECK: smt.check

// SMTLIB: (declare-const i (_ BitVec 32))
// SMTLIB: (declare-const arg0 (Array (_ BitVec 32) Real))
// SMTLIB: (check-sat)

module {
  tt.func public @src(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f32>>
    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<f32>>
    %11 = tt.addptr %arg2, %0 : !tt.ptr<f32>, i32
    %12 = arith.addf %7, %10 : tensor<4xf32>
    %13 = tt.reshape %12 allow_reorder : tensor<4xf32> -> tensor<4xf32>
    %14 = "tt.reduce"(%13) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %15 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %15 : f32
    }) : (tensor<4xf32>) -> f32
    tt.store %11, %14 : !tt.ptr<f32>
    tt.return
  }
  tt.func public @tgt(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f32>>
    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<f32>>
    %11 = tt.addptr %arg2, %0 : !tt.ptr<f32>, i32
    %12 = tt.reshape %7 allow_reorder : tensor<4xf32> -> tensor<4xf32>
    %13 = "tt.reduce"(%12) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %r0 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %r0 : f32
    }) : (tensor<4xf32>) -> f32
    %14 = tt.reshape %10 allow_reorder : tensor<4xf32> -> tensor<4xf32>
    %15 = "tt.reduce"(%14) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %r1 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %r1 : f32
    }) : (tensor<4xf32>) -> f32
    %16 = arith.addf %13, %15 : f32
    tt.store %11, %16 : !tt.ptr<f32>
    tt.return
  }
}
