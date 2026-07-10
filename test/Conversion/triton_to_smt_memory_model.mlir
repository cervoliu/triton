// RUN: triton-opt %s --convert-triton-to-smt=memory-model=true | FileCheck %s
// RUN: triton-opt %s --convert-triton-to-smt=memory-model=true | mlir-translate --export-smtlib | FileCheck %s --check-prefix=SMTLIB

// Phase-3 block memory model: a 2x2 strided, masked tile store. Each pointer
// argument becomes a block (bv64-indexed array + non-negative symbolic size);
// tensors are materialized as concrete lanes; loads/stores carry per-lane
// 64-bit offsets (each i32 offset is sign-extended, so chained tt.addptr is
// exact). Scope 0 is a documented always-unsat placeholder; scope 1 compares
// UB domains (OOB + races + poison); scope 2 compares per-block final arrays.

// Scope 0 (vestigial placeholder, trivially unsat):
// CHECK: smt.solver
// CHECK-DAG: smt.declare_fun "pid0" : !smt.bv<32>
// CHECK-DAG: smt.declare_fun "pid1" : !smt.bv<32>
// CHECK-DAG: smt.declare_fun "arg0" : !smt.array<[!smt.bv<64> -> !smt.real]>
// CHECK-DAG: smt.declare_fun "arg0_size" : !smt.bv<64>
// CHECK-DAG: smt.declare_fun "arg1" : !smt.array<[!smt.bv<64> -> !smt.real]>
// CHECK-DAG: smt.declare_fun "arg1_size" : !smt.bv<64>
// The block sizes are constrained non-negative so the signed OOB predicate is
// exactly the out-of-allocation condition.
// CHECK: smt.bv.cmp sge
// CHECK: smt.assert
// The final constant-false placeholder assertion:
// CHECK: %[[FALSE:.+]] = smt.constant false
// CHECK: smt.assert %[[FALSE]]
// CHECK: smt.check

// Scope 1 (UB-domain equality): the OOB predicates use signed 64-bit
// comparisons against the block size, and the store lanes fold via
// smt.array.store guarded by the mask.
// CHECK: smt.solver
// CHECK-DAG: smt.bv.cmp slt
// CHECK-DAG: smt.bv.cmp sge
// CHECK-DAG: smt.array.store
// CHECK: smt.distinct
// CHECK: smt.check

// Scope 2 (equivalence): final arrays of the written block are compared.
// CHECK: smt.solver
// CHECK: smt.array.store
// CHECK: smt.distinct
// CHECK: smt.check

// SMTLIB: (set-logic ALL)
// SMTLIB: (declare-const arg0 (Array (_ BitVec 64) Real))
// SMTLIB: (declare-const arg0_size (_ BitVec 64))
// SMTLIB: (check-sat)
// SMTLIB: (reset)
// SMTLIB: (check-sat)
// SMTLIB: (reset)
// SMTLIB: (check-sat)

module {
  tt.func public @src(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %M: i32, %sm: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
    %c2 = arith.constant 2 : i32
    %pm = tt.get_program_id x : i32
    %pn = tt.get_program_id y : i32
    %bm = arith.muli %pm, %c2 : i32
    %rm = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %sbm = tt.splat %bm : i32 -> tensor<2xi32>
    %om = arith.addi %sbm, %rm : tensor<2xi32>
    %om2 = tt.expand_dims %om {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %on2 = tt.expand_dims %rm {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %ssm = tt.splat %sm : i32 -> tensor<2x1xi32>
    %omm = arith.muli %om2, %ssm : tensor<2x1xi32>
    %bom = tt.broadcast %omm : tensor<2x1xi32> -> tensor<2x2xi32>
    %bon = tt.broadcast %on2 : tensor<1x2xi32> -> tensor<2x2xi32>
    %offs = arith.addi %bom, %bon : tensor<2x2xi32>
    %sM = tt.splat %M : i32 -> tensor<2x1xi32>
    %mm = arith.cmpi slt, %om2, %sM : tensor<2x1xi32>
    %mask = tt.broadcast %mm : tensor<2x1xi1> -> tensor<2x2xi1>
    %sx = tt.splat %x : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %px = tt.addptr %sx, %offs : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %v = tt.load %px, %mask, %cst : tensor<2x2x!tt.ptr<f32>>
    %two = arith.constant dense<2.000000e+00> : tensor<2x2xf32>
    %d = arith.mulf %v, %two : tensor<2x2xf32>
    %so = tt.splat %o : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %po = tt.addptr %so, %offs : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    tt.store %po, %d, %mask : tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
  tt.func public @tgt(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %M: i32, %sm: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
    %c2 = arith.constant 2 : i32
    %pm = tt.get_program_id x : i32
    %pn = tt.get_program_id y : i32
    %bm = arith.muli %pm, %c2 : i32
    %rm = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %sbm = tt.splat %bm : i32 -> tensor<2xi32>
    %om = arith.addi %sbm, %rm : tensor<2xi32>
    %om2 = tt.expand_dims %om {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %on2 = tt.expand_dims %rm {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %ssm = tt.splat %sm : i32 -> tensor<2x1xi32>
    %omm = arith.muli %om2, %ssm : tensor<2x1xi32>
    %bom = tt.broadcast %omm : tensor<2x1xi32> -> tensor<2x2xi32>
    %bon = tt.broadcast %on2 : tensor<1x2xi32> -> tensor<2x2xi32>
    %offs = arith.addi %bon, %bom : tensor<2x2xi32>
    %sM = tt.splat %M : i32 -> tensor<2x1xi32>
    %mm = arith.cmpi slt, %om2, %sM : tensor<2x1xi32>
    %mask = tt.broadcast %mm : tensor<2x1xi1> -> tensor<2x2xi1>
    %sx = tt.splat %x : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %px = tt.addptr %sx, %offs : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %v = tt.load %px, %mask, %cst : tensor<2x2x!tt.ptr<f32>>
    %d = arith.addf %v, %v : tensor<2x2xf32>
    %so = tt.splat %o : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %po = tt.addptr %so, %offs : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    tt.store %po, %d, %mask : tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}
