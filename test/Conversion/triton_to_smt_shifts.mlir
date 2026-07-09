// RUN: triton-opt %s --convert-triton-to-smt | FileCheck %s
// RUN: triton-opt %s --convert-triton-to-smt | mlir-translate --export-smtlib | FileCheck %s --check-prefix=SMTLIB

// Shifts encode as total SMT bv ops; each shift contributes an exact poison
// condition (amt >=u width) that becomes UB when it reaches a memory
// operation. Three scopes: 0 = addressing, 1 = UB-domain equality
// (assert UB_src != UB_tgt), 2 = equivalence gated on neither side being UB.

// Scope 0 (addressing):
// CHECK: smt.solver
// CHECK: smt.distinct %{{.*}}, %i : !smt.bv<32>
// Scope 1 (UB-domain equality): poison conditions amt >= 32, UB predicates
// compared for disequality.
// CHECK: smt.solver
// CHECK: smt.bv.cmp uge %{{.*}}, %{{.*}} : !smt.bv<32>
// CHECK: smt.distinct %{{.*}}, %{{.*}} : !smt.bool
// CHECK: smt.assert
// Scope 2 (equivalence):
// CHECK: smt.solver
// CHECK-DAG: smt.bv.shl
// CHECK-DAG: smt.bv.lshr
// CHECK: smt.distinct %{{.*}}, %{{.*}} : !smt.bv<32>
// CHECK: smt.not
// CHECK: smt.assert

// SMTLIB: ; solver scope 0
// SMTLIB: ; solver scope 1
// SMTLIB: bvuge
// SMTLIB: ; solver scope 2
// SMTLIB-DAG: bvshl
// SMTLIB-DAG: bvlshr
// SMTLIB: (check-sat)

module {
  tt.func public @src(%a: !tt.ptr<i32>, %o: !tt.ptr<i32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c31 = arith.constant 31 : i32
    %c1 = arith.constant 1 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i32>, i32
    %x = tt.load %pa, %m, %c0 : !tt.ptr<i32>
    %amt = arith.andi %x, %c31 : i32
    %sh = arith.shli %x, %amt : i32
    %r = arith.shrui %sh, %c1 : i32
    %po = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %po, %r, %m : !tt.ptr<i32>
    tt.return
  }
  tt.func public @tgt(%a: !tt.ptr<i32>, %o: !tt.ptr<i32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c31 = arith.constant 31 : i32
    %c1 = arith.constant 1 : i32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<i32>, i32
    %x = tt.load %pa, %m, %c0 : !tt.ptr<i32>
    %amt = arith.andi %x, %c31 : i32
    %sh = arith.shli %x, %amt : i32
    %r = arith.shrui %sh, %c1 : i32
    %po = tt.addptr %o, %idx : !tt.ptr<i32>, i32
    tt.store %po, %r, %m : !tt.ptr<i32>
    tt.return
  }
}
