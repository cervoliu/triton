// RUN: triton-opt %s --split-input-file --convert-triton-to-smt | FileCheck %s
// RUN: triton-opt %s --split-input-file --convert-triton-to-smt | mlir-translate --split-input-file --export-smtlib | FileCheck %s --check-prefix=SMTLIB

// Real division, math.absf, and boolean (`i1`) mask conjunction. @src guards
// with `(pid < n) andi (pid >= 0)` and computes |a/b| via math.absf; @tgt
// builds the same mask and computes |a/b| as select(a/b < 0, -(a/b), a/b).

// CHECK: smt.solver
// CHECK: smt.and
// CHECK: smt.real.div
// CHECK: smt.solver
// CHECK-DAG: smt.real.div
// CHECK-DAG: smt.real.neg
// CHECK-DAG: smt.ite
// CHECK: smt.distinct %{{.*}}, %{{.*}} : !smt.real

// SMTLIB: (/
// SMTLIB: (check-sat)

module {
  tt.func public @src(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m1 = arith.cmpi slt, %idx, %n : i32
    %m2 = arith.cmpi sge, %idx, %c0 : i32
    %m = arith.andi %m1, %m2 : i1
    %pa = tt.addptr %a, %idx : !tt.ptr<f32>, i32
    %av = tt.load %pa, %m, %cst : !tt.ptr<f32>
    %pb = tt.addptr %b, %idx : !tt.ptr<f32>, i32
    %bv = tt.load %pb, %m, %cst : !tt.ptr<f32>
    %q = arith.divf %av, %bv : f32
    %r = math.absf %q : f32
    %po = tt.addptr %o, %idx : !tt.ptr<f32>, i32
    tt.store %po, %r, %m : !tt.ptr<f32>
    tt.return
  }
  tt.func public @tgt(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : i32
    %idx = tt.get_program_id x : i32
    %m1 = arith.cmpi slt, %idx, %n : i32
    %m2 = arith.cmpi sge, %idx, %c0 : i32
    %m = arith.andi %m1, %m2 : i1
    %pa = tt.addptr %a, %idx : !tt.ptr<f32>, i32
    %av = tt.load %pa, %m, %cst : !tt.ptr<f32>
    %pb = tt.addptr %b, %idx : !tt.ptr<f32>, i32
    %bv = tt.load %pb, %m, %cst : !tt.ptr<f32>
    %q = arith.divf %av, %bv : f32
    %neg = arith.negf %q : f32
    %isneg = arith.cmpf olt, %q, %cst : f32
    %r = arith.select %isneg, %neg, %q : f32
    %po = tt.addptr %o, %idx : !tt.ptr<f32>, i32
    tt.store %po, %r, %m : !tt.ptr<f32>
    tt.return
  }
}

// -----

// Boolean outputs: @src stores the i1 comparison directly into the `i1`
// buffer; @tgt widens it to i8 (`arith.extui`) and stores through a
// `tt.bitcast`-reinterpreted pointer — the Inductor idiom. The store-side
// coercion reads the byte back as `!= 0`, so both final states are booleans.

// CHECK: smt.declare_fun "arg2" : !smt.array<[!smt.bv<32> -> !smt.bool]>
// CHECK: smt.solver
// CHECK-DAG: smt.bv.constant
// CHECK-DAG: smt.ite
// CHECK: smt.distinct %{{.*}}, %{{.*}} : !smt.bool

// SMTLIB: (Array (_ BitVec 32) Bool)
// SMTLIB: (check-sat)

module {
  tt.func public @src(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %o: !tt.ptr<i1>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<f32>, i32
    %av = tt.load %pa, %m, %cst : !tt.ptr<f32>
    %pb = tt.addptr %b, %idx : !tt.ptr<f32>, i32
    %bv = tt.load %pb, %m, %cst : !tt.ptr<f32>
    %eq = arith.cmpf oeq, %av, %bv : f32
    %po = tt.addptr %o, %idx : !tt.ptr<i1>, i32
    tt.store %po, %eq, %m : !tt.ptr<i1>
    tt.return
  }
  tt.func public @tgt(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %o: !tt.ptr<i1>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pa = tt.addptr %a, %idx : !tt.ptr<f32>, i32
    %av = tt.load %pa, %m, %cst : !tt.ptr<f32>
    %pb = tt.addptr %b, %idx : !tt.ptr<f32>, i32
    %bv = tt.load %pb, %m, %cst : !tt.ptr<f32>
    %eq = arith.cmpf oeq, %av, %bv : f32
    %w = arith.extui %eq : i1 to i8
    %po = tt.addptr %o, %idx : !tt.ptr<i1>, i32
    %pc = tt.bitcast %po : !tt.ptr<i1> -> !tt.ptr<i8>
    tt.store %pc, %w, %m : !tt.ptr<i8>
    tt.return
  }
}
