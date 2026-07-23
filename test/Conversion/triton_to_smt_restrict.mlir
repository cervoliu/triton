// RUN: not triton-opt %s --convert-triton-to-smt=memory-model=true 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: triton-opt %s --convert-triton-to-smt='memory-model=true assume-restrict=true' | FileCheck %s --check-prefix=ACCEPT
// RUN: triton-opt %s --convert-triton-to-smt='memory-model=true assume-restrict=true' | mlir-translate --export-smtlib | FileCheck %s --check-prefix=SMTLIB

// Restrict / aliasing soundness gate (phase-3 block memory model). Each
// !tt.ptr argument is a DISTINCT SMT array, so distinct pointer args are
// disjoint by construction -- but TTIR carries no restrict/const guarantee.
// The kernel below stores to block %out and then LOADS block %in: a memory op
// addressing one pointer-arg block after a store to a different one. Under an
// in-place launch (in == out) that load would observe the store, so the
// disjoint block model only gives a sound verdict if the caller promises
// non-aliasing.
//
// By default that disjointness is load-bearing and the pass REJECTS rather
// than risk a false EQUIVALENT (this REJECT run cannot share a file with the
// memory-model whole-file success runs, hence the dedicated file):
// REJECT: disjointness of pointer arguments is load-bearing for this verdict
// REJECT: assume-restrict=true to obtain a verdict conditional on non-aliasing
//
// With assume-restrict=true the pass encodes under the disjointness assumption
// and emits the usual three solver scopes in fixed order (the driver labels
// the resulting EQUIVALENT verdict EQUIVALENT_UNDER_RESTRICT):
// ACCEPT: smt.solver
// ACCEPT: smt.check
// ACCEPT: smt.solver
// ACCEPT: smt.check
// ACCEPT: smt.solver
// ACCEPT: smt.check

// SMTLIB: (check-sat)
// SMTLIB: (reset)
// SMTLIB: (check-sat)
// SMTLIB: (reset)
// SMTLIB: (check-sat)

module {
  tt.func public @src(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pout = tt.addptr %out, %idx : !tt.ptr<f32>, i32
    tt.store %pout, %cst, %m : !tt.ptr<f32>
    %pin = tt.addptr %in, %idx : !tt.ptr<f32>, i32
    %v = tt.load %pin, %m, %cst : !tt.ptr<f32>
    %pout2 = tt.addptr %out, %idx : !tt.ptr<f32>, i32
    tt.store %pout2, %v, %m : !tt.ptr<f32>
    tt.return
  }
  tt.func public @tgt(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %idx = tt.get_program_id x : i32
    %m = arith.cmpi slt, %idx, %n : i32
    %pout = tt.addptr %out, %idx : !tt.ptr<f32>, i32
    tt.store %pout, %cst, %m : !tt.ptr<f32>
    %pin = tt.addptr %in, %idx : !tt.ptr<f32>, i32
    %v = tt.load %pin, %m, %cst : !tt.ptr<f32>
    %pout2 = tt.addptr %out, %idx : !tt.ptr<f32>, i32
    tt.store %pout2, %v, %m : !tt.ptr<f32>
    tt.return
  }
}
