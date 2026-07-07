#!/usr/bin/env bash
# One-command verification for the SMT translation-validation tool.
#
# Builds triton-opt, probes that the environment is actually usable (the pass
# is registered, mlir-translate carries the Real-sort patch, z3 exists), then
# runs the lit tests and the pytest suite. Every failure mode is loud: a
# verifier whose tests silently skip is indistinguishable from a green one.
#
# Overrides: TRITON_BUILD_DIR, TRITON_OPT, MLIR_TRANSLATE, Z3.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

die() { echo "smt-verify: ERROR: $*" >&2; exit 1; }
step() { echo "smt-verify: $*"; }

# --- 1. Build triton-opt -----------------------------------------------------
if [ -z "${TRITON_BUILD_DIR:-}" ]; then
    for d in "$REPO_ROOT"/build/cmake.*/; do
        [ -e "$d/build.ninja" ] && TRITON_BUILD_DIR="${d%/}" && break
    done
fi
[ -n "${TRITON_BUILD_DIR:-}" ] || die "no configured build dir under build/cmake.*; set TRITON_BUILD_DIR"
step "building triton-opt in $TRITON_BUILD_DIR"
ninja -C "$TRITON_BUILD_DIR" triton-opt

TRITON_OPT="${TRITON_OPT:-$TRITON_BUILD_DIR/bin/triton-opt}"
[ -x "$TRITON_OPT" ] || die "triton-opt not found at $TRITON_OPT"

# --- 2. Probe: the pass is registered ---------------------------------------
# An empty module fails the pass for a legitimate reason ("expected exactly two
# functions"); only an unknown-argument error means the pass is missing, which
# would make every rejection test pass vacuously.
probe_err="$(echo 'module {}' | "$TRITON_OPT" --convert-triton-to-smt 2>&1 >/dev/null || true)"
if echo "$probe_err" | grep -q "Unknown command line argument"; then
    die "$TRITON_OPT does not register --convert-triton-to-smt"
fi
step "triton-opt registers --convert-triton-to-smt"

# --- 3. Probe: mlir-translate carries the Real-sort patch --------------------
if [ -z "${MLIR_TRANSLATE:-}" ]; then
    for c in "$REPO_ROOT/llvm-project/build/bin/mlir-translate" \
             "$REPO_ROOT/.llvm-project/build/bin/mlir-translate"; do
        [ -x "$c" ] && MLIR_TRANSLATE="$c" && break
    done
    MLIR_TRANSLATE="${MLIR_TRANSLATE:-$(command -v mlir-translate || true)}"
fi
[ -n "$MLIR_TRANSLATE" ] && [ -x "$MLIR_TRANSLATE" ] || die "mlir-translate not found; set MLIR_TRANSLATE"
printf 'module {\n  smt.solver() : () -> () {\n    %%r = smt.declare_fun "probe" : !smt.real\n    smt.yield\n  }\n}\n' \
    | "$MLIR_TRANSLATE" --export-smtlib 2>/dev/null \
    | grep -q "(declare-const probe Real)" \
    || die "$MLIR_TRANSLATE cannot export !smt.real; apply scripts/patches/mlir-smt-real.patch and rebuild"
step "mlir-translate handles !smt.real ($MLIR_TRANSLATE)"

# --- 4. Probe: z3 -------------------------------------------------------------
Z3="${Z3:-$(command -v z3 || true)}"
for c in /opt/homebrew/bin/z3 /usr/local/bin/z3 /usr/bin/z3; do
    [ -n "$Z3" ] && break
    [ -x "$c" ] && Z3="$c"
done
[ -n "$Z3" ] || die "z3 not found; set Z3"
step "z3 found ($Z3)"

# --- 5. lit tests -------------------------------------------------------------
LIT=""
for c in "$(dirname "$MLIR_TRANSLATE")/llvm-lit" "$REPO_ROOT/.venv/bin/lit" "$(command -v lit || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && LIT="$c" && break
done
[ -n "$LIT" ] || die "no llvm-lit/lit found"
step "running lit tests (triton_to_smt*)"
"$LIT" -sv "$TRITON_BUILD_DIR/test/Conversion" --filter 'triton_to_smt'

# --- 6. pytest suite ----------------------------------------------------------
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
step "running pytest suite with $PY"
out="$(cd "$REPO_ROOT/python" && \
       TRITON_OPT="$TRITON_OPT" MLIR_TRANSLATE="$MLIR_TRANSLATE" Z3="$Z3" \
       "$PY" -m pytest test/unit/tools/test_smt_equivalence.py -q -rs 2>&1)" \
    || { echo "$out"; die "pytest suite failed"; }
echo "$out" | tail -3
echo "$out" | grep -Eq "[1-9][0-9]* passed" || die "pytest reported no passing tests"
if echo "$out" | grep -q "skipped"; then
    die "pytest skipped tests despite tool probes passing; investigate"
fi

step "OK: all probes and test suites passed"
