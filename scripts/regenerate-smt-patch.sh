#!/usr/bin/env bash
# Regenerate scripts/patches/mlir-smt-real.patch from an LLVM checkout.
#
# The patch's source of truth while iterating is the working tree of an LLVM
# checkout pinned to cmake/llvm-info.json's llvm_hash (by default the custom
# .llvm-project/src layout; scripts/build-llvm-project.sh consumers apply the
# patch instead of editing). This script diffs the SMT-related paths, rewrites
# the patch file, and verifies the result applies cleanly to the pristine
# pinned revision -- so the patch and the checkout cannot silently drift.
#
# Usage: scripts/regenerate-smt-patch.sh   (override checkout with LLVM_SRC=...)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
LLVM_SRC="${LLVM_SRC:-$REPO_ROOT/.llvm-project/src}"
PATCH="$REPO_ROOT/scripts/patches/mlir-smt-real.patch"
PINNED_REV="$(jq -r '.llvm_hash' "$REPO_ROOT/cmake/llvm-info.json")"

# The paths the SMT Real-sort patch is allowed to touch. Restricting the diff
# keeps unrelated local experiments in the checkout out of the tracked patch.
PATHSPECS=(
    "mlir/include/mlir/Dialect/SMT"
    "mlir/lib/Dialect/SMT"
    "mlir/lib/Target/SMTLIB"
)

die() { echo "regenerate-smt-patch: ERROR: $*" >&2; exit 1; }

[ -d "$LLVM_SRC/.git" ] || die "$LLVM_SRC is not a git checkout; set LLVM_SRC"
base="$(git -C "$LLVM_SRC" merge-base HEAD "$PINNED_REV" 2>/dev/null || true)"
[ "$base" = "$(git -C "$LLVM_SRC" rev-parse "$PINNED_REV")" ] \
    || die "$LLVM_SRC HEAD does not contain the pinned revision $PINNED_REV"

# Include new files (e.g. SMTRealOps.td) via intent-to-add, then diff only the
# allowed paths. The intent-to-add marks are reverted afterwards.
untracked="$(git -C "$LLVM_SRC" ls-files --others --exclude-standard -- "${PATHSPECS[@]}")"
if [ -n "$untracked" ]; then
    echo "$untracked" | while IFS= read -r f; do
        git -C "$LLVM_SRC" add -N -- "$f"
    done
fi
# Diff against the pinned revision (not HEAD) so changes committed locally on
# top of it are captured as well as working-tree edits.
git -C "$LLVM_SRC" diff "$PINNED_REV" -- "${PATHSPECS[@]}" > "$PATCH"
if [ -n "$untracked" ]; then
    echo "$untracked" | while IFS= read -r f; do
        git -C "$LLVM_SRC" reset -q -- "$f"
    done
fi
[ -s "$PATCH" ] || die "generated patch is empty; no SMT changes in $LLVM_SRC?"

# Verify the patch applies cleanly to the pristine pinned revision, in a
# throwaway worktree so the checkout itself is never touched.
worktree="$(mktemp -d "${TMPDIR:-/tmp}/smt-patch-check.XXXXXX")"
cleanup() { git -C "$LLVM_SRC" worktree remove --force "$worktree" 2>/dev/null || true; }
trap cleanup EXIT
git -C "$LLVM_SRC" worktree add -q --detach "$worktree" "$PINNED_REV"
git -C "$worktree" apply --check "$PATCH" \
    || die "regenerated patch does not apply cleanly to $PINNED_REV"

echo "regenerate-smt-patch: OK: $(basename "$PATCH") regenerated and verified against $PINNED_REV"
echo "regenerate-smt-patch: remember to rebuild (ninja -C <llvm-build> mlir-translate) and commit the patch"
