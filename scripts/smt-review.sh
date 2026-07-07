#!/usr/bin/env bash
# Run the Codex reviewer on the TritonToSMT translation-validation work.
#
# It (re)writes phase-1-review.md and prints a machine-readable verdict line
# (VERDICT: APPROVED | VERDICT: CHANGES_REQUESTED). Intended to be driven in a
# fix -> review -> fix loop until the reviewer approves.
#
# Usage:  scripts/smt-review.sh
# Env:    CODEX_SANDBOX (default: workspace-write), CODEX_MODEL (optional).
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
PROMPT_FILE="$REPO/scripts/smt-review-prompt.md"
VERDICT_FILE="$REPO/.smt-review-last.txt"
SANDBOX="${CODEX_SANDBOX:-workspace-write}"

args=(exec -s "$SANDBOX" --skip-git-repo-check -C "$REPO" -o "$VERDICT_FILE")
[ -n "${CODEX_MODEL:-}" ] && args+=(-m "$CODEX_MODEL")

echo ">> codex ${args[*]} <prompt>"
codex "${args[@]}" "$(cat "$PROMPT_FILE")"

echo "==== reviewer verdict ===="
grep -Eo 'VERDICT: (APPROVED|CHANGES_REQUESTED)' "$VERDICT_FILE" | tail -1 \
  || echo "VERDICT: UNKNOWN (no verdict line emitted)"
