#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

WORKDIR="/workspace/senpai"

echo "=== Senpai Advisor ==="
echo "Repo:     $REPO_URL (branch: $REPO_BRANCH)"
echo "Tag:      $RESEARCH_TAG"
echo "Students: $STUDENT_NAMES"

# Repo already cloned by the deployment args block
cd "$WORKDIR"

uv pip install --system -e .

# --- Git identity ---
git config user.name "senpai-advisor"
git config user.email "senpai-advisor@senpai"

# --- Create or checkout advisor branch ---
git fetch origin
if git rev-parse --verify "origin/$ADVISOR_BRANCH" >/dev/null 2>&1; then
    git checkout "$ADVISOR_BRANCH"
    git pull origin "$ADVISOR_BRANCH"
else
    git checkout -b "$ADVISOR_BRANCH"
    git push -u origin "$ADVISOR_BRANCH"
fi

# --- Install role instructions ---
cp "$WORKDIR/instructions/CLAUDE-ADVISOR.md" "$WORKDIR/CLAUDE.md"

# --- Register Weave Claude Plugin (tools already baked into Docker image) ---
export PATH="$HOME/.claude/bin:$PATH"
source "$WORKDIR/k8s/install-weave-cc-plugin.sh"

# --- Build prompt ---
PROMPT="$(envsubst < "$WORKDIR/instructions/prompt-advisor.md" | sed '/^<!--$/,/^-->$/d')"

# --- Append extra startup instructions if provided ---
if [ -n "${EXTRA_INSTRUCTIONS_B64:-}" ]; then
    PROMPT="${PROMPT}"$'\n\n# Finally, some additional instructions\n\n'"$(printf '%s' "$EXTRA_INSTRUCTIONS_B64" | base64 -d)"
fi

# --- Launch Claude Code in Ralph Loop ---
export IS_SANDBOX=1

LOGDIR="/workspace/senpai/advisor_logs"
mkdir -p "$LOGDIR"


ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Advisor Loop iteration $ITERATION ($(date)) ==="
    echo "=== Log: $LOGFILE ==="

    # Restore CLAUDE.md — branch checkouts clobber it
    cp "$WORKDIR/instructions/CLAUDE-ADVISOR.md" "$WORKDIR/CLAUDE.md"

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --model "claude-opus-4-6[1m]" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || true
    else
        claude -c -p "$PROMPT" --model "claude-opus-4-6[1m]" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || \
        claude -p "$PROMPT" --model "claude-opus-4-6[1m]" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || true
    fi

    echo "=== Advisor exited at $(date), next check in 5 minutes ==="
    sleep 300
done
