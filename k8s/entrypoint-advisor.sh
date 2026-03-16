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

# --- Stash role files from the advisor branch (after checkout so we get the right version) ---
cp "$WORKDIR/instructions/CLAUDE-ADVISOR.md" /tmp/CLAUDE-ADVISOR.md
cp "$WORKDIR/instructions/prompt-advisor.md" /tmp/prompt-advisor.md

# --- Install Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- Install kubectl ---
curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" -o /usr/local/bin/kubectl
chmod +x /usr/local/bin/kubectl

# --- Install gh CLI ---
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null
apt-get update && apt-get install -y gh
# gh uses GITHUB_TOKEN env var automatically, no explicit login needed
echo "=== gh auth ready (using GITHUB_TOKEN env var) ==="

# --- Build prompt (bash heredoc expansion — no envsubst needed) ---
PROMPT="$(eval "cat <<_PROMPT_EOF_
$(cat /tmp/prompt-advisor.md)
_PROMPT_EOF_")"

# --- Launch Claude Code in Ralph Loop ---
export IS_SANDBOX=1

LOGDIR="/workspace/senpai/advisor_logs"
mkdir -p "$LOGDIR"

# --- Start Weave thread logger in background ---
python3 "$WORKDIR/tools/weave_logger.py" --role advisor --agent-name advisor --workdir "$WORKDIR" &

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Advisor Loop iteration $ITERATION ($(date)) ==="
    echo "=== Log: $LOGFILE ==="

    # Restore CLAUDE.md each iteration — advisor git checkouts during PR review can clobber it
    cp /tmp/CLAUDE-ADVISOR.md "$WORKDIR/CLAUDE.md"

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --model "claude-opus-4-6[1m]" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || true
    else
        claude -c -p "$PROMPT" --model "claude-opus-4-6[1m]" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || \
        claude -p "$PROMPT" --model "claude-opus-4-6[1m]" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || true
    fi

    echo "=== Advisor exited at $(date), next check in 5 minutes ==="
    sleep 300
done
