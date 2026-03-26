#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

WORKDIR="/workspace/senpai"

echo "=== Senpai Student: $STUDENT_NAME ==="
echo "Repo:   $REPO_URL (branch: $REPO_BRANCH)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Repo already cloned by the deployment args block
cd "$WORKDIR"
uv pip install --system -e .

# --- Git identity for commits ---
git config user.name "senpai-$STUDENT_NAME"
git config user.email "senpai-$STUDENT_NAME@senpai"

# --- Install Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- Install Weave Claude Code Plugin ---
# Must come BEFORE settings.json so the plugin can merge its own hooks first.
source "$WORKDIR/tools/install-weave-cc-plugin.sh"

# --- Configure Claude Code: autocompact + safety-net Stop hook ---
# 1. CLAUDE_AUTOCOMPACT_PCT_OVERRIDE triggers built-in compaction at 75% of
#    the context window (~150k on a 200k model).  This is the primary mechanism.
# 2. The Stop hook (check-context-limit.py) is a safety net: if context is STILL
#    above SENPAI_TOKEN_HARD_LIMIT after autocompact, it flags a fresh restart.
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="${CLAUDE_AUTOCOMPACT_PCT_OVERRIDE:-75}"
export SENPAI_TOKEN_HARD_LIMIT="${SENPAI_TOKEN_HARD_LIMIT:-180000}"

# Merge our settings into whatever weave-claude-plugin already wrote.
# Use python to do a proper JSON merge rather than overwriting.
python3 -c "
import json, pathlib
p = pathlib.Path('$HOME/.claude/settings.json')
cfg = json.loads(p.read_text()) if p.exists() else {}
cfg['effortLevel'] = 'high'
cfg.setdefault('env', {})['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] = '${CLAUDE_AUTOCOMPACT_PCT_OVERRIDE}'
stop_hook = {
    'hooks': [{
        'type': 'command',
        'command': 'python3 $WORKDIR/tools/check-context-limit.py'
    }]
}
cfg.setdefault('hooks', {}).setdefault('Stop', []).append(stop_hook)
p.write_text(json.dumps(cfg, indent=2))
"

# --- Install gh CLI ---
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null
apt-get update && apt-get install -y gh gettext-base
# gh uses GITHUB_TOKEN env var automatically, no explicit login needed
echo "=== gh auth ready (using GITHUB_TOKEN env var) ==="

# --- Install role instructions ---
cp instructions/CLAUDE-STUDENT.md "$WORKDIR/CLAUDE.md"

# --- Launch Claude Code in Ralph Loop ---
export IS_SANDBOX=1

PROMPT="$(envsubst < "$WORKDIR/instructions/prompt-student.md" | sed '/^<!--$/,/^-->$/d')"


ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== Ralph Loop iteration $ITERATION ($(date)) ==="

    # Return to latest advisor branch so student starts from the current baseline
    git checkout "$ADVISOR_BRANCH" 2>/dev/null || true
    git pull origin "$ADVISOR_BRANCH" 2>/dev/null || true

    # Restore CLAUDE.md — branch checkouts clobber it
    cp "$WORKDIR/instructions/CLAUDE-STUDENT.md" "$WORKDIR/CLAUDE.md"

    # Start fresh on first iteration, or if last session hit the token limit
    if [ "$ITERATION" -eq 1 ] || [ -f /tmp/senpai_needs_fresh_start ]; then
        rm -f /tmp/senpai_needs_fresh_start
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    else
        claude -c -p "$PROMPT" --dangerously-skip-permissions || \
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    fi

    echo "=== Claude exited at $(date), restarting in 5s ==="
    sleep 5
done
