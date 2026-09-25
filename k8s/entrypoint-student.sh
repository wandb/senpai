#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail
umask "${SENPAI_UMASK:-0022}"
LOGDIR="/var/lib/senpai"
rm -f "$LOGDIR/openhands_state/controller-lease.json"

WORKDIR="/workspace/senpai"
GH_HISTORY_SCOPE="${GH_HISTORY_SCOPE:-branch}"
TARGET_REPO_BRANCH="${TARGET_REPO_BRANCH:-}"
export SENPAI_ROLE="student"
export TARGET_WORKDIR="$WORKDIR/$PROBLEM_DIR"
GIT_ASKPASS_FILE="/tmp/senpai-git-askpass"
mkdir -p "$LOGDIR"
if [ -z "${GITHUB_TOKEN:-}" ] && [ -n "${SENPAI_GITHUB_TOKEN_FILE:-}" ]; then
    export GITHUB_TOKEN="$(<"$SENPAI_GITHUB_TOKEN_FILE")"
fi
: "${GITHUB_TOKEN:?GitHub bootstrap token is required}"

echo "=== Senpai Student: $STUDENT_NAME ==="
echo "Runner repo:  $SENPAI_REPO_URL (revision: $SENPAI_REPO_REVISION)"
echo "Target repo:  $TARGET_REPO_URL (base branch: ${TARGET_REPO_BRANCH:-<default>}; advisor branch: $ADVISOR_BRANCH)"
echo "Problem dir:  $PROBLEM_DIR"
echo "GitHub history: $GH_HISTORY_SCOPE"
echo "GPUs:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Senpai runner repo already cloned by the deployment args block
cd "$WORKDIR"
git config --global safe.directory "$WORKDIR"
source "$SENPAI_PLUGIN/scripts/git-guard.sh"
install_senpai_git_guard "$WORKDIR" "$GIT_ASKPASS_FILE"

clone_target_repo() {
    local depth=()
    [ "$GH_HISTORY_SCOPE" = "fresh" ] && depth=(--depth 1)
    case "$GH_HISTORY_SCOPE" in
        branch|fresh) git clone --branch "$ADVISOR_BRANCH" --single-branch "${depth[@]}" --no-tags "$TARGET_REPO_URL" "$PROBLEM_DIR" ;;
        repo) git clone "$TARGET_REPO_URL" "$PROBLEM_DIR" ;;
        *) echo "ERROR: GH_HISTORY_SCOPE must be one of: branch, repo, fresh" >&2; exit 2 ;;
    esac
}

# Clone the problem-package repo into $PROBLEM_DIR (bring-your-own-repo —
# agent commits/PRs live in $TARGET_REPO_URL, not wandb/senpai).
[ -d "$PROBLEM_DIR/.git" ] || clone_target_repo
git config --global --unset-all credential.helper 2>/dev/null || true

# --- Git identity for commits (inside the problem-package repo) ---
cd "$WORKDIR/$PROBLEM_DIR"
git config user.name "senpai-$STUDENT_NAME"
git config user.email "senpai-$STUDENT_NAME@senpai"
gh repo set-default "$GH_REPO"
install_senpai_target_git_guard "$TARGET_WORKDIR"
if [ "$GH_HISTORY_SCOPE" != "repo" ]; then
    git remote set-branches origin "$ADVISOR_BRANCH"
    git config remote.origin.tagOpt --no-tags
fi

echo "=== Agent config installed ==="
ls \
    "$SENPAI_AGENT_DIR/bash-runner.md" \
    "$SENPAI_AGENT_DIR/general-purpose.md" \
    "$SENPAI_AGENT_DIR/explore.md" \
    "$SENPAI_AGENT_DIR/search.md" \
    "$SENPAI_PLUGIN/skills/wandb-primary/SKILL.md"

# --- Hivemind is intentionally disabled pending its OpenHands rewrite. ---
# source "$WORKDIR/k8s/start-hivemind.sh"
# start_hivemind

export IS_SANDBOX=1

export SENPAI_OPENHANDS_STATE_DIR="$LOGDIR/openhands_state"
export SENPAI_OPENHANDS_ROLE_FILE="$WORKDIR/system_instructions/STUDENT.md"
export SENPAI_OPENHANDS_WORKSPACE="$TARGET_WORKDIR"
export SENPAI_OPENHANDS_HARNESS_FILE="$WORKDIR/system_instructions/SENPAI-HARNESS.md"
export SENPAI_OPENHANDS_TIMEOUT_SECONDS="${SENPAI_OPENHANDS_TIMEOUT_SECONDS:-7200}"
if [ -z "${SENPAI_GITHUB_TOKEN_FILE:-}" ]; then
    export SENPAI_GITHUB_TOKEN_FILE="/tmp/senpai-supervisor-github-token"
    (umask 077; printf '%s' "$GITHUB_TOKEN" > "$SENPAI_GITHUB_TOKEN_FILE")
fi
unset GITHUB_TOKEN GH_TOKEN GIT_ASKPASS
rm -f "$GIT_ASKPASS_FILE"
export SENPAI_TARGET_PYTHON_ENV="$HOME/.venvs/senpai-target"
if [ ! -x "$SENPAI_TARGET_PYTHON_ENV/bin/python" ]; then
    "$SENPAI_PYTHON" -P -m venv --without-pip "$SENPAI_TARGET_PYTHON_ENV"
fi
CONTROLLER_SITE="$("$SENPAI_PYTHON" -P -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
# The target interpreter is agent-writable; never execute it during trusted startup.
TARGET_SITE="$("$SENPAI_PYTHON" -P -c 'import sys, sysconfig; print(sysconfig.get_path("purelib", vars={"base": sys.argv[1]}))' "$SENPAI_TARGET_PYTHON_ENV")"
printf '%s\n' "$CONTROLLER_SITE" > "$TARGET_SITE/senpai-runtime.pth"
cd "$WORKDIR"
exec "$SENPAI_PYTHON" -P -m senpai_agent.supervisor student
