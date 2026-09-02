#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail
umask "${SENPAI_UMASK:-0022}"
LOGDIR="/var/lib/senpai"
rm -f "$LOGDIR/openhands_state/controller-lease.json"
date +%s > "${SENPAI_BOOTSTRAP_STARTED_PATH:-/var/lib/senpai/.bootstrap-started}"

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
: "${SENPAI_PROGRAM_SOURCE_COMMIT:?Launch-pinned program source commit is required}"
: "${SENPAI_PROGRAM_CONTEXT_FILE:?Launch-owned program snapshot file is required}"
[ -r "$SENPAI_PROGRAM_CONTEXT_FILE" ] || {
    echo "ERROR: program snapshot file is not readable" >&2
    exit 1
}

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
CREDENTIAL_HANDOFF_DIR=""
prepare_credential_handoff_dir() {
    [ -n "$CREDENTIAL_HANDOFF_DIR" ] && return
    CREDENTIAL_HANDOFF_DIR="$(mktemp -d /tmp/senpai-supervisor.XXXXXX)"
    chmod 700 "$CREDENTIAL_HANDOFF_DIR"
}
if [ -z "${SENPAI_GITHUB_TOKEN_FILE:-}" ]; then
    prepare_credential_handoff_dir
    export SENPAI_GITHUB_TOKEN_FILE="$CREDENTIAL_HANDOFF_DIR/github-token"
    (umask 077; printf '%s' "$GITHUB_TOKEN" > "$SENPAI_GITHUB_TOKEN_FILE")
fi
if [ -z "${SENPAI_WANDB_API_KEY_FILE:-}" ] && [ -n "${WANDB_API_KEY:-}" ]; then
    prepare_credential_handoff_dir
    export SENPAI_WANDB_API_KEY_FILE="$CREDENTIAL_HANDOFF_DIR/wandb-api-key"
    (umask 077; printf '%s' "$WANDB_API_KEY" > "$SENPAI_WANDB_API_KEY_FILE")
fi
if [ -z "${SENPAI_EXA_API_KEY_FILE:-}" ] && [ -n "${EXA_API_KEY:-}" ]; then
    prepare_credential_handoff_dir
    export SENPAI_EXA_API_KEY_FILE="$CREDENTIAL_HANDOFF_DIR/exa-api-key"
    (umask 077; printf '%s' "$EXA_API_KEY" > "$SENPAI_EXA_API_KEY_FILE")
fi
if [ -z "${SENPAI_WANDB_TRAINING_API_KEY_FILE:-}" ] && [ -n "${SENPAI_WANDB_TRAINING_API_KEY:-}" ]; then
    prepare_credential_handoff_dir
    export SENPAI_WANDB_TRAINING_API_KEY_FILE="$CREDENTIAL_HANDOFF_DIR/wandb-training-api-key"
    (umask 077; printf '%s' "$SENPAI_WANDB_TRAINING_API_KEY" > "$SENPAI_WANDB_TRAINING_API_KEY_FILE")
fi
unset GITHUB_TOKEN GH_TOKEN GIT_ASKPASS WANDB_API_KEY EXA_API_KEY SENPAI_WANDB_TRAINING_API_KEY
rm -f "$GIT_ASKPASS_FILE"
export SENPAI_TARGET_PYTHON_ENV="$HOME/.venvs/senpai-target"
if [ ! -x "$SENPAI_TARGET_PYTHON_ENV/bin/python" ]; then
    "$SENPAI_PYTHON" -m venv "$SENPAI_TARGET_PYTHON_ENV"
fi
CONTROLLER_SITE="$("$SENPAI_PYTHON" -P -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
TARGET_SITE="$("$SENPAI_TARGET_PYTHON_ENV/bin/python" -P -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "$CONTROLLER_SITE" > "$TARGET_SITE/senpai-runtime.pth"
cd "$WORKDIR"
exec "$SENPAI_PYTHON" -P -m senpai_agent.supervisor student
