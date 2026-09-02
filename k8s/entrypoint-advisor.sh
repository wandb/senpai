#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail
umask "${SENPAI_UMASK:-0022}"
LOGDIR="/var/lib/senpai/$RESEARCH_TAG/advisor"
rm -f "$LOGDIR/openhands_state/controller-lease.json"
date +%s > "${SENPAI_BOOTSTRAP_STARTED_PATH:-/var/lib/senpai/.bootstrap-started}"

WORKDIR="/workspace/senpai"
GH_HISTORY_SCOPE="${GH_HISTORY_SCOPE:-branch}"
TARGET_REPO_BRANCH="${TARGET_REPO_BRANCH:-}"
export SENPAI_ROLE="advisor"
export TARGET_WORKDIR="$WORKDIR/$PROBLEM_DIR"
SOURCE_SENPAI_PLUGIN="$WORKDIR/plugins/senpai"
export SENPAI_PLUGIN="$SOURCE_SENPAI_PLUGIN"
GIT_ASKPASS_FILE="/tmp/senpai-git-askpass"
mkdir -p "$LOGDIR"
if [ -z "${GITHUB_TOKEN:-}" ] && [ -n "${SENPAI_GITHUB_TOKEN_FILE:-}" ]; then
    export GITHUB_TOKEN="$(<"$SENPAI_GITHUB_TOKEN_FILE")"
fi
: "${GITHUB_TOKEN:?GitHub bootstrap token is required}"
export SENPAI_OPENHANDS_STATE_DIR="$LOGDIR/openhands_state"
export SENPAI_OPENHANDS_ROLE_FILE="$WORKDIR/system_instructions/ADVISOR.md"

echo "=== Senpai Advisor ==="
echo "Runner repo:  $SENPAI_REPO_URL (revision: $SENPAI_REPO_REVISION)"
echo "Target repo:  $TARGET_REPO_URL (base branch: ${TARGET_REPO_BRANCH:-<default>}; advisor branch: $ADVISOR_BRANCH)"
echo "Problem dir:  $PROBLEM_DIR"
echo "Tag:          $RESEARCH_TAG"
echo "Students:     $STUDENT_NAMES"
echo "GitHub history: $GH_HISTORY_SCOPE"

# Senpai runner repo already cloned by the deployment args block
cd "$WORKDIR"
git config --global safe.directory "$WORKDIR"
source "$SOURCE_SENPAI_PLUGIN/scripts/git-guard.sh"
install_senpai_git_guard "$WORKDIR" "$TARGET_WORKDIR" "$GIT_ASKPASS_FILE"

advisor_branch_exists() {
    local status=0
    git ls-remote --exit-code --heads "$TARGET_REPO_URL" "refs/heads/$ADVISOR_BRANCH" >/dev/null || status=$?
    case "$status" in
        0) return 0 ;;
        2) return 1 ;;
        *) echo "ERROR: could not query $TARGET_REPO_URL for branch '$ADVISOR_BRANCH'" >&2; exit 1 ;;
    esac
}

# Clone the problem-package repo into $PROBLEM_DIR (bring-your-own-repo —
# agent commits/PRs live in $TARGET_REPO_URL, not wandb/senpai). The advisor
# branch is created from the base branch only when it is genuinely absent, so a
# transient clone failure can never fast-forward an existing advisor branch.
clone_target_repo() {
    local depth=() base=()
    [ "$GH_HISTORY_SCOPE" = "fresh" ] && depth=(--depth 1)
    [ -n "$TARGET_REPO_BRANCH" ] && base=(--branch "$TARGET_REPO_BRANCH")
    case "$GH_HISTORY_SCOPE" in
        repo) git clone "$TARGET_REPO_URL" "$PROBLEM_DIR" ;;
        branch|fresh)
            if advisor_branch_exists; then
                git clone --branch "$ADVISOR_BRANCH" --single-branch "${depth[@]}" --no-tags "$TARGET_REPO_URL" "$PROBLEM_DIR"
            else
                git clone "${base[@]}" --single-branch "${depth[@]}" --no-tags "$TARGET_REPO_URL" "$PROBLEM_DIR"
                (
                    cd "$PROBLEM_DIR"
                    git checkout -b "$ADVISOR_BRANCH"
                    git push -u origin "$ADVISOR_BRANCH"
                )
            fi
            ;;
        *) echo "ERROR: GH_HISTORY_SCOPE must be one of: branch, repo, fresh" >&2; exit 2 ;;
    esac
}

[ -d "$PROBLEM_DIR/.git" ] || clone_target_repo
git config --global --unset-all credential.helper 2>/dev/null || true

uv pip install --python "$SENPAI_PYTHON" --no-deps -e .

source "$SOURCE_SENPAI_PLUGIN/scripts/agent-context.sh"
AGENT_CONTEXT_ROOT="$(mktemp -d /tmp/senpai-agent-context.XXXXXX)"
export SENPAI_PLUGIN="$(
    install_senpai_agent_context \
        "$WORKDIR" "$SOURCE_SENPAI_PLUGIN" "$AGENT_CONTEXT_ROOT"
)"

# --- Git identity (inside the problem-package repo) ---
cd "$WORKDIR/$PROBLEM_DIR"
git config user.name "senpai-advisor"
git config user.email "senpai-advisor@senpai"
gh repo set-default "$GH_REPO"
install_senpai_target_git_guard "$TARGET_WORKDIR"

# --- Create or checkout advisor branch ---
if [ "$GH_HISTORY_SCOPE" != "repo" ]; then
    git remote set-branches origin "$ADVISOR_BRANCH"
    git config remote.origin.tagOpt --no-tags
fi
if git rev-parse --verify "origin/$ADVISOR_BRANCH" >/dev/null 2>&1; then
    git checkout "$ADVISOR_BRANCH"
    git pull --ff-only origin "$ADVISOR_BRANCH"
else
    if [ -n "$TARGET_REPO_BRANCH" ]; then
        git fetch origin "$TARGET_REPO_BRANCH"
        git checkout -B "$ADVISOR_BRANCH" "origin/$TARGET_REPO_BRANCH"
    else
        git checkout -b "$ADVISOR_BRANCH"
    fi
    git push -u origin "$ADVISOR_BRANCH"
fi

echo "=== Agent config installed ==="
ls \
    "$HOME/.agents/agents/bash-runner.md" \
    "$HOME/.agents/agents/general-purpose.md" \
    "$HOME/.agents/agents/explore.md" \
    "$HOME/.agents/agents/search.md" \
    "$SENPAI_PLUGIN/skills/wandb-primary/SKILL.md"

# --- Hivemind is intentionally disabled pending its OpenHands rewrite. ---
# source "$WORKDIR/k8s/start-hivemind.sh"
# start_hivemind

export IS_SANDBOX=1
export SENPAI_OPENHANDS_WORKSPACE="$TARGET_WORKDIR"
export SENPAI_OPENHANDS_HARNESS_FILE="$WORKDIR/system_instructions/SENPAI-HARNESS.md"
export SENPAI_OPENHANDS_TIMEOUT_SECONDS="${SENPAI_OPENHANDS_TIMEOUT_SECONDS:-7200}"
if [ -z "${SENPAI_GITHUB_TOKEN_FILE:-}" ]; then
    export SENPAI_GITHUB_TOKEN_FILE="/tmp/senpai-supervisor-github-token"
    (umask 077; printf '%s' "$GITHUB_TOKEN" > "$SENPAI_GITHUB_TOKEN_FILE")
fi
unset GITHUB_TOKEN GH_TOKEN GIT_ASKPASS
rm -f "$GIT_ASKPASS_FILE"
exec python -m senpai_agent.supervisor advisor
