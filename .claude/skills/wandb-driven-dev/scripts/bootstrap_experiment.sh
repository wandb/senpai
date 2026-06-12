#!/usr/bin/env bash
# Phase 0 bootstrap: spin up a worktree ready for an experiment.
#
# Replaces the manual sequence we hit in every experiment session:
#   1. git worktree add ...
#   2. notice /experiments/ is gitignored (per-repo, may already be fixed)
#   3. uv sync so the worktree's venv has pyyaml + everything (skipped if no pyproject.toml)
#   4. mkdir experiments/<slug>/logs
#   5. write a stub plan.md
#
# Usage:
#   bootstrap_experiment.sh <slug> [worktree_root]
#
# Defaults worktree_root to "../$(basename $main_repo)-exp". Prints the
# resulting worktree path on the last line of stdout so the caller can:
#   cd "$(.../bootstrap_experiment.sh <slug> | tail -1)"
set -euo pipefail

SLUG="${1:?usage: bootstrap_experiment.sh <slug> [worktree_root]}"

# Slug format check: YYYYMMDD-<kebab> per the skill's convention
if ! [[ "$SLUG" =~ ^[0-9]{8}-[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  echo "FAIL: slug '$SLUG' must match YYYYMMDD-<kebab> (e.g. 20260429-3model-bench)" >&2
  exit 2
fi

# Resolve repo root + default worktree parent
REPO_ROOT="$(git rev-parse --show-toplevel)"
DEFAULT_PARENT="$(dirname "$REPO_ROOT")/$(basename "$REPO_ROOT")-exp"
PARENT="${2:-$DEFAULT_PARENT}"
WT="$PARENT/$SLUG"
BRANCH="exp/$SLUG"

# Refuse to clobber
if [ -d "$WT" ]; then
  echo "FAIL: $WT already exists" >&2
  exit 3
fi
if git -C "$REPO_ROOT" rev-parse --verify --quiet "$BRANCH" >/dev/null; then
  echo "FAIL: branch $BRANCH already exists" >&2
  exit 4
fi

mkdir -p "$PARENT"
git -C "$REPO_ROOT" worktree add "$WT" -b "$BRANCH" >&2

# Copy the gitignored project config into the worktree so read_config()
# resolves the same settings the main checkout uses.
CFG_REL=".claude/wandb-driven-dev.local.md"
if [ -f "$REPO_ROOT/$CFG_REL" ]; then
  mkdir -p "$WT/.claude"
  cp "$REPO_ROOT/$CFG_REL" "$WT/$CFG_REL"
  echo "Copied $CFG_REL into worktree" >&2
fi

# Un-ignore experiments/ on this branch if needed (a no-op when main already fixed it)
if grep -qE '^/?experiments/?$' "$WT/.gitignore" 2>/dev/null; then
  sed -i.bak -E '/^\/?experiments\/?$/d' "$WT/.gitignore"
  rm -f "$WT/.gitignore.bak"
  echo "Removed /experiments/ from .gitignore on $BRANCH" >&2
fi

# Logs directory committed-in-place
mkdir -p "$WT/experiments/$SLUG/logs"

# Stub plan.md per the SKILL.md scaffold (just enough to anchor Phase 1).
TODAY="$(date -u +%Y-%m-%d)"
cat > "$WT/experiments/$SLUG/plan.md" <<EOF
# $SLUG

**Date:** $TODAY
**Question:** TODO — one sentence.

## Hypothesis
TODO — what we expect to see, stated as a prediction.

## Falsifier
TODO — concrete observation that would prove the hypothesis wrong.
Measurable on wandb: metric key + threshold + which runs.

## Success criteria
TODO — quantitative bar for "variant wins".

## Baseline
- **Status:** TODO (to-be-created | fresh re-run | pinned existing run)
- **Wandb run:** TBD — recorded in Phase 4
- **Why this is the right control:** TODO

## Variant
TODO — the change(s) from baseline. Default: one knob, two runs. For multi-variant
benchmarks (≥3 runs comparing one named dimension), list each variant by name.

## Metrics
Inherited from \`.claude/wandb-driven-dev.local.md\` unless overridden here.
- **Decision:** (override list, or "use config default")
- **Health:** (override list, or "use config default")

## Report Columns
<!-- Optional. Bullet lines only; prose is ignored by the parser. -->
<!-- List config inputs changed and summary outputs you care about. -->
<!-- Example: -->
<!-- - config.lr, config.model.depth, val/loss -->

## Design        (filled in Phase 2)
## Smoke         (filled in Phase 3)
## Runs          (filled in Phase 4)
## Result        (filled in Phase 5)
EOF

# Sync the worktree's venv so pyyaml + all training deps are present.
# Skipped silently if the project doesn't use uv.
if [ -f "$WT/pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
  echo "Running uv sync in $WT ..." >&2
  ( unset VIRTUAL_ENV && cd "$WT" && uv sync ) >&2
fi

echo "Bootstrap complete." >&2
echo "  branch:   $BRANCH" >&2
echo "  worktree: $WT" >&2
echo "  plan:     $WT/experiments/$SLUG/plan.md" >&2
echo "Next: cd into the worktree and start Phase 1 (interview)." >&2

# Last line is the worktree path — pipe-friendly for cd "$(... | tail -1)"
echo "$WT"
