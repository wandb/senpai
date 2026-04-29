#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -euo pipefail

WORKDIR="/workspace/ml-intern-benchmark"
ML_INTERN_DIR="$WORKDIR/ml-intern"
TARGET_DIR="$WORKDIR/target"
PROMPT_FILE="$WORKDIR/prompt.md"
LOGDIR="$WORKDIR/logs"
GIT_CREDENTIAL_FILE="$WORKDIR/.git-credentials"

: "${TARGET_REPO_URL:?TARGET_REPO_URL is required}"
: "${BASE_REF:?BASE_REF is required}"
: "${TARGET_BRANCH:?TARGET_BRANCH is required}"
: "${REPLICATE:?REPLICATE is required}"
: "${ML_INTERN_REPO_URL:?ML_INTERN_REPO_URL is required}"
: "${ML_INTERN_REPO_REF:?ML_INTERN_REPO_REF is required}"
: "${ML_INTERN_MODEL:?ML_INTERN_MODEL is required}"
: "${ML_INTERN_PROMPT_B64:?ML_INTERN_PROMPT_B64 is required}"
: "${ML_INTERN_TIMEOUT_SECONDS:?ML_INTERN_TIMEOUT_SECONDS is required}"
: "${ML_INTERN_WALL_CLOCK_SECONDS:?ML_INTERN_WALL_CLOCK_SECONDS is required}"
: "${ML_INTERN_MAX_ITERATIONS:?ML_INTERN_MAX_ITERATIONS is required}"
: "${ML_INTERN_DEFAULT_EPOCHS:?ML_INTERN_DEFAULT_EPOCHS is required}"
: "${GPUS_PER_REPLICATE:?GPUS_PER_REPLICATE is required}"

mkdir -p "$WORKDIR" "$LOGDIR"
umask 077
printf 'https://x-access-token:%s@github.com\n' "$GITHUB_TOKEN" > "$GIT_CREDENTIAL_FILE"
git config --global credential.helper "store --file=$GIT_CREDENTIAL_FILE"
git config --global user.name "ml-intern-$REPLICATE"
git config --global user.email "ml-intern-$REPLICATE@senpai"

START_EPOCH="$(date +%s)"
DEADLINE_EPOCH="$((START_EPOCH + ML_INTERN_WALL_CLOCK_SECONDS))"
ML_INTERN_DEADLINE_EPOCH="$((START_EPOCH + ML_INTERN_TIMEOUT_SECONDS))"
{
  printf 'pod_start_epoch=%s\n' "$START_EPOCH"
  printf 'ml_intern_deadline_epoch=%s\n' "$ML_INTERN_DEADLINE_EPOCH"
  printf 'pod_kill_deadline_epoch=%s\n' "$DEADLINE_EPOCH"
  printf 'pod_kill_deadline_utc=%s\n' "$(date -u -d "@$DEADLINE_EPOCH" +%Y-%m-%dT%H:%M:%SZ)"
} > "$WORKDIR/deadline.txt"

echo "=== ML Intern TandemFoilSet-Balanced benchmark replicate $REPLICATE ==="
echo "Target repo:    $TARGET_REPO_URL"
echo "Base ref:       $BASE_REF"
echo "Target branch:  $TARGET_BRANCH"
echo "ML Intern repo: $ML_INTERN_REPO_URL ($ML_INTERN_REPO_REF)"
echo "Model:          $ML_INTERN_MODEL"
echo "W&B:            ${WANDB_ENTITY:-}/${WANDB_PROJECT:-}"
echo "Wall clock:     ${ML_INTERN_WALL_CLOCK_SECONDS}s"
echo "Agent timeout:  ${ML_INTERN_TIMEOUT_SECONDS}s"
echo "SENPAI timeout: ${SENPAI_TIMEOUT_MINUTES:-unset} minutes"
echo "Visible GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

echo "=== Installing Hugging Face ML Intern ==="
git init "$ML_INTERN_DIR"
git -C "$ML_INTERN_DIR" remote add origin "$ML_INTERN_REPO_URL"
git -C "$ML_INTERN_DIR" fetch --depth 1 --no-tags origin "$ML_INTERN_REPO_REF"
git -C "$ML_INTERN_DIR" checkout --detach FETCH_HEAD
echo "ML Intern resolved commit: $(git -C "$ML_INTERN_DIR" rev-parse HEAD)"
uv pip install --system -e "$ML_INTERN_DIR"

echo "=== Cloning target repo ==="
target_auth_url="$(printf '%s' "$TARGET_REPO_URL" | sed "s#https://github.com/#https://${GITHUB_TOKEN}@github.com/#")"
git clone --branch "$BASE_REF" --single-branch --no-tags "$target_auth_url" "$TARGET_DIR"
cd "$TARGET_DIR"
git remote set-url origin "$TARGET_REPO_URL"
git config user.name "ml-intern-$REPLICATE"
git config user.email "ml-intern-$REPLICATE@senpai"

if git ls-remote --exit-code --heads origin "$TARGET_BRANCH" >/dev/null 2>&1; then
  git fetch origin "$TARGET_BRANCH"
  git checkout -B "$TARGET_BRANCH" "origin/$TARGET_BRANCH"
else
  git checkout -B "$TARGET_BRANCH"
  git push -u origin "$TARGET_BRANCH"
fi

if [ -f pyproject.toml ]; then
  uv pip install --system -e .
elif [ -f requirements.txt ]; then
  uv pip install --system -r requirements.txt
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export SENPAI_TIMEOUT_MINUTES="${SENPAI_TIMEOUT_MINUTES:-720}"
printf '%s' "$ML_INTERN_PROMPT_B64" | base64 -d > "$PROMPT_FILE"
cp "$PROMPT_FILE" "$WORKDIR/prompt.replicate-$REPLICATE.md"

echo "=== Starting ML Intern ==="
echo "Working directory: $TARGET_DIR"
echo "Deadline file:     $WORKDIR/deadline.txt"
LOGFILE="$LOGDIR/ml-intern-replicate-$REPLICATE-$(date +%Y%m%d_%H%M%S).log"
PROMPT="$(cat "$PROMPT_FILE")"

set +e
timeout --preserve-status --kill-after=60s "${ML_INTERN_TIMEOUT_SECONDS}s" \
  ml-intern \
    --model "$ML_INTERN_MODEL" \
    --max-iterations "$ML_INTERN_MAX_ITERATIONS" \
    --no-stream \
    "$PROMPT" 2>&1 | tee "$LOGFILE"
ML_INTERN_EXIT="${PIPESTATUS[0]}"
set -e
export ML_INTERN_EXIT_CODE="$ML_INTERN_EXIT"

echo "=== ML Intern exited with code $ML_INTERN_EXIT ==="

cd "$TARGET_DIR"
mkdir -p research
python3 - <<'PY'
import json
import os
import time
from pathlib import Path

metadata = {
    "runner": "ml-intern",
    "replicate": os.environ["REPLICATE"],
    "research_tag": os.environ["RESEARCH_TAG"],
    "target_repo_url": os.environ["TARGET_REPO_URL"],
    "base_ref": os.environ["BASE_REF"],
    "target_branch": os.environ["TARGET_BRANCH"],
    "model": os.environ["ML_INTERN_MODEL"],
    "wandb_entity": os.environ.get("WANDB_ENTITY", ""),
    "wandb_project": os.environ.get("WANDB_PROJECT", ""),
    "gpus_per_replicate": int(os.environ["GPUS_PER_REPLICATE"]),
    "senpai_timeout_minutes": float(os.environ["SENPAI_TIMEOUT_MINUTES"]),
    "default_epochs": int(os.environ["ML_INTERN_DEFAULT_EPOCHS"]),
    "wall_clock_seconds": int(os.environ["ML_INTERN_WALL_CLOCK_SECONDS"]),
    "ml_intern_timeout_seconds": int(os.environ["ML_INTERN_TIMEOUT_SECONDS"]),
    "ml_intern_exit_code": int(os.environ["ML_INTERN_EXIT_CODE"]),
    "finished_epoch": int(time.time()),
}
Path("research").mkdir(exist_ok=True)
Path("research/MLINTERN_RUN_METADATA.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
)
summary = Path("research/MLINTERN_SUMMARY.md")
if not summary.exists():
    summary.write_text(
        "# ML Intern Replicate Summary\n\n"
        "ML Intern did not leave a detailed summary before the entrypoint harvest step. "
        "Inspect W&B and pod logs for the full run transcript.\n"
    )
results = Path("research/MLINTERN_RESULTS.jsonl")
if not results.exists():
    results.write_text(json.dumps({"type": "entrypoint_harvest", **metadata}, sort_keys=True) + "\n")
PY

git status --short
git add -A -- . \
  ':!wandb' ':!wandb/**' \
  ':!session_logs' ':!session_logs/**' \
  ':!*.pt' ':!*.pth' ':!*.ckpt' ':!*.safetensors'
if git diff --cached --quiet; then
  echo "=== No target repo changes to commit ==="
else
  git commit -m "Record ML Intern replicate $REPLICATE results"
fi

git push origin "HEAD:$TARGET_BRANCH"
echo "=== Harvest pushed to $TARGET_BRANCH ==="
exit 0
