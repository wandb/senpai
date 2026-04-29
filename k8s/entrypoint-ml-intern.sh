#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -euo pipefail

WORKDIR="/workspace/ml-intern-benchmark"
ML_INTERN_DIR="$WORKDIR/ml-intern"
ML_INTERN_VENV="$WORKDIR/ml-intern-venv"
TARGET_DIR="$WORKDIR/target"
PROMPT_FILE="$WORKDIR/prompt.md"
LOGDIR="$WORKDIR/logs"
ML_INTERN_CONFIG_FILE="$WORKDIR/cli_agent_config.json"
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

terminate_target_processes() {
  local pids=()
  local proc pid cwd cmd
  for proc in /proc/[0-9]*; do
    pid="${proc#/proc/}"
    if [ "$pid" = "$$" ]; then
      continue
    fi
    cwd="$(readlink "$proc/cwd" 2>/dev/null || true)"
    if [[ "$cwd" != "$TARGET_DIR"* ]]; then
      continue
    fi
    cmd="$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)"
    if [ -z "$cmd" ]; then
      continue
    fi
    pids+=("$pid")
  done

  if [ "${#pids[@]}" -eq 0 ]; then
    echo "No remaining target-repo processes to terminate."
    return
  fi

  echo "Terminating remaining target-repo processes before artifact harvest: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 15
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing target-repo process $pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

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
uv python install 3.12
uv venv --python 3.12 "$ML_INTERN_VENV"
uv pip install --python "$ML_INTERN_VENV/bin/python" -e "$ML_INTERN_DIR" tzdata
"$ML_INTERN_VENV/bin/python" --version

echo "=== Cloning target repo ==="
target_auth_url="$(printf '%s' "$TARGET_REPO_URL" | sed "s#https://github.com/#https://${GITHUB_TOKEN}@github.com/#")"
git clone --branch "$BASE_REF" --single-branch --no-tags "$target_auth_url" "$TARGET_DIR"
cd "$TARGET_DIR"
git remote set-url origin "$TARGET_REPO_URL"
git config user.name "ml-intern-$REPLICATE"
git config user.email "ml-intern-$REPLICATE@senpai"

if git ls-remote --exit-code --heads origin "$TARGET_BRANCH" >/dev/null 2>&1; then
  git fetch origin "$TARGET_BRANCH"
  git checkout -B "$TARGET_BRANCH" FETCH_HEAD
else
  git checkout -B "$TARGET_BRANCH"
  git push -u origin "$TARGET_BRANCH"
fi
if [ "$(git branch --show-current)" != "$TARGET_BRANCH" ]; then
  echo "ERROR: expected to be on target branch $TARGET_BRANCH before starting ML Intern" >&2
  exit 1
fi
echo "Using isolated target branch: $TARGET_BRANCH"

if [ -f pyproject.toml ]; then
  uv pip install --system -e .
elif [ -f requirements.txt ]; then
  uv pip install --system -r requirements.txt
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export SENPAI_TIMEOUT_MINUTES="${SENPAI_TIMEOUT_MINUTES:-720}"
printf '%s' "$ML_INTERN_PROMPT_B64" | base64 -d > "$PROMPT_FILE"
cp "$PROMPT_FILE" "$WORKDIR/prompt.replicate-$REPLICATE.md"
cat > "$ML_INTERN_CONFIG_FILE" <<'JSON'
{
  "save_sessions": true,
  "session_dataset_repo": "smolagents/ml-intern-sessions",
  "auto_save_interval": 1,
  "heartbeat_interval_s": 30,
  "auto_file_upload": true
}
JSON
export ML_INTERN_CLI_CONFIG="$ML_INTERN_CONFIG_FILE"

echo "=== Starting ML Intern ==="
echo "Working directory: $TARGET_DIR"
echo "Deadline file:     $WORKDIR/deadline.txt"
LOGFILE="$LOGDIR/ml-intern-replicate-$REPLICATE-$(date +%Y%m%d_%H%M%S).log"
export ML_INTERN_LOGFILE="$LOGFILE"
export ML_INTERN_WORKDIR="$WORKDIR"
export ML_INTERN_CONFIG_FILE
export ML_INTERN_START_EPOCH="$START_EPOCH"
PROMPT="$(cat "$PROMPT_FILE")"

set +e
timeout --preserve-status --kill-after=60s "${ML_INTERN_TIMEOUT_SECONDS}s" \
  "$ML_INTERN_VENV/bin/ml-intern" \
    --model "$ML_INTERN_MODEL" \
    --max-iterations "$ML_INTERN_MAX_ITERATIONS" \
    --no-stream \
    "$PROMPT" 2>&1 | tee "$LOGFILE"
ML_INTERN_EXIT="${PIPESTATUS[0]}"
set -e
export ML_INTERN_EXIT_CODE="$ML_INTERN_EXIT"

echo "=== ML Intern exited with code $ML_INTERN_EXIT ==="
terminate_target_processes

cd "$TARGET_DIR"
mkdir -p research
python3 - <<'PY'
import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path

try:
    from agent.core.redact import scrub
except Exception:
    def scrub(value):
        return value

TEXT_SUFFIXES_TO_COMPRESS = {".log", ".txt", ".out", ".err"}
SECRET_PATTERNS = [
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{40,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{36,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{20,}"), "Bearer [REDACTED]"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def redact_text(text: str) -> str:
    text = scrub(text)
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    for name in ("GITHUB_TOKEN", "ANTHROPIC_API_KEY", "HF_TOKEN", "WANDB_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, f"[REDACTED_{name}]")
    return text


def copy_text_artifact(src: Path, dest_dir: Path, name: str | None = None) -> dict | None:
    if not src.exists() or not src.is_file():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = name or src.name
    text = src.read_text(errors="replace")
    redacted = redact_text(text)
    compress = src.suffix in TEXT_SUFFIXES_TO_COMPRESS or len(redacted) > 8_000_000
    dest = dest_dir / base
    if compress:
        dest = dest.with_suffix(dest.suffix + ".gz")
        with gzip.open(dest, "wt", encoding="utf-8") as f:
            f.write(redacted)
    else:
        dest.write_text(redacted)
    return {
        "source": str(src),
        "artifact": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "compressed": compress,
    }


def load_session(path: Path) -> dict | None:
    try:
        data = scrub(json.loads(path.read_text()))
        return json.loads(redact_text(json.dumps(data)))
    except Exception as e:
        return {"path": str(path), "parse_error": str(e)}


target_dir = Path.cwd()
workdir = Path(os.environ["ML_INTERN_WORKDIR"])
start_epoch = int(os.environ.get("ML_INTERN_START_EPOCH", "0") or "0")
artifact_root = target_dir / "research" / "MLINTERN_ARTIFACTS"
conversation_dir = artifact_root / "session_logs"
stdout_dir = artifact_root / "stdout_logs"
tmp_dir = artifact_root / "tmp_logs"
manifest_records = []
seen_sources = set()

for src, dest_dir, name in [
    (Path(os.environ["ML_INTERN_LOGFILE"]), stdout_dir, None),
    (workdir / "prompt.md", artifact_root, "prompt.md"),
    (workdir / "deadline.txt", artifact_root, "deadline.txt"),
    (Path(os.environ["ML_INTERN_CONFIG_FILE"]), artifact_root, "cli_agent_config.json"),
]:
    resolved = str(src.resolve()) if src.exists() else str(src)
    if resolved in seen_sources:
        continue
    seen_sources.add(resolved)
    record = copy_text_artifact(src, dest_dir, name)
    if record:
        manifest_records.append(record)

for src in sorted((workdir / "logs").glob("*")):
    resolved = str(src.resolve())
    if resolved in seen_sources:
        continue
    seen_sources.add(resolved)
    record = copy_text_artifact(src, stdout_dir)
    if record:
        manifest_records.append(record)

session_files = []
for session_dir in (target_dir / "session_logs", workdir / "session_logs"):
    if session_dir.exists():
        session_files.extend(sorted(session_dir.glob("session_*.json")))
session_files = sorted({path.resolve() for path in session_files})

conversation_rows = []
for src in session_files:
    data = load_session(src)
    if data is None:
        continue
    conversation_rows.append(data)
    dest_name = f"{src.parent.name}-{src.name}" if src.parent != target_dir / "session_logs" else src.name
    dest = conversation_dir / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    manifest_records.append({
        "source": str(src),
        "artifact": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "compressed": False,
        "kind": "ml_intern_session_trajectory",
    })

conversation_jsonl = target_dir / "research" / "MLINTERN_CONVERSATION.jsonl"
conversation_jsonl.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in conversation_rows)
)
manifest_records.append({
    "source": "session_logs/session_*.json",
    "artifact": str(conversation_jsonl),
    "bytes": conversation_jsonl.stat().st_size,
    "sha256": sha256(conversation_jsonl),
    "compressed": False,
    "kind": "ml_intern_conversation_jsonl",
})

for pattern in ("bash_output_*.txt", "output*.log", "*.out", "*.err", "*train*.log"):
    for src in sorted(Path("/tmp").glob(pattern)):
        if start_epoch and src.stat().st_mtime < start_epoch - 60:
            continue
        resolved = str(src.resolve())
        if resolved in seen_sources:
            continue
        seen_sources.add(resolved)
        record = copy_text_artifact(src, tmp_dir)
        if record:
            manifest_records.append(record)

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
    "conversation_session_count": len(conversation_rows),
    "artifact_count": len(manifest_records),
    "finished_epoch": int(time.time()),
}
Path("research").mkdir(exist_ok=True)
Path("research/MLINTERN_RUN_METADATA.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
)
manifest = {
    **metadata,
    "artifacts": manifest_records,
}
Path("research/MLINTERN_ARTIFACT_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
summary = Path("research/MLINTERN_SUMMARY.md")
if not summary.exists():
    summary.write_text(
        "# ML Intern Replicate Summary\n\n"
        "ML Intern did not leave a detailed summary before the entrypoint harvest step.\n\n"
        f"The entrypoint harvested {len(conversation_rows)} ML Intern session trajectory file(s) "
        f"and {len(manifest_records)} artifact file(s). See `research/MLINTERN_CONVERSATION.jsonl`, "
        "`research/MLINTERN_ARTIFACT_MANIFEST.json`, and `research/MLINTERN_ARTIFACTS/`.\n"
    )
results = Path("research/MLINTERN_RESULTS.jsonl")
with results.open("a") as f:
    f.write(json.dumps({"type": "entrypoint_harvest", **metadata}, sort_keys=True) + "\n")
    f.write(json.dumps({
        "type": "entrypoint_artifacts",
        "artifact_manifest": "research/MLINTERN_ARTIFACT_MANIFEST.json",
        "conversation_jsonl": "research/MLINTERN_CONVERSATION.jsonl",
        "conversation_session_count": len(conversation_rows),
        "artifact_count": len(manifest_records),
    }, sort_keys=True) + "\n")
PY

git status --short
git add -A -- . \
  ':(exclude)wandb' ':(exclude)wandb/**' \
  ':(exclude)session_logs' ':(exclude)session_logs/**' \
  ':(exclude)*.pt' ':(exclude)*.pth' ':(exclude)*.ckpt' ':(exclude)*.safetensors'
if git diff --cached --quiet; then
  echo "=== No target repo changes to commit ==="
else
  git commit -m "Record ML Intern replicate $REPLICATE results"
fi

git push origin "HEAD:$TARGET_BRANCH"
echo "=== Harvest pushed to $TARGET_BRANCH ==="
exit 0
