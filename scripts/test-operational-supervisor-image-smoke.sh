#!/usr/bin/env bash

# Run the production entrypoint in the actual advisor image without external
# authority. A derived canary image replaces only the final controller exec.

set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DOCKER_BIN=${DOCKER_BIN:-docker}
ADVISOR_IMAGE=${ADVISOR_IMAGE:?ADVISOR_IMAGE must name the loaded production image}
SMOKE_ID=${SENPAI_SUPERVISOR_SMOKE_ID:-local-$$}
SMOKE_ID=$(printf '%s' "$SMOKE_ID" | tr -cd '[:alnum:]-' | cut -c1-40)
[[ -n "$SMOKE_ID" ]] || { echo "invalid supervisor smoke ID" >&2; exit 2; }
CANARY_IMAGE="senpai-operational-supervisor-entrypoint-smoke:$SMOKE_ID"
WORK_DIR=$(mktemp -d)
SOURCE_DIR="$WORK_DIR/source"

cleanup() {
  local status=$?
  trap - EXIT
  chmod -R u+w "$WORK_DIR" 2>/dev/null || true
  rm -rf "$WORK_DIR"
  exit "$status"
}
trap cleanup EXIT

mkdir -p \
  "$SOURCE_DIR/k8s" \
  "$SOURCE_DIR/system_instructions" \
  "$SOURCE_DIR/plugins"
cp "$ROOT/k8s/entrypoint-operational-supervisor.sh" \
  "$ROOT/k8s/handoff-operational-supervisor-secrets.sh" \
  "$SOURCE_DIR/k8s/"
cp "$ROOT/system_instructions/OPERATIONAL_SUPERVISOR_HARNESS.md" \
  "$ROOT/system_instructions/OPERATIONAL_SUPERVISOR.md" \
  "$SOURCE_DIR/system_instructions/"
cp -R "$ROOT/plugins/senpai" "$SOURCE_DIR/plugins/senpai"
chmod -R a-w "$SOURCE_DIR"

"$DOCKER_BIN" build \
  --file "$ROOT/tests/kubernetes/operational-supervisor-entrypoint.Dockerfile" \
  --build-arg "BASE_IMAGE=$ADVISOR_IMAGE" \
  --tag "$CANARY_IMAGE" \
  "$ROOT/tests/kubernetes"

OUTPUT=$("$DOCKER_BIN" run --rm \
  --read-only \
  --network none \
  --tmpfs /tmp:rw,nosuid,nodev,size=16m,mode=1777 \
  --tmpfs /var/lib/senpai:rw,nosuid,nodev,size=16m,uid=10001,gid=10001,mode=0700 \
  --mount "type=bind,src=$SOURCE_DIR,dst=/workspace/senpai,readonly" \
  --env RESEARCH_TAG=entrypoint-smoke \
  --env SENPAI_SKIP_EDITABLE_INSTALL=1 \
  --env SENPAI_OPENHANDS_TIMEOUT_SECONDS=37 \
  --env SENPAI_OPENHANDS_MAX_TURNS=11 \
  --env GITHUB_TOKEN=SENPAI_CI_DUMMY_GITHUB \
  --env WANDB_API_KEY=SENPAI_CI_DUMMY_WANDB \
  --env OPENAI_API_KEY=SENPAI_CI_DUMMY_OPENAI \
  --entrypoint /bin/bash \
  "$CANARY_IMAGE" \
  /workspace/senpai/k8s/entrypoint-operational-supervisor.sh)
[[ "$OUTPUT" == operational-supervisor-entrypoint-ok ]]

echo "Operational supervisor production entrypoint smoke passed"
