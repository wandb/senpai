#!/usr/bin/env bash

# Exercise the actual CUDA student image without rebuilding it or requiring a GPU.

set -Eeuo pipefail

DOCKER_BIN=${DOCKER_BIN:-docker}
STUDENT_IMAGE=${STUDENT_IMAGE:?STUDENT_IMAGE must name the loaded production image}
SMOKE_ID=${SENPAI_STUDENT_SMOKE_ID:-local-$$}
SMOKE_ID=$(printf '%s' "$SMOKE_ID" | tr -cd '[:alnum:]-' | cut -c1-40)
[[ -n "$SMOKE_ID" ]] || { echo "invalid student smoke ID" >&2; exit 2; }
CONTAINER="senpai-student-smoke-$SMOKE_ID"
EXECUTOR=/usr/local/bin/senpai-repair-executor
EXECUTOR_SOCKET=/tmp/senpai-repair-executor-smoke/executor.sock
PYTHON=/opt/senpai-venv/bin/python

cleanup() {
  local status=$?
  trap - EXIT
  "$DOCKER_BIN" rm --force "$CONTAINER" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

"$DOCKER_BIN" run --detach --name "$CONTAINER" \
  --entrypoint "$PYTHON" "$STUDENT_IMAGE" \
  -I "$EXECUTOR" serve --socket "$EXECUTOR_SOCKET" >/dev/null

for _ in $(seq 1 30); do
  if "$DOCKER_BIN" exec "$CONTAINER" "$PYTHON" -I "$EXECUTOR" \
    health --socket "$EXECUTOR_SOCKET"; then
    break
  fi
  sleep 0.2
done
"$DOCKER_BIN" exec "$CONTAINER" "$PYTHON" -I "$EXECUTOR" \
  health --socket "$EXECUTOR_SOCKET"

"$DOCKER_BIN" exec "$CONTAINER" /bin/sh -ceu '
  test "$(id -u)" = 10001
  test ! -w /opt/senpai-venv/bin/python
  test ! -w /usr/local/bin/senpai-repair-executor
  package=$(/opt/senpai-venv/bin/python -I -c \
    "import pathlib,senpai_agent; print(pathlib.Path(senpai_agent.__file__).parent)")
  case "$package" in /opt/senpai-venv/*) ;; *) exit 1 ;; esac
  test ! -w "$package"
'

RESULT=$(printf 'printf student-repair-ok' | \
  "$DOCKER_BIN" exec -i "$CONTAINER" "$PYTHON" -I "$EXECUTOR" client \
    --socket "$EXECUTOR_SOCKET" --cwd /tmp --timeout 5)
[[ "$RESULT" == *'"exit_code":0'* && "$RESULT" == *'"stdout":"student-repair-ok"'* ]]

CHILD_RESULT=$(printf 'sleep 300 & printf "$!"' | \
  "$DOCKER_BIN" exec -i "$CONTAINER" "$PYTHON" -I "$EXECUTOR" client \
    --socket "$EXECUTOR_SOCKET" --cwd /tmp --timeout 5)
CHILD_PID=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["stdout"])' \
  "$CHILD_RESULT")
[[ "$CHILD_PID" =~ ^[0-9]+$ ]]
"$DOCKER_BIN" exec "$CONTAINER" test ! -e "/proc/$CHILD_PID"
"$DOCKER_BIN" exec "$CONTAINER" "$PYTHON" -I "$EXECUTOR" \
  health --socket "$EXECUTOR_SOCKET"

echo "Student production image smoke passed: immutable runtime and isolated repair lifecycle"
