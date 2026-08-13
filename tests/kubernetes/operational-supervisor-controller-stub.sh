#!/bin/sh

set -eu

test "$#" -eq 1
test "$1" = operational-supervisor
test "$(id -u)" = 10001
test "$PWD" = /workspace/senpai
test "$SENPAI_ROLE" = supervisor
test "$SENPAI_OPENHANDS_WORKSPACE" = /workspace/senpai
test "$SENPAI_OPENHANDS_STATE_DIR" = \
  /var/lib/senpai/entrypoint-smoke/operational-supervisor/openhands_state
test "$SENPAI_SUPERVISOR_STATE_DIR" = \
  /var/lib/senpai/entrypoint-smoke/operational-supervisor
test "$SENPAI_OPENHANDS_HARNESS_FILE" = \
  /workspace/senpai/system_instructions/OPERATIONAL_SUPERVISOR_HARNESS.md
test "$SENPAI_OPENHANDS_ROLE_FILE" = \
  /workspace/senpai/system_instructions/OPERATIONAL_SUPERVISOR.md
test "$SENPAI_PLUGIN" = /workspace/senpai/plugins/senpai
test "$SENPAI_OPENHANDS_TIMEOUT_SECONDS" = 37
test "$SENPAI_OPENHANDS_MAX_TURNS" = 11
test ! -w /workspace/senpai
test -r "$SENPAI_OPENHANDS_HARNESS_FILE"
test -r "$SENPAI_OPENHANDS_ROLE_FILE"
test -d "$SENPAI_PLUGIN"
test -w "$SENPAI_SUPERVISOR_STATE_DIR"

test -z "${GITHUB_TOKEN+x}"
test -z "${GH_TOKEN+x}"
test -z "${WANDB_API_KEY+x}"
test -z "${OPENAI_API_KEY+x}"
test -z "${ANTHROPIC_API_KEY+x}"
test -d "$SENPAI_SUPERVISOR_SECRET_DIR"
test "$(stat -c %a "$SENPAI_SUPERVISOR_SECRET_DIR")" = 700
test "$(cat "$SENPAI_SUPERVISOR_SECRET_DIR/GITHUB_TOKEN")" = \
  SENPAI_CI_DUMMY_GITHUB
test "$(cat "$SENPAI_SUPERVISOR_SECRET_DIR/WANDB_API_KEY")" = \
  SENPAI_CI_DUMMY_WANDB
test "$(cat "$SENPAI_SUPERVISOR_SECRET_DIR/OPENAI_API_KEY")" = \
  SENPAI_CI_DUMMY_OPENAI
test ! -e "$SENPAI_SUPERVISOR_SECRET_DIR/ANTHROPIC_API_KEY"
test "$(stat -c %a "$SENPAI_SUPERVISOR_SECRET_DIR/GITHUB_TOKEN")" = 600

touch "$SENPAI_SUPERVISOR_STATE_DIR/entrypoint-smoke-ok"
printf '%s\n' operational-supervisor-entrypoint-ok
