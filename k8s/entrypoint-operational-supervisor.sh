#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail
umask "${SENPAI_UMASK:-0077}"

source /workspace/senpai/k8s/handoff-operational-supervisor-secrets.sh
handoff_operational_supervisor_secrets

WORKDIR="/workspace/senpai"
STATE_DIR="/var/lib/senpai/$RESEARCH_TAG/operational-supervisor"
mkdir -p "$STATE_DIR"

export SENPAI_ROLE="supervisor"
export SENPAI_OPENHANDS_WORKSPACE="$WORKDIR"
export SENPAI_OPENHANDS_STATE_DIR="$STATE_DIR/openhands_state"
export SENPAI_OPENHANDS_HARNESS_FILE="$WORKDIR/system_instructions/OPERATIONAL_SUPERVISOR_HARNESS.md"
export SENPAI_OPENHANDS_ROLE_FILE="$WORKDIR/system_instructions/OPERATIONAL_SUPERVISOR.md"
export SENPAI_PLUGIN="$WORKDIR/plugins/senpai"
export SENPAI_SUPERVISOR_STATE_DIR="$STATE_DIR"
export SENPAI_OPENHANDS_TIMEOUT_SECONDS="${SENPAI_OPENHANDS_TIMEOUT_SECONDS:-900}"
export SENPAI_OPENHANDS_MAX_TURNS="${SENPAI_OPENHANDS_MAX_TURNS:-80}"

cd "$WORKDIR"
if [ "${SENPAI_SKIP_EDITABLE_INSTALL:-0}" != "1" ]; then
    uv pip install --python "$SENPAI_PYTHON" --no-deps -e .
fi

exec /usr/local/bin/senpai-run-controller operational-supervisor
