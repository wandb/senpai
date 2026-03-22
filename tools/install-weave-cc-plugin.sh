#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Install the Weave Claude Code plugin.
# Requires WANDB_ENTITY, WANDB_PROJECT, and WANDB_API_KEY in the environment.
# WANDB_ENTITY and WANDB_PROJECT come from the K8s ConfigMap (via launch.py).
# WANDB_API_KEY comes from the senpai-secrets K8s secret.

if ! command -v npm &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
npm install -g weave-claude-plugin
export WEAVE_PROJECT="${WANDB_ENTITY}/${WANDB_PROJECT}"
# WANDB_API_KEY is already exported by the pod — re-export to make it explicit
export WANDB_API_KEY="${WANDB_API_KEY}"
weave-claude-plugin install || echo "Warning: weave-claude-plugin install failed (non-fatal)"
