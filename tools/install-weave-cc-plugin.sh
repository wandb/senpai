#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Install the Weave Claude Code plugin.
# Requires WANDB_ENTITY, WANDB_PROJECT, and WANDB_API_KEY in the environment.
# WANDB_ENTITY and WANDB_PROJECT come from the K8s ConfigMap (via launch.py).
# WANDB_API_KEY comes from the senpai-secrets K8s secret.

# Configure SSH for GitHub using the mounted key
mkdir -p ~/.ssh
cp /run/secrets/ssh/id_ed25519 ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null

if ! command -v npm &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
apt-get install -y netcat-openbsd
npm install -g weave-claude-plugin
export WEAVE_PROJECT="${WANDB_ENTITY}/${WANDB_PROJECT}"
# WANDB_API_KEY is already exported by the pod — re-export to make it explicit
export WANDB_API_KEY="${WANDB_API_KEY}"
weave-claude-plugin install || echo "Warning: weave-claude-plugin install failed (non-fatal)"
