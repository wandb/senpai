#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Install the Weave Claude Code plugin.
# Requires WANDB_ENTITY and WANDB_PROJECT to be set in the environment.

if ! command -v npm &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
npm install -g weave-claude-plugin
export WEAVE_PROJECT="${WANDB_ENTITY}/${WANDB_PROJECT}"
weave-claude-plugin install
