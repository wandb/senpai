#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Register the Weave Claude Code plugin at runtime.
# The Docker image already has the npm package, timeout patch, and base config.
# This script registers the marketplace + plugin with the `claude` CLI and
# writes the runtime-specific config values.
#
# Requires WANDB_ENTITY, WANDB_PROJECT, WANDB_API_KEY, and GITHUB_TOKEN.

# Make git use GITHUB_TOKEN for HTTPS + SSH-style GitHub URLs (needed to clone
# the private wandb/claude_code_weave_plugin marketplace repo).
git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "git@github.com:"

# Export WEAVE_PROJECT so the install command picks it up (skips interactive prompt).
export WEAVE_PROJECT="${WANDB_ENTITY}/${WANDB_PROJECT}"

# Register marketplace and install plugin (fully non-interactive with env vars set).
claude plugin marketplace add wandb/claude_code_weave_plugin || true
claude plugin install weave@weave-claude-plugin --scope user || true

# Persist project and API key into settings.json so the daemon reads them
# when started as a background subprocess (env vars may not be inherited).
weave-claude-plugin config set weave_project "${WANDB_ENTITY}/${WANDB_PROJECT}" || true
weave-claude-plugin config set wandb_api_key "${WANDB_API_KEY}" || true
