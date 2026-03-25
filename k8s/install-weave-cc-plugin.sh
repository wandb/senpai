#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Register the Weave Claude Code plugin at runtime.
# The Docker image already has the npm package, timeout patch, and base config.
# This script registers the marketplace + plugin with the `claude` CLI and
# persists the weave_project into settings.json for the daemon.
#
# Requires GITHUB_TOKEN and WANDB_API_KEY in the environment (from k8s secrets).
# WANDB_API_KEY is persisted to settings.json so the daemon subprocess can
# read it (env vars may not be inherited by background processes).

# Make git use GITHUB_TOKEN for HTTPS + SSH-style GitHub URLs (needed to clone
# the private wandb/claude_code_weave_plugin marketplace repo).
git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "git@github.com:"

# Register marketplace and install plugin (fully non-interactive).
claude plugin marketplace add wandb/claude_code_weave_plugin || true
claude plugin install weave@weave-claude-plugin --scope user || true

# Persist weave_project and API key into settings.json for the daemon.
weave-claude-plugin config set weave_project "${WANDB_ENTITY}/${WANDB_PROJECT}" || true
weave-claude-plugin config set wandb_api_key "${WANDB_API_KEY}" || true
