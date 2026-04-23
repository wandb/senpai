#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Register the Weave Claude Code plugin at runtime.
# The Docker image already has the npm package and timeout patch
# (Dockerfile: `npm install -g weave-claude-plugin`).
# This call creates ~/.weave_claude_plugin/settings.json, registers the
# plugin with the Claude Code CLI, and persists weave_project for the daemon.
#
# WANDB_API_KEY is already in the pod env (from senpai-secrets) and is picked
# up automatically by `--non-interactive`.

WEAVE_PROJECT="${WANDB_ENTITY}/${WANDB_PROJECT}" \
    weave-claude-plugin install --non-interactive
