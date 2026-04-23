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

# The installer invokes `git clone git@github.com:...` internally. Fresh pods
# have no known_hosts entry for github.com, so rewrite SSH URLs to the
# already-token-authenticated HTTPS form set up in the deployment entrypoint.
git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "git@github.com:"

WEAVE_PROJECT="${WANDB_ENTITY}/${WANDB_PROJECT}" \
    weave-claude-plugin install --non-interactive
