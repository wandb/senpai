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
# WANDB_API_KEY is already available from the launch backend's secret and is
# picked up automatically by `--non-interactive`.

# The installer invokes `git clone git@github.com:...` internally. Fresh pods
# have no known_hosts entry for github.com, so rewrite SSH URLs to the
# already-token-authenticated HTTPS form set up in the deployment entrypoint.
git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "git@github.com:"

WEAVE_PROJECT="${WANDB_ENTITY}/${WANDB_PROJECT}" \
    weave-claude-plugin install --non-interactive

# `--non-interactive` reads WEAVE_PROJECT from the env for the install flow
# itself but does NOT persist it into settings.json (weave_project stays null),
# so the daemon won't actually start tracing. Write it explicitly.
weave-claude-plugin config set weave_project "${WANDB_ENTITY}/${WANDB_PROJECT}"
