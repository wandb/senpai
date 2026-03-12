#!/bin/bash

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -e
set -o pipefail

# Install project dependencies (skip if no pyproject.toml yet)
if [ -f pyproject.toml ]; then
  uv pip install --system -e .
fi

# install node + codex
export NVM_DIR="$HOME/.nvm"
if [ ! -d "$NVM_DIR" ]; then
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
fi

# shellcheck disable=SC1090
. "$NVM_DIR/nvm.sh"
nvm install --lts
nvm use --lts

npm i -g @openai/codex

# install claude code
curl -fsSL https://claude.ai/install.sh | bash

echo "Done!"
