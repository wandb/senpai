# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

FROM ghcr.io/tcapelle/dev_box:latest

# Install senpai Python dependencies
COPY pyproject.toml /tmp/senpai/pyproject.toml
RUN uv pip install --system -e /tmp/senpai

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

# The repo is cloned at startup (not baked in) so agents always get fresh code
COPY k8s/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
