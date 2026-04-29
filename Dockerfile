FROM ghcr.io/coreweave/ml-containers/torch-extras:bc8c66e-base-cuda13.2.1-ubuntu24.04-torch2.11.0-vision0.26.0-audio2.11.0-abi1

# Install Node.js 22 + yq
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs netcat-openbsd gettext-base && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq

# Install kubectl
RUN curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
      -o /usr/local/bin/kubectl && chmod +x /usr/local/bin/kubectl

# Install uv
RUN pip install uv

# Install project Python dependencies from pyproject.toml.
# Keep the CoreWeave image's prebuilt CUDA/PyTorch stack instead of replacing it
# with PyPI torch/CUDA wheels.
COPY pyproject.toml /tmp/senpai/
RUN cd /tmp/senpai && \
    uv pip compile pyproject.toml --format requirements.txt \
      --no-header \
      --no-annotate \
      --no-emit-package torch \
      --no-emit-package torchvision \
      --no-emit-package torchaudio \
      --no-emit-package triton \
      | grep -Ev '^(torch|torchvision|torchaudio|triton|nvidia-)' \
      > requirements.txt && \
    uv pip install --system -r requirements.txt && \
    python - <<'PY'
import sys
import torch
import torchvision

assert sys.version_info >= (3, 12), sys.version
assert torch.__version__.startswith("2.11.0"), torch.__version__
assert torchvision.__version__.startswith("0.26.0"), torchvision.__version__
PY

# Install Claude Code + gh
RUN curl -fsSL https://claude.ai/install.sh | bash || true && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null && \
    apt-get update && apt-get install -y gh && rm -rf /var/lib/apt/lists/*

# Install weave-claude-plugin and patch inactivity timeout (10 min → 12 h).
# `weave-claude-plugin install` must run at runtime (needs GitHub access to
# clone the marketplace repo), so entrypoint scripts handle that step.
RUN npm install -g weave-claude-plugin && \
    (sed -i "s/const INACTIVITY_TIMEOUT_MS = 10 \* 60 \* 1_000;/const INACTIVITY_TIMEOUT_MS = 12 * 60 * 60 * 1_000;/" \
      "$(npm root -g)/weave-claude-plugin/dist/daemon.js" || true)

RUN mkdir -p /root/.weave_claude_plugin/logs && \
    cat > /root/.weave_claude_plugin/settings.json <<'EOF'
{
  "log_file": "/root/.weave_claude_plugin/logs/daemon.log",
  "weave_project": null,
  "wandb_api_key": null,
  "debug": false,
  "version": "0.1.0",
  "daemon_socket": "/root/.weave_claude_plugin/daemon.sock"
}
EOF

# Add local bin to PATH
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspaces
