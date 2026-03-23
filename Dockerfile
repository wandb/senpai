FROM ghcr.io/coreweave/ml-containers/torch-extras:es-cuda-13-dev-99be449-base-cuda13.2.0-ubuntu22.04-torch2.10.0-vision0.25.0-audio2.10.0-abi1

# Install Node.js 22 + yq
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs netcat-openbsd && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq

# Install uv
RUN pip install uv

# Install Claude Code + gh
RUN curl -fsSL https://claude.ai/install.sh | bash || true && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null && \
    apt-get update && apt-get install -y gh && rm -rf /var/lib/apt/lists/*

# Install weave-claude-plugin, patch inactivity timeout (10 min → 12 h),
# register the hook, and write default config.
RUN npm install -g weave-claude-plugin && \
    (sed -i "s/const INACTIVITY_TIMEOUT_MS = 10 \* 60 \* 1_000;/const INACTIVITY_TIMEOUT_MS = 12 * 60 * 60 * 1_000;/" \
      "$(npm root -g)/weave-claude-plugin/dist/daemon.js" || true) && \
    (weave-claude-plugin install || true)

RUN mkdir -p /root/.weave_claude_plugin/logs && \
    cat > /root/.weave_claude_plugin/settings.json <<'EOF'
{
  "log_file": "/root/.weave_claude_plugin/logs/daemon.log",
  "weave_project": "wandb-applied-ai-team/senpai-v1",
  "wandb_api_key": null,
  "debug": false,
  "version": "0.1.0",
  "daemon_socket": "/root/.weave_claude_plugin/daemon.sock"
}
EOF

# Add local bin to PATH
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspaces
