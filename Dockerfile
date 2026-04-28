FROM ghcr.io/coreweave/ml-containers/torch-extras:es-cuda-13-dev-99be449-base-cuda13.2.0-ubuntu22.04-torch2.10.0-vision0.25.0-audio2.10.0-abi1

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

# Install project Python dependencies into the image from the lockfile.
COPY pyproject.toml uv.lock /tmp/senpai/
RUN cd /tmp/senpai && \
    uv export --frozen --no-dev --no-emit-project --format requirements.txt > requirements.txt && \
    uv pip install --system -r requirements.txt

# Install Claude Code + gh
RUN curl -fsSL https://claude.ai/install.sh | bash || true && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null && \
    apt-get update && apt-get install -y gh && rm -rf /var/lib/apt/lists/*

# Add local bin to PATH
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspaces
