#!/usr/bin/env bash
# Build the Gemma Challenge Senpai runner image.
#
# This image uses the official HF benchmark base (`vllm/vllm-openai`) and adds
# Senpai agent tooling. Push requires a Docker login with GHCR package write
# access for ghcr.io/wandb/senpai.

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/wandb/senpai:gemma-vllm}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai}"
PUSH="${PUSH:-false}"

docker build \
  --pull \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${IMAGE}" \
  .

if [ "${PUSH}" = "true" ]; then
  docker push "${IMAGE}"
fi
