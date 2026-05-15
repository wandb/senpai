#!/usr/bin/env bash
set -euo pipefail

target_repo_url="https://github.com/morganmcg1/modded-nanogpt-senpai.git"
target_repo_branch="senpai-launch-20260515"
extra_instructions="launches/auto-nanogpt-extra-instructions.md"
wandb_entity="wandb-applied-ai-team"
wandb_project="modded-nanogpt-senpai"
timeout_minutes="30240"
max_epochs="100000"

env_file="${SENPAI_ENV_FILE:-}"
if [[ -z "${env_file}" ]]; then
  if [[ -f ".env" ]]; then
    env_file=".env"
  elif [[ -f "${HOME}/ML/senpai/.env" ]]; then
    env_file="${HOME}/ML/senpai/.env"
  fi
fi
if [[ -n "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

if [[ -n "${SENPAI_PYTHON:-}" ]]; then
  launch_cmd=("${SENPAI_PYTHON}")
elif python -c "import simple_parsing" >/dev/null 2>&1; then
  launch_cmd=(python)
elif command -v uv >/dev/null 2>&1; then
  launch_cmd=(uv run python)
else
  echo "Could not find Python with simple_parsing. Install the Senpai deps or set SENPAI_PYTHON." >&2
  exit 1
fi

for rep in 1 2 3 4 5; do
  tag="auto-nanogpt-r${rep}"
  "${launch_cmd[@]}" k8s/launch.py \
    --tag "${tag}" \
    --advisor \
    --target_repo_url "${target_repo_url}" \
    --target_repo_branch "${target_repo_branch}" \
    --advisor_branch "${tag}" \
    --gh_history_scope fresh \
    --n_students 8 \
    --student_prefix "r${rep}" \
    --gpus_per_student 8 \
    --timeout_minutes "${timeout_minutes}" \
    --max_epochs "${max_epochs}" \
    --wandb_entity "${wandb_entity}" \
    --wandb_project "${wandb_project}" \
    --extra_instructions "${extra_instructions}" \
    "$@"
done
