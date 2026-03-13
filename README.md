<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai

Autonomous neural network research on CFD surrogates, powered by Claude Code agents running on Kubernetes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                    │
│                                                              │
│  ┌──────────────────┐                                        │
│  │   Orchestrator    │  No GPU, lightweight                  │
│  │   (Claude Code)   │  Reads journals, queries W&B          │
│  │                   │  Launches/stops agent pods             │
│  └────────┬──────────┘                                       │
│           │ kubectl                                          │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────────┐   │
│  │              Agent Pods (one per GPU node)              │  │
│  │                                                        │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐               │  │
│  │  │  pepe   │  │  agata  │  │ claudio │  ...           │  │
│  │  │ 8x GPU  │  │ 8x GPU  │  │ 8x GPU  │               │  │
│  │  │         │  │         │  │         │               │  │
│  │  │ Claude  │  │ Claude  │  │ Claude  │               │  │
│  │  │ Code    │  │ Code    │  │ Code    │               │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘               │  │
│  └───────┼────────────┼────────────┼─────────────────────┘  │
│          │            │            │                         │
│  ┌───────▼────────────▼────────────▼─────────────────────┐  │
│  │                  Shared PVC                            │  │
│  │                                                        │  │
│  │  /mnt/new-pvc/                                         │  │
│  │    datasets/tandemfoil/    ← training data             │  │
│  │    senpai/journals/        ← research journals (*.md)  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    Weights &       │
                    │    Biases          │
                    │                   │
                    │  Metrics, runs,   │
                    │  tags, groups     │
                    └───────────────────┘
```

### Components

**Agent pods** — Each agent is a full GPU node (8x GPU) running Claude Code autonomously. It reads `program.md`, modifies the model/training code, runs experiments, and iterates. Agents use git worktrees to run multiple experiments in parallel across their 8 GPUs.

**Orchestrator** — A lightweight pod (no GPU) that manages the fleet. It reads agent journals, queries W&B for metrics, and can launch/stop agents via kubectl.

**Shared PVC** — Persistent volume mounted on all pods. Holds the training dataset and agent research journals.

**W&B** — All training runs log to a shared W&B project. Agents use `--agent <name>` (stored in config and tags) for filtering. `--wandb_group` groups iterations on the same idea.

**Research journals** — Each agent maintains a markdown journal at `/mnt/new-pvc/senpai/journals/<agent>.md` with hypotheses, results, and plans. Agents read each other's journals to avoid duplicating work.

## Quick start

### 1. Create the K8s secret

```bash
kubectl create secret generic senpai-secrets \
  --from-literal=wandb-api-key=$WANDB_API_KEY \
  --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY \
  --from-literal=github-token=$GITHUB_TOKEN
```

### 2. Launch research agents

```bash
# Launch 3 agents by name
python k8s/launch.py --tag mar13 --names "pepe,agata,claudio" \
  --wandb_entity capecape --repo_branch k8s-service

# Launch 4 agents (picks from the name pool)
python k8s/launch.py --tag mar13 --n_agents 4 --wandb_entity capecape
```

### 3. Deploy the orchestrator (optional)

```bash
kubectl apply -f k8s/orchestrator-rbac.yaml
kubectl apply -f k8s/orchestrator-pod.yaml
```

### 4. Monitor

```bash
# Check running agents
kubectl get jobs -l app=senpai

# Watch an agent's logs
kubectl logs -f job/senpai-pepe

# Check what an agent is doing
kubectl exec -it <pod-name> -- ps aux | grep python

# Read research journals
kubectl exec <pod-name> -- cat /mnt/new-pvc/senpai/journals/pepe.md

# Stop all agents
kubectl delete jobs -l research-tag=mar13
```

## Dev Environment: Devpod

The devpod has global python installed, so you can directly run scripts with `python my_script.py`, no need to use `uv` or `venv`.

1. Start the devbox:
```bash
devpod up . --id senpai
```

If the pod does not schedule, set the provider options once (CPU allocatable is slightly under 128 on the 8x GPU nodes, so we cap at 120). Also ensure the pod template mounts `/dev/shm`:
```bash
devpod provider set-options cw-cfd -o POD_MANIFEST_TEMPLATE=.devcontainer/pod-template.yaml
devpod provider set-options cw-cfd -o RESOURCES=limits.nvidia.com/gpu=8,requests.nvidia.com/gpu=8,limits.cpu=120,requests.cpu=120,limits.memory=960Gi,requests.memory=960Gi
```
The pod template adds a 32Gi `/dev/shm` mount for the devpod container.

2. Stop the devbox

Stop when not in use to release GPU resources:
```bash
devpod stop senpai
```
This deletes the pod (frees GPU) but keeps your data. Run `devpod up` again to resume.

To delete everything (pod + data):
```bash
devpod delete senpai
```

## References
`TandemFoilSet: Datasets for Flow Field Prediction of Tandem-Airfoil Through the Reuse of Single Airfoils`
is distributed by CC-BY-4.0.
```bibtex
@inproceedings{
lim2026tandemfoilset,
title={**TandemFoilSet**: Datasets for Flow Field Prediction of Tandem-Airfoil Through the Reuse of Single Airfoils},
author={Wei Xian Lim and Loh Sher En Jessica and Zenong Li and Thant Zin Oo and Wai Lee Chan and Adams Wai-Kin Kong},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=4Z0P4Nbosn}
}
```
