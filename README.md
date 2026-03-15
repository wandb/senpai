<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai

Autonomous neural network research on CFD surrogates, powered by Claude Code agents coordinated through GitHub PRs.

## The idea

We want to run autonomous ML research at scale: many hypotheses explored in parallel by AI agents, with all coordination happening through GitHub. No custom dashboards, no message queues — just PRs, labels, and code review.

An **advisor** agent (no GPU) acts as the research lead. It decides what to try, creates a GitHub PR for each hypothesis with detailed instructions, and assigns it to a **student** agent. Each student (full GPU node) picks up its assigned PR, implements the hypothesis, runs experiments, and reports results back on the PR. The advisor reviews: merge the winners into main, send promising ideas back for iteration, close dead ends.

GitHub is the single source of truth. Every hypothesis, every result, every decision is a PR. W&B tracks the experiment metrics.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                         │
│                                                                   │
│  ┌───────────────────┐                                            │
│  │     Advisor Pod    │  No GPU, lightweight                      │
│  │   (Claude Code)    │  Creates hypothesis PRs                   │
│  │                    │  Reviews results, merges/closes            │
│  └────────┬───────────┘                                           │
│           │ GitHub PRs (draft → review → merge/close)             │
│           │                                                       │
│  ┌────────▼───────────────────────────────────────────────────┐   │
│  │           Student Deployments (one per GPU node)            │   │
│  │                                                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │   │
│  │  │ frieren  │  │   fern   │  │ tanjiro  │  ...             │   │
│  │  │ 8x GPU   │  │ 8x GPU   │  │ 8x GPU   │                │   │
│  │  │          │  │          │  │          │                 │   │
│  │  │ Polls PR │  │ Polls PR │  │ Polls PR │                 │   │
│  │  │ Implements│  │ Implements│  │ Implements│                │   │
│  │  │ Reports  │  │ Reports  │  │ Reports  │                 │   │
│  │  └──────────┘  └──────────┘  └──────────┘                 │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
         │                              │
┌────────▼─────────┐          ┌─────────▼─────────┐
│     GitHub        │          │   Weights &        │
│                   │          │   Biases            │
│  PRs = hypotheses │          │                    │
│  Labels = routing │          │  Metrics, runs,    │
│  Reviews = coord  │          │  tags, groups      │
└───────────────────┘          └────────────────────┘
```

### Advisor

A lightweight pod (no GPU) running Claude Code. It does **not** write code or run experiments. Its job:

1. **Survey** — query W&B for current best metrics, list open PRs
2. **Review** — check PRs labeled `status:review`. Merge winners (squash), request changes on promising ideas, close dead ends
3. **Create hypotheses** — for each idle student, create a draft PR on an `exp/<name>` branch with a hypothesis, concrete instructions, and baseline metrics
4. **Repeat** every 5 minutes

See `advisor.md` for the full protocol.

### Students

Long-running K8s Deployments (one per GPU node) running Claude Code. Each student does **not** freelance — it only works on assigned PRs. Its job:

1. **Poll** — check for PRs labeled `student:<name>` + `status:wip`
2. **Implement** — check out the branch, follow the PR instructions, modify only `train.py` / `transolver.py`
3. **Experiment** — run training, collect metrics
4. **Report** — update the PR body with results, analysis, and suggested follow-ups
5. **Submit** — push, mark PR ready for review, swap label to `status:review`
6. **Wait** for the next assignment

See `student.md` for the full protocol.

### PR lifecycle

```
Advisor creates draft PR ──→ student:frieren + status:wip
        │
        ▼
Student picks up PR, implements, runs experiments
        │
        ▼
Student pushes results ──→ status:review
        │
        ▼
Advisor reviews:
  ├── Merge (squash) ──→ improvement lands on main
  ├── Request changes ──→ status:wip (student iterates)
  └── Close ──→ dead end, branch deleted
```

### Coordination via labels

| Label | Meaning |
|-------|---------|
| `senpai` | All senpai PRs |
| `student:<name>` | Assigned to this student |
| `status:wip` | Student is working on it |
| `status:review` | Ready for advisor review |

### Key files

| File | Purpose |
|------|---------|
| `program.md` | Shared context: problem, constraints, metrics, training output |
| `advisor.md` | Advisor workflow: create hypotheses, review results, merge/close |
| `student.md` | Student workflow: poll PRs, implement, experiment, report |
| `train.py` | Training script (modifiable by students) |
| `transolver.py` | Model architecture (modifiable by students) |
| `prepare.py` | Data loading (read-only) |

## Quick start

### 1. Create the K8s secret

```bash
kubectl create secret generic senpai-secrets \
  --from-literal=wandb-api-key=$WANDB_API_KEY \
  --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY \
  --from-literal=github-token=$GITHUB_TOKEN
```

### 2. Set up RBAC (for advisor pod)

```bash
kubectl apply -f k8s/orchestrator-rbac.yaml
```

### 3. Create GitHub labels

Create these labels on the repo: `senpai`, `status:wip`, `status:review`, and one `student:<name>` per student (e.g. `student:frieren`).

### 4. Launch

```bash
# Launch advisor + 3 students
python k8s/launch.py --tag mar13 --names "frieren,fern,tanjiro" \
  --wandb_entity wandb-applied-ai-team --repo_branch main

# Students only (no advisor)
python k8s/launch.py --tag mar13 --n_students 4 --students_only \
  --wandb_entity wandb-applied-ai-team

# Dry run to preview manifests
python k8s/launch.py --tag mar13 --names "frieren" --dry_run
```

### 5. Monitor

```bash
# Check running pods
kubectl get deployments -l app=senpai
kubectl get pod senpai-advisor

# Watch a student's logs
kubectl logs -f deployment/senpai-frieren

# Check PRs
gh pr list --label "senpai"

# Stop everything
kubectl delete deployments -l research-tag=mar13
kubectl delete pod senpai-advisor
```

## Dev Environment: Devpod

The devpod has global python installed, so you can directly run scripts with `python my_script.py`.

```bash
devpod up . --id senpai
devpod stop senpai     # stop (keeps data)
devpod delete senpai   # delete everything
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
