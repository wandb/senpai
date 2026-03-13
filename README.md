<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai

Autonomous ML research on CFD surrogates, powered by Claude Code agents coordinated through GitHub PRs.

![val/loss over time](scatter_plot.png)

[W&B Dashboard](https://wandb.ai/capecape/senpai)

## The problem

We're training a neural network surrogate for computational fluid dynamics (CFD) on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn) dataset. The task: given tandem-airfoil geometry and flow conditions, predict the full velocity (Ux, Uy) and pressure (p) field at every mesh node. Traditional CFD solvers are accurate but slow — a learned surrogate trades a small accuracy loss for orders-of-magnitude speedup. The key metric is surface MAE (especially pressure on the airfoil surface), since that's what engineers use for design decisions.

The model is a [Transolver](https://arxiv.org/abs/2402.02366) with physics-aware attention over irregular meshes.

## How it works

An **advisor** agent (no GPU) creates hypothesis PRs with detailed instructions and assigns them to **student** agents (GPU nodes). Students implement, run experiments, and report results on the PR. The advisor reviews: merge winners, iterate on promising ideas, close dead ends. Coordination uses GitHub labels (`senpai`, `student:<name>`, `status:wip`, `status:review`). W&B tracks metrics.

See `advisor.md`, `student.md`, and `program.md` for the full protocols.

## Architecture

```mermaid
graph TD
    subgraph K8s["Kubernetes Cluster"]
        A["Advisor Pod<br/>(Claude Code, no GPU)<br/>Creates hypothesis PRs<br/>Reviews results, merges/closes"]
        subgraph Students["Student Deployments (one per GPU node)"]
            S1["frieren<br/>8x GPU"]
            S2["fern<br/>8x GPU"]
            S3["tanjiro<br/>8x GPU"]
            S4["..."]
        end
        A -->|"GitHub PRs<br/>(draft → review → merge/close)"| Students
    end
    K8s --> GH["GitHub<br/>PRs = hypotheses<br/>Labels = routing"]
    K8s --> WB["Weights & Biases<br/>Metrics, runs, groups"]
```

### PR lifecycle

```mermaid
graph TD
    A["Advisor creates draft PR"] -->|"student:name + status:wip"| B["Student picks up PR"]
    B --> C["Implements hypothesis, runs experiments"]
    C -->|"status:review"| D["Advisor reviews"]
    D -->|Merge| E["Improvement lands on main"]
    D -->|Request changes| F["status:wip — student iterates"]
    D -->|Close| G["Dead end, branch deleted"]
    F --> B
```

## Key files

| File | Purpose |
|------|---------|
| `program.md` | Shared context: problem, constraints, metrics |
| `advisor.md` | Advisor protocol: hypotheses, review, merge/close |
| `student.md` | Student protocol: poll, implement, experiment, report |
| `train.py` / `transolver.py` | Training script and model (modifiable by students) |
| `.claude/skills/wandb-primary/` | W&B query skill (advisor + students) |
| `.claude/skills/list-experiments/` | Experiment log skill (advisor only, stripped from students) |

## References

`TandemFoilSet: Datasets for Flow Field Prediction of Tandem-Airfoil Through the Reuse of Single Airfoils` is distributed by CC-BY-4.0.
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
