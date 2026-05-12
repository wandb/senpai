# Instructor Guide

Workshop: **PRs as Hypotheses, W&B as Memory: Building Observable Autoresearch for Physical AI**

Duration: 3 hours

## Preparation

Open these files before teaching:

- `workshop/slides.md`
- `workshop/notebooks/00_environment_check.py`
- `workshop/notebooks/01_llm_calls_to_hypotheses.py`
- `workshop/notebooks/02_tool_calls_and_contracts.py`
- `workshop/notebooks/03_single_agent_student_loop.py`
- `workshop/notebooks/04_multiagent_advisor_student_flow.py`
- `workshop/notebooks/05_wandb_and_weave_as_memory.py`
- `workshop/notebooks/06_k8s_autoresearch_dry_run.py`
- `workshop/notebooks/07_physical_ai_claim_review.py`
- `workshop/notebooks/08_full_autoresearch_trace.py`
- `workshop/resources/artifact-pack/`

Open repo anchors:

- `README.md`
- `k8s/launch.py`
- `k8s/entrypoint-advisor.sh`
- `k8s/entrypoint-student.sh`
- `plugins/senpai/scripts/senpai-gh.sh`
- `plugins/senpai/scripts/senpai-pr-guard.py`
- `system_instructions/CLAUDE-ADVISOR.md`
- `system_instructions/CLAUDE-STUDENT.md`
- `papers/icml/AGENT_FAILURE_MODES1.2.md`

## Timing Plan

| Time | Segment | Slides | Notebook |
| --- | --- | --- | --- |
| 0:00-0:12 | Executive framing | 1-7 | none |
| 0:12-0:30 | LLM calls | 8-12 | `01` |
| 0:30-0:55 | Tool calls | 13-18 | `02` |
| 0:55-1:20 | Single agents | 19-24 | `03` |
| 1:20-1:30 | Break | none | none |
| 1:30-1:55 | Multiagent flow | 25-30 | `04` |
| 1:55-2:20 | Observability memory | 31-36 | `05` |
| 2:20-2:45 | K8s runtime | 37-40 | `06` |
| 2:45-3:00 | Claim review and capstone | 41-42 | `07` / `08` |

## Opening Script

Say:

> This workshop is about letting AI agents participate in real ML research without losing control of the research record. The hard part is not getting an LLM to suggest a hyperparameter. The hard part is building a system where hypotheses, code changes, GPU runs, metrics, and decisions remain auditable after thousands of autonomous actions.

Then anchor:

> Autoresearch is a distributed ML control system, not a long prompt.

Avoid:

- “Fully autonomous scientist.”
- “The agent figures everything out.”
- “This replaces researchers.”

Use:

- “Autonomous research harness under human-auditable review.”
- “Agent as co-worker inside a ledger.”
- “Workflow invariants moved into tools.”

## Audience-Specific Emphasis

### W&B-Heavy

Spend more time on:

- `workshop/notebooks/05_wandb_and_weave_as_memory.py`
- `workshop/notebooks/08_full_autoresearch_trace.py`
- `workshop/resources/artifact-pack/wandb-run-summary.json`
- `workshop/resources/artifact-pack/weave-trace-excerpt.json`

Message: W&B is metric/config/provenance truth; Weave is agent/tool trace truth.

### CoreWeave-Heavy

Spend more time on:

- `workshop/notebooks/06_k8s_autoresearch_dry_run.py`
- `k8s/student-deployment.yaml`
- `k8s/student-claude-watchdog.sh`

Message: runtime is research policy: GPUs, PVCs, restarts, dry-runs, and teardown shape the safe autonomy boundary.

### Research Lead

Spend more time on:

- `workshop/notebooks/07_physical_ai_claim_review.py`
- `analysis/AIRFRANS_BENCHMARK.md`
- `analysis/DRIVAERML_BENCHMARK.md`
- `analysis/TANDEMFOILSET_BENCHMARK.md`

Message: completed training is not the same as a defensible physical-AI claim.

### DevRel

Spend more time on:

- `workshop/slides.md`
- `workshop/resources/optional-live-demo.md`
- `workshop/resources/legacy-labs.md`

Message: keep the story candid, technical, and failure-aware.

## Transition Lines

From LLM calls to tools:

> A hypothesis is cheap. The first engineering question is how the model observes the world and acts without breaking workflow state.

From tools to agents:

> Once tools exist, the next question is who can use which tool, in what loop, and when they stop.

From agents to multiagent flow:

> A single student can run one experiment. A research program needs a queue, a reviewer, and a way to keep GPUs busy without losing attribution.

From multiagent flow to observability:

> Once many agents are acting, memory cannot live in any one context window.

From observability to K8s runtime:

> The ledger tells us what happened. The runtime determines what can happen safely at scale.

## Common Misconceptions

| Misconception | Correction |
| --- | --- |
| A good model can reason out the best experiment. | It can propose candidates; tools and ledgers supply current evidence. |
| More autonomy means fewer constraints. | Production autonomy requires sharper role boundaries. |
| Multiagent means agents chat with each other. | SENPAI agents coordinate through PRs, labels, W&B, and git. |
| The trace is the truth. | The trace explains behavior; W&B and git hold metric/code truth. |
| Kubernetes is deployment detail. | In autoresearch, runtime is policy. |

## Closing Script

Use:

> If you remember one thing, remember this: autoresearch becomes credible when every autonomous action leaves behind a human-auditable artifact. The agent can propose, edit, run, and review, but the system has to preserve the hypothesis, code, metrics, trace, and decision in places both humans and agents can inspect.
