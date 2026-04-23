# W&B Primary Training Diagnostics Adoption Audit

Date: 2026-04-23

## Scope

This audit covers the available `conversation_logs` window around the request date:

- Present date folders: `2026-04-21`, `2026-04-22`
- Missing date folders in the requested window: `2026-04-20`, `2026-04-23`
- Archives scanned: `137` `.claude.tgz`
- Embedded transcript files scanned: `1,388` `.jsonl`

Method:

1. Enumerate transcript-bearing archives across both dates.
2. Search assistant, user, tool-result, and subagent records separately.
3. Exclude the injected `wandb-primary` skill body itself when asking whether agents *used* the diagnostics helpers.
4. Treat quoted PR text, GitHub comments, or other copied external text as weaker evidence than assistant-authored reasoning or commands.

## Headline Findings

### 1. The skill is being launched, but the packaged diagnostics helpers are not visibly being executed.

High-confidence findings:

- `41` assistant-side `wandb-primary` skill launches were found in the audited window.
- `0` assistant-side invocations/imports were found for the concrete helper surface:
  - `plot_single_run_overview`
  - `compare_runs_curves`
  - `plot_run_comparison`
  - `curve_features`
  - `lr_schedule_features`
  - `grad_norm_features`
  - `grad_histogram_features`
  - `list_candidate_step_keys`
  - `guess_step_key_from_workspace`

Interpretation:

- The skill is successfully triggered.
- The current skill/reference shape is not reliably converting that trigger into actual helper usage.

### 1b. There is also a concrete environment mismatch that can block helper adoption.

Audit-time verification in the detected `uv` environment showed:

- `wandb`: available
- `weave`: available
- `numpy`: available
- `matplotlib`: available
- `pandas`: missing

Why this matters:

- `training_diagnostics.py` and `curve_plots.py` import `pandas` at module import time.
- A missing `pandas` means the diagnostics helper stack fails before any actual analysis begins.

Interpretation:

- Some of the observed non-adoption may be behavioral.
- Some may be plain runtime friction: the helpers are not fully available in the detected project environment.

### 2. The heuristics from `TRAINING_DIAGNOSTICS.md` *are* being used conceptually, often with high fidelity.

Strong or medium evidence showed repeated manual use of:

- step-axis correction and per-step scheduler reasoning
- LR-peak spike analysis
- plateau vs convergence distinctions
- overfit after best-checkpoint reasoning
- stability-aware run ranking
- distinguishing normal restart blips from real instability
- late-training grad-norm spike / "gradient storm" reasoning

Interpretation:

- Conceptual adoption is high.
- Tool-level adoption is low.

## Representative Evidence

### Direct skill-launch evidence

Representative archives with repeated `wandb-primary` launches:

- `.agents/skills/wandb-primary` launched from advisor review subagents inside:
  - `conversation_logs/2026-04-21/pai-2/advisor_restart_1822/advisor.claude.tgz`
  - `conversation_logs/2026-04-21/pai-2/advisor_restart_2233/senpai-advisor-668b68b9f5-7bshd_root_.claude.tgz`
  - `conversation_logs/2026-04-21/pai-2/advisor_restart_2356/senpai-advisor-9887f4d66-sbg2l_root_.claude.tgz`
  - `conversation_logs/2026-04-22/k8s_pod_archive_185147/shouko/root_.claude.tgz`

What did **not** show up after excluding injected skill text:

- no assistant-authored `plot_single_run_overview(...)`
- no assistant-authored `compare_runs_curves(...)`
- no assistant-authored `plot_run_comparison(...)`
- no assistant-authored `from training_diagnostics import ...`
- no assistant-authored `from curve_plots import ...`

### Heuristic adoption evidence

#### Step-axis / x-axis semantics

- `conversation_logs/2026-04-21/pai-2/luffy__senpai-luffy-787df5ff8c-lmxsw__claude.tgz`
  - `.claude/projects/-workspace-senpai/bec791d0-8f08-4d65-8375-4a9f6cbb1a64.jsonl`
  - `2026-04-21T05:04:24.299Z`
  - The agent explicitly corrects the interpretation to scheduler steps per training step rather than per epoch and recalculates effective cosine-cycle frequency.

- `conversation_logs/2026-04-22/k8s_pod_archive_185147/edward/root_.claude.tgz`
  - `.claude/projects/-workspace-senpai/d7b38b16-3d3d-48b6-bc73-3ea736251bc5.jsonl`
  - `2026-04-21T12:40:37.803Z`
  - The agent revises trough analysis because the schedule operates per-step rather than per-epoch.

#### Instability / spike reasoning

- `conversation_logs/2026-04-22/k8s_pod_archive_185147/shouko/root_.claude.tgz`
  - `.claude/projects/-workspace-senpai/eedbb179-9614-463e-b810-c2abc2555c32.jsonl`
  - `2026-04-21T13:05:30.468Z`
  - The agent ties a large loss spike and grad-norm spike directly to an LR peak.

- `conversation_logs/2026-04-22/k8s_pod_archive_185147/chrome/root_.claude.tgz`
  - `.claude/projects/-workspace-senpai/f88f2795-8aa8-4568-9ca7-0e1e482b9a10.jsonl`
  - `2026-04-22T00:35:58.701Z`
  - The agent diagnoses a late "gradient storm" and ranks runs by continued convergence vs disruption.

- `conversation_logs/2026-04-22/k8s_pod_archive_185147/chrome/root_.claude.tgz`
  - `.claude/projects/-workspace-senpai/f88f2795-8aa8-4568-9ca7-0e1e482b9a10.jsonl`
  - `2026-04-22T03:05:32.980Z`
  - The agent separates normal restart spikes from true instability by checking immediate recovery vs sustained damage.

#### Plateau / convergence / overfit reasoning

- `conversation_logs/2026-04-21/pai-2/advisor__senpai-advisor-7649cf9685-b6zjc__claude.tgz`
  - `.claude/projects/-workspace-senpai/0d93f336-86f5-4b78-818d-220df0606f82.jsonl`
  - `2026-04-21T10:05:46.573Z`
  - The advisor segments the run into rapid descent, oscillating descent, and near-convergence, while noting slight overfit after the best epoch.

- `conversation_logs/2026-04-21/pai-2/kill_immediate_1816/emma.claude.tgz`
  - `.claude/projects/-workspace-senpai/f78343e3-0d09-4f4a-af66-1e8391b5d88e.jsonl`
  - Multiple assistant-authored records describe the trough envelope as plateauing around `0.011-0.014` and explicitly compare basin quality across runs.

- `conversation_logs/2026-04-22/k8s_pod_archive_185147/nezuko/root_.claude.tgz`
  - `.claude/projects/-workspace-senpai/f793db55-c95b-4fab-ac18-dd470ad14ffa.jsonl`
  - `2026-04-22T03:49:51.635Z`
  - The agent compares successive troughs and labels a slight overfitting trend.

#### Structured assistant-authored curve writeups

- `conversation_logs/2026-04-21/pai-2/fern__senpai-fern-5c9766557c-7f8xv__claude.tgz`
  - `.claude/projects/-workspace-senpai/ca0d1e4a-1a54-4aa7-965f-52bfa9fdde3d.jsonl`
  - `2026-04-21T07:39:08.774Z`
  - Assistant-authored PR comment uses a verdict, trough-envelope progression, grad-clip stats, and a plateau diagnosis.

- `conversation_logs/2026-04-21/pai-2/fern__senpai-fern-5c9766557c-7f8xv__claude.tgz`
  - `.claude/projects/-workspace-senpai/8adc937e-3a08-4a36-b1ee-2e649f9c92b9.jsonl`
  - `2026-04-21T08:32:15.572Z`
  - Assistant-authored comment includes a cosine-trough table, grad-norm trajectory table, and "no divergence / slower convergence" diagnosis.

- `conversation_logs/2026-04-21/pai-2/senpai-tanjiro-75b9747f59-wspwz.claude.tgz`
  - `.claude/projects/-workspace-senpai/e9a7e93f-576d-4c45-b9f2-f35c6afa8b05.jsonl`
  - `2026-04-21T03:05:33.024Z`
  - Assistant-authored results comment breaks the run into warmup vs post-warmup phases and ties best points to LR troughs.

- `conversation_logs/2026-04-21/pai-2/senpai-shoya-7bcc8d6d47-sb677.claude.tgz`
  - `.claude/projects/-workspace-senpai/8135e86a-f825-4f6f-9146-66bbc8fa7a29.jsonl`
  - `2026-04-21T07:15:47.659Z`
  - Assistant-authored comment includes a grad-norm stabilization table and a still-descending trough envelope diagnosis.

- `conversation_logs/2026-04-22/k8s_pod_archive_185147/megumi/root_.claude.tgz`
  - `.claude/projects/-workspace-senpai/9e00b322-16cd-4680-88ac-897a72c8189e/subagents/agent-a1643199da48054f5.jsonl`
  - `2026-04-22T07:28:52.373Z`
  - Assistant gives a healthy three-phase diagnosis, identifies a noisy plateau, and prefers the best checkpoint over the final one.

## Diagnosis: Why adoption is uneven

### Gap 1. The skill is easy to launch but too easy to use passively.

Evidence:

- `41` launches, `0` visible helper calls.
- Agents often stop after querying W&B and then manually narrate the curves.

Theory:

- The current skill explains the workflow well, but it does not create enough pressure to *execute* it.
- In practice, "I know what good and bad curves look like" beats "I should run the helper suite first."

### Gap 2. The step-axis trap is real, but the current workflow does not feel mandatory enough.

Evidence:

- Luffy and Edward both had to correct their own analysis once they reasoned through per-step schedule semantics manually.

Theory:

- Agents are learning the lesson conceptually, but not associating it strongly enough with the `step_axis.py` helpers.
- When the skill says "confirm `step_key`" without a hard failure mode, some agents still reason from epoch timing first.

### Gap 3. Agents use natural phrases that are not directly mapped to helper calls.

Evidence:

- Common log phrases included "trough envelope plateaued," "gradient storm," "restart disruption," "still descending at cutoff," and "best checkpoint not final."

Theory:

- Those are exactly the right intuitions, but the skill mostly names the helper API surface, not the phrase-to-tool mapping.
- If the agent cannot translate "gradient storm" into `grad_norm_features + lr_schedule_features + overview plot`, they keep analyzing manually.

### Gap 4. There is no strong completion gate.

Evidence:

- Many assistant-authored analyses are thoughtful and detailed, but omit one or more of:
  - explicit `step_key`
  - helper-derived feature table
  - generated PNG
  - verdict tied to concrete step ranges

Theory:

- Without a completion gate, agents can stop once the narrative feels good enough.

## Changes in This PR

### 1. Add a one-shot diagnostics CLI

File:

- `.agents/skills/wandb-primary/scripts/curve_diagnostics_cli.py`

Why:

- The logs suggest that the current helper surface is too high-friction for routine use.
- A single command that validates `step_key`, prints helper tables, and renders PNGs lowers the activation energy from "I know the workflow" to "I actually ran it."

### 2. Make the workflow more mandatory

Changes to:

- `.agents/skills/wandb-primary/SKILL.md`

Why:

- The new wording makes curve analysis a default workflow rather than a polite suggestion.
- The new completion gate is intended to prevent "query W&B, then narrate by hand" from counting as done.

### 3. Add stop signs and phrase-to-helper mapping

Changes to:

- `.agents/skills/wandb-primary/SKILL.md`
- `.agents/skills/wandb-primary/references/TRAINING_DIAGNOSTICS.md`

Why:

- This meets agents where the logs show they already are.
- The skill now explicitly maps natural research phrases like "gradient storm" and "trough envelope plateaued" onto the helper calls that should back those claims.

### 4. Strengthen step-axis and restart guidance

Changes to:

- `.agents/skills/wandb-primary/references/TRAINING_DIAGNOSTICS.md`

Why:

- Step-axis mistakes and restart interpretation showed up multiple times in the logs.
- The updated reference now says those are mandatory checks, not just background nuance.

### 5. Add explicit dependency preflight guidance

Changes to:

- `.agents/skills/wandb-primary/SKILL.md`
- `.agents/skills/wandb-primary/scripts/curve_diagnostics_cli.py`

Why:

- The audit found that the detected `uv` environment currently lacks `pandas`, even though the diagnostics stack depends on it.
- The skill now tells agents to verify imports before using the helpers, and the CLI now fails with a clear install message instead of a raw traceback.

## Expected Impact

If the changes work as intended, we should see:

1. More assistant-side helper execution and fewer manual-only curve diagnoses.
2. More analyses that explicitly name `step_key`.
3. More outputs with helper-derived tables plus PNGs, rather than only prose.
4. Fewer epoch-based misreads of step-based schedulers.

## Caveat

This is a log-based audit. It measures what was visible in conversation and tool transcripts. If an agent ever executed diagnostics logic in a way that never surfaced in the logs, this audit would not see it. Even with that caveat, the signal here is strong: conceptual adoption is already happening, while packaged-helper adoption is not yet happening reliably.
