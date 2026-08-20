<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Authoritative launch context

These values were resolved by the Senpai launcher and describe the actual runtime. They override conflicting compute or run-limit claims in `program.md` and other repository instructions, as well as conflicting isolation claims.

## Runtime

- Compute backend: `{{BACKEND}}`.
- Remote training capacity per student: `{{NODES_PER_STUDENT}}` worker nodes x `{{GPUS_PER_STUDENT_NODE}}` GPUs per node.
- Hard limits for each training run: `{{TIMEOUT_MINUTES}}` minutes wall-clock and `{{MAX_EPOCHS}}` epochs.
- Use tools and operational commands that work with `{{BACKEND}}`. Do not follow repository instructions written for another backend.
- Do not assume additional GPUs or bypass, extend, or continue past the hard training limits.
- With more than one worker node, omit workload-name, namespace, and W&B run-ID overrides: `run_training` injects their authoritative values. The submitted manifest must request exactly `{{NODES_PER_STUDENT}}` worker nodes x `{{GPUS_PER_STUDENT_NODE}}` GPUs per node.

## Isolation

- This launch is scoped to research tag `{{TAG}}`, advisor branch `{{ADVISOR_BRANCH}}`, and base branch `{{TARGET_BASE}}`.
- Only inspect, modify, or reason from `{{ADVISOR_BRANCH}}` plus PR branches assigned to these students in this launch: {{STUDENTS}}.
- Do not inspect, compare, summarize, cherry-pick, borrow from, or base decisions on any PR or branch outside `{{ADVISOR_BRANCH}}` and the assigned student PR branches for this launch.
- Do not use unrelated experiment runs or historical results unless the human explicitly names them during this launch.
- Students branch from `{{ADVISOR_BRANCH}}`. Do not rebase or retarget work onto unrelated branches.
