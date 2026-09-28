<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Authoritative launch context

These values were resolved by the Senpai launcher and describe the actual runtime. They override conflicting compute or run-limit claims in `program.md` and other repository instructions, as well as conflicting isolation claims.

## Runtime identity

- Role: `{{ROLE}}`.
- GitHub repository: `{{GH_REPO}}`.
- Advisor branch: `{{ADVISOR_BRANCH}}`.
- W&B project: `{{WANDB_ENTITY}}/{{WANDB_PROJECT}}`.
- Students in scope: `{{STUDENTS}}`.

## Runtime

- Compute backend: `{{BACKEND}}`.
- Training capacity per student: `{{NODES_PER_STUDENT}}` worker nodes x `{{GPUS_PER_STUDENT_NODE}}` GPUs per node.
- Training execution: {{TRAINING_EXECUTION}}.
- Training image: `{{TRAINING_IMAGE}}`.
- Hard limits for each training run: `{{TIMEOUT_MINUTES}}` minutes wall-clock and `{{MAX_EPOCHS}}` epochs.
- Use tools and operational commands that work with `{{BACKEND}}`. Do not follow repository instructions written for another backend.
- Do not assume additional GPUs or bypass, extend, or continue past the hard training limits.
- Student controllers have no GPUs. Pass an ordinary training command to `run_training`; Senpai creates and supervises the worker workload.
- Training runs in a separate worker using the selected training image `{{TRAINING_IMAGE}}` and your committed code. Use the project's normal dependency files or include any required setup in the submitted command. {{TRAINING_IMAGE_INSPECTION}} The standard image provides a writable target environment for project dependencies.
- Your command runs once per worker node. For distributed training, use `NODE_RANK`, `NNODES`, `MASTER_ADDR`, `MASTER_PORT`, and `GPUS_PER_NODE` with your framework's launcher; Senpai does not automatically parallelize a single-process script.
- Before training, check that the datasets specified in `program.md` are present and readable. Write checkpoints and outputs beneath the worker's `SENPAI_TRAINING_OUTPUT_DIR` on the shared volume; this launch's outputs live under `{{TRAINING_OUTPUT_ROOT}}`. The advisor and student can read those outputs after the worker exits.
- Use `get_cluster_capacity` for an advisory snapshot when an observer is configured. Check its observation time, age, and all worker resources. Unknown or stale data does not establish availability; resource-fit counts do not reserve nodes or authorize a launch. The scheduler remains authoritative. The kubectl proxy cannot run cluster-read helpers; use `get_training_status` for your existing run.
- Omit workload-name, namespace, and W&B run-ID overrides: `run_training` supplies these values and the configured compute allocation.

## Isolation

- This launch is scoped to research tag `{{TAG}}`, advisor branch `{{ADVISOR_BRANCH}}`, and base branch `{{TARGET_BASE}}`.
- Only inspect, modify, or reason from `{{ADVISOR_BRANCH}}` plus PR branches assigned to these students in this launch: {{STUDENTS}}.
- Do not inspect, compare, summarize, cherry-pick, borrow from, or base decisions on any PR or branch outside `{{ADVISOR_BRANCH}}` and the assigned student PR branches for this launch.
- Do not use unrelated experiment runs or historical results unless the human explicitly names them during this launch.
- Students branch from `{{ADVISOR_BRANCH}}`. Do not rebase or retarget work onto unrelated branches.
