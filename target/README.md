<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# target/

This directory is **empty by design**. senpai itself is problem-agnostic — all problem-specific code (model, training script, data pipeline, agent instructions) lives in a separate repo that you "bring" and mount here.

## How problem packages get here

At pod launch, `k8s/entrypoint-{advisor,student}.sh` clones the problem-package repo into `target/<name>/` using two env vars wired up by `k8s/launch.py`:

- `TARGET_REPO_URL` — the repo the agents commit to (defaults come from `senpai.yaml`, override with `--target_repo_url`)
- `TARGET_WORKING_BRANCH` — integration branch inside that repo (`--target_working_branch`)

The active problem path is `senpai.yaml`'s `problem:` field (e.g. `target/tandemfoil2`) — the entrypoint clones `TARGET_REPO_URL` into that path.

## Local development

Clone your problem-package repo here manually:

```bash
git clone -b <branch> https://github.com/<owner>/<repo>.git target/<name>
```

Then run `target/<name>/train.py` as usual. Everything under `target/*` (except this README) is gitignored in the senpai repo, so your local clone won't show up as dirty.

## Problem-package layout

A problem package is a normal repo with:

```
<problem>/
├── train.py               # training script + model (entry point for students)
├── data.py or data/       # data pipeline
├── program.md             # research context, metrics, file-edit boundaries
└── instructions/
    ├── prompt-advisor.md  # task-specific prompt template read by the advisor entrypoint
    └── prompt-student.md  # task-specific prompt template read by the student entrypoint
```

See [`morganmcg1/tandemfoil2`](https://github.com/morganmcg1/tandemfoil2) for a reference.
