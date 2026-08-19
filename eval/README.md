<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Fast Senpai agent eval

This eval runs one advisor and one student against each fixed target:

- `morganmcg1/modded-nanogpt-senpai`
- `morganmcg1/TandemFoilSet-Balanced`

Every Senpai model profile uses `openai/gpt-5.6-luna` at `high` effort. Each
training process has a hard 20-minute process-group timeout by default. One
absolute six-hour deadline starts when the launcher creates the manifest and
includes preflight, provisioning, readiness, and agent work. A cluster-side
cutoff proves that it can reach Kubernetes and the shared PVC before any GPU
resources launch. It opens a shared start gate after all four pods become Ready
and removes both tracks at the absolute deadline. Readiness consumes the same
six-hour budget; it does not extend it.

The harness pins each target branch and makes the launcher create the advisor
branch from that exact commit. NanoGPT is pinned to `master`. TandemFoil is
temporarily pinned to the compatibility branch from
[PR #4631](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/4631),
which makes `WANDB_RUN_GROUP` a deterministic default. After that PR merges,
move the pin back to `main` and record its merge commit intentionally.

The eval launches real target-repository advisor branches and allows the agents
to create normal experiment PRs. Use a GitHub credential with write access to
both targets.

## Launch

Build and publish immutable Senpai advisor, student, and cutoff images for the
revision under test. Put the advisor/student image references, cluster, PVC,
W&B project, and credentials in `senpai.local.yaml` as for a normal launch.
Then run:

```bash
uv run python eval/run.py launch \
  --config-path senpai.local.yaml \
  --training-timeout-minutes 20 \
  --total-timeout-hours 6 \
  --no-web-search
```

Built-in browser, Exa, AlphaXiv, and delegated search-agent access is off by
default for the eval. Pass `--web-search` for the search-enabled variant. This
does not install a Kubernetes egress policy; generic terminal network access
remains available.

`launch` returns after it creates the cutoff and both tracks. It writes the run
manifest under the gitignored `eval/results/` directory and prints the matching
report command. Add `--wait` to keep the local command attached until the
cluster-side cutoff completes and then generate the report automatically.
Both target configs complete image-pin validation plus credential, repository,
and W&B preflight before the cutoff or target resources are created. A failed
or interrupted partial launch immediately removes both tagged tracks. If
target cleanup fails, the independent cutoff remains armed. GitHub advisor
branches and shared routing labels created before a launch failure remain for
operator inspection.

Use `--cutoff-image IMAGE` when the cutoff image cannot be derived from the
current checkout. Use `--dry-run` to render all resources without accessing the
cluster or credentials.

## Report

After the cutoff, run the command printed by `launch`, or:

```bash
uv run python eval/run.py report --run-id eval-YYYYMMDD-HHMMSS-abcdef
```

The reporter queries the exact W&B group assigned to each target, writes JSON
and Markdown locally, and logs one aggregate W&B run. It publishes to W&B only
after Kubernetes confirms that the cutoff completed. Pass `--no-wandb` for a
local partial preview. Reports record role and cutoff images, Senpai and target
revisions, the evaluator hash, deadline, cutoff outcome, and readiness counts.

The NanoGPT score is the first step to target from one completed trial, gated
by the final validation loss and the repository's statistical-significance
rule. `-1`, best-intermediate loss, multi-trial runs, and incomplete histories
do not score. The TandemFoil score is the full held-out
`test_avg/mae_surf_p`; the reporter requires all four finite test split values
and recomputes the equal-weight mean. It never substitutes validation MAE for a
missing test result.

These are cooperative development evals. The agents can edit the target
training and metric code, so the results detect regressions but are not a
tamper-resistant benchmark. A protected benchmark would score emitted
predictions or checkpoints outside the editable target repository.
