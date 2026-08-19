<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Fast Senpai agent eval

This eval runs independent advisor/student replications against each fixed
target:

- `morganmcg1/modded-nanogpt-senpai`
- `morganmcg1/TandemFoilSet-Balanced`

Every Senpai model profile uses `openai/gpt-5.6-luna` at `high` effort. Each
training process has a hard 20-minute process-group timeout by default. One
absolute six-hour deadline starts when the launcher creates the manifest and
includes preflight, provisioning, readiness, and agent work. A cluster-side
cutoff proves that it can reach Kubernetes and the shared PVC before any GPU
resources launch. The default three trials create six GPU students and six
advisors. The cutoff opens one shared start gate after all 12 pods become Ready
and removes every trial at the absolute deadline. Trials run in parallel and
share the same six-hour budget; the budget is not multiplied by the trial
count. Readiness consumes that budget and does not extend it.

The harness pins each target branch and makes the launcher create a distinct
advisor branch for every trial from that exact commit. Each trial also has a
unique student name, Kubernetes research tag, exact W&B group, and deterministic
seed. Both roles receive the zero-based replication index and seed as
`SENPAI_TRIAL_INDEX` and `SENPAI_TRIAL_SEED`. NanoGPT is temporarily pinned to
the eval-contract branch from
[PR #2474](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2474).
TandemFoil is temporarily pinned to the eval-contract branch from
[PR #4632](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/4632).
After each PR merges, move its pin back to the repository default branch and
record the merge commit intentionally.

The eval launches real target-repository advisor branches and allows the agents
to create normal experiment PRs. Use a GitHub credential with write access to
both targets.

## Launch

Build and publish immutable Senpai advisor, student, and cutoff images for the
revision under test. Put the advisor/student image references, cluster, PVC,
and credentials in `senpai.local.yaml` as for a normal launch. Eval training
runs and the aggregate report always use W&B entity
`wandb-applied-ai-team` and project `senpai_eval`. Then run:

```bash
uv run python eval/run.py launch \
  --config-path senpai.local.yaml \
  --training-timeout-minutes 20 \
  --total-timeout-hours 6 \
  --n-trials 3 \
  --no-web-search
```

Built-in browser, Exa, AlphaXiv, and delegated search-agent access is off by
default for the eval. Pass `--web-search` for the search-enabled variant. This
does not install a Kubernetes egress policy; generic terminal network access
remains available.

`launch` returns after it creates the cutoff and every trial. It writes the run
manifest under the gitignored `eval/results/` directory and prints the matching
report command. Add `--wait` to keep the local command attached until the
cluster-side cutoff completes and then generate the report automatically.
Every trial config completes image-pin validation plus credential, repository,
and W&B preflight before the cutoff or target resources are created. The
launcher then creates the isolated trials concurrently. A failed or interrupted
partial launch waits for all concurrent launch commands to settle and removes
every tagged trial. If
target cleanup fails, the independent cutoff remains armed. GitHub advisor
branches and shared routing labels created before a launch failure remain for
operator inspection.

The eval passes no extra operator instructions. Each pinned target's discovered
`program.md` is the complete scientific contract. The launcher supplies only
dynamic controls through environment variables: group, trial identity, seed,
and hard timeout.

Each trial uses `gh_history_scope: fresh`. The advisor and student clone only
that trial's advisor branch with `--single-branch --depth 1 --no-tags`, then
keep the remote restricted to that branch. This removes the earlier Git graph
from the model's local checkout for a clean-history ablation. It is not a
security boundary: the current files, GitHub API, and W&B evidence remain
available through their normal capabilities.

Use `--cutoff-image IMAGE` when the cutoff image cannot be derived from the
current checkout. Use `--dry-run` to render all resources without accessing the
cluster or credentials.

## Report

After the cutoff, run the command printed by `launch`, or:

```bash
uv run python eval/run.py report --run-id eval-YYYYMMDD-HHMMSS-abcdef
```

The reporter queries the exact W&B group assigned to each target trial, writes
JSON and Markdown locally, and logs one aggregate W&B run. It publishes to W&B
only after Kubernetes confirms that the cutoff completed. Pass `--no-wandb`
for a local partial preview. Reports record role and cutoff images, Senpai and
target revisions, every trial's group, branch, seed and resource identity, the
evaluator and adjudicator hashes, deadline, cutoff outcome, and readiness
counts. A completed report fails before scoring if the current reporter or
adjudicator bytes do not match the hashes recorded at launch.

The NanoGPT score is the first step to target from one completed trial, gated
by the final validation loss and the repository's statistical-significance
rule. `-1`, best-intermediate loss, multi-trial runs, and incomplete histories
do not score. The TandemFoil score is the full held-out
`test_avg/mae_surf_p`; the reporter requires all four finite test split values
and recomputes the equal-weight mean. It never substitutes validation MAE for a
missing test result.

Metric-valid W&B runs are candidates, not final results. A candidate must be a
finished full-data run with the exact trial group, seed, timeout, clean Git
commit, protected data hashes, and target metric contract. Each trial manifest
starts with this unresolved decision:

```json
{
  "status": "pending",
  "selected_run_id": null,
  "evidence": {}
}
```

After the cutoff, the reporter freezes and atomically persists every exact
advisor-branch head before it starts adjudication. It inspects only PRs merged
into that branch and only schema-valid
`senpai-result` markers written by the authenticated GitHub identity. A result
must be successful; match the exact repository, assignment, PR head and source
commit; report the declared metric and direction; improve its recorded
baseline; and reference one matching finished W&B candidate with the same
score. The selected result is the unique ancestry-latest qualifying merge in
the frozen branch. A cleanup PR without valid result evidence cannot replace a
winner. Spoofed, unmerged, test, reduced-data, stale-contract, and ambiguous
results remain unscored.

This uses the advisor's guarded merge as the semantic review: the advisor has
already inspected the code and W&B evidence before changing the research
baseline. A second same-family OpenHands judge would repeat that judgment and
add cost and variance, so it is not part of the default eval. If the advisor
merges code that silently changes metric semantics, that is an agent-eval
failure rather than a reporter override. The deterministic reporter records the
full decision evidence and never promotes the raw group minimum by itself.
The first persisted semantic decision is immutable. A later report reuses the
same frozen head and fails if GitHub or W&B changes would alter that decision.

Final distributions contain at most one accepted selection per trial, so
agents that emit more runs receive no extra weight. The JSON report includes
the selected-score list, mean, median, range, population variance, and
population standard deviation. Markdown summarizes accepted counts, the mean,
standard deviation, and range. The aggregate W&B run logs a native trial table
and a per-target score scatter plot when accepted results exist. Dashboard
counters distinguish accepted trials, which contribute to score distributions,
from adjudicated trials, which include both accepted and rejected decisions.
Pending trials count as neither.

These are cooperative development evals. The agents can edit the target
training and metric code, so the merge review and provenance gates detect many
mistakes but do not form a tamper-resistant benchmark. A protected TandemFoil
benchmark would score prediction artifacts with an evaluator outside the
editable target repository. A protected NanoGPT benchmark would separate the
editable optimizer/model section from a pinned validation loop and scorer.
