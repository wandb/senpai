# Benchmark Integrity Patterns

Use this reference when writing no-nos, constraints, and results contracts. Be
specific: a benchmark rule that sounds obvious to a human can be invisible to an
agent under pressure.

## Split And Data Integrity

Pattern:

```markdown
The split manifest is part of the benchmark contract. Do not regenerate, edit,
rebalance, filter, or replace it during normal experiment PRs. A result that
changes the split is not comparable to the baseline.
```

Use when the repo has manifest files, hidden test labels, preprocessed shards,
or fixed public benchmark partitions.

## Hidden Test Access

Pattern:

```markdown
Do not inspect hidden test labels or tune against test metrics. The approved
final evaluator may compute test metrics at the end of a run; those metrics are
for reporting and merge decisions, not for adaptive per-seed or per-config
selection.
```

Use when test labels exist locally for automated evaluation but should not guide
experiment selection.

## Metric Semantics

Pattern:

```markdown
Do not rename, rescale, average, filter, or otherwise change the primary metric
to make a result appear better. If a new diagnostic is useful, log it under a
new key while preserving the existing primary metric contract.
```

Use when students may edit trainers or evaluators.

## Seed And Trial Cherry-Picking

Pattern:

```markdown
For final claims, predeclare the step count, seed count, and configuration, then
report all non-cherry-picked runs. Do not use validation loss to select the best
seed, stop time, or member of a final seed batch.
```

Use for stochastic benchmarks, optimizer speedruns, and statistical success
rules.

## Benchmark-Equivalent Training

Pattern:

```markdown
Keep the benchmark equivalent: do not change the dataset, model class, batch
size, number of forward/backward passes per optimizer step, or training budget
unless the advisor explicitly assigns a benchmark-contract experiment.
```

Use when only certain levers are allowed.

## Protected Evaluator Files

Pattern:

```markdown
Evaluation files are protected. Read them to understand the metric, but do not
edit them in normal experiment PRs. If an evaluator bug is suspected, ask the
advisor before changing it and report the issue separately from the experiment
result.
```

Use for scoring scripts, submission validators, manifest generation, hidden
ground-truth joins, and official benchmark helpers.

## External Source Restrictions

Pattern:

```markdown
The following sources are banned during this launch. Do not open, fetch, browse,
clone, cite, summarize, or use them for implementation ideas: <sources>. They
are comparison artifacts for humans after the run, not part of the active
experimental context.
```

Use when contamination or unfair post-launch information is a risk.

## Early Kill Gates

Pattern:

```markdown
Early kill gates are acceptable for crashes, non-finite losses, exploding
gradients, OOMs, or hopeless debug runs. They are not acceptable as a way to
hide required final metrics for serious confirmation runs.
```

Use when trainers expose kill thresholds or students can manually stop runs.
