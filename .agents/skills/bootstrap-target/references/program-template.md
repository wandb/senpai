# program.md Template

Use this as an annotated skeleton, not a rigid form. Keep the voice direct,
specific, and operational.

## Title And Target Summary

Name the target plainly. In the first paragraph, say what inputs come in, what outputs are predicted or optimized, and what research problem `program.md` governs.

Example:

```markdown
# <Target Name>

Research target for <domain/task>. Given <inputs>, predict or optimize
<outputs>. The baseline is <model/system>. Beat it while preserving <benchmark
contract>.
```

## Mission

State the real objective in human terms. Do not only say "minimize loss." Say
whether the goal is to beat a public benchmark, reduce step count, improve OOD
generalization, preserve benchmark equivalence, or discover a better modeling
recipe.

Make the metric direction explicit and distinguish steering metrics from final
claim metrics.

Useful phrasing:

```markdown
The goal is not merely to improve the starter model. The goal is to find
changes that reduce `<test metric>` while preserving the split, data, and
evaluation contract. Validation metrics are for steering; final claims must use
`<final test metric>`.
```

## Codebase

List important files with edit boundaries and why they matter. This is not a
directory tour; it is a guardrail against invalid experiments.

Use labels like:

- **Primary editable entrypoint.**
- **Editable for experiment PRs.**
- **Read-only during normal experiment PRs.**
- **Protected. Never touch this file.**
- **Dependency file. Add packages only in the same PR that uses them.**

Mention benchmark docs and historical records if advisors should read them
before assigning work.

## Data

Describe the data location, sample structure, feature channels, target channels,
normalization, masks/padding, and anything domain-specific that helps an agent
reason about errors.

Include tables for feature dimensions and target channels when the code exposes
structured tensors. Explain split design and which files define it. If the split
is part of the benchmark contract, say that changing it invalidates results.

## Model, Training, And Evaluation Contract

Document the callable interface and invariants that evaluation depends on:

- input and output tensor shapes
- normalized versus original target spaces
- masking and padding behavior
- checkpoint-selection metric
- final validation/test evaluation behavior
- whether models may change architecture
- whether training scripts may change optimizer, schedule, or data sampling

Explain why the invariants matter. For example, if metrics exclude padding,
state that losses and metrics computed over padded rows are invalid.

## Running

Give exact command patterns. Include setup steps only when they are genuinely
needed for Senpai pods or reproducibility.

Use W&B naming in the command:

```bash
cd "$PROBLEM_DIR" && <train command> \
  --wandb_name "$STUDENT_NAME/<short-description>" \
  --wandb_group "<hypothesis-or-pr>"
```

Explain debug, screening, and confirmation run lengths. Resource ceilings such
as `SENPAI_TIMEOUT_MINUTES`, `SENPAI_MAX_EPOCHS`, and GPU counts are hard
limits, not instructions to run until exhaustion.

## W&B Metrics And Telemetry

List required metric keys exactly as logged. Separate:

- primary validation/checkpoint metric
- final test metric
- secondary diagnostics
- health telemetry such as gradients, slopes, memory, non-finite counts, or
  benchmark-specific success margins

Tell agents to preserve metric names when possible so advisors can compare runs
across PRs.

## Metrics And Success Criteria

Name the primary ranking metric and direction. If a statistical rule, public
reference, or SOTA target defines a meaningful win, write it down.

If secondary metrics matter, explain how to treat tradeoffs:

```markdown
A run that improves the aggregate but regresses `<critical metric>` has not
solved the problem. The advisor may still use it as a follow-up direction, but
should not treat it as a clean winner.
```

## Benchmark Integrity

Spell out the no-nos. Common examples:

- do not modify split manifests or hidden labels
- do not peek at test data except through the approved final evaluator
- do not change metric definitions to make results look better
- do not use per-seed early stopping or validation peeking to cherry-pick runs
- do not change architecture, batch size, data, or training budget when the
  benchmark forbids it
- do not use banned external sources during the run

Be concrete. If a banned source or file is known, name it.

## Experiment Length And Resource Guidance

Explain how advisors and students should choose run lengths:

- tiny debug runs verify code, memory, logging, and finite gradients
- short screening runs compare uncertain variants
- longer confirmation runs support final claims or seed batches

Warn against both failure modes: only short runs can discard ideas too early,
while only long runs wastes throughput.

## Results Contract

Define the required result marker. Keep the JSON single-line and name the
metric keys used by this target:

```markdown
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<run-id>"],"primary_metric":{"name":"<metric>","value":<number>},"test_metric":{"name":"<metric>","value":<number>}}
```

Also require the exact command, W&B run IDs, baseline comparison, metric table,
known caveats, what happened, and suggested follow-ups.

## Advisor Guidance

Give the advisor a research personality for this target. Mention whether to
favor exploitation, big bets, literature-inspired ideas, ablations, or balanced
portfolios.

Good guidance is specific:

```markdown
Keep the portfolio balanced. Retune learning rate and weight decay when a new
optimizer idea deserves fair treatment, but do not spend the whole run on scalar
hyperparameter search.
```

## Roles

End with the target's coordination model when it adds useful context:

```markdown
Research is coordinated through GitHub PRs with an advisor/student model. GitHub
Issues are used for communication with the human researcher team.
```
