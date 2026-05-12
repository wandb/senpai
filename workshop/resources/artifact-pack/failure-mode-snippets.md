# Failure-Mode Snippets

## Context-Amplified Monitoring

Bad pattern:

```bash
tail -f /tmp/trial1.log /tmp/trial2.log |
  grep -E --line-buffered '"epoch"|Traceback|OOM|NaN|best_checkpoint'
```

Why it fails:

- It can match every epoch.
- Each event wakes the model.
- Each wake can reload a large cached context.

Better pattern:

```bash
python scripts/summarize_training_status.py \
  --emit-on new_best,error,complete,timeout \
  --min-interval-seconds 300
```

## Prompt-Induced GitHub Scope Error

Bad prompt example:

```bash
gh pr view 2417 --comments
```

Better helper:

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
pr_all_comments 2417
```

## State Drift Across Ledgers

Conflicting evidence:

- PR label says `status:wip`.
- W&B run finished.
- PR has no terminal `SENPAI-RESULT`.
- Compaction summary says training is still running.

Correct response:

- Re-read live PR labels and comments.
- Query W&B state.
- Treat compaction summary as a cache.

## Validation-Only Win

Evidence:

- Validation improves from `4.62` to `4.42`.
- Test regresses from `6.24` to `6.31`.

Correct decision:

- Do not merge under a held-out test merge contract.
