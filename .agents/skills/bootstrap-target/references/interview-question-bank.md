# Interview Question Bank

Ask only the questions needed to resolve durable ambiguity. Prefer questions
that expose tradeoffs over questions that ask the user to summarize files you
can read.

## Success And Research Strategy

- What does success look like?
- What would make this research run feel like a clear win rather than a modest
  cleanup?
- What balance should Senpai strike between tuning known-good ideas and taking
  bigger research bets?
- Are there related fields, domains, papers, benchmarks, or research
  communities the researcher agent should draw inspiration from when proposing
  experiments?
- Are there approaches you already believe are promising, over-explored, or
  off-limits?

## Metrics

- What exactly should Senpai optimize?
- What is the primary metric, exact logged key/name, and direction?
- Is that metric already measured by the repo? If not, where should it be
  added?
- What validation metric selects checkpoints?
- What test metric supports final claims?
- Are there secondary metrics that must not regress, even if the primary score
  improves?
- Is there a statistical rule, seed policy, or confidence threshold for final
  claims?

## Benchmark Integrity

- Which files define the split, hidden labels, evaluator, or benchmark contract?
- What changes would invalidate a result even if the metric improves?
- Are students allowed to change the model architecture, data processing,
  optimizer, batch size, number of steps, dependencies, or evaluation code?
- Are any external sources, upstream repos, papers, branches, or post-launch
  updates banned during the run?
- Is validation peeking, seed selection, early stopping, or adaptive trial
  selection allowed?

## Data And Domain Context

- Where does data live in Senpai pods?
- What are the input and target shapes?
- What do feature and target channels mean?
- Which splits test in-distribution performance, OOD generalization, or
  paper-facing claims?
- Are there known hard regimes, rare cases, domain constraints, or failure
  modes that advisors should account for?

## Operations

- What command should students run for debug, screening, and confirmation runs?
- What GPU count, wall-clock timeout, epoch/step budget, and memory limit apply?
- What W&B entity/project/group naming should be used?
- What baseline should new PRs compare against?
- What result fields must appear in student PR comments?

## File Boundaries

- Which files are primary editable entrypoints?
- Which files are read-only context?
- Which files are protected and should never be touched in normal experiment
  PRs?
- Are dependency changes allowed? If yes, where should they be recorded?
