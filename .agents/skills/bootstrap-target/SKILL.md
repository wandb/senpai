---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: bootstrap-target
description: >
  Create or improve a Senpai target repository's program.md. Use this skill
  whenever the user wants to point Senpai at a fresh ML or research target
  repository, define the research objective, primary metric, benchmark contract,
  allowed edit boundaries, W&B reporting contract, or prepare a repo for
  autonomous advisor/student experiment loops.
argument-hint: "<target-repo-path-or-url>"
effort: high
---

# bootstrap-target

Turn an arbitrary ML or research repository into a Senpai-ready target package by writing `program.md`, the authoritative research contract. It is not generic documentation: it is the shared launch briefing for an autonomous research lab. Write it so the advisor knows what science to direct, the student knows what is legitimate to edit and run, and both roles agree on a valid result.

## References

Load these only when they help the current task:

- `references/program-template.md` - annotated `program.md` section template.
- `references/interview-question-bank.md` - question sets for ambiguous targets.
- `references/benchmark-integrity-patterns.md` - common benchmark no-nos and how
  to phrase them.

## Workflow

1. **Inspect the repository before writing.**
   Read the README, training and evaluation scripts, configs, data loaders,
   dependency files, benchmark docs, tests, and any existing experiment logs.
   Identify the actual commands, metric keys, data split logic, and files likely
   to be edited by experiment PRs.

2. **Infer the research contract.**
   Extract the objective, metric direction, validation/test distinction,
   benchmark constraints, data contract, model interface, resource limits,
   allowed levers, protected files, W&B/logging behavior, and result reporting
   needs. Prefer facts from code and docs over guesses.

3. **Interview the user when durable decisions are unclear.**
   Ask before encoding assumptions that will steer the research loop. Start with
   the few questions that decide whether Senpai will optimize the right thing:

   - What does success look like?
   - What exactly should Senpai optimize?
   - What is the primary metric, exact logged key/name, and direction?
   - Is that metric already measured by the repo? If not, where should it be
     added?
   - What validation metric selects checkpoints, and what test metric supports
     final claims?
   - What balance should Senpai strike between tuning known-good ideas and
     taking bigger research bets?
   - Are there related fields, domains, papers, benchmarks, or research
     communities the researcher agent should draw inspiration from when
     proposing experiments?
   - What changes are strictly forbidden: data leakage, benchmark rule changes,
     external sources, architecture changes, dependency changes, cherry-picking,
     hidden test access?
   - What files may students edit, and which files are protected?
   - What command should students run, and what time, GPU, or resource limits
     matter?
   - What baseline, SOTA, public reference, or statistical rule defines a
     meaningful win?

   If the target is still ambiguous after the initial repo inspection, read
   `references/interview-question-bank.md` and choose the smallest set of
   questions that resolves the missing metric, benchmark, data, operations, or
   file-boundary decisions. Do not ask the user to restate facts the repository
   already makes obvious.

4. **Write `program.md` as the research contract.**
   `program.md` is the document every advisor and student will use to decide
   what work is legitimate, what result matters, and what "better" means. It
   should feel like a senior researcher briefing an autonomous lab before the
   first experiment wave: concrete enough that a student can run the target
   without guessing, and opinionated enough that the advisor can design useful
   hypotheses instead of generic tweaks.

   Cover the target summary, mission, codebase map, data contract, model and
   training contract, run commands, metrics, benchmark integrity rules, result
   reporting, resource guidance, and advisor strategy. Explain why critical
   rules exist, especially around splits, metric finality, protected files, and
   benchmark equivalence. For a fuller skeleton, read
   `references/program-template.md`.

5. **Validate the target package.**
   Check that referenced paths exist, command patterns are plausible, metric names match code where possible, and protected files are explicit. If commands cannot be run locally, state what was verified statically.

## Writing Principles

Prefer concrete contracts over motivational prose. A student should know what
they may edit, what command to run, what metric matters, and what result format
to report.

Give enough domain context for real research judgment. A useful `program.md`
does not just say "optimize validation loss"; it explains what the model is
trying to learn, where the hard generalization axes are, and what kinds of ideas
are likely worth trying.

Do not hide multiple objectives behind one vague score. If important secondary
metrics can regress, name them and explain how the advisor should handle the
tradeoff.

Make benchmark integrity painfully clear. Spell out data leakage risks, split
immutability, forbidden shortcuts, external-source bans, seed/cherry-picking
rules, hidden-test rules, and metric finality.

Fail loudly on ambiguity. If the primary metric, target command, protected
files, or benchmark rules are unclear, interview the user before finalizing the
files.
