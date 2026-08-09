---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: bootstrap-target
description: >
  Create or improve Senpai target-repository onboarding files: program.md plus
  instructions/prompt-advisor.md and instructions/prompt-student.md. Use this
  skill whenever the user wants to point Senpai at a fresh ML or research target
  repository, define the research objective, primary metric, benchmark contract,
  allowed edit boundaries, W&B reporting contract, advisor/student prompts, or
  prepare a repo for autonomous advisor/student experiment loops.
argument-hint: "<target-repo-path-or-url>"
model: claude-opus-4-8
effort: high
---

# bootstrap-target

Turn an arbitrary ML or research repository into a Senpai-ready target package.

The output is usually three files in the target repo:

- `program.md` - the authoritative research contract.
- `instructions/prompt-advisor.md` - the target-specific advisor overlay.
- `instructions/prompt-student.md` - the target-specific student overlay.

These files are not generic documentation. They are the launch briefing for an
autonomous research lab. Write them so the advisor knows what kind of science to
direct, the student knows what is legitimate to edit and run, and both roles
agree on what counts as a valid result.

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

5. **Write short role overlays with target flavor.**
   The files in `instructions/` should not repeat the global Senpai loop. They
   are overlays: concise, target-specific steering that helps the advisor and
   student inhabit this particular research program.

   `prompt-advisor.md` should contain only target-specific experiment-selection
   judgment: promising hypothesis families, metric tradeoffs, target-specific
   mistakes, and context worth reading first.

   `prompt-student.md` should contain only target-specific implementation
   guidance: important files, protected invariants, exact run-command patterns,
   and interpretation pitfalls.

   Do not repeat runtime identity, branch or W&B setup, generic role workflow,
   tool instructions, or reporting rules already supplied by the control plane
   and `program.md`.

6. **Validate the target package.**
   Check that referenced paths exist, command patterns are plausible, metric
   names match code where possible, protected files are explicit, and the role
   overlays do not contradict `program.md`. If commands cannot be run locally,
   state what was verified statically.

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

Keep `instructions/` light. The global Senpai role files already define the
loop; target prompts should only add target-specific setup, commands, and
judgment.

Fail loudly on ambiguity. If the primary metric, target command, protected
files, or benchmark rules are unclear, interview the user before finalizing the
files.
