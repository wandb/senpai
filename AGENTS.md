<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai - Development Context

Development of a problem-agnostic autonomous ML research loop for target ML
problem repositories. The current research programs are often CFD surrogate
experiments, but the runner should stay target-repo agnostic.

## User Clarifications

### Interviewing the developer about how to do a task:
When asked for a large piece of work that seems vague, consequential, or full
of hidden tradeoffs, ask the user detailed clarifying questions about the real
implementation choices: technical design, workflow, UX, risks, validation,
operations, and tradeoffs. Prefer non-obvious questions that expose constraints
or intent. When the answers change durable project behavior, write the learnings
to README.md or SPEC.md as appropriate.


## Coding guidelines and philosophy

- You should generate code that is simple and readable. Avoid unnecessary abstractions and complexity. This is a research codebase, so maintainability and clarity matter.
- Avoid overly defensive coding. No need for lots of `try`/`except` patterns, fallbacks, or backups. Prefer code that fails clearly when something is wrong so it can be fixed.
- Do not add demo-only flags or placeholder CLI options that gate real functionality (e.g., `--run` just to toggle execution); scripts should run their main logic directly.
- Adhere to Python 3.12+ conventions.

## Key docs

- `README.md` - operator-facing overview, launch examples, and problem-package layout.
- `SPEC.md` - target architecture and rewrite contract for the senpai orchestration loop.
- `senpai.yaml` - launch defaults, including the target repo, target branch, advisor branch, and `problem_dir`.
- `$PROBLEM_DIR/program.md` - authoritative target research context, goals, metrics, training constraints, and file boundaries. With the default config this is `target/program.md` after the target repo is cloned.
- `$PROBLEM_DIR/instructions/prompt-advisor.md` - target-specific advisor prompt.
- `$PROBLEM_DIR/instructions/prompt-student.md` - target-specific student prompt.
- `system_instructions/CLAUDE-ADVISOR.md` - advisor role workflow.
- `system_instructions/CLAUDE-STUDENT.md` - student role workflow.

## Architecture

- **Runner repo** - this repo. Owns orchestration, Kubernetes launch, role instructions, GitHub helpers, W&B integration, and operational docs.
- **Target repo** - cloned into `$PROBLEM_DIR` from `target_repo_url`. Owns the data code, training code, evaluation code, `program.md`, target prompts, and experiment branches. Agent commits and PRs land in the target repo, not in the runner repo.
- **Advisor pod** - no GPU, runs Claude Code in a loop. Queries W&B, reviews student PRs, generates new hypotheses, and creates draft PRs to assign work.
- **Student pods** - GPU workers running Claude Code. Poll for assigned PRs, implement the hypothesis inside the target repo boundaries, run training, and report results.
- **GitHub Issues** - human-to-agent communication channel. Agents poll for and respond to these alongside their normal PR workflow.
- **W&B** - canonical experiment metrics store for training runs, comparisons, and merge decisions.

## k8s layout

- `k8s/advisor-deployment.yaml` / `k8s/student-deployment.yaml` — pod specs
- `k8s/entrypoint-advisor.sh` / `k8s/entrypoint-student.sh` — startup scripts

## system_instructions/

Role-specific runtime instruction files. At pod launch, the entrypoint renders
the appropriate role file into the pod's root `CLAUDE.md` before invoking
Claude Code:

- `system_instructions/CLAUDE-ADVISOR.md` -> advisor pods
- `system_instructions/CLAUDE-STUDENT.md` -> student pods

The checked-in root `CLAUDE.md` and `AGENTS.md` share this development context
for local agent work. They are not the role-specific pod instructions.
