<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai — Development Context

Development of an Autonomous ML research loop on CFD surrogates

## User Clarifications

### Interviewing the developer about how to do a task:
When asked for a large piece of work which seems vague or needs clarification, please interview me in detail using the AskUserQuestionTool about literally anything: technical implementation, UI & UX, concerns, tradeoffs, etc. but make sure the questions are not obvious. Be very in-depth and continue interviewing me continually until it's complete, then write the learnings to README.md


## Coding guidelines and philosophy

- You should generate code that is simple and redable, avoid unnecesary abstractions and complexity. This is a research codebase so we want to be mantainable and readable.
- Avoid overly defensive coding, no need for a lot of `try, except` patterns, I want the code to fail is something is wrong so that i can fix it.
- Do not add demo-only flags or placeholder CLI options that gate real functionality (e.g., `--run` just to toggle execution); scripts should run their main logic directly.
- Adhere to python 3.12+ conventions

## Key docs

- `program.md` — research context, goals, metrics, file constraints
- `instructions/CLAUDE-ADVISOR.md` — advisor role workflow
- `instructions/CLAUDE-STUDENT.md` — student role workflow

## Architecture

- **Advisor pod** — no GPU, runs Claude Code in a loop. Queries W&B, reviews student PRs, generates new hypotheses, and creates draft PRs to assign work.
- **Student pods** — GPU workers, each running Claude Code. Poll for assigned PRs, implement the hypothesis, run training, report results.

## k8s layout

- `k8s/advisor-deployment.yaml` / `k8s/student-deployment.yaml` — pod specs
- `k8s/entrypoint-advisor.sh` / `k8s/entrypoint-student.sh` — startup scripts
- `k8s/launch.py` — helper to template and apply deployments

## instructions/

Role-specific CLAUDE.md files. The Student and Advisor both use Claude Code. At pod launch, the appropriate role-specific file is copied over this CLAUDE.md:
- `instructions/CLAUDE-ADVISOR.md` → advisor pods
- `instructions/CLAUDE-STUDENT.md` → student pods
