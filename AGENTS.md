<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai - Project Context

## Senpai users

### Creating a target program.md

When helping a user onboard a target repository, inspect an explicitly configured `program_path` first. When it is blank, look for `program.md` at the root and exactly one directory below it. If there is no usable file, coach the user through creating one by following the [`grilling-autoresearch`](.agents/skills/grilling-autoresearch/SKILL.md) skill (`$grilling-autoresearch`). Inspect the repository before interviewing them, establish facts yourself, ask the user to decide the remaining intent and tradeoffs, and wait for shared understanding before drafting the file. Multiple auto-discovered files are ambiguous; do not choose one silently.

`program.md` is appended to every Senpai model's system prompt, so keep it concise, plain-language, and high-signal. It should clearly define:

- the project goal and the exact primary metrics, including how each metric is calculated, which direction is better, and which split or benchmark decides success;
- the data paths, shapes, sizes, train/validation/test splits, exclusions, leakage risks, and important footguns;
- operational guardrails such as commands, budgets, allowed edits, protected artifacts, and result-reporting expectations; and
- optional research avenues, papers, models, and libraries that provide useful starting points without forcing a narrow solution path.

Favor high-level goals and guardrails that let research agents discover the details. Avoid micromanaging methods or over-prompting one idea unless that narrow focus is the user's explicit goal. The [`bootstrap-target`](.agents/skills/bootstrap-target/SKILL.md) guide and its template can turn the confirmed decisions into the target contract.

Reference examples:

- [TandemFoilSet-Balanced](https://github.com/morganmcg1/TandemFoilSet-Balanced/blob/main/program.md)
- [DrivAerML](https://github.com/morganmcg1/DrivAerML/blob/main/program.md)
- [MLXFast challenge](https://github.com/morganmcg1/mlxfast-challenge_senpai/blob/main/senpai/program.md)
- [autoresearch](https://github.com/karpathy/autoresearch/blob/master/program.md)

## Senpai developers

Development of a problem-agnostic autonomous ML research loop for target ML problem repositories. The runner and its guidance must stay target-repo agnostic.

### Clarifying development work

When asked for a large piece of work that seems vague, consequential, or full
of hidden tradeoffs, ask the user detailed clarifying questions about the real
implementation choices: technical design, workflow, UX, risks, validation,
operations, and tradeoffs. Prefer non-obvious questions that expose constraints
or intent. When the answers change durable project behavior, write the learnings
to README.md or SPEC.md as appropriate.

### Coding guidelines and philosophy

- You should generate code that is simple and readable. Avoid unnecessary abstractions and complexity. This is a research codebase, so maintainability and clarity matter.
- Avoid overly defensive coding. No need for lots of `try`/`except` patterns, fallbacks, or backups. Prefer code that fails clearly when something is wrong so it can be fixed.
- Do not add demo-only flags or placeholder CLI options that gate real functionality (e.g., `--run` just to toggle execution); scripts should run their main logic directly.
- Adhere to the repository's Python 3.13 runtime.

### Key docs

- `README.md` - operator-facing overview, launch examples, and problem-package layout.
- `SPEC.md` - target architecture and rewrite contract for the senpai orchestration loop.
- `senpai.yaml` - launch defaults for the Senpai runner, target branch, advisor branch, and `problem_dir`; supply the required target repository by CLI or local config.
- `$PROBLEM_DIR/program.md` - conventional authoritative target research context, goals, metrics, training constraints, and file boundaries. A blank `program_path` requires exactly one `program.md` across the repository root and directories one level below; an explicit value selects a target-repository-relative `program.md`.
- `system_instructions/SENPAI-HARNESS.md` - shared OpenHands harness contract.
- `system_instructions/ADVISOR.md` - advisor role workflow.
- `system_instructions/STUDENT.md` - student role workflow.
- `system_instructions/SENPAI-LAUNCH-CONTEXT.md` - authoritative per-launch runtime and isolation rules.

### Architecture

- **Runner repo** - this repo. Owns orchestration, Kubernetes launch, role instructions, GitHub helpers, W&B integration, and operational docs.
- **Target repo** - cloned into `$PROBLEM_DIR` from `target_repo_url`. Owns the data code, training code, evaluation code, `program.md`, project context, and experiment branches. Agent commits and PRs land in the target repo, not in the runner repo.
- **Advisor pod** - lightweight, no GPU, keeps one durable OpenHands
  conversation and uses typed control-plane tools for GitHub and generic
  child-agent dispatch.
- **Student pods** - heavy GPU workers, use one OpenHands conversation per
  assignment revision, implement one assigned PR, run supervised training, and
  resume the same conversation for actionable monitor events.
- **Cross-node communication** - GitHub PR labels and human-tagged Issues only;
  Senpai requires no RPC service or cluster-specific network setup.
- **GitHub Issues** - human-to-agent communication channel. Agents poll for and respond to these alongside their normal PR workflow.
- **W&B** - canonical experiment metrics store for training runs, comparisons, and merge decisions.

### k8s layout

- `k8s/advisor-deployment.yaml` / `k8s/student-deployment.yaml` — pod specs
- `k8s/entrypoint-advisor.sh` / `k8s/entrypoint-student.sh` — startup scripts
- `k8s/launch.py` — helper to template and apply deployments

### system_instructions/

The OpenHands base prompt is extended with one stable system suffix, assembled in this order:

- `system_instructions/SENPAI-HARNESS.md`
- `system_instructions/ADVISOR.md` or `system_instructions/STUDENT.md`
- the selected target `program.md`, with its repository-relative path in the header
- the rendered `system_instructions/SENPAI-LAUNCH-CONTEXT.md`

The suffix keeps the Markdown headings and wraps these four fragments in
`<SENPAI_HARNESS>`, `<SENPAI_ROLE>`, `<SENPAI_PROGRAM>`, and
`<SENPAI_LAUNCH_CONTEXT>` sections, in that order.

The runner loads this complete suffix once when the agent process starts and does not refresh it during the session. Optional human operator instructions remain user context.

Target skills are loaded explicitly through OpenHands skill context. Target and runner `AGENTS.md`, `AGENT.md`, or `CLAUDE.md` instruction files are human-facing development context and are not loaded as Senpai project context; the checked-in root `CLAUDE.md` links to this canonical guide.
