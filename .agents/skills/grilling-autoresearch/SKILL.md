---
name: grilling-autoresearch
description: Coach the user through a rigorous, repository-grounded interview to create or improve program.md. Use when a target repository lacks program.md, its goals, metrics, data, or guardrails are unclear, or the user asks to be grilled about an autoresearch setup.
---

# Grilling autoresearch

Interview the user relentlessly until you reach a shared understanding. Map
this as a **design tree**: every decision branches into the decisions that hang
off it.

Before asking questions, inspect the repository's documentation, data loaders,
training and evaluation code, configuration, and any existing `program.md`.
Finding facts is your job, never the user's. Ask the user only for decisions
and intent.

Work the tree in **rounds**. The **frontier** is every decision whose
prerequisites are already settled: the questions you can ask now without
guessing at answers you have not heard yet. Ask the whole frontier in one
round. Number each question, give your recommended answer, and then wait for
the user's answers before the next round.

Format each question like this:

```text
❓ **Q1** - **<question title>**: <question body, including choices when useful>

➡️ <your recommended answer>
```

Each round reshapes the tree. Settled decisions push the frontier outward and
unblock dependent questions. Recompute the frontier and ask the next round. A
question whose answer depends on another question still open in this round
belongs to a later round.

When a frontier question needs a fact from the environment, dispatch a
subagent to find it. Do not ask the user for anything you could look up
yourself, and do not block the independent questions: a running investigation
is only an unsettled prerequisite for its downstream branch. The decisions are
the user's; put each to them and wait.

For `program.md`, keep the design tree centered on:

- the objective, exact primary metric names and definitions, optimization
  direction, evaluation split, and any secondary gates;
- data paths, shapes, sizes, splits, exclusions, leakage risks, and footguns;
- commands, budgets, result artifacts, editable boundaries, and benchmark
  integrity constraints; and
- useful research avenues, papers, models, or libraries without unnecessarily
  narrowing the search space. Narrow it only when that is the user's explicit
  intent.

The session is done when the frontier is empty: every branch of the design
tree has been visited and nothing material remains silently assumed. Do not
act on it until the user confirms that you have reached a shared understanding.

Then draft a `program.md` that is concise, plain-language, and high-signal
because Senpai appends it to every model's system prompt. Make it specific
enough to run and evaluate the research correctly, while leaving implementation
choices and exploration to the research agents. Prefer high-level goals and
guardrails over step-by-step micromanagement, and verify every repository path,
command, data claim, and metric definition against the repository.

Useful examples:

- [TandemFoilSet-Balanced](https://github.com/morganmcg1/TandemFoilSet-Balanced/blob/main/program.md)
- [DrivAerML](https://github.com/morganmcg1/DrivAerML/blob/main/program.md)
- [MLXFast challenge](https://github.com/morganmcg1/mlxfast-challenge_senpai/blob/main/senpai/program.md)
- [autoresearch](https://github.com/karpathy/autoresearch/blob/master/program.md)

Adapted from [Matt Pocock's grilling skill](https://github.com/mattpocock/skills/blob/main/skills/productivity/grilling/SKILL.md), used under the MIT License.
