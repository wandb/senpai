# Report contract

A report explains progress toward the target repository's research mission
and what needs action. Lead with what the reader must decide or fix.

## Read the mission first

Before interpreting or choosing any scientific evidence, resolve and read the
target repository's selected `program.md` using the rules in `SKILL.md`.
Cite its repository-relative path and the revision (commit or blob) when known.
From that file, derive:

- the core mission: what the programme is trying to find out, fix or achieve;
- the scientific questions it asks, including any required order;
- the evidence that decides success, including metrics, direction, selection
  rules, replicates and acceptance gates where the programme defines them; and
- constraints that affect interpretation, such as fixed protocols, protected
  code or data, budgets and changes that require approval.

Do not reconstruct the mission from recent PRs, advisor summaries or
remembered thresholds; report any conflict with `program.md`. If the file
cannot be read, do not invent a mission. An operations-only report can still
help, but it must say **Mission assessment unavailable** and give the reason.
For a historical report, read the contract corresponding to its snapshot and
label the saved copy. Surface differences between repository and loaded launch
contracts before judging compliance; do not assume agents received later edits.

## Shape

Use a recognizable shape with headings that fit the content: title, snapshot
scope and time, mission and TL;DR; actions and operational detail next when
action is needed; a compact lineage when useful; questions with evidence and
"What it means"; next useful evidence; and optional supporting evidence. Do not
impose a fixed H2 order, a set number of questions, routine roster tables, run
inventories or a forced internal/external split.

## Mission and TL;DR

State the mission in one or two plain sentences and link the programme
file. A quotation alone is not enough. The science verdict, the questions,
each "What it means", the lineage and the next evidence must all explain
progress toward that mission.

Keep the TL;DR to a few scannable bullets: action status, scientific progress
and any pending decision. Separate what is now known from unresolved causes and
from demonstrated solutions.

A mission may be an investigation rather than an optimization. Establishing a
phenomenon, narrowing a mechanism, useful negatives, reproduction and better
measurement all count as progress. Do not invent a scalar objective, stages or
a progress percentage that the programme does not define.

## Actions and operational issues

Give each material issue an explicit status: **repair needed**, **decision
needed**, **watch**, **no action needed** or **unknown**. For each one, state:

- what is happening, separating observed impact from plausible risk;
- the owner, if known, or a clearly proposed role;
- the recommended action, and how to verify it worked; and
- escalation triggers, when they help.

Do not set arbitrary deadlines or call an issue "not an emergency" without
evidence. Restarts are not harmless just because GPUs keep computing:
supervision, results and stop decisions may still fail. Limit absence claims
to what you inspected. A deliberate scientific stop is a result, not an
infrastructure failure.

Checks are read-only. Never post to research PRs unless the user explicitly
asks. Write recommendations so they cannot be mistaken for completed actions.

## Lineage

Verify how PRs relate using assignments, advisor decisions, configuration and
code. Name the relationship: scientific follow-up, reused implementation,
comparator or control, parallel sibling, or Git ancestry. Reusing code is not
the same as responding to a result. A few linked bullets or a short timeline
should explain why each current experiment exists, without repeating the full
history. Mark inferences and unknowns.

## Questions, evidence and "What it means"

Choose questions from the mission's open uncertainties, not from a list of
PRs. Under each, give the evidence, then an explicit **What it means** that
ties the experiment to a concrete consequence for the mission. Use this
reasoning, not labels: *this tests [a mission-relevant uncertainty]; the result means [a
consequence]; the next evidence will change [a decision].*

When external results matter, discuss them under the related question. Keep
protocol differences and the evidence state clear: not configured, not
queried, inaccessible, zero results, pending or scored. Include a small metric
table only when it is faster to read than prose.

## Scientific accuracy

Apply these where relevant, not as a checklist for every paragraph.

- Use the programme's exact metric names and directions. Compare against
  matched controls. Give signed deltas with units, using percentage points for
  differences between percentages.
- State seeds or replicates, uncertainty, the selection rule and window, and
  whether each run is finished or incomplete.
- Mark analyses as prospective or retrospective, and comparisons as direct or
  suggestive. Use causal language only when the design supports it.
- Separate a raw peak from a selected, evaluation-compatible result.
  Integrated code is not scientific proof. Beating a failing comparator is not
  meeting the success contract.
- A negative result covers only the budget tested; it cannot show that later
  recovery was impossible.

## Next useful evidence

Name the results that would change a decision, and say which decision. Restate
the programme's final success contract only when it matters to the reader.

## Supporting evidence

Put collection, accounting, query scope and minor gaps here or in linked
files. Raise a gap to the top only when it changes a conclusion.
