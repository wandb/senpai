---
name: check-senpai-experiments
description: Check scientific progress and operational health of a Senpai research program against the mission in its target repository's program.md. Use for experiment updates, health checks, current advisor/student activity, promising or failed experiments, W&B/GitHub reconciliation, evaluation status, and next research priorities. Produce a fresh, read-only, evidence-backed report with clear implications and actions. Do not use for cluster-capacity-only checks or to mutate research state.
---

# Check Senpai experiments

Explain what the research has learned toward its mission, what remains unknown,
and whether the operator needs to act. Establish that mission from the selected
target repository's `program.md` on every check. Recent PRs and metric movements
are evidence to interpret against that mission, not a substitute for it.

Keep the check read-only. Do not restart workloads, alter secrets or assignments,
merge changes, approve results, submit evaluations, or publish status artifacts
unless separately requested. Never post comments, reviews, or status messages
on research PRs without an explicit request to post; permission to monitor or
notify an agent does not authorize PR posts.

## Read the programme first

Resolve the target repository and launch configuration far enough to locate the
selected `program.md`, then read it before choosing or interpreting scientific
evidence. Use an explicitly configured `program_path` first. If it is blank,
look at the target repository root and exactly one directory below it; require
one usable match. Do not silently choose among multiple files or substitute a
default file when the configured path cannot be read.

Establish from that file:

- the core mission: what the research must discover, explain, demonstrate, or
  improve, and why;
- the evidence that decides success: metrics and direction where applicable,
  deciding data or benchmark, controls, selection and aggregation rules,
  replication requirements, and any external evaluation;
- the constraints that determine whether a finding can contribute to the goal:
  allowed changes, protected artifacts, budgets, stop conditions, and reporting
  obligations.

Record the repository, selected path, inspected revision when available, and
read time. For a live check, reread the repository file; do not rely on a previous
report or remembered objective. For a requested historical report or sample,
use the contract corresponding to its evidence and label the saved snapshot.
If the launch loaded a different contract revision, surface that mismatch
before judging compliance. Do not silently treat a later edit as instructions
that were already in force for an older experiment.

If the file is missing, ambiguous, or inaccessible, state that mission assessment
is unavailable. Continue with useful operational evidence, and ask for a selector
only when needed to resolve the ambiguity. Do not invent the mission from run
names or issue titles, or create a programme as part of this check.

## Establish scope and collect evidence

Resolve the research tag, branches, runtime context and namespace, configured
metric store, advisor, and complete expected student roster from configuration
or live state. Record snapshot time and source freshness. Keep a named project
scoped to that project; do not substitute remembered launch identities.

Use [the evidence playbook](references/evidence-playbook.md) for source-specific
checks. Collect independent read-only sources in parallel when safe:

- **Repository and GitHub:** research base, assignments, relevant PRs, linked
  runs, decisions, lineage, integrated changes, and evaluation records.
- **Runtime and controllers:** readiness, restarts, warnings, placement,
  resource pressure, leases/state, recent actions, supervised processes,
  provider errors, and bounded logs. For cluster-wide GPU capacity, use the
  operator's own capacity tooling when available; it complements, not replaces,
  launch-specific checks.
- **W&B or the configured metric store:** bounded metadata, configs, summaries,
  selected histories, artifacts, heartbeats, and stable run links.
- **External evaluator:** authoritative results when the programme defines one
  or the user asks. Do not invent a separate evaluator for an internal gate.

Prefer APIs and structured state. Bound queries to the launch and evidence that
can change the scientific or operational assessment. Do not expose credentials.

## Reconcile observations

Account for the advisor and every expected student, including idle, missing,
blocked, and unobserved agents. Readiness alone does not establish training;
require recent computation evidence and investigate stale or duplicate
heartbeats. Separate active defects, recovered incidents, risks, and deliberate
scientific stops. Allocation does not establish utilization or spare capacity.

Use each source for its authority: the programme for the scientific contract,
metrics for measured results, GitHub for assignments and integration, and the
runtime for workload lifecycle. State contradictions and material evidence gaps
where they limit the conclusion. Scope absence claims to inspected sources and
time windows; distinguish zero results, not configured, not queried,
inaccessible, and pending.

When counting runs, reconcile retries using stable logical identities. Keep raw
records separate from logical experiments and independent hypotheses. Record
the deduplication rule; completed, active, terminal, and unresolved logical
counts must partition the total. Keep accounting in supporting evidence unless
it changes a decision.

## Evaluate progress toward the mission

Choose the research questions from `program.md`, then relate the experiments to
those questions. Explain what a finding changes: a phenomenon established, a
hypothesis narrowed, a method ruled out at its tested budget, a measurement
made more reliable, or a solution supported. Recognize useful exploratory
progress without presenting it as a confirmed cause or remedy. Flag work whose
connection to the mission is unclear or whose result conflicts with its
constraints.

For material quantitative claims, preserve metric meaning and direction,
matched controls, signed effects and units, evaluation protocol, selection
rule, seeds or replicates, uncertainty, and relevant thresholds. Distinguish
prospective from retrospective evidence and direct from suggestive comparisons.
Keep raw peaks, selected results, evaluation-compatible candidates, integrated
changes, and externally verified outcomes distinct without requiring a section
for each. A merge is not proof; an incomplete screen is not confirmation.

Explain verified experiment lineage: what question or result motivated the
follow-up, what changed, and what it can distinguish. Separate reused code,
comparators, concurrent tests, and scientific follow-ups from Git ancestry.

## Write and verify

Use [the report guidance](references/report-contract.md). Lead with a brief
mission statement and a TL;DR of scientific progress and action needed. Keep
operational issues visible. Organize science around questions and evidence,
with explicit **What it means** explanations tied to the mission. Add a short
lineage or timeline when it explains why current experiments exist.

Before returning, check that:

- the mission comes from an actually read, identified `program.md`;
- the scientific verdict, implications, and next evidence answer that mission;
- useful learning, unresolved questions, and final success are distinguished;
- action needed, observed impact, and evidence limits are easy to scan;
- comparisons and lineage are supported, and every expected agent was checked;
- counts, current defects, scientific stops, and historical failures are not
  conflated; and
- recommendations are proposals, not claims that work was performed.

Keep collection rigor without forcing the reader through a full roster,
accounting table, exact heading sequence, or internal/external split.
