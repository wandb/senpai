# Evidence playbook

Use the sources that exist for the current program. Do not require Kubernetes,
W&B, GitHub, or an external evaluator when the launch uses another system.

## Source discovery

Resolve and read the target repository's selected `program.md` first, using the
discovery rules in `SKILL.md`. Record its path, revision when available, and read
time. Extract the mission, deciding evidence, and constraints before selecting
experiments to inspect. A prior report, advisor summary, or PR title is not a
substitute for this read.

For live checks, obtain the repository file at the relevant launch/ref. For a
historical report, use the contract corresponding to that snapshot and label
it. If the repository contract differs from the version loaded by the running
agents, report both versions and the consequence; do not assume the agents
received the edit. A missing or ambiguous contract limits mission assessment,
but need not prevent a useful operational check.

Then resolve the remaining live scope rather than using remembered values.
Useful sources include:

- the selected `program.md` and launch configuration;
- environment variables such as the target repository, advisor branch,
  research tag, W&B entity and project, runtime context, and namespace;
- runtime labels, deployment names, and controller state;
- the target repository's remote, base branch, PRs, result logs, and evaluation
  ledger; and
- the external evaluator named by the program.

Resolve the expected advisor and student roster from the launch configuration
or rendered runtime. Do not infer the expected count only from currently
visible workloads because a missing agent is itself a health finding.

## Runtime and controller evidence

For Kubernetes launches, query the configured context and namespace with the
research tag or another launch-specific selector. Avoid broad all-namespace
dumps. Inspect:

- desired, available, and ready replicas;
- pod phase, readiness, restart count, termination reason, and age;
- recent warning events;
- node placement and Ready/pressure conditions;
- requested GPUs, CPU, and memory, plus current pressure or OOM evidence; and
- active processes or bounded recent logs when activity matters.

For cluster-wide GPU and job accounting, use the operator's own capacity
tooling when available. Its snapshot complements this workflow; it does not
replace launch-specific health checks.

Interpret states precisely:

- `Running` means scheduled and alive.
- `Ready` means the readiness check passes.
- `training` requires recent computation evidence.
- a W&B heartbeat can outlive the useful training process and is not sufficient
  on its own when duplicate or stale runs exist.
- high utilization can be healthy; high memory with cgroup pressure or OOM
  events is a risk. Report both limit and observed evidence.

Inspect leases, controller state, and bounded logs for recent actions and
errors. A sleeping controller can be healthy between polls. A previous error
is recovered only after later evidence proves the affected operation works.

## W&B evidence

Use bounded API queries. Start from run IDs linked by active or recently
completed assignments, then widen only enough to establish experiment counts
and the scientific frontier.

Preserve both counts:

- **raw run records:** every W&B record, including retries and duplicates;
- **logical experiments:** the intended experimental identities after retries
  are reconciled.

Prefer a durable experiment ID. If none exists, state the fallback, such as
exact run name plus seed and defining configuration. Do not deduplicate merely
because two runs look similar. For each logical identity, retain successful
evidence when a retry finished, but also record the earlier failure when it
matters operationally.

Classify logical experiments as:

- **completed:** at least one authoritative successful result exists;
- **active:** no successful result exists and a fresh run is genuinely active;
- **terminal:** all known attempts stopped or failed without a successful
  result; or
- **unresolved:** evidence conflicts or is too stale to classify.

The four logical counts must sum to the logical total. Report stale heartbeats
and duplicate active/finished identities separately.

Use the metric key and selection rule defined by `program.md`. Distinguish a
run's final summary, history peak, selected checkpoint metric, and aggregate
comparison; they answer different questions.

## GitHub and repository evidence

Inspect the configured advisor branch and relevant PRs. Record:

- assignment owner and routing labels;
- draft, review, merged, closed, or blocked state;
- current head and base relationship when it affects validity;
- linked runs and structured result comments;
- advisor acceptance or revision decisions; and
- current implementation on the research base.

Treat a PR claim as a claim until its run evidence is verified. Treat a merged
change as integrated code, not automatic proof that its scientific conclusion
generalizes.

For lineage, inspect assignments and decisions for the motivation, then source
or config differences for what changed. Distinguish an explicit follow-up from
a comparator, reused implementation, or concurrent sibling. Use timestamps
before claiming one result triggered another experiment. Do not infer either
scientific or Git ancestry from PR numbering. Explain the connection to the
programme's mission; label an inferred or unverified relationship.

## External evaluation

Use the evaluator's authenticated read-only API or official result page when
configured. Cross-check any repository ledger, but do not let a local file
override the external source.

Use one of these exact states:

- **not configured:** the program defines no external evaluator;
- **not queried:** the evaluator exists but was outside this check's scope;
- **inaccessible:** the query failed or credentials are unavailable;
- **zero results:** the authoritative query succeeded and returned none;
- **pending:** an evaluation exists without a final result; or
- **scored:** a final score or outcome is verified.

Do not submit, approve, or modify evaluator state during a status check.

## Evidence precedence and contradictions

Use each source for what it directly observes. When sources disagree, name both
facts, their timestamps, and the practical consequence. Lower confidence rather
than creating a synthetic certainty. Examples include:

- a Ready pod with a failed training process;
- a W&B run marked active after a successful retry;
- a PR labeled in progress after its result finished;
- a local result ledger without a corresponding external evaluation; or
- a strong raw checkpoint that the evaluation pipeline cannot load.
