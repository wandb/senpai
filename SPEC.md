# Senpai OpenHands runtime contract

Status: implemented on the OpenHands rewrite branch.

## Objective

Senpai is a small deterministic Python control plane around OpenHands.
OpenHands owns research judgment, code changes, evidence interpretation, and
bounded delegation. Python owns operations that should not depend on an LLM
composing fragile tool calls:

- GitHub polling, workflow operations, and verification;
- assignment branch publication;
- generic long-running process supervision and optional W&B metric monitoring;
- conversation selection and durable local events;
- command policy and stop checks; and
- cadence, retry, deadlines, and shutdown.

The rewrite preserves the advisor/student research workflow while reducing raw
data copied through model history and removing Claude Code runtime
dependencies.

## Invariants

1. Agent commits and PRs land in the target repository, never in this runner.
2. GitHub and W&B are the durable research records.
3. GitHub PRs and Issues are the only cross-node protocol.
4. An LLM does not poll, sleep, tail logs, supervise processes, or assemble a
   multi-call GitHub transaction.
5. GitHub mutations are typed, preconditioned, convergent on replay, and
   verified against remote state.
6. The advisor uses one durable conversation UUID. A student uses one UUID per
   assignment revision, and job-monitor wakes continue the owning conversation.
7. Conversation and generated artifact state cannot fall back into the target
   checkout.
8. Senpai does not prune conversation history.
9. Only the student image carries CUDA, PyTorch, and the training stack.
10. Secrets are passed at narrow executor boundaries and redacted before
    monitored content is attached.
11. Hivemind is disabled, not redesigned, in this change.
12. The campaign operational supervisor is separate from advisor/student
    research prompts, scopes every observation and write to the immutable
    launch inventory, and never turns missing evidence into a healthy zero.

## Control loop and remote protocol

```text
entrypoint
  clone/configure
  exec python -m senpai_agent.supervisor advisor|student

Python supervisor
  start one controller worker process group
  restart crashes with bounded exponential backoff
  TERM/KILL a worker whose current phase lease expires

Python controller worker
  poll GitHub + local durable monitor/event state
  reconcile the target checkout
  start one bounded OpenHands turn
  verify durable state
  sleep/backoff/jitter
```

The worker publishes an atomic lease containing its PID, current phase, hard
deadline, and completed-turn counter. The supervisor resets bounded restart
backoff only after a turn is successfully acknowledged; process uptime and
idle sleep do not count as progress. The supervisor is independent of
OpenHands and Kubernetes.
Kubernetes liveness and Docker health checks inspect the same lease, while the
supervisor provides the same recovery on a plain host.

The core controller imports no Kubernetes API and needs no Service, port, DNS
record, ServiceAccount, RBAC, cross-node token, or tailnet.

### Campaign operational supervisor

Kubernetes launches optionally add one independent operational-supervisor
Deployment. It is not the per-role crash supervisor above and it does not join
advisor/student research conversations. Every 15 minutes it deterministically
collects a timestamped campaign snapshot, retains the latest three snapshots,
and starts one fresh bounded in-memory OpenHands conversation over those
snapshots. Only the snapshot state and mutation audit persist locally.

Every managed role Deployment records its exact Senpai source revision. An
incremental supervisor launch is rejected unless the existing exact-tag advisor
and every student not replaced in the same launch use that revision and advisor
branch, and the advisor's configured student inventory is unchanged. This
prevents a supervisor from starting against an absent, incompatible, or
differently scoped role-control protocol.

The snapshot scope is fixed at launch:

- GitHub returns only open PRs whose current base ref exactly equals the
  configured advisor branch. Each PR includes age and paginated counts for
  issue comments, submitted reviews, and inline comments.
- W&B resolves only exact run IDs discovered in the configured students'
  role-local training state. Experiment code may freely choose its own W&B
  group without breaking campaign ownership.
- Kubernetes selects exactly one running pod for each configured role using
  the research-tag, role, and student labels. Role-local inspection returns the
  controller lease, current conversation UUID, completed turns, running
  training count, bounded utilization, reset status, and recent structured
  error markers. Raw log and training-error text never leaves the role.

Each failed evidence source remains `unknown` with a typed evidence gap. In
particular, a failed W&B query is not reported as zero running jobs. Repeated
`SENPAI_TURN_DEFERRED` log markers survive in the bounded three-snapshot trend.

The supervisor receives OpenHands' native terminal and one typed operations
tool. Unlike advisor/student terminals, the native terminal is not wrapped by
Senpai's command policy. It can run arbitrary commands permitted by the
container's Unix identity and ServiceAccount. The exact runtime and instruction
checkout is populated by an init container and mounted read-only into the model
container, and user-skill loading is disabled. The typed tool remains the
preferred surface for inspecting a role, enqueueing a deduplicated role event,
queueing a context reset, or restarting the controller because targets can name
only the configured advisor or students; callers cannot supply hosts, pods,
namespaces, working directories, environments, or credentials. Typed mutations
have durable idempotency keys, per-incident cooldowns, and metadata-only audit
records. The enforced cooldown identity is derived from the anomaly category,
mutation kind, and exact role target; changing a free-form incident label
cannot bypass it. Role inspection is always fresh and is never replayed from
the mutation ledger. Each fresh supervisor turn receives the 12 most recent
mutation targets, categories, timestamps, and outcomes.

A context reset is an owner-consumed request. The external supervisor records
the expected conversation UUID, controller identity, raw-event prefix digest
and count, and pending-event keys. Only that role's controller may claim it at
a quiescent turn boundary. The controller calls
`run_openhands(..., reset_context=True)`, records completion before ordinary
event acknowledgement, and keeps the same UUID, workspace, complete append-only
event trace, and pending events. External code never instantiates a second
`LocalConversation` over live state or deletes individual events.

A controller restart is refused while an advisor or student supervised job or
delegated agent is running, or when either activity inventory cannot be proven.
It signals only the verified controller PID; the role's existing crash
supervisor performs the restart. Kubernetes RBAC grants no AWS, node,
pod-delete, or Deployment-patch verbs. The terminal receives no GitHub token,
so authenticated branch and PR mutations remain with advisor/student tools.

Every six hours, the next wake runs a second fresh review against the currently
deployed `system_instructions/ADVISOR.md`. It may inject one concise reminder
into the existing advisor conversation only for clear sustained strategic
drift. This periodic review does not modify the advisor prompt and does not
micromanage ordinary scientific judgment.

GitHub state is level-triggered:

- `status:wip` plus exactly one `student:<name>` label is an assignment;
- trusted human comments and reviews on one assigned open `status:wip` or
  `status:review` PR wake its exact student assignment conversation;
- `status:review` is a durable advisor wake;
- when the configured research base changes from an active assignment's
  recorded base SHA, `research_base_changed` gives the advisor
  `required_base_sha`, `current_base_sha`, and a compare URL without cancelling
  the student;
- `status:blocked`, `status:needs-rebase`, missing or duplicate student labels,
  stale WIP, and duplicate assignments are advisor-action events; and
- an open Issue labeled `human` plus `team`, the advisor branch, or one student
  label is a human message.

Human Issue events use the exact latest human-authored body/comment ID as their
dedupe key and `human_message_id`. An agent reply updates the Issue but does not
create a new wake for its own comment. `respond_to_human_issue` verifies the exact
human message before writing an idempotent response.
Launches with human-Issue handling disabled skip that GitHub query entirely.

Assigned-PR issue comments, submitted reviews, and inline comments each use
their immutable GitHub ID as a level-triggered event key. Senpai accepts GitHub
users associated as repository owners, members, or collaborators. A comment by
the authenticated actor containing a Senpai protocol marker is automation, not
human feedback, except for the explicit `senpai-assignment-feedback` operation.
Every accepted event carries its first-seen assignment and revision identity,
so monitor and feedback events resume one student UUID.
Successful turns atomically acknowledge immutable feedback keys in a small JSON
ledger. Oldest unacknowledged events are delivered in bounded count/byte
batches; immediate post-turn polls drain later batches without dropping them.

While an OpenHands turn is running, `ActiveGitHubWatcher` polls the same GitHub
state. It enqueues all newly visible advisor events, and only PR feedback bound
to the currently running student UUID, in the role's local event store.
OpenHands 1.40 supports concurrent `send_message`; `AdvisorEventPump` injects at
its state lock boundary without cancelling unrelated work. Successfully
injected student feedback is acknowledged in `github-feedback.json` only when
the enclosing student turn succeeds.

Generic child results use a local SQLite WAL event store because parent and
child run on the same advisor or student instance. That is not an inter-node
protocol.

Role SQLite databases are `advisor-events.sqlite3`, for unacknowledged advisor
watcher/child/supervisor events; `student-events.sqlite3`, for unacknowledged
student feedback/child/supervisor events; `context-resets.sqlite3`, for the
owner-consumed reset queue; and `training/monitors.sqlite3`, for role-local job
monitor policy, samples, and deduplicated actionable signals. The `training/`
state-directory name is retained so live persisted conversations and supervised
jobs remain recoverable across the generalized job surface. The operational
supervisor separately stores `operations.sqlite3` for metadata-only action
audit. OpenHands conversation history remains a file-backed per-UUID event log.

## State and conversations

Advisor state:

```text
/var/lib/senpai/<research-tag>/advisor/openhands_state/
├── advisor-conversation-id
├── controller-lease.json
├── advisor-events.sqlite3
├── context-resets.sqlite3
├── conversation-state.json
├── training/
│   ├── <job-id>.json
│   ├── <job-id>.log
│   ├── monitors.sqlite3
│   └── monitors/<job-id>.json
├── github/
└── conversations managed by OpenHands
```

The advisor UUID is created once and reused. Its conversation may cover several
ideas and monitoring threads concurrently.

Student state:

```text
/var/lib/senpai/openhands_state/
├── controller-lease.json
├── github-feedback.json
├── student-conversations.json
├── student-events.sqlite3
├── context-resets.sqlite3
├── conversation-state.json
├── training/
│   ├── <job-id>.json
│   ├── <job-id>.log
│   ├── monitors.sqlite3
│   └── monitors/<job-id>.json
├── github/
└── conversations managed by OpenHands
```

`student-conversations.json` maps one `(assignment_id, revision_id)` to one
UUID. `conversation-state.json` records, per UUID, both successful initial
instruction delivery and the digest of the delivered merged system context.
The controller replaces this one document atomically after a successful turn,
so a restart cannot observe those two facts at different revisions. A
`job_monitor` carries its original conversation UUID and therefore resumes,
rather than replaces, the owning advisor or student conversation.

`github-feedback.json` records every immutable PR feedback key's first-seen
assignment revision, then marks it acknowledged only after its student turn
succeeds. This prevents pending or completed feedback from replaying or
rebinding to a later assignment revision after a restart.

When `conversation-state.json` does not yet exist, startup atomically migrates
the previous `started-conversations.json` and
`system-context-revisions.json` files. A conversation caught between those
legacy files' two writes resumes without replaying its initial brief and
receives the current system context once.

OpenHands stores base state and individual events beneath that UUID. A killed
worker resumes from the last persisted event. An in-flight response or tool
call without a durable event is retried from the preceding event.

The controller marks a conversation's initial instructions delivered and
records its current system-context digest in the same atomic update, only after
the OpenHands turn succeeds. A crash or nonzero first turn therefore retries
the complete programme and assignment prompt instead of incorrectly
continuing from instructions that were never delivered.

Role state uses pod-local storage and survives controller or container restarts
within the same pod. Replacing or rescheduling a pod starts fresh local state;
the PR, branch, typed result, W&B runs, and Weave trace remain the durable
handoff.

No default path may be relative to the current workspace. Senpai removes only
its generated PR Markdown artifacts after 24 hours. It does not delete
OpenHands conversations or impose a retention count.

## Prompt and progressive disclosure

The model receives:

1. OpenHands' native base system prompt and tool schemas.
2. One stable system suffix assembled from:
   - `system_instructions/SENPAI-HARNESS.md`; and
   - the rendered advisor or student role charter.
3. Applicable target `AGENTS.md` and compatible `CLAUDE.md` project context.
4. A compact skill catalog whose bodies are loaded only when invoked.
5. User turns containing `program.md`, target role instructions, current state,
   and current UTC time.

The operational supervisor instead uses the minimal
`system_instructions/OPERATIONAL_SUPERVISOR_HARNESS.md`; it does not inherit
target-workspace, research, or subagent instructions.

Harness and role remain separate source documents because they have different
owners, but are merged into one system suffix so the agent knows both the
OpenHands operating contract and its Senpai role. The complete role is not
periodically duplicated in user messages; OpenHands includes the system suffix
on every inference. A persisted merged-context hash detects a changed deployed
harness or role and injects the current text once into the same conversation
UUID. Current time is rendered for every controller wake.

File-based subagents are discovered from `.agents/agents`. Skill bodies are not
concatenated into agent definitions. The OpenHands fork's `main` branch applies
each agent definition's `reasoning_effort` override after resolving its
inherited LLM or stored model profile.

## Prompt caching

The SDK and tools track the `main` branch of
[`morganmcg1/software-agent-sdk`](https://github.com/morganmcg1/software-agent-sdk)
and are based on OpenHands SDK 1.40.0. `uv.lock` records the exact `main` commit
used for reproducible image builds, while runtime CI installs directly from
`main` to verify the current fork head.

`prompt_cache_configuration()` sets:

- Anthropic: `prompt_cache_ttl="1h"`;
- GPT-5.6: one explicit cache breakpoint on the stable system block,
  `prompt_cache_options.mode="explicit"`, and a 30-minute TTL;
- older compatible OpenAI models: `prompt_cache_retention="24h"`; and
- other providers: no provider-specific cache option.

The fork emits an Anthropic cache-control `ttl` only when explicit Anthropic
caching is active. Its tests prove the five-minute wire form remains unchanged,
the one-hour TTL is forwarded, and OpenAI retention continues to work without
receiving an Anthropic TTL parameter. Laminar is an optional SDK extra and is
not part of Senpai's locked runtime; Weave is the agent observability
integration.

Direct `openai/*` models use a stored Responses API chain. The active branch's
latest `resp_*` ID is recovered from the durable OpenHands event log after
every process restart, passed as `previous_response_id`, and paired only with
inputs created after that response. System instructions and tools remain
explicit on every request.

Senpai sets `reasoning_context="all_turns"` and `reasoning_summary="auto"` so
supported models can reuse server-side private reasoning and return the most
detailed available summary. The default main effort is `xhigh`; GPT-5.6 also
accepts `max`, which uses API `max` effort with Responses
`reasoning.mode: pro`. Automatic OpenAI compaction starts at
200,000 rendered tokens. The OpenHands condenser is disabled for that provider
chain, but its complete local event log remains durable and is used to recover
the latest response ID after restart.

Direct Anthropic models use native server-side compaction with a 200,000-input-
token trigger. OpenHands persists the returned typed compaction block in the
normal event log and replays it first in each later request, including after a
process restart. The local condenser is disabled for these conversations.
Other providers retain the high-quality OpenHands condenser.

Every advisor root turn appends one controller-derived liveness invariant to
the system-message suffix, outside condensed history. It states that the
campaign is active, that this runtime has no campaign round limit, and that
`max_turns` limits one OpenHands turn rather than the research programme. Round
labels, final-round claims, and summaries cannot authorize stopping.

The complete durable transcript remains available as plain event JSON under
`$SENPAI_OPENHANDS_STATE_DIR/$SENPAI_CONVERSATION_ID/events/`. The harness
directs the model to use `rg` and bounded reads because the directory can be
large. No dedicated history-search tool duplicates shell capabilities. A
dispatched child receives `$SENPAI_PARENT_CONVERSATION_HISTORY_DIR`, allowing a
main advisor or student to delegate broad history recovery without copying the
full parent context.

## Typed tools

### `get_prs`

One function accepts explicit numbers, an inclusive creation-date range, and/or
a GitHub search expression. Every selected PR contains its full body, all issue
comments, all submitted reviews, and all inline review comments across
pagination.

`max_inline_prs` defaults to five. At or below the limit, Markdown is returned
in context. Above it, the same Markdown is written to one deterministically
named mode-0600 artifact outside the target checkout, and the model receives a
compact manifest and path. Raising the inline limit above five warns about
context pollution. There is no duplicate JSON artifact and no hidden
summarizing subagent.

### GitHub workflow tools

GitHub mutations are separate, operation-specific tools without a union wrapper.
There is no operation discriminator or model-supplied repository. The runtime
binds repository, role, credentials, workspace, and configured branches outside
the model-facing schema. It also canonicalizes every Senpai-authored comment to
an `ADVISOR:` or `STUDENT:` prefix from that trusted role; models supply plain
comment text and cannot impersonate the other role through a payload.

Advisor operations that act on an assignment share this object:

```json
{
  "pr_number": 123,
  "assignment_id": "assignment-id",
  "revision_id": "current-revision-id",
  "expected_pr_head_sha": "CURRENT_PR_HEAD_SHA"
}
```

| Tool | Role | Input beyond the shared `assignment` object |
|---|---|---|
| `create_assignment` | advisor | `assignment_id`, `revision_id`, `student`, `expected_base_sha`, `head_branch`, `title`, `body`; the base is the configured advisor branch |
| `publish_advisor_branch` | advisor | `remote_branch_sha_before_push`, `local_commit_sha` |
| `repair_assignment_routing` | advisor | `working_state` (`wip` or `review`) and a `blockers` list containing only `blocked`, `hold`, or `needs-rebase` |
| `send_assignment_feedback` | advisor | `feedback_id`, `comment` |
| `request_assignment_revision` | advisor | `new_revision_id`, `required_base_sha`, `comment` |
| `accept_result_on_current_base` | advisor | `expected_current_base_sha`, `reason` |
| `merge_experiment` | advisor | `expected_current_base_sha`, `merge_method` |
| `close_experiment` | advisor | `reason` |
| `respond_to_human_issue` | advisor or student | `issue_number`, `human_message_id`, `response` |
| `submit_experiment_result` | student | `branch`, `remote_branch_sha_before_push`, `result` |

Student publication happens only inside `submit_experiment_result`, which
derives the PR and proposed local head from the structured result, then validates
repository, assignment, revision, student, and current remote head before it can
push. Marker comments are trusted only when authored by the authenticated token
actor.

Assignment creation checks the remote base SHA, creates an isolated empty
assignment commit with `git commit-tree`, publishes with force-with-lease,
refuses a second active assignment for the student, creates or reconciles one
draft PR, embeds a typed assignment marker, and verifies routing state.

Advisor feedback carries exact assignment, revision, and PR-head preconditions.
It creates one immutable feedback ID without changing the assignment marker,
draft state, or routing labels, so a nudge reaches the current conversation
without creating a new revision UUID. Exact replay converges; changed guidance
uses a new ID and therefore a new durable GitHub comment event.

Routing repair declares the desired working state and blocker set; the tool
computes and verifies the corresponding labels. It cannot restore `review`
without the exact authenticated terminal result for that assignment revision
and head. Revision requests bind the new revision to an exact required
research-base SHA rather than leaving that base implicit.

Student submission requires a clean assignment branch, lease-pushes the local
commit, upserts the typed result, marks the PR ready, reconciles
`status:review`, and verifies all postconditions. The label itself is the
cross-node notification. A schema-valid result is immutable for its assignment
revision and head: canonical-identical duplicates are one idempotent result,
while different evidence must use a new commit or revision. Result records are
append-only across revision/head identities, and gates select only the record
for the live assignment; stale workers therefore cannot rewrite newer evidence.
Distinct valid results at the same identity fail closed.

Research-base movement is a general property of concurrent research, not a
target-specific benchmark rule. A changed base does not cancel an in-flight
assignment. When deciding a terminal result whose required base differs from
the live base, the advisor must either request a new revision on that live SHA
or call `accept_result_on_current_base`. Acceptance records a durable reason
bound to the exact assignment, revision, result head, canonical structured
result, and live base SHA. It becomes stale when any of those identities or the
result payload changes.

Immediately before a first merge mutation, `merge_experiment` reads the live
Git ref for the assignment's base branch and compares it with
`expected_current_base_sha`. The merge proceeds only when the result's required
base equals that live SHA or an exact matching acceptance exists. Replay of an
already verified merge returns before this ref lookup.

All assignment mutations issued by one workflow instance, plus that worker's
advisor-branch publication and the student's complete preflight/push/result
transaction, share one runtime lock. This closes races among sibling tool calls
in the same process. Separate advisor and student workers still rely on exact
GitHub identities, branch leases, immutable result evidence, and post-mutation
verification; a stale result that loses a revision race restores the current
revision to WIP before failing. GitHub's merge endpoint can precondition the PR
head but not the base SHA, so deployments with external writers need strict
up-to-date branch protection or a merge queue for an atomic cross-process base
guarantee.

Definitive HTTP failures fail clearly. An ambiguous transport failure after a
mutation is resolved by reading and verifying desired state before any retry.

This tool split is a breaking schema change. The removed multi-operation action
has no alias, adapter, or event-log migration. A deployment upgraded across this
boundary must start with fresh OpenHands conversation state; historical GitHub
and W&B records remain durable outside that state.

### Subagent lifecycle

```text
spawn_agents(
  batch_key: str,
  tasks: [{
    key: str | null = null,
    task: str,
    agent: general-purpose | explore | search_general_web |
           search_research_publications | bash-runner = general-purpose,
    model: fast | smart | frontier = smart,
    include_context: bool = false,
  }],
) -> {tasks: [{task_id, key, status, agent, model, result?, error?}]}

await_agents(
  task_ids: [str],
  join: all | first | quorum | change = all,
  quorum: int | null = null,
  timeout_seconds: float,
) -> {join, satisfied, timed_out, changed_task_ids, waited_seconds, guidance,
      tasks: [{task_id, key, status, agent, model, result?, error?}]}

agent_status(
  task_ids: [str] | null = null,
) -> {tasks: [{task_id, key, status, agent, model, result?, error?}]}

cancel_agents(
  task_ids: [str],
) -> {tasks: [{task_id, key, status, agent, model, result?, error?}]}
```

Task status is `queued`, `running`, `finished`, `failed`, or `cancelled`.

Spawning and collection are deliberately separate. `spawn_agents` starts one
batch of Markdown-defined agents in separate process groups and fresh
OpenHands conversations, then returns stable task IDs without waiting for a
model result. `batch_key` is required and stable within the caller
conversation. A task `key` is optional; when omitted, its stable list index is
used. Replaying the same batch and specification returns the same task records;
it never launches duplicate children. Reusing a batch key with a different
task specification fails clearly.

`await_agents` is the only blocking delegation operation. `all` waits for every
selected task to reach a terminal state, `first` waits for any one, `quorum`
waits for the requested number, and `change` returns when any selected task
changes state or immediately when one has an uncollected terminal result. Its
timeout is required and capped at 300 seconds; expiry returns
`satisfied=false`, current records, elapsed time, and next-step guidance without
cancelling unfinished work. Any terminal results included in that response are
marked collected so a later event does not repeat them. `agent_status` is a
non-blocking snapshot. With no
task IDs, it returns up to eight direct tasks that are active or have an
uncollected terminal result; explicit task IDs can retrieve older history.
`cancel_agents` terminates selected pending or running process groups and
durably records their cancelled outcome. Completed results remain collectable
through status or a later await. All three operations accept only task IDs
owned by the calling conversation.

Root advisor and student conversations may continue unrelated work or finish a
turn while tasks remain active. A terminal child result or error is persisted
and resumes the exact root conversation. A nested child must await or cancel
all of its descendants before returning; it cannot detach background work.

One root spawn batch and all descendants form a delegation tree. The tree may
admit at most eight tasks over its lifetime, a single spawn batch is limited to
eight, and the role registry allows at most eight active tasks concurrently
across all trees. Root tasks consume that lifetime budget, so callers must
leave capacity when a General Purpose child needs helpers. A later sequential
root batch forms a new tree. The root is depth zero. It may spawn any registered
agent at depth one, and a depth-one General Purpose agent may spawn leaf helpers
at depth two. Explore, Search, Bash Runner, and every depth-two agent are leaves.
This makes chains such as Explore -> Explore impossible without constraining a
later research phase to the first batch's lifetime budget.

The tree inherits one absolute root-turn deadline. Each task also has a tier
runtime cap: 600 seconds for `fast`, 1,800 for `smart`, and 3,600 for `frontier`.
The effective deadline is the earlier of that cap and the inherited root
deadline. Reaching it interrupts the complete process group and records a
terminal timeout; no descendant survives the tree deadline.

Each tier selects one explicit model-and-effort profile. `model=fast` defaults
to `openai/gpt-5.6-luna` at `high` for mechanical search, command execution,
and extraction. `model=smart` defaults to `openai/gpt-5.6-sol` at `xhigh` for
ordinary review, literature research, synthesis, and failure diagnosis.
`model=frontier` defaults to `openai/gpt-5.6-sol` at `max` with Responses
`reasoning.mode: pro` for the hardest
quality-first work. The provider prefix determines the required credential
(`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`); model-facing calls never select
credential names.

Reasoning effort is validated against the selected model. Provider-specific
request configuration maps GPT-5.6 `max` to Responses Pro mode; invalid
combinations fail clearly. The built-in file agents inherit the selected
profile's effort.

`explore` searches code, data, PR artifacts, and durable history and returns
concise conclusions with paths and line numbers. `search_general_web` uses
Exa's general index with agent-oriented defaults, while
`search_research_publications` uses Exa's publication index and primary papers.
`general-purpose` handles mixed terminal investigation, code editing, task
tracking, tests, and one controlled level of leaf delegation. It is the default
frontier agent, so a frontier task is generalist unless the caller deliberately
selects `explore`, one of the explicit search forms, or `bash-runner`.
`bash-runner` has only the terminal and runs tests, builds, linters, formatters,
dependency commands, Git inspection, or system checks. It normally uses the
fast model and returns counts and actionable failures rather than raw command
output.

With `include_context=false`, the child receives the merged system prompt and
task and may search the parent's durable history path. With
`include_context=true`, it also receives the complete model-visible parent
history, including progressively disclosed skill content.

Each child receives only the tools and progressively disclosed skills declared
by its Markdown definition. Bash Runner is terminal-only. Explore, Search, and
Bash Runner have no delegation tools. A depth-one General Purpose child can use
the lifecycle tools for depth-two leaf work, subject to the same tree budget
and deadline. Children receive neither GitHub credentials nor GitHub
read/write tools; the parent prepares any large PR Markdown artifact and owns
every typed GitHub operation. They do not receive job tools.

When `review_ready` arrives during other advisor work, the advisor can spawn a
smart, full-context General Purpose review and continue unrelated work. Every
terminal record includes its root conversation identity, allowing the
controller to resume the exact advisor or student conversation after its turn.

### Long-running jobs and monitoring

Advisor and student root conversations receive:

```text
run_job(spec: JobSpec) -> JobResult
get_job_status(job_id: str) -> JobResult
cancel_job(job_id: str) -> JobResult
monitor_job(
  job_id,
  wandb_metric=None,
  direction=None,
  gates=(),
  poll_interval_seconds=60,
  stale_after_seconds=600,
) -> MonitorJobObservation
```

The process supervisor owns one process group, the configured timeout ceiling,
TERM/KILL cleanup, restart identity checks using PID/PGID/create-time, a bounded
8 KiB error tail, streamed 64 KiB log parsing, persisted state, and discovered
W&B run IDs. Run IDs are persisted while the job is still running so optional
metric monitoring can begin immediately.

The generic job supervisor confines `cwd` to the role workspace. Jobs declare
`workspace_access` as `mutable` or `read_only`. A student mutable job requires a
clean worktree at launch and holds an exclusive workspace lease, so assignment
hydration, feedback checkout, and branch reconciliation wait until it reaches a
terminal state. Advisor jobs and truly passive student watchers may be
`read_only` and remain usable while notes are being edited. Jobs receive a
scrubbed environment and may request only registered credentials explicitly;
the current public grant is `WANDB_API_KEY`. Every successful `run_job` call
immediately registers a terminal-state monitor bound to the current
conversation. `monitor_job` sets or replaces optional W&B metric gates and
staleness detection for an already-running job; it never disables terminal
wakes.

The timeout is a total wall-clock ceiling, not merely the point at which
shutdown begins. TERM is sent early enough that the configured grace period
ends at the deadline, after which the complete process group is killed.
`cancel_job` follows the same process-group cleanup path and does not return
until the supervisor has persisted a terminal state. Target job code remains
responsible for handling SIGTERM and flushing external services
such as W&B before the grace period expires.

The controller polls only monitors that are due. It fetches one latest selected
metric value from W&B, evaluates deterministic threshold/change/staleness and
terminal-state rules, and persists deduplicated compact signals. Ordinary
polls use no LLM tokens. The monitor store reports its earliest due poll to the
controller, which shortens the ordinary heartbeat sleep accordingly; a
minute-scale job therefore does not wait for the default ten-minute cadence.

Metric samples reject NaN and infinities. Due monitors are ordered and processed
in a capped batch with an overall time budget. A failure in one monitor's job
status or W&B lookup advances that monitor's schedule and emits one deduplicated
`monitor_error` hard signal. A slow external request can delay the current batch
within its timeout and budget, but it cannot prevent later controller cycles,
GitHub events, child results, or an already-pending hard-failure wake. A changed
monitor policy resets its derived samples and signals to match the new persisted
policy. The SQLite monitor store is the single source of truth for ownership,
schedule, and active state; there is no second marker to reconcile.

Every persisted actionable signal directly creates a compact `job_monitor`
wake for the signal's original advisor or student conversation UUID. No
intermediate LLM call gates these events: registering the monitor policy is the
role's request to resume when one of its conditions emits a signal. The
signal remains pending until that exact conversation successfully handles it.

Controller events are partitioned by their exact conversation UUID before a
turn. Each partition is acknowledged only after its own successful turn, so a
child result for one assignment cannot consume or permanently block a job
event for another.

The advisor and student Stop hooks verify that every live supervised job has an
active SQLite monitor record. The student hook also verifies a clean worktree,
allowing its turn to end while the controller supervises the process. Advisor
and student children never receive job tools.

## Hooks, deadlines, and shutdown

The native plugin declares OpenHands `PreToolUse`, `Stop`, and `SessionEnd`
hooks. Its pre-tool hook covers both `senpai_terminal` and the raw `terminal`
used by file-defined children, so delegation cannot bypass workflow or job
boundaries. Hooks give early model-visible feedback. `senpai_terminal` also
evaluates the same pure policy in-process and fails closed if policy evaluation
fails.

The operational supervisor does not load this research-role plugin. Its native
terminal therefore has no Senpai command filter or 600-second Senpai wrapper.
OpenHands' native 30-second no-output continuation behavior, the overall turn
deadline, and infrastructure permissions still apply.
For advisor, student, and child conversations, denied patterns include raw
GitHub mutations, raw `git push`, direct training launches, sleeps, polling
loops, `watch`, and `tail -f`, including nested shell and `env` wrappers.

Every OpenHands turn has a controller-configured hard deadline. The deadline
interrupts the conversation, produces a non-success result, and leaves durable
events unacknowledged. The controller then retries with bounded exponential
backoff. Controller termination interrupts and closes the current conversation,
cancels active supervised jobs, closes local stores, and flushes Weave
before the controller exits. Standalone and child runners flush Weave at runner
exit.

## Secrets and Weave

The entrypoint uses the GitHub write token only for bootstrap, writes it to a
private mode-0600 file under the pod-local `/tmp`, removes the askpass helper,
clears all raw token environment variables, and execs the supervisor. The
supervisor consumes and unlinks that bootstrap file into typed in-process
memory. Before each controller restart it creates a one-shot inherited pipe;
the worker reads and closes that pipe before tool initialization. No raw token
is written to conversation/dataset storage. The long-lived PID 1 environment,
model-facing tool schemas, and agent terminal contain no GitHub token.

Generic child processes receive no GitHub token and no GitHub tools. Main-role
GitHub operations remain typed and lease/state guarded. Terminal and hook
policies are behavioral guardrails, not a credential-containment boundary.

Git operations use a temporary askpass helper rather than a persistent
credential store. The runner repository cannot push, and a target pre-push hook
enforces the exact role/branch matrix. Images run as an unprivileged user, and
the Kubernetes containers drop every Linux capability, disallow privilege
escalation, and use the runtime-default seccomp profile.

Weave content capture applies a longest-first transform over all configured
API keys, tokens, passwords, secrets, credentials, and the selected custom
model credential before content is sent. The pinned `weave-openhands`
integration is initialized before OpenHands imports. Each conversation run is
an agent trace with child LLM and tool spans, all carrying the durable
OpenHands conversation ID. These OTLP records are stored in Weave Agent
Observability and queried with `get_agent_spans()`, not the legacy Calls API;
`OPENHANDS_RUN.weave_url` links directly to the conversation.

## Images and launch acceptance

Three images are built from the same exact source commit:

- advisor: Python/OpenHands, GitHub CLI, Chromium, and pinned `kubectl`; no
  PyTorch or CUDA. Only the separate supervisor pod receives campaign RBAC;
- student: the CUDA/PyTorch stack plus the same OpenHands and Chromium runtime;
- cutoff: a minimal shell/Python runtime with one checksum-verified, pinned
  `kubectl`.

Advisor and student build Chromium and run a browser smoke test. The student
image validates CUDA architecture support. The launcher and cutoff arming
script accept only matching full source-SHA tags or immutable digests and check
out that exact revision.

Launch preflight verifies:

- target-repository push and branch access;
- the Anthropic key;
- the Exa key with one `type="instant"`, publication-category, one-result
  search; and
- the W&B key with a minimal viewer query.

Exa is a progressive skill/script integration, not an always-connected MCP
server.

The Kubernetes launcher creates the fixed role Secret only when it launches an
advisor or students. The operational supervisor owns a separate least-
privilege Secret and ConfigMap. Both supervisor resources are immutable and
content-addressed; a new release creates new objects and leaves the prior
bundle available to the previous ReplicaSet. A supervisor-only launch never
rewrites the role Secret. After applying the supervisor Deployment, the
launcher waits for rollout readiness for the configured timeout and prints an
exact namespace/context-qualified `kubectl rollout undo` command on failure.

When enabled, the launcher also creates one dedicated supervisor
ServiceAccount, namespace-scoped Role, RoleBinding, and Deployment. Kubernetes
RBAC cannot constrain pod list/log/exec by label. The typed tool enforces exact
campaign selectors, but the native terminal can use those verbs anywhere in
its namespace. A campaign that enables the supervisor therefore requires a
dedicated namespace for hard campaign isolation. The Role has no AWS, node,
pod-deletion, or Deployment-mutation verbs. The launcher creates no Service or
general cluster RBAC. Docker and local hosts need no shared network for
advisor/student communication; another deployment backend may implement the
same typed supervisor operation protocol.

The Kubernetes container launcher transfers GitHub, W&B, and model credentials
through a mode-checked, one-use directory, removes them from the environment,
and execs Python. Python consumes and unlinks the directory before OpenHands is
imported. Weave receives W&B authentication only during its synchronous
initialization and the ambient environment is restored immediately afterward.
The native terminal therefore receives no credentials through inheritance or
the Python process's Linux initial environment.

Docker, AWS GPU, and AWS Mac operational-supervisor transports are deliberately
not implemented in this revision. Docker should use a narrow host-side broker
bound to the exact planned container IDs rather than mounting the Docker socket
into the model container. AWS GPU can reuse that broker on its single EC2 host
without granting AWS lifecycle credentials. AWS Mac should run the supervisor
beside the advisor on host zero and reach exact student LaunchDaemons through
campaign-scoped forced-command SSH identities. It must never receive instance
stop/termination or Dedicated Host release authority; upgrades preserve the
allocated Macs, and it must not reuse #3472's broad bootstrap SSH key. The
supervisor retains an unrestricted native terminal locally. Cross-container and
cross-host access uses one fixed `senpai role-control` transport client that can
carry an arbitrary command to an exact configured role. Its broker scopes
reachable campaign runtimes rather than filtering Git or shell syntax,
authenticates through a private per-launch Unix socket or forced SSH command,
loads an immutable role-to-runtime map from the launch plan, rejects unrecorded
containers, hosts, and labels, bounds output and execution time, cleans up
orphaned children, and audits every request and outcome. All transports reuse
the same snapshot, ledger, prompt, and role-control protocol and must pass
scope, replay, restart-safety, and no-host-release tests before their launcher
backend accepts the supervisor flag.

Hivemind startup remains commented with a clear note. The Python controller
waits for the optional cluster start gate while continuously refreshing a
`start-gate` lease; readiness therefore cannot deadlock gated launch. Cluster
launch and cutoff CLIs accept a gate only when it is an absolute normalized
file path beneath their shared PVC mount. Cluster cutoff arms as soon as all
expected resources are Ready or when its bounded readiness window expires,
whichever comes first, and opens the optional start gate in either case. One
missing or crash-looping pod therefore cannot prevent the runtime budget from
starting. At the persisted deadline it deletes launch resources; all
conversation harvest/archive code is removed.

## Removed code

Removed:

- Claude Code and its image install;
- `.claude/` runtime resources;
- Claude-named and OpenHands shell watchdog/supervisor loops;
- the Exa MCP configuration;
- the HTTP advisor service, bearer token, port, probes, and Kubernetes RBAC;
- shell GitHub polling and pod-process inspection;
- cutoff conversation harvesting;
- obsolete tool-role instructions; and
- full skill-body inlining for subagents.

Retained intentionally:

- Agent skills and their model/effort metadata under `.agents`;
- OpenHands Browser, task tracker, and the high-quality default
  condenser for providers not using stored OpenAI Responses continuation or
  Anthropic native compaction;
- the pinned `weave-openhands` agent, LLM, and tool tracing integration; and
- only a small bootstrap shell path for clone, identity, and Git push guards.

The task tracker is described as optional persisted coordination memory for parallel
work, delegated agents, and long-running jobs; it does not impose a single
`in_progress` item. The legacy model-visible `think` scratchpad is omitted from
root and child tool surfaces while provider-native reasoning stays enabled.

## Acceptance

The change is acceptable when:

- unit and local integration tests pass;
- shell scripts pass `bash -n`;
- manifests render matching immutable source revisions without Service/RBAC;
- browser smoke succeeds in both image builds;
- no operational prompt advertises a missing tool or service;
- no runtime role requires Claude Code semantics;
- secrets do not appear in serialized tool specs or captured content;
- monitor wakes resume the original student UUID;
- cutoff arming completes after a bounded readiness window even when a pod
  never becomes Ready; and
- a live credential preflight plus GitHub read-only smoke succeeds before
  production rollout.
