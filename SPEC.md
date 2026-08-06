# Senpai OpenHands runtime contract

Status: implemented on the OpenHands rewrite branch.

## Objective

Senpai is a small deterministic Python control plane around OpenHands.
OpenHands owns research judgment, code changes, evidence interpretation, and
bounded delegation. Python owns operations that should not depend on an LLM
composing fragile tool calls:

- GitHub polling, workflow transitions, and verification;
- assignment branch publication;
- training process supervision and W&B metric monitoring;
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
   assignment revision, and monitor wakes continue it.
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

The supervisor receives one typed operations tool, not a general cluster or
role terminal. It may inspect, enqueue a deduplicated role event, queue a
context reset, or restart the controller process. Targets can name only the
configured advisor or students; callers cannot
supply hosts, pods, namespaces, working directories, environments, or
credentials. Mutations have durable idempotency keys, per-incident cooldowns,
and metadata-only audit records. The enforced cooldown identity is derived from
the typed anomaly category, mutation kind, and exact role target; changing a
free-form incident label cannot bypass it. Role inspection is always fresh and
is never replayed from the mutation ledger. Each fresh supervisor turn receives
the 12 most recent mutation targets, categories, timestamps, and outcomes.

A context reset is an owner-consumed request. The external supervisor records
the expected conversation UUID, controller identity, raw-event prefix digest
and count, and pending-event keys. Only that role's controller may claim it at
a quiescent turn boundary. The controller calls
`run_openhands(..., reset_context=True)`, records completion before ordinary
event acknowledgement, and keeps the same UUID, workspace, complete append-only
event trace, and pending events. External code never instantiates a second
`LocalConversation` over live state or deletes individual events.

A controller restart is refused while a student training process or delegated
agent is running, or when either activity inventory cannot be proven. It
signals only the verified controller PID; the role's existing crash supervisor
performs the restart. The operational supervisor has no AWS, node, pod-delete,
Deployment-patch, experiment-cancel, branch, or PR-mutation authority.

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
- when the live advisor branch moves beyond an active assignment's recorded
  base SHA, `baseline_advanced` gives the advisor both exact SHAs and a compare
  URL without cancelling the student;
- `status:blocked`, `status:needs-rebase`, missing or duplicate student labels,
  stale WIP, and duplicate assignments are advisor-action events; and
- an open Issue labeled `human` plus `team`, the advisor branch, or one student
  label is a human message.

Human Issue events use the exact latest human-authored body/comment ID as their
dedupe key and `human_message_id`. An agent reply updates the Issue but does not
create a new wake for its own comment. `respond_to_issue` verifies the exact
human message before writing an idempotent response.
Launches with human-Issue handling disabled skip that GitHub query entirely.

Assigned-PR issue comments, submitted reviews, and inline comments each use
their immutable GitHub ID as a level-triggered event key. Senpai accepts GitHub
users associated as repository owners, members, or collaborators. A comment by
the authenticated actor containing a Senpai protocol marker is automation, not
human feedback, except for the explicit `senpai-assignment-feedback` transition.
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
owner-consumed reset queue; and `training/monitors.sqlite3`, for student monitor
policy, samples, and deduplicated actionable signals. The operational
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
│   ├── <training-id>.json
│   ├── <training-id>.log
│   ├── monitors.sqlite3
│   └── monitors/<training-id>.json
├── github/
└── conversations managed by OpenHands
```

`student-conversations.json` maps one `(assignment_id, revision_id)` to one
UUID. `conversation-state.json` records, per UUID, both successful initial
instruction delivery and the digest of the delivered merged system context.
The controller replaces this one document atomically after a successful turn,
so a restart cannot observe those two facts at different revisions. A
`training_monitor` event carries its original conversation UUID and therefore
resumes, rather than replaces, the student conversation.

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

Student state is ephemeral by default. Losing it is acceptable after the
assignment ends because the PR, branch, typed result, W&B runs, and Weave trace
are durable. The advisor state is persisted by the deployment.

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
accepts `max` and the `ultra` profile, which uses API `max` effort with
Responses `reasoning.mode: pro`. Automatic OpenAI compaction starts at
200,000 rendered tokens. The OpenHands condenser is disabled for that provider
chain, but its complete local event log remains durable and is used to recover
the latest response ID after restart.

Direct Anthropic models use native server-side compaction with a 200,000-input-
token trigger. OpenHands persists the returned typed compaction block in the
normal event log and replays it first in each later request, including after a
process restart. The local condenser is disabled for these conversations.
Other providers retain the high-quality OpenHands condenser.

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

### `github_transition`

One discriminated tool owns:

- `create_assignment`;
- `push_branch`;
- `reconcile_labels`;
- `request_revision`;
- `send_assignment_feedback`;
- `respond_to_issue`;
- `submit_result`;
- `close_experiment`; and
- `merge_experiment`.

Student publication happens only inside `submit_result`, which validates
repository, PR, assignment, revision, student, current remote head, and
proposed result head before it can push. Assignment identity is required for
feedback, revision, label, close, and merge transitions. Marker comments are
trusted only when authored by the authenticated token actor.

Assignment creation checks the remote base SHA, creates an isolated empty
assignment commit with `git commit-tree`, publishes with force-with-lease,
refuses a second active assignment for the student, creates or reconciles one
draft PR, embeds a typed assignment marker, and verifies routing state.

Advisor feedback carries exact assignment, revision, and head preconditions. It
creates one immutable feedback ID without changing the assignment marker,
draft state, or routing labels, so a nudge reaches the current conversation
without creating a new revision UUID. Exact replay converges; changed guidance
uses a new ID and therefore a new durable GitHub comment event.

Student submission requires a clean assignment branch, lease-pushes the local
commit, upserts the typed result, marks the PR ready, reconciles
`status:review`, and verifies all postconditions. The label itself is the
cross-node notification.

Immediately before a first merge mutation, `merge_experiment` reads the live
Git ref for the assignment's base branch. If it no longer equals the base SHA
recorded in the assignment marker, the transition refuses the merge unless the
advisor deliberately supplies `accepted_base_sha` equal to that exact live
SHA. This keeps the rerun decision scientific while making stale-baseline
acceptance explicit and race-checked. Replay of an already verified merge
returns before this ref lookup.

Definitive HTTP failures fail clearly. An ambiguous transport failure after a
mutation is resolved by reading and verifying desired state before any retry.

### Subagent lifecycle

```text
spawn_agents(
  batch_key: str,
  tasks: [{
    key: str | null = null,
    task: str,
    agent: general-purpose | explore | search | bash-runner = general-purpose,
    model: fast | smart | frontier = smart,
    include_context: bool = false,
    search_mode: general-web | research-publications | null = null,
  }],
) -> {tasks: [{task_id, key, status, agent, model, result?, error?}]}

await_agents(
  task_ids: [str],
  join: all | first | quorum = all,
  quorum: int | null = null,
  timeout_seconds: float,
) -> {join, satisfied, timed_out, tasks: [{task_id, key, status, agent, model, result?, error?}]}

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
selected task to reach a terminal state, `first` waits for any one, and
`quorum` waits for the requested number. Its timeout is required and capped at
300 seconds; expiry returns `satisfied=false` plus the current records without
cancelling unfinished work. `agent_status` is a non-blocking snapshot. With no
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
runtime cap: 300 seconds for `fast`, 600 for `smart`, and 1,500 for `frontier`.
The effective deadline is the earlier of that cap and the inherited root
deadline. Reaching it interrupts the complete process group and records a
terminal timeout; no descendant survives the tree deadline.

Each tier selects one explicit model-and-effort profile. `model=fast` defaults
to `openai/gpt-5.6-luna` at `high` for mechanical search, command execution,
and extraction. `model=smart` defaults to `openai/gpt-5.6-sol` at `xhigh` for
ordinary review, literature research, synthesis, and failure diagnosis.
`model=frontier` defaults to `openai/gpt-5.6-sol` at the `ultra` profile
(`max` effort with Responses `reasoning.mode: pro`) for the hardest
quality-first work. The provider prefix determines the required credential
(`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`); model-facing calls never select
credential names.

Reasoning effort is validated against the selected model and passed through
unchanged. Invalid combinations fail clearly rather than being clamped or
translated. The built-in file agents inherit the selected profile's effort.

`explore` searches code, data, PR artifacts, and durable history and returns
concise conclusions with paths and line numbers. `search` requires exactly one
mode: `general-web` uses Exa's general index with agent-oriented defaults,
while `research-publications` uses Exa's publication index and primary papers.
`general-purpose` handles mixed terminal investigation, code editing, task
tracking, tests, and one controlled level of leaf delegation. It is the default
frontier agent, so a frontier task is generalist unless the caller deliberately
selects `explore`, `search`, or `bash-runner`. `bash-runner` has only the
terminal and runs tests, builds, linters, formatters, dependency commands, Git
inspection, or system checks. It normally uses the fast model and returns
counts and actionable failures rather than raw command output.

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
every typed GitHub operation. They do not receive training tools.

When `review_ready` arrives during other advisor work, the advisor can spawn a
smart, full-context General Purpose review and continue unrelated work. Every
terminal record includes its root conversation identity, allowing the
controller to resume the exact advisor or student conversation after its turn.

### Training and monitoring

Students receive:

```text
run_training(spec: TrainingSpec) -> TrainingResult
get_training_status(training_id: str) -> TrainingResult
cancel_training(training_id: str) -> TrainingResult
monitor_training(
  training_id,
  metric=None,
  direction=None,
  gates=(),
  poll_interval_seconds=60,
  stale_after_seconds=600,
) -> MonitorTrainingObservation
```

`TrainingSupervisor` owns one process group, the configured timeout ceiling,
TERM/KILL cleanup, restart identity checks using PID/PGID/create-time, a bounded
8 KiB error tail, streamed 64 KiB log parsing, persisted state, and discovered
W&B run IDs. Run IDs are persisted while training is still running so metric
monitoring can begin immediately.

The student commits the exact implementation and cleans the worktree before an
expensive launch. Every successful `run_training` launch immediately registers
a terminal-state monitor bound to the current conversation. `monitor_training`
is an optional policy upgrade for useful metric gates or staleness detection;
repeating it replaces the default or previous policy.

The timeout is a total wall-clock ceiling, not merely the point at which
shutdown begins. TERM is sent early enough that the configured grace period
ends at the deadline, after which the complete process group is killed.
`cancel_training` follows the same process-group cleanup path and does not
return until the supervisor has persisted a terminal state. Target training
code remains responsible for handling SIGTERM and flushing external services
such as W&B before the grace period expires.

The controller polls only monitors that are due. It fetches one latest selected
metric value from W&B, evaluates deterministic threshold/change/staleness and
terminal-state rules, and persists deduplicated compact signals. Ordinary
polls use no LLM tokens.

Metric samples reject NaN and infinities. A failure in one monitor's training
status or W&B lookup advances that monitor's schedule and emits one
deduplicated `monitor_error` hard signal; it cannot block other monitors,
GitHub events, child results, or an already-pending hard-failure wake. A changed
monitor policy resets its derived samples and signals to match the new marker.

Every persisted actionable signal directly creates a compact
`training_monitor` wake for the signal's original student conversation UUID.
No intermediate LLM call gates these events: registering the monitor policy is
the student's request to resume when one of its conditions emits a signal. The
signal remains pending until that exact conversation successfully handles it.

Controller events are partitioned by their exact conversation UUID before a
turn. Each partition is acknowledged only after its own successful turn, so a
child result for one assignment cannot consume or permanently block a training
event for another.

The Stop hook verifies the automatic monitor marker and a clean worktree,
allowing the student turn to end while the controller supervises the process.
The advisor and advisor children never receive training tools.

## Hooks, deadlines, and shutdown

The native plugin declares OpenHands `PreToolUse`, `Stop`, and `SessionEnd`
hooks. Its pre-tool hook covers both `senpai_terminal` and the raw `terminal`
used by file-defined children, so delegation cannot bypass workflow or training
boundaries. Hooks give early model-visible feedback. `senpai_terminal` also
evaluates the same pure policy in-process and fails closed if policy evaluation
fails.

Denied patterns include raw GitHub mutations, raw `git push`, direct training
launches, sleeps, polling loops, `watch`, and `tail -f`, including nested shell
and `env` wrappers.

Every OpenHands turn has a controller-configured hard deadline. The deadline
interrupts the conversation, produces a non-success result, and leaves durable
events unacknowledged. The controller then retries with bounded exponential
backoff. Controller termination interrupts and closes the current conversation,
cancels active supervised training, closes local stores, and flushes Weave
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

The Kubernetes launcher creates one Secret, ConfigMaps, role Deployments, and,
when enabled, one dedicated supervisor ServiceAccount, namespace-scoped Role,
RoleBinding, and Deployment. Kubernetes RBAC cannot constrain pod list/log/exec
by label, so the typed backend enforces exact campaign selectors inside that
namespace. Deploy campaigns into separate namespaces when hard authorization
isolation between campaigns is required. The Role has no AWS, node,
pod-deletion, or Deployment-mutation verbs. The launcher creates no Service or
general cluster RBAC. Docker and local hosts need no shared network for
advisor/student communication; another deployment backend may implement the
same typed supervisor operation protocol.

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
- OpenHands Browser, task tracker, Think, and the high-quality default
  condenser for providers not using stored OpenAI Responses continuation or
  Anthropic native compaction;
- the pinned `weave-openhands` agent, LLM, and tool tracing integration; and
- only a small bootstrap shell path for clone, identity, and Git push guards.

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
