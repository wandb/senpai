# Senpai OpenHands runtime contract

Status: implemented on the OpenHands rewrite branch.

## Objective

Senpai is a small deterministic Python control plane around OpenHands.
OpenHands owns research judgment, code changes, evidence interpretation, and
bounded delegation. Python owns operations that should not depend on an LLM
composing fragile tool calls:

- GitHub polling, workflow operations, and verification;
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

The only SQLite databases are `advisor-events.sqlite3`, for unacknowledged
advisor watcher/child events; `student-events.sqlite3`, for unacknowledged
student feedback/child events; and `training/monitors.sqlite3`, for student
monitor policy, samples, and deduplicated actionable signals. OpenHands
conversation history is a separate file-backed per-UUID event log.

## State and conversations

Advisor state:

```text
/var/lib/senpai/<research-tag>/advisor/openhands_state/
├── advisor-conversation-id
├── controller-lease.json
├── advisor-events.sqlite3
├── started-conversations.json
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
├── started-conversations.json
├── training/
│   ├── <training-id>.json
│   ├── <training-id>.log
│   ├── monitors.sqlite3
│   └── monitors/<training-id>.json
├── github/
└── conversations managed by OpenHands
```

`student-conversations.json` maps one `(assignment_id, revision_id)` to one UUID. `started-conversations.json` records the UUIDs that successfully received their initial controller context. A `training_monitor` event carries its original conversation UUID and therefore resumes, rather than replaces, the student conversation.

`github-feedback.json` records every immutable PR feedback key's first-seen
assignment revision, then marks it acknowledged only after its student turn
succeeds. This prevents pending or completed feedback from replaying or
rebinding to a later assignment revision after a restart.

OpenHands stores base state and individual events beneath that UUID. A killed
worker resumes from the last persisted event. An in-flight response or tool
call without a durable event is retried from the preceding event.

The controller marks a conversation's initial controller context delivered only after the OpenHands turn succeeds. A crash or nonzero first turn therefore retries that context instead of incorrectly continuing from information that was never delivered.

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
   - the rendered advisor or student role charter, including its non-secret
     runtime identity; and
   - the selected target-repository `program.md` under
     `# program.md - <path>`; and
   - the rendered `system_instructions/SENPAI-LAUNCH-CONTEXT.md`, containing
     authoritative runtime and isolation rules after `program.md`. A blank
     `program_path` searches root
     `program.md` and one-level `*/program.md` paths and requires exactly one
     total match.
3. Explicit project and Senpai skills through OpenHands skill context. Agent Skills bodies are loaded only when invoked. Repository `AGENTS.md`, `AGENT.md`, and `CLAUDE.md` instruction files are not loaded as project context.
4. User turns containing optional human operator instructions, current state, and current UTC time.

Before constructing a model worker, the supervisor resolves the configured
program path and renders the role's `{{VARIABLE}}` placeholders once from an
explicit non-secret allowlist. A missing referenced value fails the launch;
unrelated environment variables and credentials are never considered. The
rendered role is persisted in role state and reused across worker restarts.

At process startup, the runner loads the harness, rendered role, `program.md`,
and authoritative launch context into one immutable
`SenpaiSystemInstructions` value. Its prompt is the stable system suffix for
that process and is never reread, monitored, or refreshed during the agent
session. Delegated children inherit the rendered role snapshot, resolved
repository-relative program path, and exact launch context, then build their
own immutable value. Runtime identity and `program.md` are not duplicated in
ordinary user messages. Optional operator instructions remain user context;
use GitHub Issues for live human direction. OpenHands includes the system
suffix on every inference, and current time is rendered for every controller
wake. Operators must start fresh role state to apply a changed identity,
program, or role charter.

File-based subagents are discovered from `.agents/agents`. Live advisor and
student skills come only from `plugins/senpai/skills`; `.agents/skills` is for
human operators and Senpai developers and is not installed into pods. Target
repositories may still supply their own project skills. Skill bodies are not
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

Claude Fable 5, Opus 5, and Sonnet 5 profiles pass `max` through as
provider-native `output_config.effort: max` with adaptive thinking. Senpai
never adds the OpenAI-only `reasoning.mode: pro` request body to Anthropic
calls.

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

When a student has more than one configured node, `KubernetesTrainingSupervisor`
keeps the same tool contract while supervising one remote MPIJob. It creates an
atomic Git bundle for the clean `HEAD` on the shared PVC, generates the workload
and W&B identities, launches the target submitter through the local process
path, then persists and polls the broker-created UID. The broker replaces
target-provided init logic with a fixed local-copy and exact-commit checkout, so
bundle mutation fails before training starts. Cancellation, timeout, and restart
recovery remain UID-bound; uncertain deletion retains the broker reservation for
deadline cleanup rather than releasing ownership early.

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

Four images are built from the same exact source commit:

- advisor: Python/OpenHands, GitHub CLI, and Chromium; no PyTorch, CUDA, or
  Kubernetes tooling;
- student: the CUDA/PyTorch stack plus the same OpenHands and Chromium runtime;
- executor: a minimal Python broker with no model runtime or `kubectl`;
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

The Kubernetes launcher creates one Secret, ConfigMaps, and Deployments. For
multi-node students it also creates a namespaced ServiceAccount, Role, and
RoleBinding limited to creating/getting/deleting Jobs or MPIJobs and reading
their pod logs. The controller is CPU-pinned and tokenless. A separate executor
sidecar alone mounts a projected token and validates the exact node/GPU and
CPU/memory allocation, deadline, source/W&B evidence, volumes, pod security,
and workload ownership across a Unix socket. Docker and local hosts need no
shared network for Senpai communication.

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

- runtime skills and their model/effort metadata in the Senpai plugin;
- human and developer guides under `.agents/skills`, outside pod context;
- OpenHands Browser, task tracker, Think, and the high-quality default
  condenser for providers not using stored OpenAI Responses continuation or
  Anthropic native compaction;
- the pinned `weave-openhands` agent, LLM, and tool tracing integration; and
- only a small bootstrap shell path for clone, identity, and Git push guards.

## Acceptance

The change is acceptable when:

- unit and local integration tests pass;
- shell scripts pass `bash -n`;
- manifests render matching immutable source revisions and scoped multi-node RBAC;
- remote workloads remain suspended until their exact created UID is confirmed;
- browser smoke succeeds in both image builds;
- no operational prompt advertises a missing tool or service;
- no runtime role requires Claude Code semantics;
- secrets do not appear in serialized tool specs or captured content;
- monitor wakes resume the original student UUID;
- cutoff arming completes after a bounded readiness window even when a pod
  never becomes Ready; and
- a live credential preflight plus GitHub read-only smoke succeeds before
  production rollout.
