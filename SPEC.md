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
10. Secret values are passed at narrow executor boundaries and redacted before
    monitored content is attached. Custom secret names are explicit.
11. Hivemind is disabled, not redesigned, in this change.

## Control loop and remote protocol

```text
entrypoint
  clone/configure
  exec python -m senpai_agent.supervisor advisor|student

Python supervisor
  start one controller worker process group
  forget stored credentials after handoff
  TERM/KILL descendants and exit when the worker stops or its lease expires

Python controller worker
  poll GitHub + local durable monitor/event state
  reconcile the target checkout
  start one bounded OpenHands turn
  verify durable state
  sleep/backoff/jitter
```

The worker publishes an atomic lease containing its PID, current phase, hard
deadline, completed-turn counter, and active LLM request timestamps. A
non-model-visible heartbeat updates `llm_request_heartbeat_at` while preserving
the request's original `llm_request_started_at`. It does not add conversation
events or renew the hard deadline. The supervisor starts exactly one worker.
A worker crash, clean exit, or expired lease causes descendant cleanup and a
nonzero supervisor exit. An operator stop returns zero. An external process
manager owns restart and backoff. The supervisor is independent of OpenHands
and Kubernetes.
OpenHands events renew the root turn's lease; its configured timeout measures
inactivity rather than total elapsed time. Provider, tool, training, and child
deadlines remain hard.
The supervisor serves `/healthz` on all IPv4 interfaces, using port 8080 by
default. It returns HTTP 200 for a live worker lease and HTTP 503 otherwise.
Kubernetes startup and liveness probes query this endpoint on fixed port 8080
without creating a process inside the container. The images have no Docker
`HEALTHCHECK`. Standalone launchers can set `SENPAI_HEALTH_PORT` and configure
an external monitor with startup grace and retries. After persistent failure,
the monitor restarts the container or repeats host bootstrap. Restarting the
complete entrypoint recreates the one-use credential handoffs.

The health listener monitors one supervisor. Advisor/student communication
continues through GitHub. The core controller imports no Kubernetes API.
Cross-node coordination needs no Service, listening port, DNS record,
ServiceAccount, RBAC, cross-node token, or tailnet.

GitHub state is level-triggered:

- `status:wip` plus exactly one `student:<name>` label is an assignment;
- trusted human comments and reviews on one assigned open `status:wip` or
  `status:review` PR wake its exact student assignment conversation;
- `status:review` is a durable advisor wake;
- a configured student with no open assignment labeled `status:wip` or
  `status:review` emits `student_available_for_assignment`. This event describes
  assignment routing, not the student process or GPU state. A later successful
  GitHub poll retracts the event while it is still queued and unclaimed if the
  student now has an open assignment labeled `status:wip` or `status:review`;
- when the configured research base changes from an active assignment's
  recorded base SHA, `research_base_changed` gives the advisor
  `required_base_sha`, `current_base_sha`, and a compare URL without cancelling
  the student;
- `status:blocked`, `status:needs-rebase`, missing or duplicate student labels,
  stale WIP, and duplicate assignments are advisor-action events; and
- an open Issue labeled `human` plus `team`, the advisor branch, or one student
  label is a human message.

Both roles filter PRs before deriving events or student availability. A PR must
have a head in the target repository and an author with current effective write
or admin access, as reported by the collaborator permission API. Repository and
author names are compared without case sensitivity. Fork heads and missing or
malformed PR trust metadata are rejected. Rejected PRs contribute no assignment,
review, or feedback events; unrelated human Issues retain their existing rules.
Each poll checks each distinct same-repository author once and does not reuse
permissions across polls. A failed permission request or invalid permission
response raises `GitHubReadError` and invalidates the whole GitHub snapshot.
Availability reconciliation therefore leaves queued state unchanged, and
`CompositeMailbox` continues serving other sources.

Human Issue events use the exact latest human-authored body/comment ID as their
dedupe key and `human_message_id`. Each controller delivers an exact version
until one turn processes and acknowledges it, then never delivers that version
again. Failed, interrupted, and recovery turns retain the same delivery. New
trusted human text or a changed versioned title creates a new version and wake.
Trusted messages may share the authenticated actor's GitHub identity; an
authoritative Senpai protocol marker distinguishes agent output and prevents it
from creating a new wake. `respond_to_human_issue` reapplies the same
classification to the exact message before writing an idempotent response.
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
state. It enqueues newly visible GitHub events except student-assignment
availability, which the foreground poll reconciles before the next turn. For
students, it maps authenticated human Issues and assignment-bound PR feedback
into the active UUID. Authenticated humans are the interrupt tier: tools get up
to 60 seconds to finish before Senpai interrupts and resumes the run, even when
its inbox batch is full. Student assignments and trusted PR feedback share a
FIFO queue tier; feedback waits for the next completed agent step without
cancelling it.
Ordinary events remain FIFO. Turn formation and non-human attachments are
bounded to 16 events or 64 KiB; prioritized overflow remains pending to lead
the next turn.
Successfully injected student feedback is acknowledged in
`github-feedback.json` only when the enclosing student turn succeeds.

Generic child results use a local SQLite WAL event store because parent and
child run on the same advisor or student instance. That is not an inter-node
protocol.

The only SQLite databases are `advisor-events.sqlite3`, for unacknowledged
advisor watcher/child events; `student-events.sqlite3`, for unacknowledged
student feedback/child events; and `training/monitors.sqlite3`, for student
monitor policy, samples, and deduplicated actionable signals. OpenHands
conversation history is a separate file-backed per-UUID event log.

A completed tool observation resets the three-attempt no-progress budget. A
separate 36-inference-start backstop applies to each turn branch across worker
restarts without limiting one productive run. Either exhausted budget enters
bounded fresh-branch recovery and then quarantine. Only authenticated human
steering can reopen quarantine; trusted PR feedback remains pending.

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
   - the rendered advisor or student role charter; and
   - the selected target-repository `program.md` as a research-policy snapshot
     with its path, exact source commit, and content digest; and
   - the rendered `system_instructions/SENPAI-LAUNCH-CONTEXT.md`, containing
     authoritative runtime identity, limits, and isolation rules after
     `program.md`. A blank
     `program_path` searches root
     `program.md` and one-level `*/program.md` paths and requires exactly one
     total match.
3. Explicit project and Senpai skills through OpenHands skill context. Agent Skills bodies are loaded only when invoked. Repository `AGENTS.md`, `AGENT.md`, and `CLAUDE.md` instruction files are not loaded as project context.
4. User turns containing optional human operator instructions, current state, and current UTC time.

Before creating pods, the launcher captures one Git-advertised advisor-branch
head in an isolated clone. It recomputes the commit, tree, and program blob IDs
and stores the policy in a separate immutable, content-addressed Secret. Pods
mount only its program-context key as a read-only file. The ConfigMap supplies
the selected path, commit, and content SHA-256; the supervisor verifies all
three against the mounted snapshot before constructing a worker.

The selected path supports printable UTF-8, including spaces and Unicode,
without backslashes or traversal. The committed program must be a regular file
of at most 256 KiB of UTF-8 data; the encoded Secret value may not exceed 1 MiB.
The prompt content omits the SPDX header and outer whitespace.

The supervisor renders the role's `{{VARIABLE}}` placeholders from an explicit
non-secret allowlist. A missing referenced value fails startup; unrelated
environment variables and credentials are never considered. It persists the
rendered role and complete system snapshot. The snapshot digest covers the
components and the exact rendered suffix, so changed wrapper templates also
invalidate persisted context.

The launcher renders `timeout_minutes` and `max_epochs` into the launch context
as agent policy. It does not export dedicated timeout or epoch environment
variables, and the training supervisor has no launch-wide timeout default or
ceiling. Each training run supplies its own positive `timeout_seconds` value.

At process startup, the runner verifies the complete system snapshot against
the digest held by its supervisor or parent. It also checks the configured
program path and commit, harness, rendered role, and launch context against
that snapshot. The resulting `SenpaiSystemInstructions` value stays fixed for
the session. Delegated children inherit that exact value through a file and
an independently supplied digest. Restarts verify persisted context against
trusted launch inputs, without reading the target workspace's current policy.
Runtime identity and `program.md` are not duplicated in
ordinary user messages. Optional operator instructions remain user context;
use GitHub Issues for live human direction. OpenHands includes the system
suffix on every inference, and current time is rendered for every controller
wake. Operators must start fresh role state to apply a changed identity,
program, or role charter. Snapshot integrity does not prohibit publishing
operator-authored `program.md` changes through advisor synchronization.

Every desired role Deployment and live controller Pod under a tag binds the
same program Secret. Terminal Pods do not retain a binding; terminating Pods
retain it until they reach a terminal phase. An advisor-only apply preserves
the original commit and encoded snapshot when normalized policy path and
content match. Launches that include students require an unused tag and cannot
extend an existing launch. Changed policy or legacy roles without a binding
also require a new tag. Advisor-only applies for one cluster, namespace, and
tag must be serialized: the program-binding check and subsequent apply are
not an atomic reservation. A reused program Secret must be immutable, belong
to the tag, and match its content-addressed name. These checks trust
operator-controlled Kubernetes resources; they do not defend against a
Kubernetes administrator replacing the launch inputs.

File-based subagents are discovered from `.agents/agents`. Live advisor and
student skills come only from `plugins/senpai/skills`; `.agents/skills` is for
human operators and Senpai developers and is not installed into pods. Target
repositories may still supply their own project skills. Skill bodies are not
concatenated into agent definitions. The OpenHands fork's `main` branch applies
each agent definition's `reasoning_effort` override after resolving its
inherited LLM or stored model profile.

The images install the runner as a non-editable package and make its Python
environment, built-in agent definitions, and plugin assets root-owned and
read-only. `SENPAI_AGENT_DIR` selects the installed built-in definitions;
`SENPAI_PLUGIN` selects the installed plugin. The supervisor, controller,
delegated children, and plugin hooks use trusted Python with `-P`, so a target
working directory cannot shadow the installed runner. These controls protect
runtime imports and assets; they do not sandbox target code or freeze the
operator's system-instruction files.

`SENPAI_TARGET_PYTHON_ENV` selects a writable target venv for terminals and
training. Its site-packages include the trusted environment through a `.pth`
path entry. Target packages can override those shared packages without writing
to the trusted environment. The image includes pip so additive target installs
can resolve packages on the shared path. uv resolves a separate target package
set and does not inspect that path. The image compiles runtime bytecode before
making the environment read-only. The images copy uv from its versioned,
digest-pinned official image. Bootstrap computes both paths with trusted Python
and uses `uv venv` with that interpreter, without project/config discovery,
Python downloads, or pip bootstrapping. It preserves existing target files and
never executes target Python. Terminal and training environments select the
target through PATH and uv settings. Training and terminal setup remove inherited
`PYTHONSAFEPATH` so project imports work normally. File-defined child terminals
use the same routing.
Each native terminal session receives target settings after shell startup,
including new and recovered tmux panes. Later commands can change that
session's environment. This adapter uses the pinned SDK's environment-export
callback and preserves native parallel terminal execution.
Bootstrap also creates missing target launchers for the trusted environment's
console scripts. Each launcher executes the original read-only script with
target Python, so shared commands and their Python workers see target packages.
Bootstrap preserves existing target scripts and never executes target Python.

The bundled plugin remains explicitly loaded for root and child conversations.
Its skill helpers run in the target environment with `uv run --no-sync`; they
must not rewrite the read-only installed skill files. Target lock updates and
dependency syncs are explicit experiment changes, not a side effect of reading
experiment results.

OpenHands ambient plugin discovery is disabled before root or child
conversations are created. Only the explicitly supplied trusted plugin loads
through the plugin loader. Explicit target skills, unreserved target/user
agents, and their declared MCP configurations remain supported. This removes
automatic plugin hooks, MCP servers, and skills from writable user/project
plugin directories. The adapter depends on the pinned SDK's discovery call;
SDK upgrades must retain the executable plugin-isolation test.

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
detailed available summary. The standalone runner and launcher default to
Claude Opus 5.5 at `xhigh` for advisors and `high` for students. Explicit
GPT-5.6 profiles also accept `max`, which uses API `max` effort with Responses
`reasoning.mode: pro`. Automatic OpenAI compaction starts at
the `compaction_trigger_tokens` value from `senpai.yaml`, which defaults to
200,000 rendered tokens. The OpenHands condenser is disabled for that provider
chain, but its complete local event log remains durable and is used to recover
the latest response ID after restart.

Claude Opus 5.5, Fable 5.1, Fable 5, Opus 5, and Sonnet 5 profiles pass `max`
through as provider-native `output_config.effort: max` with adaptive thinking. Senpai
never adds the OpenAI-only `reasoning.mode: pro` request body to Anthropic
calls.

Direct Anthropic models use native server-side compaction with the same
`compaction_trigger_tokens` input-token trigger. OpenHands persists the returned
typed compaction block in the normal event log and replays it first in each
later request, including after a process restart. Anthropic performs the token
count after provider rendering; Senpai does not load a local tokenizer. This
trigger is not a context-size cap. LiteLLM's normalized `prompt_tokens` can
exceed it because that field sums the compaction and post-compaction sampling
iterations; diagnose compaction from the raw iteration usage and returned
compaction block. Senpai leaves Anthropic's compaction instructions unset so
the provider uses its model-specific native prompt. The local condenser is
disabled for these conversations.
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

Assignment-scoped advisor and student operations share this object:

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
| `post_assignment_comment` | student | `comment_id`, `comment` |
| `request_assignment_revision` | advisor | `new_revision_id`, `required_base_sha`, `comment` |
| `accept_result_on_current_base` | advisor | `expected_current_base_sha`, `reason` |
| `merge_experiment` | advisor | `expected_current_base_sha`, `merge_method` |
| `close_experiment` | advisor | `reason` |
| `respond_to_human_issue` | advisor or student | `issue_number`, `human_message_id`, `response` |
| `submit_experiment_result` | student | `branch`, `remote_branch_sha_before_push`, `result` |

Interim student communication happens through `post_assignment_comment`. The
runtime binds the configured student identity and validates the exact open WIP
or review assignment, revision, and PR head before posting an immutable typed
comment. Exact replay is a no-op; changed text uses a new `comment_id`. The
operation does not push or change the PR head, draft state, or labels, and its
trusted marker wakes the advisor without entering the student's own feedback
inbox. A comment that races with a revision request retains its original
revision identity and is still delivered.

Terminal student publication happens only inside `submit_experiment_result`, which
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

Student submission verifies the assignment branch and exact local result commit,
publishes that commit with a remote-head lease, upserts the typed result, marks
the PR ready, reconciles `status:review`, and verifies all postconditions.
Uncommitted worktree changes are not published. Training separately requires a
clean worktree. The label itself is the cross-node notification. A schema-valid result is immutable for its assignment
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
    model: fast | smart | frontier,
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

Every task must select `fast`, `smart`, or `frontier` explicitly. There is no
implicit model tier.

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

Children are told they can use approximately 1,500 tokens for conclusions and
evidence pointers. If a report exceeds 15,000 tokens, Senpai stores the complete
report under the role state, asks the same child conversation for one concise
summary, and persists only that summary and the local artifact path. A failed
summary returns an error with the artifact path; it never sends the oversized
report to the parent conversation.

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

Each task has an absolute tier runtime cap: 1,200 seconds for `fast`, 3,600 for
`smart`, and 7,200 for `frontier`. A descendant's effective deadline is the
earlier of that cap and its inherited ancestor deadline. Reaching it interrupts
the complete process group and records a terminal timeout; no descendant
outlives an ancestor deadline.

Each tier selects one explicit model-and-effort profile. `model=fast` defaults
to `anthropic/claude-sonnet-5` at `medium` for mechanical search, command
execution, and extraction. `model=smart` defaults to
`anthropic/claude-opus-5-5` at `xhigh` for ordinary review, literature research,
synthesis, and failure diagnosis. `model=frontier` defaults to
`anthropic/claude-opus-5-5` at `max` for the hardest quality-first work. The
provider prefix determines the required credential
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

When `review_ready` arrives during other advisor work, the harness pauses the
advisor at the next safe agent-step boundary and delivers the event into the
same durable conversation without interrupting an active tool. The advisor can
delegate or defer the review, then resume the displaced work. Every terminal
record includes its root conversation identity, allowing the controller to
resume the exact advisor or student conversation after its turn.

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

Every student uses `KubernetesTrainingSupervisor` to supervise one remote
Job for single-node training or MPIJob for multi-node training. It creates an
atomic Git bundle for the clean `HEAD` on the shared PVC, generates the workload
and W&B identities, constructs the Kubernetes workload for the supplied command, then persists
and polls the broker-created UID. The broker replaces
target-provided init logic with a fixed local-copy and exact-commit checkout, so
bundle mutation fails before training starts. Cancellation, timeout, and restart
recovery remain UID-bound; uncertain deletion retains the broker reservation for
deadline cleanup rather than releasing ownership early.

Multiple Senpai instances may share `WANDB_API_KEY`; no per-student key is required.
The executor returns raw diagnostic components over its private socket. The
controller redacts each component before formatting, truncating, or persisting
it. The executor does not receive the W&B key.

`run_training` accepts ordinary command arguments, not a Kubernetes submission
script. It supplies the selected student image, configured resources and PVC
mount, W&B identity, and exact committed source. The worker starts a writable
project environment over the image's read-only runtime. Project setup belongs in
normal dependency files or the command; Senpai requires no setup manifest.
The launch context exposes the training image and how to inspect its base
packages. Multi-node commands run once per node with rank/rendezvous information;
the target remains responsible for its framework's distributed execution.

Checkout and default workers use UID/GID 10001. The checkout trusts only its
exact workspace path for Git ownership checks and retains full commit
verification. No dataset ownership repair is performed. The full configured
PVC uses the same mount path in every role. Checkpoints live below the supplied
`SENPAI_TRAINING_OUTPUT_DIR` and survive worker deletion.

Before agent deployment, storage preflight uses temporary CPU-only Pods in the
selected role images to exercise checkpoint write/flush/read/rename/delete and
advisor/student reads of worker output. Multi-node preflight requires distinct
writer/reader hosts. Failure stops launch before GitHub or controller mutations.
Probes use unique identities and UID-preconditioned cleanup. Dataset paths remain
in `program.md` and are verified by the agent before its first training run.

Controller shutdown detaches from a running Kubernetes workload. It leaves
the remote workload running, keeps the
durable result in the `RUNNING` state, and retains the workload UID and broker
reservation. A restarted controller in the same Pod reserves the same training
identity and re-adopts only that UID before it resumes monitoring. Recovery
requires retained state and the same controller Pod UID; the default state
volumes do not survive Pod replacement. Ordinary controller Pod deletion or
replacement also garbage-collects its owned Job or MPIJob and terminates remote
training. Explicit cancellation and
timeout still delete the remote workload and persist a terminal result before
releasing ownership.

The student commits the exact implementation and cleans the worktree before an
expensive launch. Before reserving resources or starting a process, `run_training`
reads the current open WIP assignment from GitHub and checks that its revision
maps to this conversation in `student-conversations.json`. Missing, ambiguous,
unreadable, or superseded assignments prevent a new launch. This admission check
does not affect monitoring, cancellation, or terminal delivery for existing runs.
Every successful `run_training` launch immediately registers
a terminal-state monitor bound to the current conversation. `monitor_training`
is an optional policy upgrade for useful metric gates or staleness detection;
repeating it replaces the default or previous policy.

Each run's requested timeout covers submission and training. Expiry triggers
UID-bound workload deletion; the broker independently enforces the launch's
maximum training deadline. `cancel_training` uses the same deletion path and
does not return until the supervisor has persisted a terminal state. Target
training code remains responsible for handling Kubernetes termination and
flushing external services such as W&B.

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

The Stop hook always verifies the automatic monitor marker and normally
requires a clean worktree. While queued PR feedback waits for a safe boundary,
a role-local marker waives only the clean-worktree check; the pump clears it
before delivery and on entry and exit.
The advisor and advisor children never receive training tools.

## Hooks, deadlines, and shutdown

The native plugin declares OpenHands `PreToolUse`, `Stop`, and `SessionEnd`
hooks. Its pre-tool hook covers both `senpai_terminal` and the raw `terminal`
used by file-defined children, so delegation cannot bypass workflow or training
boundaries. Hooks give early model-visible feedback. `senpai_terminal` also
evaluates the same pure policy in-process and fails closed if policy evaluation
fails.

Recognized denied patterns include raw GitHub mutations, raw `git push`, direct
training launches, sleeps, `watch`, and `tail -f`, including nested shell and
`env` wrappers.

The terminal policy parses Bash syntax before checking nested commands and
recognized command runners. It rejects malformed syntax, startup-file loading,
explicit shell callbacks, aliases, and variable-name reevaluation. Shell startup
and prompt variables are also reserved against custom-secret injection.
These checks enforce workflow boundaries without
prescribing research methods or requiring an allowlist of data formats and
analysis languages. Heredoc input to ordinary programs remains data. The
original Bash syntax still exposes expansions in unquoted input for checking;
input fed to recognized shells receives recursive shell checks. A function
that overrides the actual consumer, or output routed through an opaque `exec`
redirect, retains conservative checking. Unrelated functions and process
substitutions do not disable ordinary program input. Dynamic output paths are
allowed unless recognized shell execution in the same command makes that
stream ambiguous. Shell loops, arithmetic, variable executable names, and
variable timeout durations are allowed. Commands visible inside loop bodies,
conditions, and substitutions remain checked. The policy does not resolve
variable contents or prove loop termination; operations selected indirectly
through variables can fall outside its recognition. Common wrappers inspect
the child command visible in the submitted syntax and preserve its data
arguments. Unsupported wrapper grammars and unclear shell streams can still
reject valid commands. These
checks do not inspect arbitrary executable files or Python code, reconstruct
prior terminal state, or establish a shell sandbox.

Every OpenHands turn has a controller-configured hard deadline. The deadline
interrupts the conversation, produces a non-success result, and leaves durable
events unacknowledged. The controller then retries with bounded exponential
backoff. Controller termination interrupts and closes the current conversation.
It detaches from active Kubernetes training so the next controller in the same
Pod can re-adopt the same remote UID. It then closes local
stores and flushes Weave before it exits. Standalone and child runners flush
Weave at runner exit.

## Secrets and Weave

The entrypoint uses the GitHub write token only for bootstrap, writes it to a
private mode-0600 file under the pod-local `/tmp`, removes the askpass helper,
clears all raw token environment variables, and execs the supervisor. The
supervisor consumes and unlinks that bootstrap file into typed in-process
memory. It creates one-use inherited descriptors for one controller, then drops
its stored credentials and copied worker environment. The worker reads and
closes the descriptors before tool initialization. Bootstrap also hands off
W&B and Exa through private files. The controller restores those service keys
at startup before importing the runner so research access and import-time Weave
tracing continue to work. Runner configuration retains Exa in trusted runtime
memory and removes it from the environment before agent tools run. W&B remains
in the environment for research tools and tracing. Standalone launches may omit
services they do not use. When a
service key is set at supervisor startup, its private file handoff is required;
a raw key without that handoff fails startup. No raw token is written to
conversation/dataset storage. The long-lived PID 1 environment, model-facing
tool schemas, and agent terminal contain no GitHub token.

Authenticated Git publication runs `/usr/bin/git` in a disposable bare repository.
The controller supplies the GitHub URL from its configured repository, disables
hooks and credential helpers, ignores global and system Git configuration, and
clears inherited Git and proxy settings. The network process never reads the
checkout's Git configuration. Push staging shares only the checkout's object
directory and shallow boundaries, then verifies the staged commit SHA.
Assignment creation fetches the base at depth one; idempotent replay fetches the
assignment at depth two to verify its parent, tree, and message. These fetches do
not change the advisor checkout. Pushes retain expected-SHA checks, ancestry
checks, exact ref leases, and post-push verification. After verified publication,
including an idempotent retry, a credential-free local Git command updates
`refs/remotes/<remote>/<branch>` to the published SHA. It compares the ref with
its value before network work and preserves concurrent changes. It does not
follow symbolic refs, run hooks, or move the working branch, HEAD, index, or files.
Other local update failures report that publication succeeded so a retry can
repair the tracking ref. The bootstrap runner and
target pre-push hooks remain behavioral guards; typed publication bypasses them
and applies its own branch and lease checks. Before creating a remote branch,
the typed assignment tool requires a configured student and a `<student>/`
branch prefix.

Delegated model credentials use a bounded JSON bundle in an unnamed file. The
parent passes its descriptor to the child and closes its copy after spawning.
The child closes the descriptor after reading and resolves configuration from
an in-memory mapping. The same private bundle carries Exa to every delegated
runtime so general-purpose children can delegate to search grandchildren.
Exa stays outside shell environments, tool parameters, and conversation secrets.
The root and search agents expose `exa_search`; all child runtimes retain the
key in trusted process memory, including those that can delegate onward. This
boundary does not protect against compromise of the trusted runtime or a
privileged host process.
W&B conversation secrets remain available to child tools. W&B inference still
shares `WANDB_API_KEY` until the W&B identity
cutover. No per-student W&B key is required by this handoff change.

The supervisor, controller, and runner disable process dumping on Linux,
including standalone runner invocations. This also disables core dumps and
ptrace-based debugging of those processes. After capturing the worker
environment, the supervisor removes current model-provider values and discards
its environment copies. Removing an `os.environ` entry does not erase the
kernel's original startup environment.
Model values can remain there until process exit. A same-UID process can also
race a delegated child's inherited descriptor before Python disables dumping.
These measures reduce exposure; they do not establish complete same-UID secrecy.

The supervisor cleans up the worker and observed descendants under one shared
60-second grace allowance, below the 75-second liveness termination grace.
It reserves the smaller of one second or half that allowance for SIGKILL and
reaping. Worker and adopted-child TERM waits share the earlier cutoff.
When it runs as container PID 1, it also terminates adopted descendants. On a
host where it is not PID 1, detached children can become orphans between polls.
Before restarting the entrypoint, the host process manager must terminate every
descendant process group, including groups created by detached children, or
terminate the workload's cgroup.
Existing post-SIGKILL waits can still depend on kernel process termination.
Kubernetes or another process manager must restart the complete entrypoint;
restarting only the Python supervisor cannot recreate consumed handoff files.
The supplied manifests use separate pod-local `emptyDir` volumes for role state,
the target checkout at `/workspace/senpai/$PROBLEM_DIR`, and the writable target
environment at `/home/senpai/.venvs/senpai-target`. A container restart in the
same Pod retains all three. Bootstrap preserves the existing target branch,
uncommitted files, unpushed commits, and installed dependencies. It does not
checkout, pull, or reset an existing advisor checkout. The runner checkout and
the rest of HOME are recreated. Pod replacement starts fresh role state,
checkout, and target environment; the dataset PVC follows its own lifetime.
An existing target without a valid HEAD commit fails bootstrap with an
operator-repair message. Bootstrap preserves all files and refs instead of
deleting or resetting an incomplete checkout. Isolated Python Git commands trust
only their exact resolved working directory through command-scope configuration;
they continue to ignore global and system Git configuration.

Generic child processes receive no GitHub token and no GitHub tools. Main-role
GitHub operations remain typed and lease/state guarded. Terminal and hook
policies are behavioral guardrails, not a credential-containment boundary.

`custom_secret_env_names` is an explicit, shared list of additional
environment-variable names. Names must be unique and match
`[A-Za-z_][A-Za-z0-9_]*`. Built-in launch credential names and names beginning
with `GH_`, `GITHUB_`, or `SENPAI_` are reserved. Launcher-owned and
process-control environment-variable names are also reserved. The launcher
resolves each value from the shell and then the repository-root `.env`. It
reads `.env` values literally without variable interpolation. A missing listed
value fails the launch. It writes values only to the per-launch Kubernetes
Secret and injects them into every advisor and student environment. The
corresponding names, but not the values, are model-visible so agents can
reference them during tool execution. OpenHands makes the values available at
execution boundaries and propagates them to delegated children. Dry-run
manifests validate names and contain deterministic placeholders instead of
resolved values. Custom secrets receive no service-specific
authentication preflight.

Git operations use a temporary askpass helper rather than a persistent
credential store. The runner repository cannot push, and a target pre-push hook
enforces the exact role/branch matrix. Images run as an unprivileged user, and
the Kubernetes containers drop every Linux capability, disallow privilege
escalation, and use the runtime-default seccomp profile.

Weave content capture applies a longest-first transform over all configured
API keys, tokens, passwords, secrets, credentials, custom secrets, and the
selected custom model credential before content is sent. Custom secrets do not
depend on naming conventions for redaction. The pinned
`weave-openhands` integration is initialized before OpenHands imports. Each
conversation run is an agent trace with child LLM and tool spans, all carrying
the durable OpenHands conversation ID. These OTLP records are stored in Weave
Agent Observability and queried with `get_agent_spans()`, not the legacy Calls
API; `OPENHANDS_RUN.weave_url` links directly to the conversation.

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
- every model-provider credential referenced by the configured profiles;
- the Exa key with one `type="instant"`, publication-category, one-result
  search;
- the W&B key with a minimal viewer query;
- the presence of every configured custom secret, without attempting
  a service-specific authentication check;
- a matching published image set, resolved to immutable references; and
- checkpoint writes and cross-role reads on the configured volume using
  temporary non-root Pods, with cross-node access for multi-node launches.

Exa uses a credential-isolated native `exa_search` tool with progressive skill
guidance. It preserves the standalone script's request controls and web/publication
counts of 10/30 results, with up to 100 per call. Both modes default to
`deep-reasoning`. Unless `no_content` is set, the tool requests
`text={"verbosity": "full"}` without a character limit and defaults to
`max_age_hours=0` so the extraction setting applies to a fresh crawl. Explicit
cache-age values remain supported. Returned text is retained with its original
line breaks; missing text is reported. These are Exa's extracted contents, not
original PDF/HTML files or a guarantee of complete-paper coverage. The legacy
operator script keeps its original defaults. The root tool remains declared when
a standalone runtime has no Exa key so persisted conversations can resume.
Calls without configured credentials or conversation persistence fail
before contacting Exa. Every response persists the complete Markdown in the
conversation's observations directory and returns its file path and character
count. Responses above 30,000 characters return an explicit preview. A failed
write fails the tool call instead of losing evidence. Local conversation cleanup
retains these files, so parents can read results from completed search children.
The preview fits below the pinned SDK's 50,000-character tool-message limit; SDK serializers
and other tools retain their existing limits. The legacy terminal path also
previews at 30,000 characters. Complete evidence is available through bounded
file reads; it is not sent to a model in one unbounded message.

The Kubernetes launcher creates a credential Secret, a separate immutable
program-context Secret, ConfigMaps, and Deployments. A student launch first scans every requested name for an existing Deployment,
controller Pod, or nonterminal labeled Job. It scans MPIJobs only when the API
exists and requires that API for multi-node students. After all scans pass, it
creates the per-tag credential Secret as an atomic, durable, immutable reservation before
any GitHub write. An advisor-only apply cannot change the Secret data after a
student launch wins the reservation race. It then creates each student resource
separately, with the Deployment last. Student resources are new-only; the
launcher never applies over them. A concurrent launch or an orphan from a
failed launch therefore fails closed and requires explicit operator cleanup
before retry.

For every student the launcher also creates a namespaced ServiceAccount,
Role, and RoleBinding that allow creating, getting, patching, and deleting Jobs;
getting and listing Pods; reading pod logs; and listing Events. Multi-node
students also receive the same workload permissions for MPIJobs.
The launcher's operator identity needs get access to Deployments, list access
to Pods and Jobs, API discovery, and create access to the rendered resources.
Creating the Role and RoleBinding also requires every delegated permission in
that namespace, or explicit `escalate` permission for the Role and `bind`
permission on the referenced Role, respectively.
When the MPIJob API is installed,
every launch needs list access to MPIJobs so it can reject orphaned workloads;
multi-node preflight also requires the API to exist. The controller requests
CPU and memory without GPUs and has no Kubernetes token. Controller node
placement is unconstrained unless the operator sets `controller_node_selector`.
A separate executor sidecar alone mounts a projected token and validates the
exact node/GPU and CPU/memory allocation, deadline,
source/W&B evidence, volumes, pod security, and workload ownership across a
Unix socket. Docker and local hosts need no shared network for Senpai
communication.

An opt-in capacity observer is separate from the research controllers and
training executor. It lists Nodes and nonterminal Pods through a dedicated
credentialed process and updates only its named, precreated snapshot ConfigMap.
The advisor and students mount that sanitized snapshot read-only and expose
`get_cluster_capacity` without Kubernetes credentials or mutation inputs.
The observer needs explicit cluster-scoped read RBAC; namespace-only executor
permissions stay unchanged. Observation shape, selectors, tolerations, and any
CoreWeave verification exemption are operator-owned configuration. All three
worker resources must fit on the same eligible node. Missing, incomplete,
malformed, or stale snapshots return unknown. This is timestamped advisory
capacity, never a reservation or assignment transition; scheduling remains
Kubernetes' responsibility. Raw Pod specifications, environments, logs, and
unrelated project identifiers never enter the snapshot. Cluster-scoped observer
RBAC cleanup remains an operator action rather than expanding cutoff authority.

The snapshot includes the complete observation configuration. Optional
`expected_requirements` compares resource shape, node selectors, tolerations,
and preemption policy with that configuration. Toleration order and duplicates
do not affect the comparison. A mismatch returns unknown, preserves the observed
configuration, and removes capacity counts. Matching requirements do not assess
affinity, topology, quotas, or PVC placement. Observation defaults to empty
tolerations for all topologies. Operators must configure observation tolerations
to match target-owned worker manifests; CPU student controllers do not define
worker placement. Explicit observation settings do not change worker placement.

Hivemind startup remains commented with a clear note. The Python controller
waits for the optional cluster start gate while continuously refreshing a
`start-gate` lease; readiness therefore cannot deadlock gated launch. Cluster
launch and cutoff CLIs accept a gate only when it is an absolute normalized
file path beneath their shared PVC mount. Cluster cutoff arms as soon as all
expected resources are Ready or when its bounded readiness window expires,
whichever comes first, and opens the optional start gate in either case. One
missing or crash-looping pod therefore cannot prevent the runtime budget from
starting. The operator fixes the readiness deadline and latest cutoff time
when arming the Job. The runtime budget begins on readiness or timeout, capped
by that latest cutoff time across restarts. The Job authenticates persisted
JSON state with a per-arm key and never sources shared files. It rejects
symlinks and non-regular state files. State reads and temporary writes use
nonblocking opens to avoid FIFO hangs. The Job keeps an in-memory deadline
when state persistence fails. Failed
start-gate writes retry only until the cutoff deadline.

At the deadline, the Job deletes matching Deployments. It runs as UID/GID
10001 with a read-only root filesystem, no added capabilities, and no privilege
escalation. Its namespace Role permits pod observation and Deployment deletion;
it grants no Secret or ConfigMap access. ConfigMaps, Secrets, PVC data, and
other launch resources remain for explicit operator cleanup. Use the
`research-tag` selector to delete retained launch ConfigMaps and Secrets as
documented in README.md. The cutoff Job, its script ConfigMap, and its shared
RBAC resources are separate from those launch labels. All conversation
harvest/archive code is removed.

## Removed code

Removed:

- Claude Code and its image install;
- `.claude/` runtime resources;
- Claude-named and OpenHands shell watchdog/supervisor loops;
- the Exa MCP configuration;
- the old HTTP advisor service and its bearer token, port, probes, and Kubernetes RBAC;
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
- manifests render matching immutable source revisions and scoped training RBAC;
- remote workloads remain suspended until their exact created UID is confirmed;
- browser smoke succeeds in both image builds;
- no operational prompt advertises a missing tool or service;
- no runtime role requires Claude Code semantics;
- secret values do not appear in serialized tool specs or captured content;
- every configured custom secret reaches advisor, student, and child tool
  execution with its output and trace content redacted;
- monitor wakes resume the original student UUID;
- cutoff arming completes after a bounded readiness window even when a pod
  never becomes Ready; and
- a live credential preflight plus GitHub read-only smoke succeeds before
  production rollout.
