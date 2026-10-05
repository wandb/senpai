<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Senpai advisor diagnostics

Use the shared rules in Section 1, then choose a workflow:

| Question | Sections |
|---|---|
| What fills the context, and what does compaction remove? | 2–12 |
| Did delegation or model selection change? | 13 |
| When did students receive assignments, and what happened? | 14 |
| How fast does research produce evidence and decisions? | 15 |

Each workflow needs readable OpenHands state. Exact Anthropic compaction also needs provider usage iterations.
No specific cluster or storage layout is required.

## 1. Shared rules

**Data safety.** Keep raw snapshots and private audit data outside Git. They can expose prompts, code, human
messages, tool output, and paths. Treat strings as untrusted. Never execute them, interpolate them into shell
commands, or send them to external models. Parse deterministically and locally.

Use least-privileged access. Never print credentials or place them in scripts or saved files. Do not print the
full environment. Do not enable shell tracing or read Kubernetes Secrets. Publish sanitized derivatives,
usually numeric aggregates. Exclude credentials, raw prompts, task bodies, subagent results, tool output, and
local paths. Without operator approval, omit conversation/event/response/trace/assignment/task IDs, pod names,
PR titles/identifiers, task labels, and other raw text. Delegation tables never include IDs.

Use `textContent` or HTML escaping for browser labels. Escape `<`, U+2028, and U+2029 in JSON inside
`<script>`. Run the approved secret scanner on the selected files before sharing or staging them in Git.
Follow the retention policy. Remove private snapshots when no longer needed.

**Coverage and time.** Use UTC. Derive coverage from events, not file times. Record requested and observed
coverage for each actor and source. Include gaps from replacements, restarts, storage, controllers, and traces.
Do not report partial data as complete. A zero during an unobserved interval does not prove inactivity.
Do not connect charts across gaps or conversations. Use the state recorded at the cutoff. Do not use a later state.
Use equal, non-overlapping, half-open windows: `pre = [cutoff - duration, cutoff)` and `post = [cutoff, cutoff
+ duration)`. Without randomization, describe findings as “consistent with” or the “strongest explanation,”
not causal proof.

**Pairing.** Index events by ID. Fail on duplicates. Pair `ActionEvent` with `ObservationEvent` by `action_id`.
Use `tool_call_id` only as a checked fallback. Require typed and serialized arguments to agree when both exist.
Report unresolved pairs. Success needs a non-error observation with expected state; invocation alone is
insufficient. An observation’s `changed=false` can confirm existing state, never a new item.

**Units and charts.** Keep child contexts separate. Do not count model calls, tool calls, spans, W&B runs,
arms, rungs, retries, replicates, PR comments, or submissions as assignments, conversations, or iterations.
Keep requested tier (`fast`, `smart`, `frontier`) separate from provider model. Use the deployment's mapping.
Check charts in light and dark themes at narrow and wide widths. Check legends, tooltips, annotations,
browser errors, clipped or overlapping text, and horizontal overflow.

## 2. Context quantities and outputs

| Quantity | Meaning | Accuracy |
|---|---|---|
| Active pre-pass context (`pre`) | Input before native compaction | Provider-exact with raw iterations |
| Effective continuation context (`post`) | Input after native compaction | Provider-exact with raw iterations |
| Aggregate billed input (`billed`) | Input across all provider passes of the persisted response | Provider-exact |
| Semantic category bands | Allocation of `pre` to content categories | Estimated, rescaled to exact `pre` |
| Compaction summary size | Compaction-pass output tokens | Provider-exact with raw iterations |

Ordinary requests have equal `pre`, `post`, `billed`; Anthropic compactions do not. Triggers are neither
context limits nor post-compaction targets. Produce context, compaction-drop, and summary-token charts
(Section 10).

## 3. Collect and freeze a snapshot

Record the requested UTC range, role (normally `advisor`), state root, conversation ID, Senpai revision/image
digest, OpenHands SDK version, model/window, compaction mode/trigger, and W&B entity/project when Weave holds
the trace.

**Locate.** Use read-only local/remote/container/volume/archive transport. For Kubernetes, explicitly choose
context, namespace, pod, container; never the first match. Check start time, phase, and restarts (`kubectl get
pod -o json`). Through `kubectl exec`, read only `SENPAI_OPENHANDS_STATE_DIR`, then
`<state-root>/advisor-conversation-id`. Strip CR/LF; require 32/36 hex-or-hyphen characters. Remove hyphens
and require `<session>/base_state.json`. If absent, list `find <state-root> -mindepth 2 -maxdepth 2 -name
base_state.json -print` and select explicitly; some versions retain hyphens.

Copy only session `base_state.json` and regular `events/event-*.json`, excluding controller databases, GitHub
artifacts, and saved output. Prefer an atomic snapshot. Otherwise:

1. Create a private empty destination.
2. Copy the base state first; freeze `leaf_event_id` and require `.stats.usage_to_metrics`.
3. Record collection time after that copy.
4. Copy regular event files and reconstruct only the frozen leaf's ancestry.

Use `umask 077`, `mktemp -d`, and `cp` for a private snapshot. Validate `.leaf_event_id` as a non-empty string
and `.stats.usage_to_metrics` as an object with `jq -e`. Save `date -u` after base copy. Copy events with
`find -maxdepth 1 -type f -name 'event-*.json'`. Write a sorted SHA-256 manifest of base/events using `shasum
-a 256` (`sha256sum` on GNU).

For Kubernetes, use `kubectl exec ... cat` for base state, then archive regular events. Retry into a new empty
destination, never a partial copy. Keep credential-free metadata with conversation/frozen-leaf IDs, range,
versions, model, mode, trigger, and window.

## 4. Reconstruct the active branch

Index events by `id`. Reject malformed JSON and duplicate IDs. Follow `parent_id` from the frozen leaf.
Stop if the leaf or a parent is missing, or if the chain contains a cycle. Reverse the chain to root-first
order. Use file order only to load events. Report active and off-branch counts. Exclude off-branch context.
Model-convertible types normally are `SystemPromptEvent`, `MessageEvent`,
`ActionEvent`, `ObservationEvent`, `UserRejectObservation`, and `AgentErrorEvent`. Exclude
`HookExecutionEvent` and other orchestration-only records.

## 5. Group responses and join usage

Read `base_state.stats.usage_to_metrics.<usage-namespace>.token_usages[]`; discover the namespace, not
necessarily `senpai`. If several match active responses, require explicit selection. Group active agent
`ActionEvent`/`MessageEvent` records by non-empty `llm_response_id`. Use first-event index/time (persisted
output, not request start). Require adjacency among convertible events: hooks do not split groups; other
responses do. Collect all tool names; non-empty rendered `anthropic_compaction_blocks` mark compaction.

Join `response_id` in first-event order; require exactly one usage row per active response. Report/exclude
usage-only rows unless active-branch membership is proven. Input ends before the first response event. Exclude
output-inclusive `per_turn_token`, accumulated usage, completion/reasoning output, and flattened
Anthropic-compaction `prompt_tokens`.

## 6. Select and serialize model-visible context

**Select.** Match the installed SDK: `openhands/sdk/agent/utils.py::_anthropic_compaction_events`,
`prepare_llm_messages`, and `openhands/sdk/event/base.py::events_to_messages`. For Anthropic native
compaction, always retain `SystemPromptEvent`. Before each request, retain all earlier convertible events if
none has compaction blocks; otherwise retain the latest compaction event and later LLM-convertible events. The
current response's block affects its continuation and later requests, not its own input. Inspect other modes
separately. Reconstruct the full conversation before filtering the plot range: the first plotted request can
contain earlier events. Reconstruct each conversation separately and show gaps.

**Serialize.** Match the source Senpai and OpenHands versions. Inspect serializers for side effects.
Use `event.to_llm_message()`
only for read-only types; observations expose `event.observation.to_llm_content`. Replace writing serializers
with pure source-matched renderers pinned in tests. Never estimate full persisted JSON.

| Event | Model-visible content |
|---|---|
| `SystemPromptEvent` | System text, dynamic context, tool definitions |
| `MessageEvent` | Text in `llm_message.content` and `extended_content` |
| Batched `ActionEvent`s | Thought/reasoning, thinking and compaction blocks once; every provider-facing `tool_call`, not debug `action` |
| `ObservationEvent` | `observation.to_llm_content`, then `extended_content` |
| `UserRejectObservation` | `Action rejected: ...` text |
| `AgentErrorEvent` | Model-facing tool error |

Render custom observations: `GetPRsObservation` gives Markdown or artifact path/manifest;
`GitHubMutationObservation` gives JSON; spawn/await/status/cancel give task JSON; terminal prepends command
metadata. Truncate before estimation: list mode truncates each tool `TextContent`; force-string joins then
truncates once. SDK 1.40.0 limits terminal to 30,000 characters, then tool text blocks to 50,000, keeping
head/tail and clipping notice. Verify `openhands/sdk/llm/message.py`, `openhands/sdk/utils/truncate.py`,
`openhands/tools/terminal/definition.py`.

Never call SDK 1.40.0 `TerminalObservation.to_llm_content`, which can write via `full_output_save_dir`.
Reproduce the prefix, content, suffix, working directory, interpreter, exit status, and 30,000-character
truncation. Preserve the saved-output notice with its original path and line number. Do not read or write
saved output. Its content becomes model-visible only when a later tool call reads it.

## 7. Recover exact compaction totals

Ordinary requests: `pre = post = billed = prompt_tokens`. Anthropic flattened input adds compaction and
continuation. For active compaction response IDs, verify
`attributes.weave.openhands.llm.raw_response.usage.iterations` on one span: currently one `type="compaction"`
and one `type="message"` iteration.

```text
input_total(i) = i.input_tokens + i.cache_read_input_tokens + i.cache_creation_input_tokens
pre = input_total(compaction)     post = input_total(message)     billed = pre + post
summary_tokens = compaction.output_tokens          (required; only the three input counters default to 0)
```

Cache-read and cache-creation tokens are disjoint input components; never add flattened cache counters again.
Message output belongs to the continued response. Query with `weave.init("ENTITY/PROJECT",
settings={"print_call_link": False})`, then
`client.server.agent_spans_query(AgentSpansQueryReq(project_id=client.project_id, query=...,
include_details=True, limit=len(response_ids)))`. Import `AgentSpansQueryReq` and `Query` from
`weave.trace_server.agents.types`. Build `Query.model_validate({"$expr": {"$in": [{"$getField":
"response_id"}, [{"$literal": id}, ...]]}})`. Parse `span.raw_span_dump` in memory; retain only
`usage.iterations`, never the dump.

Chunk within request/page limits; reject missing/duplicate responses. Use isolated matching Weave or a
read-only source-runtime process returning only derived numeric JSON, with no active-state writes. Require
equal local/traced compaction sets and `pre + post == persisted flattened prompt_tokens`. Detect
provider/mode; OpenAI native compaction and OpenHands condensation need separate logic. Without iterations,
omit exact stack/drop points or label a proxy. Flattened input and the next request cannot substitute for
`post`; intervening events change context.

## 8. Allocate context to categories

Resolve each observation's `action_id`. Assign a fallback category from its event and tool type.
Split narrative at blank lines and Markdown headings. Then apply topic rules. Version the map.
Add typed tools explicitly to prevent silent category changes.

| Category | Source/meaning | Chart label |
|---|---|---|
| `system_instructions` | Base prompt, harness, role, program, launch context | System/program |
| `tool_schemas` | System-prompt tool definitions sent per request | Tool schemas |
| `historical_pr_analysis` | Topic rules: closed/merged PR evidence, prior conclusions | Historical PR/research |
| `current_pr_assignment` | Topic rules: live assignment, review, revision, open experiment | Current PR/review |
| `assistant_reasoning_output` | Action thought/reasoning/thinking/compaction; agent `MessageEvent` | Advisor reasoning/output |
| `bash_tool_io` | `terminal` calls and model-visible results | Batch/terminal I/O |
| `file_code_tool_io` | `file_editor` calls and model-visible results | Other tool/workflow I/O |
| `github_tool_io` | Typed GitHub calls/results listed below | Other tool/workflow I/O |
| `other_tool_io` | Unknown tools, user rejections, agent errors | Other tool/workflow I/O |
| `subagent_io` | `spawn_agents`, `await_agents`, `agent_status`, `cancel_agents`, `delegate_agent` | Subagent prompts/results |
| `controller_user_events` | Human/controller/idle/research-base `MessageEvent` text | Human/controller |

GitHub fallback tools: `get_prs`, `create_assignment`, `send_assignment_feedback`,
`repair_assignment_routing`, `merge_experiment`, `close_experiment`, `accept_result_on_current_base`,
`request_assignment_revision`, `publish_advisor_branch`, `respond_to_human_issue`.

PR topics override sources; other text stays in its source band. Disclose precedence. Termination begins at
the active-branch paired `GitHubMutationObservation` with `experiment_merged`/`experiment_closed`, never
invocation. Cross-check `action.assignment.pr_number` and `observation.resource_url`. Apply first match:

1. Historical if every referenced PR was terminal before this request.
2. Current if any referenced PR remained live. Split mixed fragments or classify them current.
3. Without structured PR identity, use explicit historical/current markers.
4. Otherwise retain the source category.

Prefer structured assignments, PR URLs, and typed manifests; bare `#123` may mean an Issue. Version marker
lists: historical = prior round, accepted frontier, negative result, closed axis, merged result, research
ledger, confidence interval, paired bootstrap, retrospective; current = assignment/revision ID, head SHA, in
flight, review-ready, pending review, open experiment, current review. Mark compaction-summary allocation as
lossy because summaries lose provenance.

Proxy: `tokens(text) = 0` for empty text; otherwise `max(len(text) / 4, lexical * 0.82, 1.0)`. `lexical`
counts `[A-Za-z0-9_]+|[^\w\s]` matches. Providers supply no per-fragment counts.

Estimate tokens from compact JSON sent to the provider. Classify a decoded copy, then scale its categories
to the serialized token estimate. For each request, sum all raw category estimates. Require a positive total.
Multiply each category by `pre / total`. Round each value down. Assign the remainder to the largest scaled
category. Require nonnegative integers totaling `pre`. Shares stay estimated; normalization absorbs images,
encrypted reasoning, role and threading tokens, and serialization overhead. If the system or tool-schema
bands collapse unexpectedly, check whether the analyzer counted complete saved output instead of the clipped
result shown to the model.

## 9. Derive parent-visible subagent boundaries

Use only the parent's spawn, await, status, and cancel events. Key tasks by `task_id`. Set `called_at` to the
spawn action's timestamp and `returned_at` to the first terminal parent observation's timestamp.
Resolve the spawn action through `action_id`. Extend active tasks to the cutoff. Deduplicate await and
status observations. Count parent-visible prompts, envelopes, status, and reports as `subagent_io`.

## 10. Write the dataset and draw charts

Filter after full reconstruction. Each row contains `timestamp`, `model`, `context_window`, `billed`, `pre`,
`post`, `compact`, `summary_tokens`, `summary_words`, and integer category columns. `summary_tokens` is
provider-exact; `summary_words` counts words, not tokens, in private local block text. Keep response IDs only in
private audit data. Require constant model/window per segment; split or annotate changes and draw time-local
window lines. Handle trigger history likewise; otherwise label it snapshot-time configuration. Record
requested/observed/collected bounds; versions/model; mode/trigger/window; active/off-branch counts;
complete/unmatched responses; exact-versus-estimated method; gaps/resets.

- Main chart: stack categories to `pre`; draw solid `post`, dashed `billed`, and a horizontal trigger. At
  compaction, connect `pre` to `post` vertically and mark `post`. Optionally add a two-lane child
  call/return strip.
- Detail: show each compaction's `pre`→`post` and summary tokens over time with a median line. Optional
  cache panel: cache-read, cache-creation, uncached input divided by aggregate `billed`; state that
  denominator.
- For D3, include category toggles and a nearest-request tooltip containing all visible series, including
  `pre`, `post`, `billed`.

Use token axes, stable colors, and subtitle `Provider totals and compaction iterations are exact; semantic
categories are estimated.` Lines connect samples; they do not represent continuous measurement.

## 11. Extend a chart without moving its start

Save first response ID/time privately. Freeze again; require that response on the active branch. Reconstruct
from new leaf, filter from saved start (not `now - window`), query the complete compaction set in the new
snapshot, rebuild, and assert unchanged first ID/time. Separate new conversations with gaps; disclose missing
predecessors.

## 12. Validate context charts

Check Sections 1–11, leaf time ≤ collection time, and ancestry to root/system. Prominently report
off-branch/unmatched counts. Require positive summary tokens; inspect/report non-shrinking compactions. Use
runtime/model window/trigger values, not defaults. Verify no doubled cache or output counted as input.

## 13. Analyze advisor delegation and model use

Use evidence in this order: registry, paired OpenHands, root Weave, child spans.

| Unit | Definition/evidence |
|---|---|
| Delegation decision | One successful `spawn_agents` action, confirmed by paired observation; retain failed attempts separately |
| Requested child task | Unique registry `task_id`; proves acceptance, tier, status, launch time, depth, parent. Recursive means non-null `parent_task_id`. |
| Started child conversation | One child root `invoke_agent`; proves traced start, provider model, duration, status |
| Provider request | Child `chat` span; explains work, adds no conversation |
| Requested tier / observed provider | Tier: registry `tasks.model` or spawn specification. Provider: root `request_model` and frozen child state. |

Source intervention from deployment/commit/controller evidence; use Section 1 windows and
advisor/registry/child-state/Weave coverage. Report accepted tasks, tasks/observed advisor hour, successful
decisions, tasks/decision, tier/provider counts/shares, recursive count/share, start rate, cutoff statuses,
`process_start_time - created_at`, root duration/error rate, and source-only tasks. Normalize by turns or
eligible events when available: round plans, plateau pivots, large reviews, difficult optimization/debugging,
conflicting-evidence reviews, expensive portfolio choices. Opportunity can increase counts without policy
change.

**Query roots.** Use Agents `agent_spans_query` (`/agents/spans/query`), not ordinary Weave: `agent_name`
(default `advisor`), `operation_name=invoke_agent`, `include_details=False`, `started_after`,
`started_before`, `limit=1000`, `offset`. Page to `total_count` or empty. Normalize `started_at` to UTC (naive
means UTC); enforce `start <= started_at < end`. Select verified child IDs from private uncommitted
registry/frozen-state data. Require one non-empty `request_model` per conversation. Count each conversation
once. Count it as an errored conversation if any of its root spans has `status_code=ERROR`.
Select each child's root `invoke_agent` spans first.
Then group them by `request_model`. Print only aggregates.

Verify the analyzed revision's `OpenHandsChildProcess` task-ID→conversation-ID rule. Without exact joins,
locally match the stable delegated-task prompt in `input_messages`; label this heuristic, print no matching
content, and validate a sample against state.

**Query registry.** Open `<delegation-root-state>/delegation/tasks.sqlite3` with `file:<url-quoted absolute
path>?mode=ro`, `uri=True`. Select `parent_task_id`, `depth`, `model`, `status`, `created_at`, `updated_at`,
`process_start_time` from `tasks` for `created_at >= start AND created_at < end` (epoch seconds). Avoid
`task`, `result`, `error`, paths, and IDs when aggregates suffice. Started means non-null `process_start_time
< end`; start rate is started/all tasks. Report median launch delay for started tasks. For past statuses, use
a cutoff snapshot or timestamped OpenHands evidence. For long reads/transfers, use SQLite backup; never copy
an active main file without its WAL.

**Decisions.** Join successful spawn observations to registry by returned task IDs; discard bodies/results.
Use `parent_task_id`/`depth` for recursion, parent event time for decisions, `created_at` for acceptance.
Without action files, label the weaker Weave fallback: `agent_name=advisor`, `operation_name=execute_tool`,
`tool_name=spawn_agents`. Verified child `conversation_id` means recursive. Extract only task count/tier from
`tool_call_arguments`, discard arguments, and count structured status/result errors.

**Reconcile.** Weave misses pre-trace failures; state can expire; registries can outlive traces. Join each
accepted task to spawn/root evidence and report all combinations, including complete records:

| Registry | Successful spawn | Root span | Usual explanation |
|---|---|---|---|
| Yes | Yes | No | Pre-trace launch failure, startup timeout, trace gap |
| No | No: failed action | No | Spawn created no task |
| No | No | Yes | Expired state, wrong window, incomplete child-ID join |

Report discrepancies, never silently select the larger count. Read raw text only when aggregate
status/timestamps cannot explain gaps. Future roots need private `task_id`, `tree_id`, `parent_task_id`,
`depth`, `spawn_operation_key`, `requested_model_tier`, `conversation_id`, and provider model for exact
lineage/creation/start joins.

| Change | Interpretation |
|---|---|
| More uptime/start success, shorter launch delay; stable decisions/eligible event | Operational improvement |
| More decisions/batches/recursion; stable start success/duration | Advisor behavior |
| More eligible events; stable decisions/eligible event | Workload mix |
| More uptime and decisions/eligible event | Mixed operational/behavior change |
| More tasks; stable Frontier share | Broad delegation increase |

Compare deployed code/prompts. Operational changes touch launch, admission, concurrency, timeouts, recovery,
storage, or telemetry. Guidance, skill visibility, triggers, and model selection change policy.

**Example: 2026-08-21 18:07:30 UTC rollout.** A local stable-prompt heuristic selected child roots without
retaining prompts. Equal 24-hour windows showed:

| Metric | Before | After |
|---|---:|---:|
| Traced conversations; observed advisor hours; conversations/hour | 10; 14.375; 0.70 | 50; 24; 2.08 |
| Fable (configured Frontier provider) | 2 (20%) | 11 (22%) |
| Finished roots; errors | 10/10; 0 | 50/50; 0 |
| Median/p95 duration (seconds) | 623/1,182 | 589/1,323 |
| Recursive decisions; tasks | 1; 3 | 8; 19 |
| Recursive tiers | 3 smart | 16 smart, 1 fast, 2 frontier |
| Registry | Missing | 65 tasks: 43 direct/22 recursive; 32 frontier/22 smart/11 fast; 62 finished/3 failed at collection |

Missing advisor readiness (9 h 37 min 30 s) reduces 5x growth to about 3x/hour. Observed launch success was
already 100%; median duration improved 5.5%, p95 worsened. Recursive requests added 16/40 (40%) of growth,
pending exact lineage; direct growth was also large. Fisher two-sided 2/10 versus 11/50 gave `p=1.0`: no
evidence of a change in Frontier share. The small before sample limits this conclusion.

Revisions `97769de0`→`5a1ae8d0` broadened triggers, required pre-round/expensive-portfolio critique and
deliberate tiers, and expanded child guidance. Skill
loading/launch/concurrency/timeouts/recovery/instrumentation stayed unchanged; deployment mapped Fable. Images
bundled retry/compaction changes and fresh pod/conversation: compare SHAs. Registry creation times, Weave
start times, pre-trace failures, missing trace task IDs, and missing before registry prevent direct count/tier
comparison. Evidence favors policy plus uptime, possibly workload/runtime; no causal percentage.

## 14. Build an advisor and student activity timeline

Freeze advisor, persistent students, and recursively discovered children (Sections 3–4). Keep activity
separate from context panels. Per actor record `role`, `actor`, `observed_start`, `observed_end`, `complete`,
`gaps`. Tag active-branch membership; retain off-branch mutations only when paired observations or joined
external systems prove execution. Report active/proven-off-branch counts separately.

**Assignments.** Require paired non-error
`create_assignment`→`GitHubMutationObservation(state=assignment_created)`. See
[senpai_agent/github/tools/contracts.py](senpai_agent/github/tools/contracts.py) and
[senpai_agent/github/tools/advisor.py](senpai_agent/github/tools/advisor.py). Private fields: `assignment_id`,
`requested_at`, `student`, `category`, `category_source`, `status_at_cutoff`, `terminal_at`, `pr_number`,
`resource_url`, `active_branch`, `source_action_event_id`, `source_observation_event_id`.

Deduplicate `assignment_id`. Use the paired `create_assignment` action time for `requested_at`.
Replace it with GitHub `createdAt` only when the PR URL or number and the trusted assignment marker both match.
Attach feedback, routing repairs, revisions, publications, and submission retries to the assignment.
Terminal requires successful `experiment_merged`/`experiment_closed` by cutoff;
otherwise active. Merge means completion, not scientific victory. Join typed `ExperimentResult.runs[].run_id`;
runs are not points.

**Children.** Join `spawn_agents` specifications to `SpawnAgentsObservation.tasks` by `key`; without keys,
first require equal lengths, then use executor order. Deduplicate returned `task_id`. Fields: `task_id`,
`parent_actor`, `requested_at`, `returned_at`, `status_at_cutoff`, `agent_type`, `model_tier`,
`provider_model`, `include_context`, `category`, `active_branch`. Types: `general-purpose`, `explore`,
`bash-runner`, `search_general_web`, `search_research_publications`. Provider model comes from frozen
`base_state.agent.llm.model`; absent means unknown. Terminal is the first paired
`finished`/`failed`/`cancelled` from `await_agents`, `agent_status`, or `cancel_agents`, or controller
evidence with structured task ID/time. Without terminal evidence by cutoff, remain active.

See [senpai_agent/delegation.py](senpai_agent/delegation.py).
[tools/senpai_tool_telemetry.py](tools/senpai_tool_telemetry.py) provides recursive discovery, role/model
inference, timestamp/pairing patterns; it lacks branches, assignment lifecycles, and child specifications.

**Categories.** Apply the first matching rule to the primary preregistered intervention. Show definitions
below the legend. Label categories as deterministic interpretations. Version the rules and the override table
keyed by assignment ID. Give one reason for each override used for a reviewed mixed case.

| Order | Category | Definition/rule |
|---|---|---|
| 1 | Validate / transfer (`validate_transfer`) | Reproduce, integrate, or test another base, host, or end-to-end path. Rule: evidence-only replication, transfer, exactness, or ship gate without a new mechanism. |
| 2 | Policy / head (`policy_head`) | Change or price draft decisions or proposal-head/readout behavior. Rule: changed shipped policy, proposal head, or readout decision. |
| 3 | Kernel / runtime (`kernel_runtime`) | Change executed kernels, dispatch, memory, or scheduling to remove cost. Rule: changed on-path kernel or runtime implementation. |
| 4 | Diagnose / model (`diagnose_model`) | Measure, attribute, or predict a bottleneck without requiring a shipped mechanism. Rule: measurement, ablation, screening, oracle analysis, attribution, or cost model. |
| 5 | `unclassified` | Insufficient evidence; do not guess or use an external classifier |

**Board.** Join milestones one-to-one by immutable receipt and candidate or source SHA. Keep evaluator states
and UTC times for queueing, rejection, cancellation, and promotion. Do not infer milestones from prose.
Interpret percentages only as their definition permits. Source configuration and harness annotations from
UTC events and exact revisions. Do not use the next tool call as deployment time.

Draw UTC Board/student lanes and one assignment at `requested_at`: circle=terminal by cutoff, diamond=active.
Color only four categories; tooltips contain approved details/state. Draw Board queue/terminal events,
optional pre-intervention shading/marker. Reconcile category/status/student totals and before/after shares
from the same rows. Show children separately with type, tier, provider model, category, status. Show
requested/observed bounds and every gap.

## 15. Measure research iteration speed

The unit is one logical assignment and its advisor decision (measurement choice below). Reuse frozen snapshots
and paired observations. Build private UTC rows from structured evidence:

| Event | Preferred evidence |
|---|---|
| Assignment created | Successful paired `create_assignment` |
| Job launched/terminated | Persisted supervisor/monitor state |
| Terminal signal delivered | Controller inbox event and delivery receipt |
| Conversation resumed | First OpenHands response after delivery |
| Result published | Successful paired `submit_experiment_result` |
| Advisor decision | Successful merge, close, or revision observation |
| W&B run started/finished | Run metadata joined through typed result |
| External submission changed | Immutable evaluator receipt |
| Frontier changed | Evaluator-reported promotion |

Keep private assignment/revision identity, PR number, commit SHA, run/training IDs, monitor dedupe key,
evaluator receipt, conversation ID, and source event IDs. Runs are evidence; local/W&B metrics are not
official scores. Rejected external results can still represent valid scientific iterations. Keep wake
boundaries separately: `job_terminal_at`, `monitor_signal_created_at`, `signal_delivered_at`,
`conversation_resumed_at`, `first_relevant_action_at`, `result_published_at`. They separate polling, delivery,
response, and follow-up; use timestamps, not log order.

Source intervention time from the deployment or fresh conversation that loaded it, not commit time alone; use
a rollout band when needed. For a live post window, stop at the latest common observation and match pre
duration. Record advisor/student/controller/GitHub/W&B/evaluator coverage; normalize by observed hours.
Include inherited/active boundary-crossing cycles. Record concurrent model, reasoning-effort, prompt,
hardware, student-count, evaluator, queue-policy, and baseline changes.

| Measure | Definition |
|---|---|
| Assignment throughput | Logical assignments created per observed advisor hour |
| Evidence throughput | Assignments with terminal evidence per observed hour |
| Decision throughput | Merge, close, or revision decisions per observed hour |
| Assignment cycle | Assignment creation to terminal advisor decision |
| Experiment time | Assignment creation to result publication |
| Review latency | Result publication to terminal advisor decision |
| Wake latency | Monitor signal creation to resumed conversation |
| Reaction latency | Signal delivery to first relevant action |
| Submission cadence | Submissions per evaluator-available hour and consecutive interval |
| Controllable handoff | Previous external terminal result to next queue time |
| Evaluator service | Queue time to terminal evaluator result |
| Merge or promotion rate | Positive outcomes divided by terminal decisions |
| Progress | Best official metric and gap to promoted frontier |

> **Measurement choice: state it in every report.** This guide does not fix it.
>
> - Decision throughput includes revisions, and cycle/review latency end at a “terminal advisor decision,”
>   but Section 14 ends assignments only at merge/close. State whether a revision ends a cycle.
> - For merge or promotion rate, state numerator (advisor merge, evaluator promotion, or both) and
>   denominator (advisor terminal decisions or evaluator terminal results). If decisions differ, report each
>   variant or justify one.

Report counts, medians, p90, and OpenHands overhead per terminal decision: model calls, input tokens,
model-visible tool-output bytes, compactions, FinishActions, response gaps. Separate evaluator service from
controllable handoff; serialized evaluation can dominate intervals despite fast reactions.

Use common rows for three views: external submissions with official metrics, terminal states, best-so-far and
intervention band; equal-window assignment/evidence/decision/merge/W&B/submission counts/rates with observed
hours; assignment→result, result→decision, terminal-job→action, handoff, evaluator-service, and response-tail
latencies. Add actor lanes only for concurrency/idle capacity. Distinguish local/W&B/external evidence and
improvement direction. Check HTML at about 360, 736, 1,024 pixels in both themes.

Pair speed with validity gates, merge/promotion rates, official progress, frontier gap. Disclose small
samples, censoring, inherited candidates, coverage gaps, evaluator availability, time-of-day effects.
