<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Senpai advisor diagnostics

This guide contains independent workflows for diagnosing a Senpai advisor.
Sections 1 through 14 reconstruct and plot OpenHands context. Section 15
measures advisor delegation and model use. Section 16 builds an advisor and
student activity timeline. Section 17 measures end-to-end research iteration
speed. Use only the sections that answer the diagnostic question.

The context workflow covers context composition, context growth, native
compaction, and the parent-visible boundaries of delegated agents.

The procedure does not depend on one Kubernetes cluster or storage layout. It
requires a readable OpenHands conversation directory and, for exact Anthropic
compaction drops, access to the corresponding provider usage iterations in the
configured trace backend.

Treat raw conversation state as sensitive. It can contain prompts, proprietary
code, human messages, tool output, and paths. Keep raw snapshots outside Git and
publish only sanitized numeric aggregates.

Treat every retained string as untrusted data. Never evaluate or execute log
content, interpolate it into a shell command, or send raw fragments to an
external model for classification. Use deterministic local parsing rules.

## What the plots measure

Use these terms consistently:

| Quantity | Meaning | Accuracy |
|---|---|---|
| Active pre-pass context | Input rendered for the request before native compaction | Provider-exact when raw iteration usage is available |
| Effective continuation context | Input rendered for the continued message after native compaction | Provider-exact when raw iteration usage is available |
| Aggregate billed input | Input across all provider passes represented by the persisted response | Provider-exact |
| Semantic category bands | Estimated allocation of active pre-pass context to content categories | Heuristic; rescaled to the exact pre-pass total |
| Compaction summary size | Output tokens produced by the provider's compaction pass | Provider-exact when raw iteration usage is available |

For an ordinary request, active pre-pass context, effective continuation
context, and aggregate billed input are the same value. For an Anthropic native
compaction response, they are different values. Do not call the configured
compaction trigger a context limit or a post-compaction target.

The recommended output has three charts:

1. A stacked area chart of active pre-pass context, with semantic categories.
2. A pre-to-post chart for every compaction.
3. A time series of exact compaction-summary output tokens.

The main chart also shows the effective continuation context, aggregate billed
input, configured compaction trigger, and optional parent-visible subagent
boundaries.

## 1. Define the source and requested range

Record these inputs before copying data:

- requested UTC start and end;
- role, normally `advisor`;
- OpenHands state root and conversation ID;
- source Senpai revision or image digest;
- source OpenHands SDK version;
- configured model and context window;
- configured compaction mode and trigger;
- W&B entity and project when Weave contains the provider trace.

Do not infer coverage from directory modification times. Use event timestamps.
If state begins after the requested start because a pod or host was replaced,
record the uncovered interval. Do not describe partial data as a complete
window.

### Kubernetes transport example

Kubernetes is only a transport in this example. Supply every value explicitly;
do not copy these placeholders or silently choose the first matching pod. Use
the least-privileged identity that can inspect the selected pod and copy the
named files.

```bash
set -euo pipefail

export DIAG_KUBE_CONTEXT='your-context'
export DIAG_NAMESPACE='your-namespace'
export DIAG_POD='your-advisor-pod'
export DIAG_CONTAINER='advisor'

kubectl --context "$DIAG_KUBE_CONTEXT" --namespace "$DIAG_NAMESPACE" \
  get pod "$DIAG_POD" \
  -o json | jq '{
    name: .metadata.name,
    start: .status.startTime,
    status: .status.phase,
    containers: (
      (.status.containerStatuses // []) |
      map({name: .name, restart_count: .restartCount})
    )
  }'

kubectl --context "$DIAG_KUBE_CONTEXT" --namespace "$DIAG_NAMESPACE" \
  exec "$DIAG_POD" -c "$DIAG_CONTAINER" -- \
  printenv SENPAI_OPENHANDS_STATE_DIR
```

Set the printed path explicitly. Read only named non-secret variables. Never
print the full environment, enable shell tracing, or retrieve Kubernetes Secret
objects.

```bash
set -euo pipefail

export DIAG_REMOTE_STATE_ROOT='/path/printed/by/the/previous/command'

DIAG_CONVERSATION_ID="$(
  kubectl --context "$DIAG_KUBE_CONTEXT" --namespace "$DIAG_NAMESPACE" \
    exec "$DIAG_POD" -c "$DIAG_CONTAINER" -- \
    cat "$DIAG_REMOTE_STATE_ROOT/advisor-conversation-id"
)"
DIAG_CONVERSATION_ID="$(printf '%s' "$DIAG_CONVERSATION_ID" | tr -d '\r\n')"
case "$DIAG_CONVERSATION_ID" in
  ''|*[!0-9A-Fa-f-]*)
    printf 'invalid conversation ID\n' >&2
    exit 1
    ;;
esac
case "${#DIAG_CONVERSATION_ID}" in
  32|36) ;;
  *)
    printf 'conversation ID is not a 32- or 36-character UUID\n' >&2
    exit 1
    ;;
esac
DIAG_SESSION_NAME="$(printf '%s' "$DIAG_CONVERSATION_ID" | tr -d '-')"
export DIAG_REMOTE_SESSION_DIR="$DIAG_REMOTE_STATE_ROOT/$DIAG_SESSION_NAME"

kubectl --context "$DIAG_KUBE_CONTEXT" --namespace "$DIAG_NAMESPACE" \
  exec "$DIAG_POD" -c "$DIAG_CONTAINER" -- \
  test -f "$DIAG_REMOTE_SESSION_DIR/base_state.json"
```

Some OpenHands versions preserve hyphens in the on-disk UUID. If the derived
path does not exist, discover candidate sessions by file structure and choose
the intended conversation explicitly:

```bash
set -euo pipefail

kubectl --context "$DIAG_KUBE_CONTEXT" --namespace "$DIAG_NAMESPACE" \
  exec "$DIAG_POD" -c "$DIAG_CONTAINER" -- \
  find "$DIAG_REMOTE_STATE_ROOT" -mindepth 2 -maxdepth 2 \
  -name base_state.json -print
```

For a local directory, remote host, container runtime, mounted volume, or
operator-supplied archive, use the equivalent read-only copy mechanism. The
analysis starts from the same two inputs:

```text
<session>/base_state.json
<session>/events/event-*.json
```

Do not copy the whole state root. It can contain controller databases, GitHub
artifacts, saved full tool output, and other data that the chart does not need.

## 2. Freeze a coherent snapshot

OpenHands event files are immutable, but the advisor can append events while a
copy is running. Use `base_state.json` as the consistency boundary:

1. Create a private, empty destination.
2. Copy `base_state.json` first and freeze its `leaf_event_id`.
3. Record the collection time after that copy completes.
4. Copy the event directory.
5. Reconstruct only the ancestry of the frozen leaf.

An atomic volume snapshot is preferable when the storage system provides one.
For a local or mounted source, copy only the base state and regular event JSON
files:

```bash
set -euo pipefail

export DIAG_SOURCE_SESSION_DIR='/absolute/path/to/conversation'

umask 077
DIAG_SNAPSHOT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/senpai-context.XXXXXX")"

cp -- "$DIAG_SOURCE_SESSION_DIR/base_state.json" \
  "$DIAG_SNAPSHOT_DIR/base_state.json"
DIAG_FROZEN_LEAF="$(
  jq -er '.leaf_event_id | select(type == "string" and length > 0)' \
    "$DIAG_SNAPSHOT_DIR/base_state.json"
)"
jq -e '.stats.usage_to_metrics | type == "object"' \
  "$DIAG_SNAPSHOT_DIR/base_state.json" > /dev/null
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$DIAG_SNAPSHOT_DIR/collected-at.txt"

mkdir "$DIAG_SNAPSHOT_DIR/events"
find "$DIAG_SOURCE_SESSION_DIR/events" -maxdepth 1 -type f \
  -name 'event-*.json' \
  -exec cp -- {} "$DIAG_SNAPSHOT_DIR/events/" \;
```

For an SSH, object-store, container-runtime, or archive transport, preserve the
same copy order and file selection. Copy the base state first. Then copy only
regular `event-*.json` files into the private destination.

The following Kubernetes alternative requires `find -print0` and a `tar`
implementation that supports `--null`. Use an equivalent regular-file-only
archive when the source runtime provides different tools.

```bash
set -euo pipefail

umask 077
DIAG_SNAPSHOT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/senpai-context.XXXXXX")"

kubectl --context "$DIAG_KUBE_CONTEXT" --namespace "$DIAG_NAMESPACE" \
  exec "$DIAG_POD" -c "$DIAG_CONTAINER" -- \
  cat "$DIAG_REMOTE_SESSION_DIR/base_state.json" \
  > "$DIAG_SNAPSHOT_DIR/base_state.json"

DIAG_FROZEN_LEAF="$(
  jq -er '.leaf_event_id | select(type == "string" and length > 0)' \
    "$DIAG_SNAPSHOT_DIR/base_state.json"
)"
jq -e '.stats.usage_to_metrics | type == "object"' \
  "$DIAG_SNAPSHOT_DIR/base_state.json" > /dev/null
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$DIAG_SNAPSHOT_DIR/collected-at.txt"

kubectl --context "$DIAG_KUBE_CONTEXT" --namespace "$DIAG_NAMESPACE" \
  exec "$DIAG_POD" -c "$DIAG_CONTAINER" -- sh -c '
    set -eu
    cd "$1"
    find events -maxdepth 1 -type f -name "event-*.json" -print0 |
      tar --null -czf - -T -
  ' sh "$DIAG_REMOTE_SESSION_DIR" \
  | tar -xzf - -C "$DIAG_SNAPSHOT_DIR"
```

If a copy fails, retry into a new empty directory. Do not merge a partial retry
with an earlier directory.

Record a checksum manifest for auditability:

```bash
set -euo pipefail

(
  cd "$DIAG_SNAPSHOT_DIR"
  shasum -a 256 base_state.json
  find events -type f -name 'event-*.json' \
    -exec shasum -a 256 {} +
) | LC_ALL=C sort > "$DIAG_SNAPSHOT_DIR/manifest.sha256"
```

Use `sha256sum` instead of `shasum -a 256` on systems that provide GNU
coreutils.

Also record the conversation ID, frozen leaf ID, requested range, source
versions, model, compaction mode, trigger, and context-window size in separate
snapshot metadata. Do not place credentials in that metadata.

## 3. Reconstruct the active branch

The event directory is a graph, not a flat transcript. Hook events and abandoned
branches can remain on disk. Follow `parent_id` backward from the frozen
`leaf_event_id`, then reverse the result.

```python
import json
from pathlib import Path

snapshot = Path("/private/snapshot")
state = json.loads((snapshot / "base_state.json").read_text())

events_by_id = {}
for path in (snapshot / "events").glob("event-*.json"):
    event = json.loads(path.read_text())
    event_id = event["id"]
    if event_id in events_by_id:
        raise RuntimeError(f"duplicate event id: {event_id}")
    events_by_id[event_id] = event

active = []
seen = set()
event_id = state["leaf_event_id"]
while event_id:
    if event_id in seen:
        raise RuntimeError(f"parent cycle at event: {event_id}")
    seen.add(event_id)
    try:
        event = events_by_id[event_id]
    except KeyError as error:
        raise RuntimeError(f"missing active event: {event_id}") from error
    active.append(event)
    event_id = event.get("parent_id")

active.reverse()
off_branch_count = len(events_by_id) - len(active)
```

Fail on duplicate IDs, a missing leaf or parent, malformed JSON, or a parent
cycle. Report the active and excluded event counts.

Only these event kinds normally contribute to model messages:

- `SystemPromptEvent`;
- `MessageEvent`;
- `ActionEvent`;
- `ObservationEvent`;
- `UserRejectObservation`;
- `AgentErrorEvent`.

Exclude `HookExecutionEvent` and other orchestration-only records. Use event
filename order only as a loading aid; use the active parent chain as the
authority.

## 4. Build request boundaries and join usage

Read usage rows from:

```text
base_state.stats.usage_to_metrics.<usage-namespace>.token_usages[]
```

Discover the usage namespace. Do not assume it is named `senpai`. If several
namespaces contain response IDs from the active branch, require an explicit
choice.

Build response groups from every active-branch, agent-generated `ActionEvent`
or `MessageEvent` with a non-empty `llm_response_id`:

- map the response ID to the index of its first response event;
- group all response events with the same response ID;
- validate that those events are adjacent after filtering to LLM-convertible
  events, and fail if another model response separates them; a hook or other
  non-LLM event does not split the group;
- use the first response event timestamp as the plotted response time, and
  document that it records persisted model output rather than request start;
- collect tool names from every action in the group;
- mark a compaction when any grouped event's rendered LLM message has non-empty
  `anthropic_compaction_blocks`.

Join response groups to usage rows by `response_id`. Order requests by the
first response event's active-branch index. Fail when an active-branch response
has zero or multiple matching usage rows. Report usage-only rows separately and
exclude them unless their relationship to the frozen active branch is
established.

The input to a request is the active context strictly before its first response
event. Never include the current response output in its own pre-pass context.

Do not use these fields as current context size:

- `per_turn_token`, which includes output;
- accumulated usage, which grows across the session;
- completion or reasoning output tokens;
- flattened `prompt_tokens` on an Anthropic compaction response.

## 5. Select the model-visible context

For Anthropic native compaction, match the installed OpenHands selection logic.
For each request boundary:

1. Take active-branch events before the request's first response event.
2. Find the latest earlier event with non-empty Anthropic compaction blocks.
3. Always retain `SystemPromptEvent`.
4. If no compaction exists, retain all earlier LLM-convertible events.
5. Otherwise, retain the latest compaction-bearing event and every later
   LLM-convertible event.

For every other compaction mode, use that source version's event-selection
logic. Do not apply the `anthropic_compaction_blocks` rules to another mode.

The compaction-bearing event is inclusive because it carries the summary that
replaces older history. A compaction block generated by the current response
affects its continuation pass and later requests, not its own pre-compaction
input.

Inspect the installed SDK implementation before relying on this rule. In the
OpenHands fork used by Senpai, the relevant code is normally in:

```text
openhands/sdk/agent/utils.py::_anthropic_compaction_events
openhands/sdk/agent/utils.py::prepare_llm_messages
openhands/sdk/event/base.py::events_to_messages
```

Load the full conversation before applying the requested plot time range. The
first plotted request can still contain events created before the plot starts.
Filtering event files first produces an incorrect starting composition.

If several conversations cover the range, reconstruct each separately. Draw a
visible gap between sessions. Do not interpolate across a restart or state-loss
boundary.

## 6. Reproduce model-visible serialization

Run the analyzer with the same OpenHands and Senpai source versions when
possible. Treat those serializers as the rendering specification, but inspect
them for side effects before calling them. Call `event.to_llm_message()` only
for event types whose implementation is read-only. For an observation, the
model content comes from the `event.observation.to_llm_content` property, not a
method. If the serializer can write data, implement a pure, source-matched
renderer and pin its behavior in tests.

Do not estimate tokens from the complete persisted event JSON. Many persisted
fields never reached the model.

| Event | Include |
|---|---|
| `SystemPromptEvent` | System-prompt text, dynamic-context text, and tool definitions |
| `MessageEvent` | Text blocks from `llm_message.content` and `extended_content` |
| Batched `ActionEvent`s | Shared thought/reasoning once, thinking and compaction blocks once, and every provider-facing tool call |
| `ObservationEvent` | `observation.to_llm_content`, followed by `extended_content` |
| `UserRejectObservation` | The model-facing `Action rejected: ...` text |
| `AgentErrorEvent` | The model-facing tool error text |

Use the provider-facing `tool_call` representation. Do not substitute the
debug-oriented `action` object when measuring serialized token mass.

Custom observations can provide specialized `to_llm_content` properties.
Current examples include:

- `GetPRsObservation`, which emits inline Markdown or an artifact path plus a
  compact manifest;
- `GitHubMutationObservation`, which emits structured JSON;
- spawn, await, status, and cancel observations, which emit parent-visible task
  JSON;
- terminal observations, which render content with command metadata before
  truncation.

Match the serializer mode used by the source request. The list serializer
truncates each tool `TextContent` block separately. The force-string serializer
joins tool text and truncates the joined string once.

Mirror the source-version truncation rules before estimating tokens. The
OpenHands SDK 1.40.0 revision used for the original diagnostics clipped each
model-facing tool text block to 50,000 characters, while terminal rendering
applied a 30,000-character limit first. The truncation kept the head and tail
and inserted a clipping notice. Verify the source revision and live constants
instead of assuming that they remain unchanged:

```text
openhands/sdk/llm/message.py
openhands/sdk/utils/truncate.py
openhands/tools/terminal/definition.py
```

Do not call `TerminalObservation.to_llm_content` directly in a read-only
analyzer. In SDK 1.40.0 it can create directories and write full output through
`full_output_save_dir`. Use a pure renderer that reproduces the source
prefix/content/suffix, working directory, interpreter, exit status,
30,000-character head-tail truncation, and persisted-output notice. Preserve
the original notice path and calculated line number without reading or writing
the saved full output.

Do not read a full-output file merely because an observation names its path.
Those bytes were not model-visible unless a later tool call read them back into
the conversation.

## 7. Recover exact compaction totals

For an ordinary request:

```text
pre = post = billed = prompt_tokens
```

For Anthropic native compaction, LiteLLM's flattened `prompt_tokens` is the sum
of the provider's compaction pass and continuation pass. Plotting that sum as
active context hides the compaction sawtooth.

Query the trace backend for only the active-branch compaction response IDs. In
W&B Weave, request detailed agent spans and read the current raw OpenHands usage
path:

```text
attributes.weave.openhands.llm.raw_response.usage.iterations
```

Inspect one span before processing the batch because this internal path can
change with dependency versions. Under the current integration, a native
Anthropic compaction response contains one `type="compaction"` iteration and
one `type="message"` iteration.

Calculate one iteration's total input as:

```python
def input_total(iteration):
    return sum(
        iteration.get(key, 0)
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )

pre = input_total(compaction_iteration)
post = input_total(message_iteration)
billed = pre + post
summary_tokens = compaction_iteration["output_tokens"]
```

Cache-read and cache-creation tokens are disjoint portions of input. Do not add
flattened cache counters to `prompt_tokens` again.

The core Weave batch query is:

```python
import json
import weave
from weave.trace_server.agents.types import AgentSpansQueryReq, Query

client = weave.init("ENTITY/PROJECT", settings={"print_call_link": False})
query = Query.model_validate(
    {
        "$expr": {
            "$in": [
                {"$getField": "response_id"},
                [{"$literal": value} for value in sorted(response_ids)],
            ]
        }
    }
)
result = client.server.agent_spans_query(
    AgentSpansQueryReq(
        project_id=client.project_id,
        query=query,
        include_details=True,
        limit=len(response_ids),
    )
)

for span in result.spans:
    raw = json.loads(span.raw_span_dump)
    usage = raw["attributes"]["weave"]["openhands"]["llm"]["raw_response"]["usage"]
    # Extract only usage.iterations here. Do not persist raw.
```

Chunk the response-ID set when it can exceed the server's request or page
limit. Detect duplicate or missing responses across chunks.

Run this query in an isolated diagnostics environment with the matching Weave
version and the least-privileged available read credential. If only the source
runtime has the required dependencies, run a read-only one-shot process there,
write no files under the active conversation directory, and return only the
derived numeric usage JSON. Never print the credential, place it in the script,
or persist `raw_span_dump`.

For every compaction, assert:

```text
pre + post == persisted flattened prompt_tokens
```

Also assert that the local compaction response set equals the traced response
set. Use the compaction iteration's output tokens as the summary size. The
message iteration's output tokens belong to the continued assistant response.

This two-iteration rule is specific to the observed Anthropic native
compaction path. Detect provider and compaction mode before applying it. For
OpenAI native compaction or OpenHands local condensation, inspect that mode's
persisted events and raw usage separately.

If exact iteration usage is unavailable, do not claim an exact drop. Omit the
compaction point from the exact stack and drop panel, or label a separate proxy
mode. Flattened `prompt_tokens` cannot substitute for either `pre` or `post`.
The next request is not an exact post-compaction substitute because additional
events can enter the context first.

## 8. Assign mutually exclusive semantic categories

Store a fine-grained source category before applying topic semantics:

| Dataset category | Meaning |
|---|---|
| `system_instructions` | Base prompt, Senpai harness, role, program, and launch context |
| `tool_schemas` | Tool definitions sent with each request |
| `historical_pr_analysis` | Closed or merged PR evidence and prior research conclusions |
| `current_pr_assignment` | Live assignment, review, revision, or open experiment |
| `assistant_reasoning_output` | Visible advisor reasoning and output, plus thinking carriers |
| `bash_tool_io` | Terminal calls and model-visible terminal results |
| `file_code_tool_io` | File-editor calls and model-visible file or code results |
| `github_tool_io` | Typed GitHub workflow calls and observations |
| `subagent_io` | Parent-visible subagent prompts, status, and reports |
| `controller_user_events` | Human, controller, idle, and research-base messages |
| `other_tool_io` | Other tools, errors, and rejections |

The chart can combine `file_code_tool_io`, `github_tool_io`, and
`other_tool_io` as **Other tool/workflow I/O**. Keep **Batch/terminal I/O** and
**Subagent prompts/results** separate.

Classify each model-visible text fragment in two stages:

1. Assign a fallback category from the event kind and tool name.
2. Split narrative text on blank lines and Markdown headings, then apply
   PR/research topic rules.

Use a deterministic, versioned fallback map. The original diagnostics used:

- `terminal` to `bash_tool_io`;
- `file_editor` to `file_code_tool_io`;
- `spawn_agents`, `await_agents`, `agent_status`, `cancel_agents`, and
  `delegate_agent` to `subagent_io`;
- `get_prs`, `create_assignment`, `send_assignment_feedback`,
  `repair_assignment_routing`, `merge_experiment`, `close_experiment`,
  `accept_result_on_current_base`, `request_assignment_revision`,
  `publish_advisor_branch`, and `respond_to_human_issue` to `github_tool_io`;
- user rejections, agent errors, and unknown tools to `other_tool_io`;
- action thought, reasoning, thinking, and compaction content to
  `assistant_reasoning_output`;
- agent-generated `MessageEvent` content to `assistant_reasoning_output`;
- controller-delivered `MessageEvent` text to `controller_user_events`.

Resolve each `ObservationEvent.action_id` to its paired action before applying a
tool-based fallback. Report an observation whose action cannot be resolved.

Review every new typed tool and add it deliberately. Do not let an unknown tool
silently change categories between runs.

PR-topic classification can override the fallback source. Therefore,
PR-specific text returned by a subagent or tool appears in a PR band, while its
non-PR text remains in the subagent or tool band. State this precedence in the
chart methodology.

Resolve PR lifecycle at each request boundary. Advance a PR to terminal only
after the paired `GitHubMutationObservation` reports `state` equal to
`experiment_merged` or `experiment_closed`. Do not infer success from the
`merge_experiment` or `close_experiment` invocation. A `changed=false` result
can still confirm an idempotent terminal state. Resolve the observation's
`action_id` to its paired action, read `action.assignment.pr_number`, and
cross-check `observation.resource_url`. Apply the terminal transition at the
observation's active-branch position. Classify a fragment as:

- historical when every referenced PR was terminal before this request;
- current when any referenced PR remained live;
- historical or current from explicit marker phrases when no structured PR
  identity is present;
- its fallback source category otherwise.

Use structured assignment objects, PR URLs, and typed manifests before a bare
`#123` expression. A bare number can refer to an Issue. Treat a mixed fragment
as current or split it more finely.

Useful historical markers include prior round, accepted frontier, negative
result, closed axis, merged result, research ledger, confidence interval,
paired bootstrap, and retrospective. Useful current markers include assignment
or revision ID, head SHA, in flight, review-ready, pending review, open
experiment, and current review. Keep the marker list explicit and versioned.

Compaction summaries preserve conclusions but lose exact source provenance.
Classify their visible text when useful, and label that allocation as lossy.

## 9. Estimate and normalize category tokens

Provider APIs do not return tokens per semantic fragment. A practical proxy is:

```python
import re

def estimated_tokens(text):
    if not text:
        return 0.0
    lexical = len(re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text))
    return max(len(text) / 4, lexical * 0.82, 1.0)
```

For provider-facing JSON, estimate mass from the actual compact serialized
text. Decode or unescape a copy for semantic classification, then scale the
classified fragments back to the serialized mass.

For each request, normalize all raw category estimates to the exact active
pre-pass total:

```python
target = exact_pre
raw_total = sum(raw_categories.values())
if raw_total <= 0:
    raise RuntimeError("no model-visible category mass for request")
scale = target / raw_total
scaled = {
    name: value * scale
    for name, value in raw_categories.items()
}
categories = {name: int(value) for name, value in scaled.items()}
remainder = target - sum(categories.values())
categories[max(scaled, key=scaled.get)] += remainder
assert sum(categories.values()) == target
assert all(value >= 0 for value in categories.values())
```

This makes the height of the stack exact. It does not make the semantic shares
provider-exact. Images, encrypted reasoning, role/threading tokens, and
provider serialization overhead do not have clean semantic ownership; the
normalization absorbs them into the nearest categories.

Unexpected collapse of the system or tool-schema bands often means the
analyzer counted complete persisted tool output instead of the clipped
model-visible result.

## 10. Derive parent-visible subagent boundaries

Do not open or add child-agent transcripts. They are separate contexts.

Use only the parent conversation's spawn, await, status, cancel, or equivalent
events:

- key tasks by `task_id`;
- resolve `called_at` from the spawn action referenced by `action_id`;
- use the first parent observation with a terminal task status as
  `returned_at`;
- extend an active task to the snapshot cutoff;
- deduplicate repeated await and status observations.

Count parent-visible task prompts, envelopes, status, and returned reports as
`subagent_io`. A boundary strip can show calls and returns without exposing task
text.

## 11. Filter and write a sanitized plotting dataset

Reconstruct the full active context first. Then filter requests by the requested
UTC range.

Keep these fields in each plotting row:

```text
timestamp
model
context_window
billed
pre
post
compact
summary_tokens
summary_words
one integer column per category
```

Derive `summary_words` from the local compaction-block text only as a secondary
diagnostic. Treat it as a text count, not a provider token count. Use
`summary_tokens` from the provider compaction iteration for the exact summary
series, and do not publish the summary text.

Keep response IDs only in a private audit dataset. Omit response IDs,
conversation IDs, pod names, PR titles, raw task labels, and raw text from a
shareable chart unless an operator explicitly approves them.

Add snapshot-level metadata:

- requested, observed, and collected UTC bounds;
- source versions and model;
- compaction mode, trigger, and context window;
- active and off-branch event counts;
- complete and unmatched response counts;
- exact-versus-estimated methodology;
- coverage gaps and reset boundaries.

Retain `model` and `context_window` from every usage row. Assert that both are
constant within a plotted segment. Split or annotate the segment when either
changes, and draw the corresponding time-local context-window reference. Treat
the compaction trigger the same way when configuration history is available.
Otherwise, disclose that the trigger line reflects snapshot-time
configuration.

Treat all retained strings as untrusted. Build browser labels with
`textContent` or escape them before inserting HTML. Escape `<`, U+2028, and
U+2029 when embedding JSON in a script element.

## 12. Draw the charts

Use UTC for every time axis and tokens for every input/output axis. Keep colors
stable across refreshed plots.

### Main composition chart

- Draw a stacked area whose total equals exact `pre`.
- Color the stack with the estimated semantic categories.
- Draw exact `post` as a solid line.
- Draw exact `billed` as a dashed line.
- Draw the configured compaction trigger as a horizontal reference.
- At each compaction, draw a vertical connector from `pre` to `post` and a mark
  at `post`.
- Optionally add a two-lane parent-visible subagent call/return strip.

At an ordinary request, `pre == post == billed`. At an Anthropic compaction,
the stack ends at the compaction-pass input, the solid connector reaches the
continuation-pass input, and the dashed line reaches their aggregate.

### Compaction detail charts

For every compaction:

1. Connect exact pre-pass input to exact continuation-pass input.
2. Plot exact compaction-summary output tokens over time and show the median as
   a reference.

An optional cache panel can divide provider input into cache-read,
cache-creation, and uncached portions. Use the aggregate billed input across
passes as its denominator and state that choice.

### Minimal Matplotlib shape

```python
category_series = {
    "System/program": rows["system_instructions"],
    "Tool schemas": rows["tool_schemas"],
    "Historical PR/research": rows["historical_pr_analysis"],
    "Current PR/review": rows["current_pr_assignment"],
    "Advisor reasoning/output": rows["assistant_reasoning_output"],
    "Batch/terminal I/O": rows["bash_tool_io"],
    "Other tool/workflow I/O": (
        rows["file_code_tool_io"]
        + rows["github_tool_io"]
        + rows["other_tool_io"]
    ),
    "Subagent prompts/results": rows["subagent_io"],
    "Human/controller": rows["controller_user_events"],
}

axes[0].stackplot(
    rows["timestamp"],
    *category_series.values(),
    labels=category_series.keys(),
)
axes[0].plot(rows["timestamp"], rows["post"], label="effective context")
axes[0].plot(
    rows["timestamp"],
    rows["billed"],
    linestyle="--",
    label="aggregate billed input",
)
axes[0].axhline(compaction_trigger, linestyle=":", label="compaction trigger")

for row in rows.loc[rows["compact"]].itertuples(index=False):
    axes[0].vlines(row.timestamp, row.post, row.pre)
    axes[1].vlines(row.timestamp, row.post, row.pre)

axes[2].plot(compactions["timestamp"], compactions["summary_tokens"])
```

For an interactive D3 chart, add category toggles and one cross-series tooltip.
Snap the pointer to the nearest plotted request. Show that request's timestamp
and values for every visible series, including the active, continuation, and
billed totals. Make the chart readable in light and dark themes and at narrow
and wide widths.

Label the chart clearly:

```text
Provider totals and compaction iterations are exact; semantic categories are estimated.
```

Do not connect lines across missing conversations. Lines between adjacent
requests are visual connections, not continuously sampled context.

## 13. Refresh a chart without moving its start

Persist the first plotted response ID and timestamp in private metadata. To
extend a chart:

1. Freeze a new snapshot from the same conversation.
2. Verify that the saved first response remains on the new active branch.
3. Reconstruct the complete new snapshot from its frozen leaf.
4. Filter from the saved start, not from `now - window`.
5. Query provider iterations for the complete new compaction response set.
6. Rebuild all category allocations and charts.
7. Assert that the first plotted timestamp and response ID did not change.

If the conversation changed, preserve each session as a separate segment and
show the uncovered interval. Do not join the two endpoints with an area or line.
If an ephemeral predecessor disappeared, state that the older interval is
unavailable.

## 14. Validate before sharing

Check every item:

- The frozen leaf exists, and its parent chain is complete and acyclic.
- The frozen leaf timestamp does not exceed the recorded collection time.
- The active branch reaches its root and system event.
- Off-branch events are counted and excluded.
- Active response IDs join one-to-one with usage rows.
- Unmatched response IDs are empty or reported prominently.
- Every native compaction has the expected provider iteration data.
- Every compaction satisfies `pre + post == prompt_tokens`.
- Every category value is nonnegative.
- Category values sum to exact `pre` for every row.
- Compaction-summary output tokens are positive.
- Every non-shrinking compaction is inspected and reported.
- The trigger and context window come from runtime configuration and model
  metadata, not hard-coded defaults.
- Model, context-window, and known trigger changes split or annotate the plot.
- Cached tokens are not counted twice.
- Completion and reasoning output tokens are not added to input context.
- Model-visible truncation runs before token estimation.
- Requested and observed coverage are both present.
- All timestamps use UTC.
- The subtitle identifies exact totals and estimated semantic allocation.
- Sanitized output contains no raw prompts, tool output, credentials, or
  unapproved identifiers.
- The organization's approved secret scanner passes on the exact JSON, HTML,
  and other files selected for sharing or Git staging.
- The private snapshot follows the approved retention policy and is removed
  when the audit no longer requires it.
- Interactive charts have no browser errors, clipped labels, overlapping text,
  or horizontal overflow in light and dark themes.

The most consequential errors are:

- plotting flattened compaction `prompt_tokens` as active context;
- counting saved full tool output that the model never saw;
- scanning every event file instead of the frozen active branch;
- filtering old events before reconstructing the first plotted context;
- applying final PR status retrospectively to earlier requests;
- adding child-agent contexts to the parent context;
- hiding reset or storage-loss gaps;
- publishing raw trace content or identifiers in the derived chart.

## 15. Analyze advisor delegation and model usage

Use this workflow to determine whether a Senpai advisor delegated more work,
whether it changed model tiers, and whether a change came from runtime
availability or advisor behavior. It applies to deployments that retain W&B
Weave spans and OpenHands conversation state. A Senpai delegation registry adds
the most reliable task and parent-child evidence.

### Define the units before counting

Keep these quantities separate:

| Quantity | Definition | Preferred evidence |
| --- | --- | --- |
| Delegation decision | One successful `spawn_agents` action | Paired OpenHands action and observation |
| Requested child task | One task accepted by the delegation controller | One unique registry `task_id` |
| Started child conversation | One child that reached `invoke_agent` | One root Weave `invoke_agent` span per child conversation |
| Provider request | One model call inside a child conversation | Weave `chat` spans |
| Requested model tier | The `fast`, `smart`, or `frontier` routing choice | Registry `tasks.model` or the spawn specification |
| Observed provider model | The concrete model that served the child | Root Weave span `request_model` and frozen child state |
| Recursive request | A task whose `parent_task_id` is not null | Registry task row |

Do not call model requests, tool spans, or provider turns “delegated
conversations.” Count each child conversation once by its root `invoke_agent`
span or unique task ID. Keep the requested tier separate from the observed
provider model. Read the tier-to-model mapping from the deployment revision;
do not infer it from a model name.

### Freeze matched comparison windows

Record the rollout or intervention timestamp from a deployment record, commit,
or controller event. Select equal-duration, non-overlapping UTC windows around
it. Use half-open intervals so a boundary event belongs to only one window:

```text
pre  = [cutoff - duration, cutoff)
post = [cutoff, cutoff + duration)
```

Record requested and observed coverage separately for the advisor, delegation
registry, child state, and Weave. Mark pod replacement, missing storage,
controller downtime, and trace-ingestion gaps. Never interpret a zero during an
unobserved interval as zero delegation.

For each window, calculate at least:

- accepted task count and tasks per observed advisor hour;
- successful spawn-decision count and mean tasks per decision;
- requested tier counts and shares;
- observed provider-model counts and shares;
- recursive task count and share;
- task start rate and terminal status counts at the selected cutoff;
- `process_start_time - created_at` launch delay;
- root-span duration and error rate; and
- tasks present in only one source.

Normalize by advisor turns or eligible decision events when those values are
available. “Eligible” means a research-round plan, plateau pivot, large review,
difficult optimization or debugging decision, conflicting-evidence review, or
expensive portfolio choice. A busier research window can increase delegation
without any policy change.

### Use each source for what it proves

Use the sources in this order:

1. The read-only delegation registry proves accepted tasks, requested tiers,
   task status, launch time, depth, and exact parent relationships.
2. Paired OpenHands actions and observations prove advisor decisions, batch
   size, and what the parent learned. They also expose failed tool calls that
   created no task.
3. Root Weave `invoke_agent` spans prove that a conversation reached the traced
   agent runtime. They provide the observed provider model, duration, and trace
   status.
4. Child `chat` and tool spans explain work inside a conversation. They do not
   add conversations to the count.

Weave cannot observe a child that fails before the first traced agent call.
OpenHands state can disappear with ephemeral storage. The registry and
conversation logs can also outlive a missing Weave span. Report discrepancies;
do not silently select the larger count.

### Query Weave without exposing prompts

The W&B Agents data plane is separate from ordinary Weave call queries. Query
`/agents/spans/query` through `agent_spans_query`, and select root
`invoke_agent` spans. Page until `total_count` is satisfied. The following
example returns normalized metadata only and never prints message content or a
raw span dump:

```python
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import weave
from weave.trace_server.agents.types import AgentSpansQueryReq, Query


def utc(name):
    value = os.environ[name].replace("Z", "+00:00")
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def equal(field, value):
    return {"$eq": [{"$getField": field}, {"$literal": value}]}


def normalized_time(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def root_invocations(client, agent_name, start, end):
    query = Query.model_validate(
        {
            "$expr": {
                "$and": [
                    equal("agent_name", agent_name),
                    equal("operation_name", "invoke_agent"),
                ]
            }
        }
    )
    spans = []
    offset = 0
    while True:
        page = client.server.agent_spans_query(
            AgentSpansQueryReq(
                project_id=client.project_id,
                query=query,
                include_details=False,
                started_after=start,
                started_before=end,
                limit=1000,
                offset=offset,
            )
        )
        spans.extend(page.spans)
        if not page.spans or len(spans) >= page.total_count:
            break
        offset += len(page.spans)

    # Enforce the half-open interval locally, independent of endpoint behavior.
    selected = []
    for span in spans:
        started_at = normalized_time(span.started_at)
        if started_at is not None and start <= started_at < end:
            selected.append(span)
    return selected


client = weave.init(
    f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}",
    settings={"print_call_link": False},
)
all_roots = root_invocations(
    client,
    os.environ.get("DIAG_AGENT_NAME", "advisor"),
    utc("DIAG_START"),
    utc("DIAG_END"),
)

child_ids = set(
    Path(os.environ["DIAG_CHILD_CONVERSATION_IDS_PATH"]).read_text().splitlines()
)
conversations = {}
for span in all_roots:
    if span.conversation_id in child_ids:
        conversations.setdefault(span.conversation_id, []).append(span)

provider_models = Counter()
error_conversations = 0
for roots in conversations.values():
    models = {root.request_model for root in roots if root.request_model}
    if len(models) != 1:
        raise ValueError(f"expected one provider model, got {len(models)}")
    provider_models[next(iter(models))] += 1
    error_conversations += any(root.status_code == "ERROR" for root in roots)

print("conversations", len(conversations))
print("provider_models", dict(provider_models))
print("error_conversations", error_conversations)
```

Set `DIAG_CHILD_CONVERSATION_IDS_PATH` to a private newline-delimited file built
from the registry and frozen child state. Do not commit it. Current Senpai
versions derive a child's conversation ID from its task ID in
`OpenHandsChildProcess`; verify that rule in the analyzed revision before using
it. If no exact join is available, a local check for the stable delegated task
prompt in `input_messages` can backfill old traces. Label that check as a
heuristic, never print the matching content, and validate a sample against
OpenHands state.

Group by `request_model` only after selecting child root invocations. A query
that counts all spans for a model will multiply long conversations by their
number of model turns.

When older OpenHands action files are unavailable, use Weave as a historical
fallback for spawn decisions. Query spans with `agent_name=advisor`,
`operation_name=execute_tool`, and `tool_name=spawn_agents`. A spawn span whose
`conversation_id` is in the verified child-ID set is recursive. Parse only the
task count and requested `model` values from `tool_call_arguments`, then discard
the arguments. Count errors from structured status and result fields. This
trace-observed count is weaker than a paired OpenHands action and observation,
so label it separately.

### Read the delegation registry without task text

The registry normally lives at
`<delegation-root-state>/delegation/tasks.sqlite3`. Open it in read-only mode.
Do not select or print `task`, `result`, `error`, paths, task IDs, or
conversation IDs when aggregate data is sufficient.

```python
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from urllib.parse import quote


def timestamp(name):
    value = os.environ[name].replace("Z", "+00:00")
    return datetime.fromisoformat(value).astimezone(timezone.utc).timestamp()


registry_path = Path(os.environ["SENPAI_DELEGATION_REGISTRY_PATH"])
registry_uri = quote(str(registry_path.expanduser().resolve()), safe="/")
database = sqlite3.connect(f"file:{registry_uri}?mode=ro", uri=True)
database.row_factory = sqlite3.Row
start = timestamp("DIAG_START")
end = timestamp("DIAG_END")
rows = database.execute(
    """
    SELECT parent_task_id, depth, model, status,
           created_at, updated_at, process_start_time
    FROM tasks
    WHERE created_at >= ? AND created_at < ?
    """,
    (start, end),
).fetchall()

launch_delays = [
    row["process_start_time"] - row["created_at"]
    for row in rows
    if row["process_start_time"] is not None and row["process_start_time"] < end
]
print("tasks", len(rows))
print("requested_tiers", dict(Counter(row["model"] for row in rows)))
print("current_statuses", dict(Counter(row["status"] for row in rows)))
print("recursive", sum(row["parent_task_id"] is not None for row in rows))
started = sum(
    row["process_start_time"] is not None and row["process_start_time"] < end
    for row in rows
)
print("start_rate", started / len(rows) if rows else None)
print("median_launch_delay_s", median(launch_delays) if launch_delays else None)
```

Snapshot the live SQLite database with the SQLite backup API when a long query
or transfer is necessary. Do not copy the main file without its write-ahead log
while the controller is active. The example prints each task's current status.
Do not back-project that value into an earlier cutoff. Use a registry snapshot
captured at the cutoff or timestamped OpenHands terminal evidence.

To count decisions and batch sizes, use the action-observation pairing rules in
Section 16. Count only successful `spawn_agents` observations. Join returned
task IDs to the registry, then discard task bodies and results. The registry's
`parent_task_id` and `depth` fields provide the exact recursive split. Use the
parent event time for the decision timestamp and the task `created_at` time for
the controller-acceptance timestamp.

### Reconcile before interpreting

Build one private row per accepted task, then join its successful OpenHands
spawn observation and Weave root span. Report these sets explicitly:

```text
registry task, successful spawn observation, root Weave span
registry task, successful spawn observation, no root Weave span
failed spawn action, no registry task, no root Weave span
root Weave span, no retained registry or OpenHands record
```

The second set usually indicates a pre-trace launch failure, timeout before
agent startup, or trace gap. The fourth usually indicates expired local state,
an incorrect evidence window, or an incomplete conversation-ID join. Inspect
aggregate status and timestamps first. Read raw text only when those fields do
not resolve the discrepancy.

For future exact joins, add these non-content attributes to the child root
span: `task_id`, `tree_id`, `parent_task_id`, `depth`, `spawn_operation_key`,
and `requested_model_tier`. Also retain `conversation_id` and the observed
provider model. These fields let an analyst distinguish task creation from
invocation start, identify pre-trace failures, and measure direct and recursive
work without scanning prompts. Treat the identifiers as private even though
they contain no task text.

Use the following interpretation rules:

| Evidence after a rollout | Most likely explanation |
| --- | --- |
| More uptime or start success, shorter launch delay, but stable decisions per eligible event | Operational improvement |
| More successful spawn decisions, larger batches, or more recursion, with stable start success and duration | Advisor behavior change |
| More eligible research events, but stable decisions per eligible event | Workload or opportunity mix |
| Restored uptime and more decisions per eligible event | Mixed operational and behavior change |
| Higher task count but stable Frontier share | More delegation, not a Frontier-specific selection shift |

These are observational diagnoses, not randomized causal estimates. Compare the
deployed code and prompt diff. Classify a change as operational only when it
touches process launch, admission, concurrency, timeouts, recovery, storage, or
telemetry. Classify guidance, skill visibility, delegation triggers, and
model-selection requirements as behavior-policy changes.

### Worked example: delegation-policy rollout

One Senpai deployment supplied a useful mixed case. The rollout occurred at
2026-08-21 18:07:30 UTC. A local stable-prompt check classified child root
invocations without retaining or printing prompt content. Equal 24-hour
windows produced:

| Metric | Before | After |
| --- | ---: | ---: |
| Provider-observed child conversations | 10 | 50 |
| Conversations served by Fable, the configured Frontier provider model | 2 | 11 |
| Fable share | 20.0% | 22.0% |
| Finished root invocations | 10/10 | 50/50 |
| Root-invocation errors | 0 | 0 |
| Median root duration | 623 seconds | 589 seconds |
| 95th-percentile root duration | 1,182 seconds | 1,323 seconds |
| Recursive spawn decisions | 1 | 8 |
| Tasks requested by recursive decisions | 3 | 19 |

The raw conversation count rose 5x. However, the before window contained 9
hours, 37 minutes, and 30 seconds with no advisor delegation activity while an
old pod was gone and its replacement was not ready. Dividing by observed active
time changed the before rate to 10 / 14.375 = 0.70 conversations per hour and
the after rate to 50 / 24 = 2.08 conversations per hour. The estimated rate
change therefore fell from 5x to about 3x.

Launch reliability was already 100% in the observed before sample. Median root
duration improved by only about 5.5%, while the 95th percentile became slower.
This result describes children that reached Weave; it cannot measure tasks that
failed before their first traced agent call.

The advisor already delegated before the rollout. The prompt-policy review
compared revisions `97769de0` and `5a1ae8d0`. The two policy changes broadened
the advisor's recurring delegation triggers and made a critique before a new
research round or expensive portfolio explicit. They also required a deliberate
model-tier choice and expanded the guidance available to delegation-capable
children. The existing child skill-loading path did not change. These revisions
did not change child launch, concurrency, timeouts, recovery, or Weave
instrumentation. Fable's mapping to the Frontier tier came from deployment
configuration, not these revisions.

Recursive children made 16 more task requests after the rollout. That increase
equals 40% of the net increase of 40 root conversations, although the registry
must confirm exact one-to-one lineage. The remaining non-recursive increase was
also large. The recursive task mix moved from three `smart` requests before the
rollout to 16 `smart`, one `fast`, and two `frontier` requests after it. The
strongest explanation is therefore a behavior-policy change, amplified by
restored uptime and possibly by a different research-opportunity mix. The
nearly flat Fable share shows that the increase was broad delegation rather
than a material Frontier-specific shift. A two-sided Fisher exact test of 2/10
against 11/50 gives `p = 1.0`; the small before sample provides no evidence of
a conditional Fable-share change.

The post-rollout registry contained 65 tasks created in the same requested
window: 43 direct and 22 recursive; 32 requested `frontier`, 22 requested
`smart`, and 11 requested `fast`. At collection time, 62 were finished and
three had failed. Do not compare 65 registry tasks directly with 50 traced
child conversations. The registry uses task-creation time, Weave uses
invocation-start time, failed tasks can be absent from Weave, and the traces did
not carry a durable task ID for an exact join. The before registry had already
disappeared, so these registry values cannot establish a before-and-after tier
change.

The deployed image also advanced across more revisions than the two reviewed
prompt-policy changes and started a fresh pod and conversation. Earlier bundled
changes included provider retry and compaction work. Compare the exact before
and after image SHAs before ruling out an operational contribution. The causal
diagnosis is therefore mixed: there is direct evidence of more delegation
decisions and recursion after the policy change, while restored uptime and
possibly bundled runtime changes increased the opportunity to complete them.
The retained data cannot assign a causal percentage to either component.

### Validate before publishing

Check every item:

- The before and after windows have equal requested durations and one sourced
  intervention timestamp.
- Coverage gaps and observed advisor uptime are explicit.
- Every child is counted once, from a task ID or root invocation.
- Provider turns and tool spans are not counted as conversations.
- Requested tier and observed provider model are separate.
- Registry tasks, successful spawn observations, and root spans reconcile.
- Pre-trace failures remain visible instead of disappearing from the rate.
- Recursive tasks use `parent_task_id` or equivalent structured evidence.
- Task rate is normalized by observed uptime and, when possible, eligible
  advisor decision events.
- The deployment diff distinguishes runtime changes from policy changes.
- Conclusions use “consistent with” or “strongest explanation” unless the
  deployment was randomized.
- Published tables contain no prompts, task bodies, results, credentials,
  conversation IDs, trace IDs, task IDs, or local paths.

## 16. Build an advisor and student activity timeline

This timeline shows when the advisor created logical experiment assignments,
which persistent student received each assignment, what kind of work it
requested, whether it was terminal at the chart cutoff, and which approved
external milestones occurred. It is an activity audit, not a context-size plot
and not a count of raw training or benchmark arms.

Reuse the snapshot, event-graph, and active-ancestry rules in Sections 1
through 3. Keep any context-size panel derived from Sections 4 through 14
separate from the activity rows. Context size and research activity have
different units and different evidence contracts.

### Freeze the evidence window

Choose one requested UTC start, end, and cutoff. Freeze a coherent OpenHands
conversation snapshot from `base_state.json` and `events/event-*.json` for
the advisor and every persistent student. Discover delegated-child
conversations recursively, but never add a child's context to its parent.

Record observed coverage for every actor:

```text
role
actor
observed_start
observed_end
complete
gaps
```

An activity mutation can remain durable after its event leaves the active
model-context branch. Tag every event with active-branch membership, but do not
discard a successful off-branch mutation automatically. Include it only when a
paired observation or the joined external system proves that the operation
executed. Report active-branch and proven off-branch counts separately.

### Pair actions with observations

Parse each event as data. Never execute log content or interpolate it into a
shell command. Normalize timestamps to UTC, index events by ID, and pair each
`ActionEvent` with its `ObservationEvent` through `action_id`. Use
`tool_call_id` only as a checked fallback. When both the typed action and the
serialized tool-call arguments exist, require them to agree.

Create one assignment row only after a `create_assignment` action has a
paired, non-error `GitHubMutationObservation` whose state is
`assignment_created`. The contracts are in
[`senpai_agent/github/tools/contracts.py`](senpai_agent/github/tools/contracts.py),
and the executors are in
[`senpai_agent/github/tools/advisor.py`](senpai_agent/github/tools/advisor.py).

Use this private normalized row:

```text
assignment_id
requested_at
student
category
category_source
status_at_cutoff
terminal_at
pr_number
resource_url
active_branch
source_action_event_id
source_observation_event_id
```

Apply these lifecycle rules:

1. Key and deduplicate assignments by `assignment_id`. An idempotent replay
   with `changed=false` is not a new experiment.
2. Use the timestamp of the paired `create_assignment` action for
   `requested_at`. A GitHub `createdAt` timestamp may replace it only after
   the PR URL or number and trusted assignment marker match.
3. Do not create another experiment point for feedback, routing repair, a
   requested revision, result publication, or a submission retry. Attach those
   events to the same assignment.
4. Mark an assignment terminal only when a successful paired observation
   reports `experiment_merged` or `experiment_closed` at or before the
   cutoff. Otherwise mark it active at that cutoff.
5. Do not apply the current GitHub state retrospectively to an earlier cutoff.
   A merged assignment means that the task completed; it does not mean that the
   experiment produced a scientific win.
6. Join W&B runs through typed `ExperimentResult.runs[].run_id` values. Treat
   runs as evidence for the logical assignment. Do not draw one point for each
   arm, rung, retry, or replicate.

### Record delegated-child requests separately

Persistent-student assignments and delegated-child tasks are different work
units. Do not mix their counts.

Extract child requests from `spawn_agents`. Join action task specifications
to `SpawnAgentsObservation.tasks` by `key`. If keys are absent, first
require equal list lengths, then use the executor-defined list order. Deduplicate
by the returned `task_id`.

Retain these private fields:

```text
task_id
parent_actor
requested_at
returned_at
status_at_cutoff
agent_type
model_tier
provider_model
include_context
category
active_branch
```

The supported `agent_type` values are `general-purpose`, `explore`,
`bash-runner`, `search_general_web`, and
`search_research_publications`. The `model_tier` value is `fast`,
`smart`, or `frontier`; it is a routing tier, not necessarily the provider
model name. Read the exact provider model from the child's frozen
`base_state.agent.llm.model`. Leave it unknown when that state is
unavailable.

Use the first terminal `finished`, `failed`, or `cancelled` state from a
paired `await_agents`, `agent_status`, or `cancel_agents` observation.
When only durable controller state proves termination, require its structured
task ID and timestamp. A task with no terminal evidence before the cutoff
remains active.

The delegation schemas are in
[`senpai_agent/delegation.py`](senpai_agent/delegation.py).
[`tools/senpai_tool_telemetry.py`](tools/senpai_tool_telemetry.py) contains
reusable patterns for recursive discovery, role and model inference, timestamp
normalization, and tool pairing. It does not reconstruct branch membership,
logical assignment lifecycles, or child task specifications, so running it
alone does not produce this timeline.

### Classify the primary research intent

Assign exactly one category from the assignment's primary preregistered
intervention. Display these one-line definitions below the chart legend:

| Category | Definition |
| --- | --- |
| Diagnose / model | Measure, attribute, or predict a bottleneck without requiring a shipped mechanism. |
| Kernel / runtime | Change executed kernels, dispatch, memory, or scheduling to remove cost. |
| Policy / head | Change or price draft decisions or proposal-head/readout behavior. |
| Validate / transfer | Reproduce, integrate, or test whether a result survives another base, host, or end-to-end path. |

Use this deterministic resolution order:

1. Evidence-only replication, transfer, exactness, or a ship gate with no new
   mechanism is `validate_transfer`.
2. A changed shipped policy, proposal head, or readout decision is
   `policy_head`.
3. A changed on-path kernel or runtime implementation is `kernel_runtime`.
4. Measurement, ablation, screening, oracle analysis, attribution, or a cost
   model is `diagnose_model`.
5. Insufficient evidence is `unclassified`. Do not guess or send raw logs to
   an external classifier.

Store the ruleset version. Use a versioned assignment-ID override table for
reviewed mixed cases, with one short rationale per override. Label category
selection as deterministic interpretation, not provider-exact fact.

### Join board and intervention events

Keep external evaluation events on a separate Board lane. Join each event by
an immutable receipt and candidate or source SHA. Preserve the evaluator's
state and exact UTC timestamps for queue, rejection, cancellation, and
promotion. Do not infer a board event from advisor prose, and do not reinterpret
an evaluator's displayed percentage unless its definition supports that use.

Add a harness or configuration marker only from a sourced UTC event and exact
revision. Treat it as an annotation, not as proof of a causal effect. Never
infer deployment time from the first later tool call.

### Render the timeline

Use one UTC axis over the requested window:

- Draw one Board lane and one lane for every persistent student.
- Draw one point per logical assignment at `requested_at`.
- Use a circle when the assignment was terminal by the cutoff and a diamond
  when it was active.
- Use color only for the four research categories.
- Put concise, approved assignment details and terminal state in the tooltip.
- Draw external queue and terminal events on the Board lane.
- Optionally shade the pre-intervention interval and draw a vertical
  intervention marker.
- Derive before/after category shares and per-student totals from the same
  assignment rows so every count reconciles.
- Put delegated-child requests in a separate strip or table. Show agent type,
  model tier, provider model when known, request category, and terminal state.
- Show requested and observed bounds and make every coverage gap visible.

Publish only a sanitized derivative. Omit conversation IDs, event IDs,
assignment IDs, task IDs, raw prompts, task bodies, subagent results, local
paths, and unapproved PR identifiers. Treat every retained string as untrusted.
Build browser labels with `textContent` or escape them before inserting HTML.

### Validate before sharing

Check every item:

- Every actor has requested and observed coverage, including restart or
  state-loss gaps.
- Event IDs are unique, timestamps are valid UTC values, and action/observation
  pairs are unambiguous.
- Every assignment mark has a successful durable creation observation.
- Assignment IDs and child task IDs are unique after deduplication.
- Status is evaluated at the selected cutoff, with no future-state leakage.
- Active-branch and proven off-branch counts are reported separately.
- Every assignment has one category or fails explicitly as `unclassified`.
- Lane points, category totals, status totals, before/after bars, and
  per-student totals reconcile.
- Model tier and provider model remain separate fields.
- Board events join one-to-one to immutable evaluator receipts.
- No raw prompts, tool output, credentials, or unapproved identifiers appear
  in the shareable dataset or chart.
- The approved secret scanner passes on the exact JSON, HTML, and Markdown
  selected for sharing or Git staging.
- Interactive charts have no browser errors, clipped labels, overlapping text,
  or horizontal overflow in light and dark themes.

## 17. Measure research iteration speed

Measure how quickly Senpai converts a hypothesis into durable evidence and,
when configured, external evaluation feedback. Reuse the frozen OpenHands
snapshots and action-observation pairing from Sections 1 through 3 and 16.

The primary unit is one logical assignment and its terminal advisor decision.
Do not treat model calls, tool calls, W&B runs, benchmark arms, retries, PR
comments, or external submissions as interchangeable research iterations.

### Normalize the lifecycle

Build one private UTC event table from structured state rather than agent
narration:

| Event | Preferred evidence |
| --- | --- |
| Assignment created | Successful paired `create_assignment` observation |
| Job launched or terminated | Persisted supervisor or monitor state |
| Terminal signal delivered | Controller inbox event and delivery receipt |
| Conversation resumed | First OpenHands response after delivery |
| Result published | Successful paired `submit_experiment_result` observation |
| Advisor decision | Successful merge, close, or revision observation |
| W&B run started or finished | Run metadata joined through the typed result |
| External submission changed state | Immutable evaluator receipt |
| Frontier changed | Evaluator-reported promoted state |

Retain immutable join keys privately: assignment and revision identity, PR
number, commit SHA, run ID, training ID, monitor dedupe key, evaluator receipt,
conversation ID, and source event IDs. Publish only sanitized aggregates.

Keep these distinctions:

- Replicates, retries, and arms belong to their logical assignment.
- A W&B run is evidence, not necessarily a complete experiment.
- Local and W&B metrics are not external or official scores.
- A rejected external result can still be a valid scientific iteration.
- Current PR, run, or evaluator state must not be projected backward.

For wake diagnostics, retain each available boundary separately:

```text
job_terminal_at
monitor_signal_created_at
signal_delivered_at
conversation_resumed_at
first_relevant_action_at
result_published_at
```

These timestamps separate monitor polling, controller delivery, model response,
and scientific follow-up. Do not infer them from log-line order when structured
timestamps exist.

### Compare fair pre/post windows

Source the intervention from the deployment or fresh conversation that
actually loaded the change. A commit timestamp alone is insufficient. Use a
deployment band when rollout took time.

Choose equal, non-overlapping half-open windows:

```text
pre  = [cutoff - duration, cutoff)
post = [cutoff, cutoff + duration)
```

For a live post window, stop at the latest common observed timestamp and give
the pre window the same duration. Record coverage for the advisor, students,
controller, GitHub, W&B, and any external evaluator. Normalize rates by
observed hours. Report active or inherited cycles that cross a boundary rather
than silently excluding them.

Record concurrent changes in model, reasoning effort, prompt, hardware,
student count, evaluator availability, queue policy, and research baseline.

### Calculate speed and outcome together

| Measure | Definition |
| --- | --- |
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
| Progress | Best official metric and gap to the promoted frontier |

Report counts, medians, and 90th percentiles. For OpenHands, also report model
calls, input tokens, model-visible tool-output bytes, compactions, FinishActions,
and response gaps per terminal decision. These measure coordination overhead,
not scientific output.

Keep evaluator service time separate from controllable handoff. A serialized
evaluator can dominate the submission interval even when Senpai reacts quickly.

### Draw and verify the charts

Use three coordinated views:

1. A production-feedback timeline with one mark per external submission,
   official metric, terminal state, best-so-far line, and intervention band.
2. An equal-window comparison of assignment, evidence, decision, merge, W&B,
   and submission counts or rates. Show raw counts and observed hours.
3. A latency view for assignment-to-result, result-to-decision,
   terminal-job-to-action, controllable handoff, evaluator service, and model
   response-tail percentiles.

Add advisor and student lanes only when concurrency or idle capacity is the
question. Derive every panel from the same normalized rows so totals reconcile.
Use UTC axes. Mark local, W&B-only, and external evidence distinctly. State
whether higher or lower is better for every measure.

For responsive HTML, verify approximately 360, 736, and 1,024 pixels in light
and dark themes. Check legends, tooltips, intervention annotations, clipped or
overlapping labels, horizontal overflow, and browser errors. Build untrusted
labels with `textContent` and embed only sanitized data.

Interpret a speedup with its scientific yield. Pair cycle-time and submission
claims with validity gates, merge rate, promotion rate, official progress, and
frontier gap. Disclose small samples, censored work, inherited candidates,
coverage gaps, evaluator availability, and time-of-day effects. Unless the
intervention was randomized, describe the result as consistent with the
deployed change rather than proven causal.
