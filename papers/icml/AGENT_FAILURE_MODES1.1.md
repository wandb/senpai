# Agent Failure Modes

Senpai fails in two qualitatively different ways. At the action level, the agent spends context on repeated observation and collides with brittle tool interfaces. At the systems level, it loses continuity: delegated work is not always awaited, derived summaries drift from PR and W&B truth, and long-running experiments outlive the sessions that launched them.

The most important result is that local action errors and cross-session state failures behave differently. `Observation_churn` is a real token and wall-clock tax, but it is not the same as incorrect tool use. Explicit `tool_interface_error` remains common even after polling and rereads are counted separately. The research-corrupting failures, however, are dominated by delegation, session mortality, and state/accounting drift.

## 1. Summary Findings

1. `Observation_churn` appears in **143/241 `noam` sessions (59%)** and **111/453 `radford` sessions (25%)**. It is most severe on `noam`, where long repeated-read and repeated-status loops are common.
2. `Tool_interface_error` appears in **191/241 `noam` sessions (79%)** and **356/453 `radford` sessions (79%)**. The dominant subtypes are wrong repo/path/cwd assumptions, GitHub scope or repo errors, blocked wait primitives, oversized reads, and cancelled parallel calls.
3. Delegation improves in rate but not in prevalence. In `noam`, `delegation_mistake` appears in **234/241 sessions (97%)**, while the per-session rate falls from **1.73 -> 1.51 -> 1.36** across P4-P6.
4. Session boundaries are the main `radford` failure surface. `Session_mortality` affects **371/453 sessions (82%)**, and `state_accounting_drift` affects **377/453 sessions (83%)**.
5. State drift is the slow-burn failure that grows with harness maturity. In `noam`, state drift rises from **27%** of P4 sessions to **44%** of P5 sessions to **80%** of P6 sessions.
6. The design lesson is boundary design: bounded monitoring, path-aware tool wrappers, idempotent result submission, awaited delegation, and direct PR/W&B reconciliation matter more than additional prompt prose.

## 2. Category Definitions

The taxonomy separates one operational cost category from four primary failure categories.

### Observation churn

`Observation_churn` means repeated polling, rereading, or status checking that produces little or no new information. It is expensive and often unnecessary, but it is not automatically incorrect tool use.

Example:
- Session `096e8190-1437-42b7-b1b3-f9b606ca700a` repeatedly runs the same process and log checks, including `ps`, `wc -l`, `nvidia-smi`, and `sleep 60`, long after the state has stopped changing.

### Tool / interface error

`Tool_interface_error` means the tool layer explicitly rejects the action or the agent uses the wrong interface: wrong repo, wrong path, blocked wait primitive, invalid arguments, oversized reads, edit guards, or cancelled parallel tool calls.

Example:
- Session `37b14a94-5d46-5773-82ea-4887f2e3f6ca` calls `gh pr view 2327 --comments` and hits a missing `read:org` scope, then queries the wrong repo (`AnswerDotAI/senpai-v1`) and gets repeated `404` responses.

### Delegation mistake

`Delegation_mistake` means the agent uses subagents or background workers incorrectly: spawning them redundantly, routing the wrong job to them, or launching useful verification work and then making a decision before consuming the result.

Example:
- In the P6 `unawaited_verification_subagent` pattern, advisors launch verification or review work in the background and then close or merge a PR before reading the subagent output.

### Session mortality

`Session_mortality` means the work outlives the session that was supposed to own it. The agent may restart, compact away active context, cross a queue boundary, or disappear while training or reporting is still in flight.

Example:
- Session `7686e311-c5b7-41cc-9577-311d9850053f` launches and monitors training, but there is no in-session results submission marker. The PR later receives a results comment anyway, which shows the work survived while the original reporting session did not.

### State / accounting drift

`State_accounting_drift` means the agent reasons from stale or derived state rather than the authoritative research ledger in PRs, PR labels, and W&B metadata.

Example:
- Advisor session `5b6f2072-741` closed PR `#2070` as post-hoc analysis, while essentially the same idea later reappeared as merged PR `#2076`. The problem is not just a bad judgment call; it is judgment made from drifted state.

## 3. Branch-Level Results

| Branch | Sessions | Observation Churn | Tool / Interface Error | Delegation | Session Mortality | State Drift | Dominant Story |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `noam` | 241 | 1137 incidents across 143 sessions | 915 across 191 sessions | 353 across 234 sessions | 18 across 18 sessions | 143 across 143 sessions | Action-level noise dominates: monitoring churn, interface friction, and delegation mistakes are layered inside the same sessions. |
| `radford` | 453 | 673 incidents across 111 sessions | 1030 across 356 sessions | 40 across 40 sessions | 411 across 371 sessions | 377 across 377 sessions | Monitoring churn is lower, but continuity failures dominate outcomes: most sessions drift or die across handoffs. |

`Noam` is primarily noisy inside the session. Observation churn, tool/interface errors, and delegation mistakes often co-occur before the work reaches a restart boundary. `Radford` is primarily noisy across boundaries. Action-level tool errors remain common, but the larger research failures come from session death and stale state, not from polling alone.

The PR reporting surface shifts in the same direction. Result-bearing `noam` PRs are mostly body-driven (**1360** body vs **463** comment), while result-bearing `radford` PRs are entirely comment-driven (**404** comment, **0** body). This is consistent with a workflow where later sessions increasingly repair or finalize reporting, although PR location alone does not prove the causal pathway.

## 4. `noam` Phase Progression

| `noam` phase | Sessions | Observation Churn Prevalence | Observation Incidents / Session | Tool / Interface Error Prevalence | Tool / Interface Error / Session | Delegation / Session | State Drift Prevalence | Mortality Prevalence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase_4` | 33 | 45% | 8.73 | 45% | 3.15 | 1.73 | 27% | 6% |
| `phase_5` | 90 | 74% | 4.42 | 84% | 3.91 | 1.51 | 44% | 13% |
| `phase_6` | 118 | 52% | 3.82 | 85% | 3.89 | 1.36 | 80% | 3% |

The phase progression is not a simple story of the harness eliminating failures. P4 contains the most intense observation churn by incident volume, but the incidents are concentrated in relatively few giant sessions. P5 and P6 spread observation churn across more sessions, especially through repeated rereads and monitor loops. Tool/interface errors rise sharply in P5 and then stay flat in P6. Delegation improves steadily, while state/accounting drift becomes the main new systems failure.

## 5. Category Patterns

### Observation churn

Observation churn matters operationally without always being a semantic failure.

- `Noam`: **143/241 sessions (59%)**, **4.72 incidents/session**
- `Radford`: **111/453 sessions (25%)**, **1.49 incidents/session**

On `noam`, churn is mostly repeated code/doc rereads and repeated status checks while the agent is still orienting or waiting. On `radford`, churn is mostly long log-monitoring loops around active training or background-task output files. This distinction matters because repeated polling is sometimes the least harmful available action during long training jobs, sparse metric emission, or queue waiting.

Observation is still not free. It overlaps strongly with other failure modes:

- `134/241` `noam` sessions contain `observation_churn`, `tool_interface_error`, and `delegation_mistake` together.
- `106/111` `radford` observation-churn sessions also contain at least one explicit tool/interface error.

The pattern is best read as a cost amplifier. Observation churn is often the medium through which other mistakes accumulate, even when it is not the root cause.

### Tool / interface error

Once repeated polling and rereads are counted separately, the remaining tool category is still large, but its content is clearer.

- `Noam`: **191/241 sessions (79%)**, **3.80 incidents/session**
- `Radford`: **356/453 sessions (79%)**, **2.27 incidents/session**

This is a broad interface-friction bucket, not a catastrophic tool-failure bucket. Many incidents are recoverable one-offs. The important point is that they are explicit tool-side rejections, not merely repeated observation.

| Branch | Top Tool / Interface Error Subtypes | Count |
| --- | --- | ---: |
| `noam` | `parallel_cancelled` | 237 |
| `noam` | `github_scope_or_repo_error` | 227 |
| `noam` | `wrong_path_or_cwd` | 119 |
| `noam` | `oversized_read` | 111 |
| `radford` | `wrong_path_or_cwd` | 342 |
| `radford` | `blocked_wait_pattern` | 294 |
| `radford` | `github_scope_or_repo_error` | 208 |
| `radford` | `parallel_cancelled` | 118 |

These are mostly operational mistakes: wrong repo, missing GitHub scopes, wrong cwd, blocked `sleep ... ; tail ...` wait patterns, cancelled parallel inspections, and oversized raw reads. They are not primarily failures of scientific reasoning. The agent often understands the experiment but still collides with fragile operational surfaces.

### Delegation mistake

Delegation is the clearest example of a failure family that improved locally without disappearing.

- In `noam`, delegation mistakes appear in **234/241 sessions (97%)**.
- The rate falls from **1.73/session in P4** to **1.36/session in P6**.
- Prevalence stays nearly flat even as the per-session rate improves.

The subpattern changes are telling. P4 is dominated by disposable polling-subagent behavior. P6 is dominated by `unawaited_verification_subagent`. The harness improved the syntax of delegation, but not the semantics of dependency management.

### Session mortality

Session mortality is a minor category in `noam` and a dominant category in `radford`.

- `Noam`: **18/241 sessions (7%)**
- `Radford`: **371/453 sessions (82%)**

The dominant `radford` subtype is `pr_switch_after_compaction_or_restart`: **371 incidents in 371 sessions**. `Missing_result_post` is smaller at **40 incidents**, but more revealing: it marks the sessions where training was launched and the in-session reporting path broke before results were posted.

The strongest evidence that this is a session-boundary failure is that **21 of the 40 `radford` `missing_result_post` incidents later acquired results on the PR anyway**, always via a later comment. The experiment was not lost; the original session's reporting channel was.

### State / accounting drift

State drift is the category that most clearly grows with harness maturity.

- `Noam` state drift prevalence rises from **27% of sessions in P4** to **44% in P5** to **80% in P6**.
- `Radford` shows state drift in **377/453 sessions (83%)**.

The role split inside `noam` is especially revealing:

- `Advisor`: state drift in **65/71 sessions (92%)**
- `Student`: state drift in **78/170 sessions (46%)**

By `radford`, drift is no longer mainly advisor-side. It becomes a full-system student-loop problem as well: **372/448 `radford` student sessions (83%)** show state drift. Once work spans multiple sessions, any cached summary that is not continuously reconciled against PR labels and W&B truth turns into a second ledger.

## 6. Failure Bundles

The categories are not independent. A small number of bundles recur across the corpus.

| Bundle | Sessions | Interpretation |
| --- | ---: | --- |
| `noam`: `observation_churn` + `tool_interface_error` + `delegation_mistake` | 134/241 | Noisy sessions are usually noisy in more than one way. |
| `noam`: `tool_interface_error` + `delegation_mistake` + `state_accounting_drift` | 119/241 | Once sessions live longer, stale state stacks on top of action-level interface friction. |
| `radford`: `tool_interface_error` + `session_mortality` + `state_accounting_drift` | 352/453 | The typical `radford` failure is a mix of shallow tool errors and broken continuity. |
| `radford`: `observation_churn` + `session_mortality` + `state_accounting_drift` | 111/453 | Heavy monitoring is usually attached to later continuity failure, not isolated from it. |

This bundle view is more useful than category totals alone. `Noam` is noisy but still mostly inside the session. `Radford` crosses session boundaries without sufficiently durable state.

## 7. Harness-Change Impact

| Change | Directly Observed Signal | Conservative Interpretation | What Remained Broken |
| --- | --- | --- | --- |
| `poll-for-work` / skills-heartbeat loop (`#1965`) | `delegation_mistake` falls **1.73 -> 1.51 -> 1.36** per `noam` session | Strongly consistent with reduced disposable polling-subagent behavior | `observation_churn` stays high (**8.73 -> 4.42 -> 3.82**) and `tool_interface_error` stays flat-to-worse after P5 (**3.15 -> 3.91 -> 3.89**) |
| large `max_turns` / longer session runway (`07a6506`) | `noam` `session_mortality` drops **12 -> 4** from P5 to P6 | Consistent with fewer abrupt mid-task deaths inside a single session | `state_accounting_drift` rises **40 -> 94** incidents over the same step |
| summary-doc state (`CURRENT_RESEARCH_STATE.md` and similar) | `noam` advisor drift reaches **65/71 sessions (92%)** while student drift is **78/170 (46%)** | Summary-led continuity helped advisors stay active across longer loops, but also created a derived-state layer prone to drift | Drift becomes more common, not less, as the system matures |
| idempotent results-submission path | **21/40** `radford` `missing_result_post` incidents later acquired PR results via comments | Supports eventual repair after a broken reporting path | Does not prevent the original handoff failure; repair still depends on later sessions |

The safest reading is that the harness got better at reducing catastrophic local redundancy, but not at preserving durable truth across sessions.

## 8. Failure Chains

### Chain A: observation churn -> token burn -> session exhaustion

The common monitoring loops are often understandable, but they still consume context and delay decision points. Long runs of `ps`, `wc -l`, `tail`, `nvidia-smi`, and repeated rereads make it more likely that the session ends before reporting or reconciliation happens.

### Chain B: tool/interface rejection -> workaround cascade

A GitHub or file call fails, the agent tries a nearby but still wrong interface, and the workaround introduces more parsing, path, or parallel-call errors. Session `37b14a94-5d46-5773-82ea-4887f2e3f6ca` is a clean example: scope error, wrong repo, broken parsing assumptions, then another wrong endpoint.

### Chain C: restart / compaction -> PR drift -> stale accounting

In `radford`, `pr_switch_after_compaction_or_restart` overlaps with `stale_summary_state` in **371 sessions**. After a boundary crossing, the agent often re-enters through stale local state instead of a fresh reconciliation against the PR ledger and W&B runs.

### Chain D: training finishes after the launching session dies

A student launches training, the session keeps polling, the session ends or is replaced, and the run completes after the original reporting context is gone. That is what **21/40** repaired `missing_result_post` incidents look like.

## 9. Case Studies

### Case 1: observation churn is expensive, but not the same as misuse

Session `096e8190-1437-42b7-b1b3-f9b606ca700a` captures the distinction cleanly. The agent spends large stretches repeatedly checking the same process, GPU, and log file state. This is wasteful, but it is not the same as calling the wrong tool. The design response is a bounded monitor primitive, not a label that treats all polling as failure.

### Case 2: the real tool failures are interface failures

Session `37b14a94-5d46-5773-82ea-4887f2e3f6ca` is representative of the explicit error bucket: missing GitHub scope, wrong repo, broken parsing assumptions, then repeated wrong endpoint calls. This is the failure mode meant by "the agent called the interface incorrectly."

### Case 3: delegation improved syntactically but not semantically

The `unawaited_verification_subagent` cluster in P6 shows the mature version of delegation failure. Advisors launch verification or review work in the background, then close or merge a PR before consuming the result. This is better than P4's disposable polling-subagent storm, but it is still a coordination failure.

### Case 4: missing results are often later repaired

`Radford` session `7686e311-c5b7-41cc-9577-311d9850053f` on PR `#2966` is representative of the mortality chain. Training launches, the session contains heavy GPU/log monitoring, and there is no in-session results-submission marker. Yet PR-level evidence later shows a results comment. The experiment succeeded; the original session failed to survive long enough to own the report.

### Case 5: PR switching is usually a handoff symptom

`Radford` session `2a741a64-7f3a-4abf-832e-524d86060fff` on PR `#2682` shows the restart/queue version of PR drift: repeated polling for new work, a ready-for-review transition, and state that spans multiple PR contexts. The important interpretation is not that the agent randomly chose a different task. It is that the handoff surface between "current assignment," "queue state," and "local summary state" is too weak.

## 10. Evidence Snapshots

### A. Observation churn

- Session: `096e8190-1437-42b7-b1b3-f9b606ca700a`
- PR: `#2623`
- Evidence ref: `conversation_logs/2026-04-21/pai-2/gen__senpai-gen-5ff7796695-9klh8__claude.tgz`
- Scope note: this is a cost-category example, not a tool/interface error example.

> `date && ps -p 5724 ... && wc -l /tmp/gen-drivaerml-200k.log (x646); sleep 60 && echo "done" (x45); date && nvidia-smi ... && wc -l /tmp/gen-drivaerml-200k.log (x12)`

### B. Tool / interface error

- Session: `37b14a94-5d46-5773-82ea-4887f2e3f6ca`
- PR: `#2327`
- Evidence ref: `analysis/scratch/hivemind_cache/transcripts/37b14a94-5d46-5773-82ea-4887f2e3f6ca.md`
- Scope note: this is the canonical example of explicit interface misuse.

> `gh pr view 2327 --comments` -> missing `read:org` scope; `gh api repos/AnswerDotAI/senpai-v1/issues/2327/comments` -> `Not Found (HTTP 404)`.

### C. Training outlives the session

- Session: `7686e311-c5b7-41cc-9577-311d9850053f`
- PR: `#2966`
- Evidence ref: `conversation_logs/2026-04-22/k8s_pod_archive_185147/gohan/root_.claude.tgz`
- Scope note: this is the `session_mortality / missing_result_post` example; later PR evidence shows a results comment even though the original session had no in-session results submission marker.

> `nvidia-smi --query-gpu=index,memory.used,utilization.gpu ... (x199); for dir in /workspace/senpai/target/icml2026/wandb/run-20260422_16... (x62)`

### D. State/accounting drift

- Session: `46eafb61-1174-4476-a0b6-0a986ae8d4e2`
- PR: `#2963`
- Evidence ref: `conversation_logs/2026-04-22/k8s_pod_archive_185147/brook/root_.claude.tgz`
- Scope note: this session also carried other labels; the excerpt below is the evidence used for `state_accounting_drift / stale_summary_state`.

> `for f in /tmp/tandemfoil_layerscale.log /tmp/airfrans_layerscale.log /tmp/drivaerml_layerscale_1e4.log ... (x81); ... (x35)`

## 11. Design Implications

- Treat `observation_churn` as a product problem. The response should be bounded monitor primitives with backoff, completion detection, and cheap "nothing changed" summaries.
- Treat `tool_interface_error` as an interface-hardening problem. The response should be path-aware file tools, canonical GitHub wrappers, and clearer blocked-wait guidance.
- Treat cross-session recovery as a first-class systems requirement. If training can outlive a session, result harvesting cannot be owned only by the launching session.
- Treat summary docs as caches, not ledgers. The authoritative state must live in PR labels, PR comments, and W&B metadata.
- Treat asynchronous delegation as unsafe by default. If a parent decision depends on a subagent result, the harness should force an await or make the dependency explicit.

## 12. Methods and Evidence Boundaries

- Evidence comes from **241** transcript-backed `noam` sessions, **453** root `radford` sessions from local `conversation_logs/` archives, joined PR records, and joined W&B records.
- The root-session corpus contains approximately **242.4 million tokens** measured with `o200k_base`: **44.8M** tokens on `noam` and **197.6M** on `radford`.
- Tool-category metrics come from `analysis/failure_modes/recompute_tool_taxonomy_primary.py`, which parses the raw `noam` transcript markdown and the exact `radford` root-session `.jsonl` members referenced by `analysis/failure_modes/data/sessions.jsonl`.
- Delegation, mortality, and drift counts use the normalized artifacts under `analysis/failure_modes/data/`.
- `Observation_churn` is a tracked cost category, not a task-fatal failure by definition.
- `Tool_interface_error` is intentionally broad: many incidents are recoverable one-offs rather than catastrophic failures.
- Weighted counts for joined-ledger categories use `incident_count`, not simple row counts.
- `Noam` and `radford` are comparable at the taxonomy level, but not every frequency difference should be read as a pure harness effect; the corpora expose different logging surfaces.
- The token-volume figure is a corpus-size measure over the root-session files used in analysis, not a sum of model billing or usage tokens.

These measurements support a narrative claim about where senpai fails and how those failures changed. They are less suited to a clean causal claim that any single prompt or harness edit alone explains the whole transition.
