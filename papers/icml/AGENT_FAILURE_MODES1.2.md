# Agent Failure Modes in a 24h SENPAI Fleet Trace

## Contribution Paragraph

We analyze a canonical 24-hour SENPAI deployment trace spanning `53,022` real Claude requests, `230,195` filtered transcript records, and `5.24B` cache-inclusive tokens. The dominant operational failure mode is not idle polling or simple tool misuse: `28,247` model-facing monitor events produced `28,246` direct model responses and consumed `3.25B` cache-inclusive tokens, because narrow training-log updates were interpreted inside full long-lived agent contexts. Strict tool-interface errors were much smaller, at `892/25,363` tool results (`3.5%`), while assignment plus human-issue polling accounted for only `3.8%` of student usage records. A negative correlation was observed between tool-use error rates and context length: strict tool-interface error rate decreased from `7.0%` below `25k` input tokens to `1.4%` at `175k-200k`. This does not imply that long context causally improves tool use; the cleaner interpretation is that long contexts are dominated by valid monitoring/status commands, while short contexts contain setup, GitHub, path, schema, and edit mistakes. The main design implication is that autonomous ML agents need bounded monitoring, durable external state, and explicit handoff protocols more than additional prompt prose.

## Discussion Paragraph

The 24-hour trace shows SENPAI operating less like a sequence of isolated LLM calls and more like a distributed training-control system. Students ran long GPU jobs, re-entered through scheduled wakeups, compacted conversations, watched logs, and reported results through PRs and W&B. This architecture worked in important ways: compactions were automatic and paired with continuation summaries, scheduled wakeups carried PR/run/PID context, and `38/39` sessions with final-like monitor signals also showed explicit PR-comment reporting. The main failure surface was therefore not that agents polled too often or could not use tools at all. It was that small pieces of fresh evidence, especially monitor events, repeatedly reloaded very large cached contexts; that session-local state had to be compressed into fragile summaries; and that PR labels, pod liveness, W&B summaries, and bot comments formed partially overlapping ledgers. Monitor command design matters, but it is not the whole story: epoch-streaming `tail -f | grep` commands created too many events, while the Claude Code architecture amplified each event into a main-context continuation. In this setting, "tool misuse" must be decomposed: routine polling and monitoring are expected coordination actions, strict tool-interface errors are explicit rejected calls, training/runtime failures are experiment outcomes or infrastructure signals, and state/accounting drift is a cross-ledger consistency problem.

## Monitor Setup Case Study

The monitor-cost problem is partly command design and partly architecture. Many `Monitor` commands were reasonable in intent but too chatty in effect: persistent `tail -f ... | grep ...` watchers matched every epoch or best-metric line, and Claude Code then delivered each matching line as a model-facing event in the main session. Better commands would have reduced event volume; they would not have eliminated the amplification mechanism, because every emitted event still caused a normal continuation of the long-lived cached context.

Gilbert's session `dc06937d-ec28-481e-8533-2b962cb9bbf9` is the clearest example. The high-volume monitor used only one persistent shell watcher:

```bash
tail -f /tmp/trial1_vol15x.log /tmp/trial2_vol20x.log 2>/dev/null |
  grep -E --line-buffered '"epoch"|Traceback|Error|FAILED|OOM|NaN|Killed|best_checkpoint|best_test_metrics'
```

This was not semantically vague: it watched two AirfRANS trials for epoch metrics, errors, checkpoints, and test metrics. But because both logs emitted JSON on every epoch, the monitor produced `1,092` model-facing events over about six hours. Gilbert used only `2` explicit `Monitor` setups and `21` wakeups, yet received `1,093` direct monitor responses consuming `134.5M` cache-inclusive tokens. The command-level lesson is to emit only milestone, new-best, error, completion, or timeout events. The architectural lesson is to add a stateful reducer between raw logs and the main agent: `min_interval`, `max_events`, `coalesce_window`, `only_on_change`, `new_best_delta`, and side-context summarization would all directly attack this failure mode.

A contrast case supports this interpretation. Gohan's session `dbf9958b-215e-4a88-972b-6743168168ed` used a more summary-shaped monitor with `check_interval=120`, PID checks, compact latest-run status, and an `ALL_DONE` terminal condition. It still used `Monitor`, but it produced only `48` direct monitor responses and `6.2M` cache-inclusive tokens. This does not eliminate the architecture issue, but it shows that monitor prompt/command design can reduce the event stream by an order of magnitude before architectural changes are made.

| Monitor Setup Pattern | Task-Linked Setups | Task-Linked Events | Interpretation |
| --- | ---: | ---: | --- |
| Commands containing `epoch` | `385` | `20,463` | Main source of event volume |
| `tail -f` plus epoch-like grep | `253` | `18,382` | Worst high-volume pattern |
| Commands parsing/summarizing JSON with Python | `82` | `936` | Much lower event volume |
| `sleep 300` throttled monitors | `23` | `313` | Lower volume because updates were rate-limited |

## Corpus Scope

| Quantity | Value |
| --- | ---: |
| Window | `2026-04-22T19:00:00Z` to `2026-04-23T19:00:00Z` |
| Fleet coverage | `1` advisor pod plus `59` student pod archives |
| Token-emitting student pods | `57` |
| Filtered transcript files | `1,292` |
| Filtered transcript records | `230,195` |
| Claude Code usage records | `53,050` |
| Real Claude model requests | `53,022` |
| Cache-inclusive total tokens | `5,244,977,135` |
| Cache-read tokens | `5,112,922,631` (`97.5%` of cache-inclusive total) |
| Uncached total tokens | `132,054,504` (`2.5%` of cache-inclusive total) |
| Student share | `5,053,788,078` tokens (`96.4%`) |
| Advisor share | `191,189,057` tokens (`3.6%`) |
| Sidechain/subagent usage | `4,823` requests, `93,745,736` tokens |
| Conversation compactions | `240`, all automatic |

The bundle is the local canonical trace at `/Users/mmcguire/ML/senpai/conversation_logs/20260422T1900Z_20260423T1900Z_24h_pai2_claude_usage`. Token totals are cache-inclusive unless otherwise stated.

## Category Definitions

`Context-amplified monitoring` is the cost pattern where a narrow external monitor event causes a normal model continuation of a long session. Example: Gilbert's session `dc06937d-ec28-481e-8533-2b962cb9bbf9` used only `2` explicit `Monitor` setups and `21` wakeups, but received `1,093` direct monitor responses consuming `134.5M` cache-inclusive tokens.

`Polling and scheduled re-entry` are expected coordination actions: checking for PR assignments, checking human issues, or asking the system to wake the agent later. Polling is not counted as tool misuse by default. Example: assignment plus human-issue polling was `1,907` student requests, only `3.8%` of student usage records.

`Tool-interface error` means the tool layer explicitly rejected the action or the agent used the wrong interface: wrong cwd/path, GitHub scope or repo error, blocked wait primitive, oversized read, edit guard, validation error, failed push, or cancelled parallel call. Example: the advisor repeatedly hit missing GitHub scopes on `gh pr view --repo wandb/senpai`, and several PR-disposition helpers failed because plugin scripts were missing in the pod path.

`Training/runtime failure signal` means the tool result or log contained experiment/runtime failure evidence such as `NaN`, `OOM`, `Traceback`, nonzero training exits, or killed processes. This is not automatically an agent-control failure; it often means the experiment failed.

`Session-boundary pressure` means useful work crosses compaction, wakeup, sidechain, or window boundaries. This is weaker than confirmed session mortality. In this bundle, many endings are deliberate scheduled re-entry rather than crashes.

`State/accounting drift` means the agent must reconcile partially inconsistent ledgers: PR labels, PR comments, W&B config/summary, bot summaries, local logs, and live pod activity. Example: PR labels listed `mitsuha` and `mugen` as WIP owners, while the canonical actor logs showed no Claude usage for those students in the window.

`Delegation pressure` means the system relies on sidechains or subagents for review and coordination. It is not inherently a failure, but it creates dependency-management and state-transfer risk.

## Quantitative Findings

| Pattern | Count | Share / Interpretation |
| --- | ---: | --- |
| Training execution or monitoring requests | `40,821` | `77.0%` of real model requests |
| Training execution or monitoring tokens | `4,448,401,515` | `84.8%` of fleet tokens |
| Student training-supervision requests | `40,275` | `80.6%` of student usage records |
| Direct monitor-event responses | `28,246` | `53.3%` of real model requests |
| Direct monitor-event tokens | `3,250,452,753` | `62.0%` of fleet tokens |
| Direct monitor-event uncached tokens | `47,581,066` | Only `1.5%` of direct monitor cache-inclusive total |
| Assignment plus human-issue polling | `2,055` fleet requests | `3.9%` of real model requests |
| Assignment plus human-issue polling, students only | `1,907` requests | `3.8%` of student usage records |
| Strict tool-interface errors | `892/25,363` tool results | `3.5%` strict error rate |
| Training/runtime failure-like tool results | `2,156/25,363` tool results | `8.5%`; mostly experiment/runtime signals |
| Scheduled wakeups | `911` | `783` (`86.0%`) were `<=5m` |
| Sidechain/subagent requests | `4,823` | `1.8%` of fleet tokens |
| Automatic compactions | `240` | All had continuation summaries |

The central decomposition is that monitoring dominates cost, but strict tool errors do not. The system is expensive because many small observations are interpreted in large cached contexts, not because the agent is constantly calling tools incorrectly.

## Pattern 1: Monitoring Dominates the Fleet

The training-control loop generated the bulk of usage. Students made `615` explicit `Monitor` calls and `845` `ScheduleWakeup` calls, but those monitors produced `28,247` model-facing monitor events and `28,246` direct model responses. The median direct monitor response reloaded about `110k` cache-read tokens while adding only `347` uncached tokens.

| Session | Direct Monitor Responses | Cache-Inclusive Tokens | Uncached Tokens | Notes |
| --- | ---: | ---: | ---: | --- |
| `robin / 737e94ea` | `2,679` | `313,864,276` | `1,402,102` | AirfRANS and training-progress monitoring |
| `violet / e529b386` | `2,637` | `300,772,952` | `1,884,434` | EMA/volume-loss monitoring, PR `#3105` prominent |
| `nezuko / c5363c62` | `2,191` | `271,952,374` | `1,689,985` | Multiple trial lifecycle monitors |
| `gilbert / dc06937d` | `1,093` | `134,463,579` | `868,208` | `2` explicit monitor setups, `21` wakeups |
| `kakashi / ea648b14` | `1,009` | `127,665,159` | `554,992` | PR/state-heavy monitoring |

The top three direct-monitor sessions alone consumed `886.6M` cache-inclusive tokens, `27.3%` of direct monitor usage. The top ten consumed `51.6%`. This is a concentration problem: a small number of long-lived training sessions dominate the token footprint.

The most common monitor summaries were ordinary training-progress watchers: epoch progress, restart progress, AirfRANS volume-weight curricula, EMA volume experiments, warmup restarts, and DrivAerML progress. Among explicit `Monitor` tool inputs, `466/615` used a tail-plus-grep shape, `436/615` filtered for error-like terms, and `384/615` referenced W&B or log paths. The monitor commands were narrow; the expensive part was the full-context interpretation after each event.

## Pattern 2: Polling Is Not the Main Cost Driver

Assignment polling and human-issue polling are visible but not dominant. Students made `1,020` assignment-polling requests and `887` human-issue-polling requests, together `3.8%` of student usage records. Fleet-wide, assignment plus issue polling was `2,055` real model requests, `3.9%` of the request count.

This matters because polling is often the correct control behavior for a distributed ML fleet. Repeated `NO_WORK` checks, human-issue checks, and short wakeups are coordination mechanisms. The report therefore treats polling as `polling_and_reentry`, not as `tool_misuse`. The more important question is whether polling carries durable state and whether it wakes at bounded intervals.

Scheduled wakeups were common: `911` total, including `845` student wakeups. Their prompts often carried assigned PR state, run IDs, PIDs, task IDs, monitor IDs, and rolling metrics. `783/911` wakeups were for `<=5m`, which explains frequent re-entry without requiring a session-crash interpretation.

## Pattern 3: Tool Errors Are Interface Friction, Not the Dominant Failure

Strict tool-interface errors were `892` out of `25,363` tool results (`3.5%`). This denominator includes only explicit tool-result rows: it includes `615` explicit `Monitor` setup calls and `911` `ScheduleWakeup` calls, but excludes the `28,246` direct monitor-response turns that drove the token cost. Removing the explicit `Monitor` setup calls leaves `24,748` non-Monitor tool results with `891` strict errors (`3.6%`); removing both `Monitor` and `ScheduleWakeup` leaves `23,837` tool results with the same `891` strict errors (`3.7%`). Tool errors are therefore not being hidden by Monitor setup calls, although the separate direct-monitor response stream dominates token usage.

| Strict Tool-Interface Error Subtype | Count | Example |
| --- | ---: | --- |
| `process_or_command_failed` | `273` | Advisor `gh pr create` failed because GitHub reported no commits between `radford` and the new branch. |
| `github_scope_or_repo_error` | `163` | Advisor `gh pr view 3115 --repo wandb/senpai` failed because the token lacked `read:org` scopes. |
| `blocked_wait_pattern` | `136` | Harness rejected `sleep 30 && gh pr list ...` and instructed the agent to use `Monitor` or an until-loop. |
| `wrong_path_or_cwd` | `114` | Advisor tried to source `/scripts/senpai-gh.sh`, which did not exist in the pod path. |
| `git_auth_or_remote` | `79` | `git push -u origin ...` failed because Git could not read a GitHub username. |
| `parallel_cancelled` | `39` | A parallel `gh pr list` call was cancelled after a sibling parallel call errored. |
| `bad_output_parse` | `27` | W&B script printed metrics, then failed on a helper signature mismatch: unexpected keyword argument `samples`. |
| `oversized_read` | `24` | `Read` attempted a task-output file with `43,203` tokens, above the `25,000` token tool limit. |
| `edit_guard` | `22` | `Edit` was rejected because the file had changed since it was last read, or because the target string was absent. |
| `other_tool_error` | `11` | `Read` attempted a `317.4KB` file, above the `256KB` file-size limit. |
| `validation_or_schema_error` | `3` | A `Monitor` call supplied unsupported parameter `timeout`, causing `InputValidationError`. |
| `command_not_found` | `1` | A Bash pipeline used `column -t`, but `column` was not installed, producing exit `127`. |

The advisor top-level session `9c9f7fde-83cb-43eb-b776-32c542a4f7a7` is a compact example. It had `34` strict tool-interface errors, including missing plugin scripts, failed branch pushes, PR creation with no commits, blocked wait patterns, and oversized task-output reads. These are operational integration failures, not failures of scientific hypothesis generation.

Rei's student session `fa37ade6-6bc4-4ac1-a779-e827aaf7f63b` shows the other side: `45` strict tool errors, `44` of them command failures, amid active PR work on `#3222`. A strict tool-error count alone does not tell us whether the research result was wrong; it tells us the session spent control bandwidth recovering from brittle interfaces.

### Why the Top Tool Errors Happened

A root-cause pass over the three largest strict tool-error subtypes (`572` incidents total) shows that many were induced by tool contracts and prompt examples, not by the model spontaneously choosing nonsensical tools.

| Error Family | Count | Most Likely Cause | Would Better Prompting or a Skill Help? |
| --- | ---: | --- | --- |
| `gh pr view ... --comments` / raw `gh pr view` scope failures | `154/163` GitHub scope errors | Role and skill docs showed `gh pr view <number> --comments`, but the deployed token had `repo` scope only; `gh` used GraphQL fields requiring `read:org`. | Yes, if the prompt removes the bad command and a helper exposes REST-backed `pr_body`, `pr_issue_comments`, `pr_review_comments`, and `pr_all_comments`. Merely adding prose is weaker because sub-agents copied exact commands from their prompts. |
| `sleep N && ...` blocked waits | `136` blocked waits, including `134` status/log/GPU checks | Prompts said "wait 60 seconds" or "wait 5 minutes" without giving the Claude Code-safe wait idiom. Agents translated natural language waiting into foreground Bash sleeps. | Yes. A short `wait idioms` skill or prompt block should say: never foreground `sleep`; use `ScheduleWakeup` for loop re-entry, `Monitor` for condition waits, and background `until` loops only for bounded local checks. |
| `gh pr create` with no commits between base and head | `71/273` process-command failures | The `assign-experiment` skill claimed to handle branch/PR mechanics but its steps created and pushed a branch without first creating an assignment commit or other diff. GitHub then rejected the PR as having no commits relative to `radford`. | Strong yes. This should be an executable wrapper, not a recipe. The wrapper should create a sentinel assignment commit or preflight `git rev-list base..head` before calling `gh pr create`. |
| Ad hoc log/JSON probes | `79/273` process-command failures | Agents wrote brittle probes such as `tail -1 .../output.log | python -c 'json.load(...)'`; empty logs, non-JSON lines, or one missing trial turned "not ready yet" into a Bash failure. | Yes. A status helper should normalize missing files, empty logs, and partial trial starts as structured states: `not_started`, `running_no_metric`, `metric_seen`, `failed`, `complete`. |
| Training/runtime or train-CLI failures | `64/273` process-command failures | Some were real experiment crashes; others reflected stale command examples such as underscore flags (`--wandb_name`, `--wandb_group`) when `train.py` rejected the arguments. | Partly. Real training failures belong outside tool-interface errors; stale CLI examples should be replaced by a generated train-command helper or by caching `train.py --help` into the role instructions. |

The GitHub scope failures are the clearest prompt-induced case. The student workflow and `poll-for-work` skill both told agents to check comments with `gh pr view <number> --comments`. In the trace, this failed repeatedly with `read:org` scope errors. Agents often recovered by using REST endpoints (`gh api repos/wandb/senpai/pulls/<pr>/comments` and `.../issues/<pr>/comments`), but fresh sub-agents kept receiving the bad command. This is why the error is concentrated in short contexts: `150/163` GitHub scope errors occurred in the `0-25k` context bucket, where newly launched agents were still following prompt examples rather than learned local repairs.

The blocked-wait failures show a different prompt-interface mismatch. The role loop asked agents to wait, but the execution environment disallowed foreground `sleep && command` waits. Most incidents were not idle polling: `113/136` waited and then read logs or task output, `14/136` waited and then checked `nvidia-smi`, and only `2/136` were advisor GitHub PR polling. The harness rejection was useful because it named replacement idioms, but the repeated failures show that the safe waiting pattern needs to be first-class, not discovered through errors.

The `process_or_command_failed` bucket should be interpreted cautiously. It mixes command-composition mistakes, brittle diagnostics, real experiment failures, and external service failures. The largest actionable subpattern is assignment PR creation: the skill text gave the model a multi-step recipe but omitted the commit/preflight needed to make the branch PR-able. The model often repaired this with `git commit --allow-empty`, which is useful evidence that the failure was not conceptual research confusion; it was a missing invariant in the helper contract.

The design lesson is that skills help most when they hide fragile mechanics, not when they merely document them. A skill that says "run these commands" still leaves quoting, branch state, token scopes, CLI schemas, and wait primitives inside the model's action space. The next version should make the boring actions executable and idempotent: assignment PR creation, PR comment retrieval, label swaps, run-status checks, and wait/re-entry should be single-purpose helpers with stable outputs.

### Context-Length Analysis

The 24-hour trace does not support the intuition that longer context causes more strict tool-interface errors. For each tool-result row, we joined the result back to the assistant tool-use turn and used the logged Claude usage metadata as the context length: `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. This produced context lengths for `25,363` tool results, including all `892` strict tool-interface errors. The error rate falls with context length, from `7.0%` below `25k` input tokens to `1.4%` in the `175k-200k` bucket.

| Context Bucket | Tool Results | Strict Errors | Error Rate |
| --- | ---: | ---: | ---: |
| `0-25k` | `4,513` | `317` | `7.0%` |
| `25-50k` | `4,537` | `201` | `4.4%` |
| `50-75k` | `4,183` | `139` | `3.3%` |
| `75-100k` | `3,188` | `86` | `2.7%` |
| `100-125k` | `2,687` | `53` | `2.0%` |
| `125-150k` | `2,465` | `50` | `2.0%` |
| `150-175k` | `2,118` | `22` | `1.0%` |
| `175-200k` | `1,672` | `24` | `1.4%` |

![Tool-interface errors by context bucket](/Users/mmcguire/ML/senpai/analysis/failure_modes/data/fleet_24h/tool_interface_context_24h_buckets.svg)

Aggregating buckets gives the same result: `5.72%` below `50k`, `3.05%` from `50k-100k`, `2.00%` from `100k-150k`, and `1.21%` above `150k`. The bucket-level correlation between context midpoint and error rate is strongly negative (`r = -0.906`). This should not be interpreted as evidence that long context improves tool use. The cleaner interpretation is workload mix: short contexts contain setup, GitHub auth/scope, path, schema, and edit mistakes, while long contexts are dominated by repeated monitoring and status commands that are costly but usually valid.

### Within-Session Tool Learning

The negative correlation is also consistent with a limited form of within-session tool learning: agents often make an interface mistake during setup, observe the tool error, and then switch to a working pattern. This is clearest for subtypes where the tool error itself contains a repair hint. The evidence is strongest for blocked waits, oversized reads, GitHub scope errors, and missing paths; it is weaker for generic command failures because those are often experiment/runtime failures rather than interface learning.

| Error Subtype | Sessions with Error | Recovery Signal | Sessions with Recovery | Sessions Repeating Same Subtype |
| --- | ---: | --- | ---: | ---: |
| `github_scope_or_repo_error` | `157` | Later successful `gh api repos/...` call within 8 tool results | `154` (`98%`) | `5` (`3%`) |
| `blocked_wait_pattern` | `120` | Later `Monitor`, `ScheduleWakeup`, background task, or until-loop within 8 tool results | `108` (`90%`) | `13` (`11%`) |
| `oversized_read` | `17` | Later bounded `Read`, `Grep`, `head`, `tail`, or `sed -n` within 8 tool results | `16` (`94%`) | `4` (`24%`) |
| `wrong_path_or_cwd` | `100` | Later path discovery or corrected path use within 8 tool results | `94` (`94%`) | `12` (`12%`) |
| `edit_guard` | `12` | Later `Read` followed by successful `Edit`/`Write` within 12 tool results | `5` (`42%`) | `3` (`25%`) |

Representative traces show the mechanism. After a GitHub scope error on `gh pr view 3101 --comments`, the Tanjiro session switched to `gh api repos/wandb/senpai/pulls/3101/comments` and `gh api repos/wandb/senpai/issues/3101/comments`, both successful. After an oversized `Read` of `train.py`, the Casca session switched to bounded reads (`limit`, `offset`) and `Grep`. After an advisor path failure for `/scripts/senpai-gh.sh`, the advisor checked `$CLAUDE_PLUGIN_ROOT`, used `Glob` to find `plugins/senpai/scripts/senpai-gh.sh`, and then successfully sourced the corrected path.

The `blocked_wait_pattern` subtype is especially interpretable because the harness error carried an explicit repair instruction. In advisor session `9c9f7fde-83cb-43eb-b776-32c542a4f7a7`, the model tried to wait for assignment PRs with `sleep 30 && gh pr list --repo wandb/senpai --label "student:franky" ...`. The harness rejected the call and told the model to use an until-loop, `Monitor`, or `run_in_background`. The next tool call repaired the pattern: `until gh pr list --repo wandb/senpai --label "student:franky" --state open --json number ...; do sleep 5; done`, launched with `run_in_background: true`. Thirty minutes later the same session repeated the same mistake for `student:gojo`, received the same rejection, and again immediately repaired it with a background until-loop. This looks less like the model discovering a deep concept and more like short-horizon interface adaptation: when the tool layer returns a concrete replacement idiom, the model usually adopts it quickly, but the habit is not perfectly retained across a long control session.

This supports a careful version of the tool-learning hypothesis. The agent does appear to adapt to explicit tool feedback inside a session, especially when the repair is local and mechanical. But the aggregate context-length trend should still be attributed primarily to session phase and workload mix: early turns are setup-heavy and error-prone; later turns are monitoring-heavy and mechanically stable.

## Pattern 4: Training Failures Should Be Separated from Tool Misuse

A broader scan found `2,156` failure-like tool results (`8.5%` of tool results), but most are not interface errors. The largest subtype is `training_failure_signal` (`1,302`), including `NaN`, `OOM`, `Traceback`, `RuntimeError`, killed processes, or divergent metrics. These are often the outcome of the experiment being tested.

This distinction is essential for paper claims. A failed training run can be scientifically useful if it cleanly closes a hypothesis. A tool-interface error is different: it is the agent or harness failing to execute its intended control action. The trace shows both, but the design response is different. Training failures need robust result capture and checkpointing; tool-interface errors need safer wrappers and path-aware affordances.

## Pattern 5: Session Boundaries Are Mostly Managed, But Still Risky

The trace contains `240` automatic compactions: `71` student-main, `68` student-subagent, `7` advisor-main, and `94` advisor-subagent. Every compaction had a continuation summary. Main student compactions were large-context events, with median `196,151` pre-compact tokens reduced to `3,031` post-compact tokens, retaining about `1.5%` of the token count. Advisor main compactions were similarly large, with median `196,761` pre-tokens and `5,530` post-tokens.

This is good engineering and a source of risk at the same time. The continuation summaries preserved important operational state in inspected cases: assigned PR, active runs, W&B IDs, baseline metrics, reporting requirements, and review criteria. But a compression from roughly `196k` tokens to a few thousand tokens makes the local transcript a derived state store. Derived state must be treated as lossy and reconciled against PR and W&B truth.

Student main session IDs were also disposable by design. There were `197` student main sessions, and `50` actors had more than one main session in the window. Inter-session gaps were short, with median about `17.5m`. The last observed category for `102/197` student main sessions was training execution or monitoring. This is not enough to call them dead sessions; many had re-armed monitors or scheduled wakeups.

Result reporting looked comparatively strong once terminal evidence appeared. Among `39` student main sessions with final/completion/done/result-like monitor summaries, `38` had explicit PR-comment evidence and `33` had ready/status-review handoff evidence. The main unresolved area is the edge of the window: `8` late-window training sessions had no explicit PR result comment before the bundle ended, and all should be treated as unfinished-at-window-boundary unless follow-on logs prove a missed report.

## Pattern 6: State and Accounting Drift Is a Cross-Ledger Problem

The 24-hour trace shows several distinct ledgers in play: PR body, PR comments, PR labels, W&B config, W&B summary, W&B history, local logs, schedule-wakeup prompts, compaction summaries, and bot-generated session summaries. Drift appears when the agent treats one derived ledger as authoritative without reconciling the others.

Observed drift patterns include:

- PR labels tracked intended assignment state, not guaranteed pod liveness. The advisor's survey listed `mitsuha` and `mugen` as WIP/draft PR owners, while the canonical actor logs showed no Claude usage for those students in the window.
- W&B summaries were usually cross-checked, but summary semantics and baselines could change conclusions. One AirfRANS review identified that a pre-`#3135` baseline made a result look positive even though it was worse against the current baseline.
- Local W&B artifacts sometimes disagreed with terminal summaries. One inspected review noted that a variant summary had been overwritten and fell back to local W&B logs.
- Bot-generated session summaries were common PR-comment noise. Raw logs contained `181` user records with `hivemind-session-summary` markers, with `260` total string occurrences; students usually recognized them as bot summaries rather than advisor instructions, but they still occupied the same channel as real guidance.

The design lesson is that PR and W&B should be authoritative ledgers, not prose summaries. Wakeup and compaction summaries can carry hints, but important decisions should re-read live PR state, labels, W&B run config, and W&B history.

## Pattern 7: Delegation Is Useful, But It Adds Dependency Pressure

Sidechains were a major coordination layer: `4,823` sidechain requests and `93.7M` cache-inclusive tokens, with `1,851` advisor sidechain requests and `2,972` student sidechain requests. The raw trace contains `1,094` sidechain transcript files and `284` `Agent` tool uses.

Advisor sidechains often did useful work: read PR comments, invoke W&B checks, inspect training metrics, and return merge/send-back/close recommendations. A representative advisor review of PR `#3206` checked comments and W&B, found Trial 2 at `3.887%` against a `3.833%` baseline, and recommended close/send-back rather than merge. This is the right pattern. The risk is dependency management: if the parent session does not consume the review result, or if subagent summaries become another stale ledger, delegation becomes a state-transfer failure.

## Harness-Change Impact

| Mechanism | Observed Benefit | Remaining Failure Surface |
| --- | --- | --- |
| `Monitor` tool | Narrow shell watchers over logs and W&B outputs; only `615` explicit student monitor calls | Event stream expands to `28,246` direct model responses and `3.25B` cache-inclusive tokens |
| Prompt caching | Makes high-frequency monitoring feasible; uncached monitor spend is only `47.6M` tokens | Each monitor response still reloads a median `110k` cached tokens |
| `ScheduleWakeup` | Carries PR/run/PID/task context across re-entry; `845` student wakeups | Frequent `<=5m` re-entry can still create many model turns and derived-state prompts |
| Automatic compaction | All `240` compactions had continuation summaries | Summaries are lossy; main student sessions retain about `1.5%` of pre-compact tokens by count |
| PR-comment result reporting | `38/39` final-like monitor sessions had explicit PR-comment evidence | Late-window sessions and moving W&B baselines still require follow-on reconciliation |
| Sidechain review | Enables parallel PR/W&B verification | Adds dependency-management risk if parent sessions act before consuming results |

The harness is not merely failing. It is doing useful distributed-systems work: waking, compacting, monitoring, and delegating. The remaining failures are boundary failures: how often to wake, how much context to reload, which ledger is authoritative, and when a compressed summary is safe to trust.

## Worked Examples

### Robin: full-context monitoring dominates cost

Session `student/robin/737e94ea-607d-4653-86fd-9480f4c30e6a` is the largest token session in the bundle: `338.6M` cache-inclusive tokens and `2,929` usage records. It received `2,679` direct monitor responses consuming `313.9M` cache-inclusive tokens but only `1.4M` uncached tokens. The failure mode is not wrong polling; it is high-frequency interpretation of small training updates inside a large cached session.

### Violet: repeated experiment monitoring with useful reporting

Session `student/violet/e529b386-7086-4919-83f2-36615647648b` consumed `324.6M` cache-inclusive tokens, including `2,637` direct monitor responses and `300.8M` direct-monitor tokens. PR `#3105` appears prominently in the session. The trace shows the normal shape of successful long-running supervision: repeated monitor callbacks, compactions, wakeups, and eventual result/reporting markers.

### Gilbert: two monitor setups produce a thousand continuations

Session `student/gilbert/dc06937d-ec28-481e-8533-2b962cb9bbf9` had only `2` explicit `Monitor` tool uses and `21` wakeups, but `1,093` direct monitor events. It compacted twice and centered on PR `#3204`. This is the cleanest example of why explicit tool-call counts understate monitoring load.

### Advisor: orchestration creates interface-error cascades

The advisor top-level session `advisor/9c9f7fde-83cb-43eb-b776-32c542a4f7a7` consumed `150.6M` tokens and used `236` `Agent` calls. It also hit `34` strict tool-interface errors: missing plugin paths, failed pushes, no-commit PR creation, blocked sleep patterns, oversized reads, and edit guards. This is the operational-control failure surface of the advisor role.

### Rei: many command failures inside one student session

Session `student/rei/fa37ade6-6bc4-4ac1-a779-e827aaf7f63b` had `45` strict tool errors, `44` classified as process or command failures, while working around PR `#3222`. This is a local execution-friction example rather than a monitoring-cost example.

### Canute: late-window uncertainty should not be overcalled

One late-window Canute session near PR `#3225` had no explicit PR result comment before the bundle ended. The final observed assistant state said one trial had diverged and another was still being monitored. This should be classified as unfinished-at-window-boundary, not confirmed missing result reporting. The 24-hour bundle is canonical for this window, but it cannot adjudicate events after `2026-04-23T19:00:00Z`.

## Design Lessons

1. Bound monitor interpretation, not just monitor commands. A narrow `tail | grep` watcher is cheap externally but expensive if every event causes a full-context model continuation.
2. Treat polling as a control primitive, not a failure. Optimize its cadence and state payload; do not collapse it into tool misuse.
3. Split strict interface errors from training failures. The former need safer wrappers; the latter need robust experiment capture and reporting.
4. Make result submission idempotent and ledger-backed. PR comments, PR labels, and W&B metadata should be joined directly, not mediated only by summaries.
5. Treat compaction and wakeup prompts as lossy caches. They should carry enough state to resume, but decisions should revalidate against authoritative ledgers.
6. Require parent sessions to consume sidechain outputs before disposition. Delegation is useful only if dependency completion is part of the control protocol.
7. Track liveness separately from assignment. A PR label can say WIP while the assigned pod has no token-emitting session in the window.
8. Interpret context-length correlations through workload mix. In this bundle, strict tool-interface error rate decreases with longer context because long contexts are dominated by valid monitoring/status commands, not because long context is inherently safer.

## Methods and Evidence Boundaries

Primary quantitative evidence comes from the canonical 24-hour bundle and the fleet token audit:

- `/Users/mmcguire/ML/senpai/conversation_logs/20260422T1900Z_20260423T1900Z_24h_pai2_claude_usage`
- `/Users/mmcguire/ML/senpai/analysis/operational_metrics/fleet_token_audit_20260424T094430Z/FLEET_24H_TOKEN_AUDIT.md`

Supporting tables produced for this report:

- `/Users/mmcguire/ML/senpai/analysis/failure_modes/data/fleet_24h/failure_modes_24h_summary.json`
- `/Users/mmcguire/ML/senpai/analysis/failure_modes/data/fleet_24h/failure_modes_24h_sessions.jsonl`
- `/Users/mmcguire/ML/senpai/analysis/failure_modes/data/fleet_24h/failure_modes_24h_examples.jsonl`
- `/Users/mmcguire/ML/senpai/analysis/failure_modes/data/fleet_24h/tool_interface_context_24h_summary.json`
- `/Users/mmcguire/ML/senpai/analysis/failure_modes/data/fleet_24h/tool_interface_context_24h_buckets.csv`
- `/Users/mmcguire/ML/senpai/analysis/failure_modes/data/fleet_24h/tool_interface_context_24h_buckets.svg`
- `/Users/mmcguire/ML/senpai/analysis/failure_modes/analyze_24h_bundle_failure_modes.py`
- `/Users/mmcguire/ML/senpai/analysis/failure_modes/analyze_tool_interface_context_24h.py`

The analysis uses exact CSV counts for model requests, token usage, direct monitor responses, and request-driver categories. It uses raw JSONL parsing for tool-result errors, monitor tool shapes, compactions, wakeups, sidechain files, and exemplar selection. Tool-interface errors are counted only when a `tool_result` is explicitly marked as an error. For the context-length analysis, context size is the logged assistant usage for the tool-use turn: `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. This uses the actual retrospective Claude usage metadata in the transcript rather than re-submitting large historical prompts to the token-counting endpoint, which is intended for preflight estimates. Training/runtime failure signals are heuristic string matches over tool results and should be interpreted as a broad diagnostic category, not a strict agent-error count.

The bundle ends at `2026-04-23T19:00:00Z`. Sessions still training near that boundary cannot be classified as missed reports without follow-on logs. The safest claim is that the canonical 24-hour trace exposes context-amplified monitoring, boundary-state pressure, and interface friction; it does not by itself prove all late-window runs were abandoned.
