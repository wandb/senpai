# Agent Failure Modes

This file rebuilds the failure-mode section from raw local evidence instead of inheriting the existing synthesized draft. The `noam` branch is reconstructed from transcript-level Hivemind cache artifacts, while `radford` is reconstructed from the local `conversation_logs/` archive JSONL snapshots. `hivemind-query` is used as a second-reader verification layer where available, not as the primary truth source.

## Methodology

- `noam` session denominator: **241** transcript-backed sessions from `analysis/scratch/hivemind_cache/` after excluding `other` role sessions.
- `radford` session denominator: **453** root sessions parsed directly from `conversation_logs/` archives.
- Base-branch PR coverage: **2009** PRs on `noam`, **694** PRs on `radford`.
- Hydrated live W&B runs linked to PR/session evidence: **120**.
- Primary taxonomy: `tool_misuse`, `delegation_mistake`, `session_mortality`, `state_accounting_drift`.
- Secondary tracked patterns: `missing_result_post`, `pr_switch_after_compaction_or_restart`, `unawaited_verification_subagent`, `premature_pr_closure`.

## Branch Comparison

| Branch | Sessions | PRs | Tool Misuse | Delegation Mistake | Session Mortality | State Drift | Missing Result Post | PR Switch | Unawaited Verification | Premature Closure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noam` | 241 | 2009 | 1077 | 353 | 18 | 143 | 8 | 0 | 123 | 21 |
| `radford` | 453 | 694 | 176 | 40 | 411 | 377 | 40 | 371 | 0 | 0 |

## `noam` Phase Table

| Phase | Sessions | Tool Misuse / Session | Delegation Mistakes / Session | Session Mortality Incidents | State Drift Incidents | Missing Result Post |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase_4` | 33 | 3.52 | 1.73 | 2 | 9 | 2 |
| `phase_5` | 90 | 4.68 | 1.51 | 12 | 40 | 5 |
| `phase_6` | 118 | 4.58 | 1.36 | 4 | 94 | 1 |

## `radford` Branch Summary

- `radford` is reconstructed from local archive JSONL rather than the older Hivemind cache, so its evidence is structurally richer but temporally narrower.
- Observed sessions: **453**.
- Most common primary category: **session_mortality** (411 incidents).
- The strongest archive-native mortality signals are PR switching after compaction/restart (371) and student sessions that launch training without an in-session result submission marker (40).

## Pooled Top Findings

- `tool_misuse` / `duplicate_poll_loop`: 685 incidents
- `tool_misuse` / `repeated_identical_tool_call`: 568 incidents
- `state_accounting_drift` / `stale_summary_state`: 499 incidents
- `session_mortality` / `pr_switch_after_compaction_or_restart`: 371 incidents
- `delegation_mistake` / `subagent_misuse`: 270 incidents
- `delegation_mistake` / `unawaited_verification_subagent`: 123 incidents
- `session_mortality` / `missing_result_post`: 48 incidents
- `state_accounting_drift` / `premature_pr_closure`: 21 incidents

## Worked Examples

### Example 1 — `tool_misuse` / `repeated_identical_tool_call`

- Branch: `noam`
- Phase: `phase_6`
- Session: `8c5fdaa1-cd88-52d8-a64e-fe5d5f9e3029`
- Severity: `moderate`
- Evidence source: `prebuilt_ledger`
- Evidence ref: `analysis/scratch/hivemind_cache/analyses/8c5fdaa1-cd88-52d8-a64e-fe5d5f9e3029.json`
- Adjudication note: Derived from prebuilt session/conversation ledger counts.
- Scope note: this session triggered multiple incident labels; the excerpt below is the evidence used for `tool_misuse` / `repeated_identical_tool_call`.

> Actually wait - the python command I tried to run was importing from train.py but that immediately starts training! The flag is being parsed correctly... I need to kill the background process `b2evcca93` since it's running a full training job with its own wandb run ("honest-dust-6972")

### Example 2 — `tool_misuse` / `duplicate_poll_loop`

- Branch: `radford`
- Session: `096e8190-1437-42b7-b1b3-f9b606ca700a`
- Severity: `moderate`
- PR: [#2623](https://github.com/wandb/senpai/pull/2623) — TandemFoil: Multi-query attention audit — more epochs via K/V reduction
- W&B run: `1kted1ug` (gen/tandem-mqa-lr2e4)
- Evidence source: `prebuilt_ledger`
- Evidence ref: `conversation_logs/2026-04-21/pai-2/gen__senpai-gen-5ff7796695-9klh8__claude.tgz`
- Adjudication note: Derived from prebuilt session/conversation ledger counts.
- Scope note: this session triggered multiple incident labels; the excerpt below is the evidence used for `tool_misuse` / `duplicate_poll_loop`.

> date && ps -p 5724 -o pcpu,etime --no-headers 2>/dev/null && wc -l /tmp/gen-drivaerml-200k.log (x646); sleep 60 && echo "done" (x45); date && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 0 && ps -p 5724 -o rss,pcpu,etime --no-headers 2>/dev/null && wc -l /tmp/gen-drivaerml-200k.log (x12)

### Example 3 — `delegation_mistake` / `unawaited_verification_subagent`

- Branch: `noam`
- Phase: `phase_6`
- Session: `25da9fd8-6fee-5358-9690-3d06928c0aa3`
- Severity: `moderate`
- Evidence source: `prebuilt_ledger`
- Evidence ref: `analysis/scratch/hivemind_cache/analyses/25da9fd8-6fee-5358-9690-3d06928c0aa3.json`
- Adjudication note: Derived from prebuilt session/conversation ledger counts.
- Scope note: this session triggered multiple incident labels; the excerpt below is the evidence used for `delegation_mistake` / `unawaited_verification_subagent`.

> sleep 60 && source "/workspace/senpai/plugins/senpai/scripts/senpai-gh.sh" && student_poll_for_work "tanjiro"

### Example 4 — `session_mortality` / `missing_result_post`

- Branch: `radford`
- Session: `7686e311-c5b7-41cc-9577-311d9850053f`
- Severity: `minor`
- PR: [#2966](https://github.com/wandb/senpai/pull/2966) — wave3: Huber Loss for Robust Regression (cross-dataset)
- Evidence source: `prebuilt_ledger`
- Evidence ref: `conversation_logs/2026-04-22/k8s_pod_archive_185147/gohan/root_.claude.tgz`
- Adjudication note: Derived from prebuilt session/conversation ledger counts.
- Scope note: this session triggered multiple incident labels; the excerpt below is the evidence used for `session_mortality` / `missing_result_post`.

> nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null (x199); for dir in /workspace/senpai/target/icml2026/wandb/run-20260422_16{0838,0842,0848,1048,0823}*; do   runid=$(basename "$dir" | sed 's/run-[0-9]*_[0-9]*-//')   logfile="$dir/files/ou (x62); nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null (x30)

### Example 5 — `session_mortality` / `pr_switch_after_compaction_or_restart`

- Branch: `radford`
- Session: `2a741a64-7f3a-4abf-832e-524d86060fff`
- Severity: `minor`
- PR: [#2682](https://github.com/wandb/senpai/pull/2682) — DrivAerML: 4L/384d + T_max=50 (slower cosine for wider model)
- W&B run: `149gczw7` (rei/drivaerml-4L384d-tmax50)
- Evidence source: `prebuilt_ledger`
- Evidence ref: `conversation_logs/2026-04-21/pai-2/rei__senpai-rei-69d9959bc9-vpjz7__claude.tgz`
- Adjudication note: Derived from prebuilt session/conversation ledger counts.
- Scope note: this session triggered multiple incident labels; the excerpt below is the evidence used for `session_mortality` / `pr_switch_after_compaction_or_restart`.

> until gh pr list --label "student:rei" --label "status:wip" --state open --json number --jq 'length' 2>/dev/null | grep -q '[1-9]'; do sleep 60; done && echo "NEW_WORK_AVAILABLE" (x5); source "/workspace/senpai/plugins/senpai/scripts/senpai-gh.sh" && mark_ready_for_review 2643 && swap_gh_pr_label 2643 "status:wip" "status:review" (x2)

### Example 6 — `state_accounting_drift` / `stale_summary_state`

- Branch: `radford`
- Session: `46eafb61-1174-4476-a0b6-0a986ae8d4e2`
- Severity: `minor`
- PR: [#2963](https://github.com/wandb/senpai/pull/2963) — wave3: LayerScale Initialization for Transformer Stability (cross-dataset)
- W&B run: `05kdcwfz` (brook/layerscale-1e4-tandemfoil-nocompile)
- Evidence source: `prebuilt_ledger`
- Evidence ref: `conversation_logs/2026-04-22/k8s_pod_archive_185147/brook/root_.claude.tgz`
- Adjudication note: Derived from prebuilt session/conversation ledger counts.
- Scope note: this session triggered multiple incident labels; the excerpt below is the evidence used for `state_accounting_drift` / `stale_summary_state`.

> for f in /tmp/tandemfoil_layerscale.log /tmp/airfrans_layerscale.log /tmp/drivaerml_layerscale_1e4.log /tmp/drivaerml_layerscale_1e5.log; do name=$(basename $f .log); count=$(grep (x81); for f in /tmp/tandemfoil_layerscale2.log /tmp/airfrans_layerscale.log /tmp/drivaerml_layerscale_1e4.log /tmp/drivaerml_layerscale_1e5.log; do name=$(basename $f .log); count=$(grep (x35); for f in /tmp/tandemfoil_

### Example 7 — `state_accounting_drift` / `premature_pr_closure`

- Branch: `noam`
- Phase: `phase_6`
- Session: `3d7f80df-f6d4-54e6-8d92-11d833a14f59`
- Severity: `minor`
- Evidence source: `prebuilt_ledger`
- Evidence ref: `analysis/scratch/hivemind_cache/analyses/3d7f80df-f6d4-54e6-8d92-11d833a14f59.json`
- Adjudication note: Derived from prebuilt session/conversation ledger counts.
- Scope note: this session triggered multiple incident labels; the excerpt below is the evidence used for `state_accounting_drift` / `premature_pr_closure`.

> Exit code 1 GraphQL: Your token has not been granted the required scopes to execute this query. The 'login' field requires one of the following scopes: ['read:org']

## Disagreements Against Existing Drafts

- `wandb_hydration_capped`: Capped live hydration to 120 runs out of 1520 referenced IDs.
- `hivemind_query_skipped`: ANTHROPIC_API_KEY missing.
- `tool_misuse_per_session`: **match** — rebuilt=[3.52, 4.68, 4.58] prior=[3.52, 4.68, 4.58]
- `delegation_mistakes_per_session`: **mismatch** — rebuilt=[1.73, 1.51, 1.36] prior=[1.73, 1.36, 1.4]
- `state_accounting_drift_counts`: **mismatch** — rebuilt=[9, 40, 94] prior=[4, 11, 17]

## Design Lessons

- Idempotent skill design is still mandatory. Repeated polling, repeated reads, and malformed GitHub calls remain common enough that tool misuse should be treated as expected behavior, not an edge case.
- Delegation gets better when it is wrapped in named skills, but verification-style subagents still fail when the parent launches them asynchronously and then makes a decision before consuming their result.
- Session mortality is not just a prompt problem. In both corpora, unfinished work frequently crosses session boundaries, which makes result-harvesting and restart-aware state recovery more important than stricter prose instructions.
- Derived summary state should stay subordinate to the PR ledger and W&B metadata. When summaries drift, the system starts reasoning from stale abstractions instead of auditable artifacts.

## `hivemind-query` Verification

- Verification was attempted but not completed: ANTHROPIC_API_KEY missing.
