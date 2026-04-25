---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

name: experiment-status-check
description: >
  Produce a fresh status report for the senpai ML experiment fleet. Use when the
  user asks for an experiment status, final status, PR/W&B/pod health check,
  stale student triage, training shutdown harvest, advisor-state audit, or a
  "what is really happening right now?" report. The report must prioritize
  paper-facing test metrics over validation metrics and compare test results to
  dataset benchmarks or targets.
---

# Experiment Status Check

Create a timestamped `analysis/STATUS_*.md` report that answers two questions:

1. What are the best paper-facing **test** results right now, and how do they
   compare to the benchmark targets?
2. Is the advisor/student system actually doing useful work, or is it stalled,
   mislabeled, sleeping, or optimizing the wrong thing?

The core distinction is "is the fleet alive?" versus "is useful science
happening?" Keep those separate throughout the report.

Use this skill with `wandb-primary` for W&B data and the GitHub skill or `gh`
for PR state.

## Why this skill exists

The status check is an intervention tool, not a newsletter. The system spends
real GPU time and real researcher attention, so the report must help decide
whether to keep training, harvest results, repair operations, or restart with
new instructions.

The recurring failure modes this skill guards against:

- **Validation drift:** agents find plausible validation wins and accidentally
  report them like paper wins. The paper needs test metrics.
- **Fleet illusion:** Kubernetes pods can be Running while student Claude
  sessions are idle, sleeping, mis-assigned, or done.
- **Assignment invisibility:** a PR without the right `student:<name>` and
  branch labels can look assigned to the advisor while being invisible to the
  student pod.
- **Advisor state staleness:** `CURRENT_RESEARCH_STATE.md` is useful steering
  context, but it can lag behind GitHub, W&B, and pod reality.
- **Experiment sprawl:** many PRs can be active while only a few have a
  credible path to improving paper-facing test results.

## Non-negotiables

- Put test metrics before validation metrics. Validation is a steering signal,
  not the paper result. This prevents the report from overstating progress.
- Always pair a test metric with its benchmark target/reference and a gap/read:
  "beats", "misses by X", "internal benchmark only", or "no external scalar".
  This makes the next decision obvious instead of leaving the reader to infer
  whether a number matters.
- Always include source and caveat columns for metrics: W&B run ID, PR number,
  full-eval vs batch-limited, best-checkpoint vs final epoch, truncated vs full.
  This protects the paper from mixing incompatible evaluation protocols.
- Do not trust advisor state files as source of truth until cross-checked
  against GitHub labels, W&B, and pod/process state. The advisor can be wrong
  precisely when a status check is most needed.
- Do not treat a Running pod as active training. Check for real Python
  `train.py` processes and recent W&B runs. A pod can be healthy while the
  experiment loop is stalled.
- Do not hide operational failures below science detail. If labels, pods, or
  polling are broken, surface that in the executive read. Broken control flow
  invalidates otherwise good scientific planning.
- Do not report "SOTA" from validation unless the benchmark contract is
  validation-based. For this programme, most paper claims are test-facing.

## Evidence order

For metric claims, prefer:

1. Direct W&B summary/history from the target W&B project.
2. PR result comments that include W&B run IDs.
3. Current `research/CURRENT_RESEARCH_STATE.md`, only after cross-checking.
4. Older `analysis/STATUS_*.md` files, only as historical context.

Why: metric values are easy to transpose across runs, PR comments, and status
files. W&B is closest to the training run; old statuses are useful for
continuity but should not override live evidence.

For operational claims, prefer:

1. Kubernetes process state on the active fleet context. Use the context named
   in the user's request or the latest status file; if unclear, inspect
   available contexts before reporting pod health.
2. Student/advisor raw `.claude` JSONL logs.
3. GitHub PR labels and timestamps.
4. Advisor state file claims.

Why: operational reality is determined by what the pods and labels are doing
now, not by what the advisor believes should be happening.

## Core questions to answer

Start by establishing scope and source freshness:

- What branch/program, GitHub repo, W&B project, and Kubernetes context are in
  scope?
- Which previous status files are the continuity anchors?
- Which sources were checked live versus reused from history?
- Did this pass modify anything, or was it read-only?

### Paper-facing science

- What is the best verified test result for each dataset?
- What exact metric contract applies to each dataset?
- Does the best test result beat the benchmark target? By how much?
- Which results are validation-only, truncated-eval, batch-limited, or otherwise
  not paper-ready?
- Which live or recent validation runs deserve final/full test harvest before
  shutdown?
- Which closed/merged PRs changed the frontier since the last status?
- Which negative results are decision-useful enough to stop whole families?

Why: this section tells the researcher whether the programme is producing
paper-useful improvements or merely accumulating plausible training stories.

### PR queue and assignment health

- How many PRs are open, review-ready, WIP, stale, draft, merged, or closed?
- Are all WIP PRs labeled with the advisor branch label and exactly one
  `student:<name>` label?
- Are any PRs invisible to students because they have `status:wip` but no
  `student:*` label?
- Are any PRs invisible to advisor accounting because they lack the branch
  label, for example `radford`?
- Do head branch prefixes imply an intended student that labels failed to
  encode?
- Are old PRs stale because the student is still training, because Claude is
  stuck in a monitor/sleep loop, or because the PR is invisible?

Why: the most expensive failure is an apparently full queue that no student can
actually see or advance.

### Fleet and raw logs

- Which Kubernetes context is actually hosting the fleet?
- How many advisor and student pods are Running/Ready?
- How many student pods have a real Python `train.py` process?
- Which pods are Running but not training?
- Do recent student logs end in "no open PR", "submitted", sleep/monitor text,
  setup failure, or active training/log tailing?
- Are W&B run names appearing for newly assigned PRs?
- Does the advisor log show recent review/assignment cycles, or repeated idle
  cycles from a false fleet model?

Why: process tables and raw logs distinguish "work is happening" from "the
automation is alive but no experiment is moving."

### Advisor state and steering

- What does `research/CURRENT_RESEARCH_STATE.md` claim?
- Which claims are contradicted by GitHub labels, W&B, or pod processes?
- Is the advisor over-weighting validation relative to test?
- Is the advisor spending capacity on low-EV breadth instead of final test
  harvest?
- Does the current queue match the user's active directive and remaining time?

Why: the advisor is part of the experiment system. Its state can amplify a good
frontier, but stale or validation-heavy state can spend the remaining budget on
the wrong questions.

## Useful commands

Adjust repo, branch, context, and project names if the user specifies different
ones.

```bash
# Status history
rg --files -g 'STATUS*.md' -g 'analysis/STATUS*.md' | sort
for f in $(rg --files -g 'STATUS*.md' -g 'analysis/STATUS*.md' | sort); do
  echo "### $f"
  rg -n '^#{1,3} ' "$f" | head -40
done

# PR queue
gh pr list --repo wandb/senpai --base radford --state open --limit 200 \
  --json number,title,state,createdAt,updatedAt,isDraft,headRefName,labels,url

# Detect WIP PRs invisible to students or advisor accounting
gh pr list --repo wandb/senpai --base radford --state open --limit 200 \
  --json number,headRefName,labels,createdAt,updatedAt \
  > /tmp/radford_open_prs.json
uv run python - <<'PY'
import json, datetime
prs = json.load(open('/tmp/radford_open_prs.json'))
now = datetime.datetime.now(datetime.timezone.utc)
for p in sorted(prs, key=lambda x: x["number"]):
    labels = sorted(l["name"] for l in p["labels"])
    if "status:wip" not in labels:
        continue
    missing_student = not any(x.startswith("student:") for x in labels)
    missing_branch = "radford" not in labels
    if missing_student or missing_branch:
        created = datetime.datetime.fromisoformat(p["createdAt"].replace("Z", "+00:00"))
        age = (now - created).total_seconds() / 3600
        intended = p["headRefName"].split("/")[0]
        print(f"#{p['number']} age={age:.1f}h intended={intended} labels={labels} head={p['headRefName']}")
PY

# Fleet overview
kubectl --context pai-2 get pods -l app=senpai -o wide
kubectl --context pai-2 get deploy -l app=senpai

# Real training process sweep
for podref in $(kubectl --context pai-2 get pods -l app=senpai,role=student -o name | sort); do
  pod=${podref#pod/}
  student=${pod#senpai-}; student=${student%-*}; student=${student%-*}
  out=$(kubectl --context pai-2 exec "$pod" -- sh -lc \
    "ps -eo pid,etime,comm,args | awk '\$3 ~ /python/ && /train.py/ {print; exit}'" 2>/dev/null || true)
  if [ -n "$out" ]; then
    printf "%-10s PYTRAIN %s\n" "$student" "$(printf "%s" "$out" | sed 's/^[[:space:]]*//; s/[[:space:]][[:space:]]*/ /g' | cut -c1-180)"
  else
    printf "%-10s NO_PYTRAIN\n" "$student"
  fi
done

# Advisor state file
gh api 'repos/wandb/senpai/contents/research/CURRENT_RESEARCH_STATE.md?ref=radford' \
  --jq .content | base64 --decode | sed -n '1,220p'

# Advisor raw log locations
kubectl --context pai-2 exec deploy/senpai-advisor -- sh -lc \
  "find /root/.claude -type f -name '*.jsonl' -printf '%T@ %p %s\n' | sort -nr | head -10"

# Student raw log locations
for podref in $(kubectl --context pai-2 get pods -l app=senpai,role=student -o name | sort); do
  pod=${podref#pod/}
  echo "### $pod"
  kubectl --context pai-2 exec "$pod" -- sh -lc \
    "find /root/.claude -type f -name '*.jsonl' -printf '%T@ %p %s\n' | sort -nr | head -3"
done

# Tail latest student Claude log messages for stuck/sleeping/active-training reads
for podref in $(kubectl --context pai-2 get pods -l app=senpai,role=student -o name | sort); do
  pod=${podref#pod/}
  echo "### $pod"
  kubectl --context pai-2 exec "$pod" -- sh -lc \
    "latest=\$(find /root/.claude -type f -name '*.jsonl' -printf '%T@ %p\n' | sort -nr | awk 'NR==1{sub(/^[^ ]+ /, \"\"); print}'); [ -n \"\$latest\" ] && tail -40 \"\$latest\""
done
```

## W&B metric scan pattern

Use the `wandb-primary` skill. Directly fetch known run IDs from PRs, then scan
recent runs only as needed. Keep dataset classification conservative; do not mix
TFP `surface_mse` with AirfRANS `surface_mse`.

Minimum fields to extract:

- DrivAerML: `test_primary/surface_rel_l2_pct`, full-eval preferred.
- TandemFoil Paper: `best_test_primary/field_mse`, `test_primary/field_mse`.
- AirfRANS: `full_test/surface_mse`, `full_test/volume_mse`,
  `best_full_test/surface_mse`, `best_full_test/volume_mse`.
- TandemFoil: `best_test_primary/surface_pressure_mae`,
  `test_primary/surface_pressure_mae`.

Always record:

- W&B run name and ID
- PR number if known
- run state
- metric key used
- selection caveat

## Status file template

Write to:

```text
analysis/STATUS_<YYYY-MM-DD-HHMM>_<branch>_<short_topic>.md
```

Use this wireframe:

```markdown
# STATUS <YYYY-MM-DD HH:MM TZ> - <branch> <short title>

Collected at `<UTC timestamp>`.

Sources checked:
- GitHub PRs on `<repo>`, base `<branch>`
- W&B project `<entity/project>`
- Kubernetes context `<context>`
- Advisor/student raw logs, if inspected
- Prior `analysis/STATUS_*.md` files

No PR labels, branches, pods, or W&B runs were modified during this pass.
<!-- If you did modify anything because the user explicitly asked, say exactly what. -->

## Executive read

<5-10 bullets. Lead with test frontier and any operational stall. Do not bury
label, pod, or advisor-state failures.>

## Test metric frontier

| Dataset | Contract | Best verified test | Target/reference | Gap/read | Source | Caveat |
| --- | --- | ---: | ---: | --- | --- | --- |
| DrivAerML | `test_primary/surface_rel_l2_pct` |  |  |  | W&B `<id>`, PR `#` | full-eval / truncated / final |
| TandemFoil Paper | `test_primary/field_mse` |  |  |  | W&B `<id>`, PR `#` | best-checkpoint / final |
| AirfRANS | `full_test/surface_mse`, `full_test/volume_mse` |  |  |  | W&B `<id>`, PR `#` | pair metric |
| TandemFoil | `test_primary/surface_pressure_mae` |  |  |  | W&B `<id>`, PR `#` | val-selected / final |

## What changed since last status

| Item | Evidence | Test impact | Action |
| --- | --- | --- | --- |

## PR queue and label audit

- Open PRs:
- `status:wip`:
- `status:review`:
- Draft:
- Older than 6h:
- Missing `student:*`:
- Missing branch label:

Include a table for malformed/stale PRs when nonzero:

| PR | Age | Intended student | Labels | Head branch | Risk | Fix |
| --- | ---: | --- | --- | --- | --- | --- |

## Fleet and raw-log health

| Student/pod group | Count | Evidence | Read |
| --- | ---: | --- | --- |
| Student pods Running/Ready |  | `kubectl` |  |
| Real `train.py` processes |  | process sweep |  |
| Running but not training |  | process sweep + `.claude` tail |  |
| Recent W&B runs |  | W&B scan |  |

Call out specific stuck pods or suspicious logs.

## Advisor state risks

- What `CURRENT_RESEARCH_STATE.md` says.
- What GitHub/W&B/pods contradict.
- Whether the advisor is emphasizing validation over test.
- Whether intervention is needed.

## Running jobs needing test harvest

| Priority | Run/PR | Current signal | Missing test metric | Why it matters | Action |
| --- | --- | --- | --- | --- | --- |

## Keep / kill / fix now

| Action | Targets | Reason | Owner/command |
| --- | --- | --- | --- |
| Keep running |  |  |  |
| Full-test harvest |  |  |  |
| Kill/close |  |  |  |
| Relabel/restart/fix control plane |  |  |  |

## Bottom line

<Short decision: are we scientifically improving, operationally healthy, both,
or neither? Name the next 1-3 moves.>
```

## Reporting guidance

- For final-hours checks, add a "shutdown harvest" section and rank only runs
  likely to improve paper-facing test.
- For stale-pod checks, put pod/process truth above PR commentary.
- For label failures, include the deterministic repair mapping but do not apply
  labels unless the user explicitly asks.
- For advisor drift, quote the state file only briefly and paraphrase the
  mismatch with evidence.
- For best metrics, include both "paper-safe" and "literal lowest observed"
  rows when selection protocol differs, and label the caveat clearly.
- Prefer small tables over long prose. The reader is deciding what to do next
  with expensive GPUs.

## Repeat this at the end

Before finishing any status check, explicitly answer these five questions:

1. **Best test results:** What is the best verified paper-facing test metric
   for each dataset, what benchmark or target is it compared to, and what is
   the gap?
2. **Useful work:** Which pods are actually training or harvesting full-test
   results right now, and which are merely Running?
3. **Queue health:** Are any WIP PRs invisible because labels, branch labels,
   draft state, or stale assignment state are wrong?
4. **High-upside actions:** Which 1-5 runs or PRs deserve the remaining GPU
   budget because they could improve a test result, not just validation?
5. **Decision:** Should the system keep running, harvest tests, relabel/restart
   stalled students, close low-value work, or restart with new instructions?

If the answer does not make the next operational decision obvious, tighten the
report before handing it back.
