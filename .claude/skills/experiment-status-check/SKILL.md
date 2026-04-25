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

Create a timestamped `analysis/STATUS_*.md` report that answers:

1. What are the best paper-facing **test** results, and how do they compare to
   benchmark targets?
2. Is the advisor/student system actually doing useful work, or is it stalled,
   mislabeled, sleeping, or optimizing the wrong thing?

The core distinction is "is the fleet alive?" versus "is useful science
happening?" Keep those separate throughout the report.

## Why this skill exists

This is an intervention tool, not a newsletter. It should help decide whether
to keep training, harvest results, repair operations, or restart with new
instructions. It guards against:

- **Validation drift:** agents find plausible validation wins and accidentally
  report them like paper wins. The paper needs test metrics.
- **Fleet illusion:** pods can be Running while student sessions are idle,
  sleeping, mis-assigned, or done.
- **Assignment invisibility:** a PR without the right `student:<name>` and
  branch labels can look assigned to the advisor while being invisible to the
  student pod.
- **Stale advisor state:** `CURRENT_RESEARCH_STATE.md` can lag behind GitHub,
  W&B, and pod reality.
- **Experiment sprawl:** many PRs can be active while only a few have a
  credible path to improving paper-facing test results.

## Non-negotiables

- Put test metrics before validation metrics. Validation is a steering signal,
  not the paper result. This prevents the report from overstating progress.
- Always pair a test metric with its benchmark target/reference and a gap/read:
  "beats", "misses by X", "internal benchmark only", or "no external scalar".
- Include source and caveat columns for metrics: W&B run ID, PR number,
  full-eval vs batch-limited, best-checkpoint vs final epoch, truncated vs full.
- Do not trust advisor state files as source of truth until cross-checked
  against GitHub labels, W&B, and pod/process state.
- Do not treat a Running pod as active training. Check for real `train.py`
  processes and recent W&B runs.
- Surface broken labels, pods, polling, or advisor state in the executive read.
- Do not report "SOTA" from validation unless the benchmark contract is
  validation-based. For this programme, most paper claims are test-facing.

## Evidence order and why

- **Metrics:** W&B summary/history first, then PR comments with run IDs, then
  advisor state, then old status files. W&B is closest to the run; old status
  files are continuity, not truth.
- **Operations:** Kubernetes process state first, then raw `.claude` JSONL logs,
  then GitHub labels/timestamps, then advisor state. Pod/process truth shows
  what is actually happening now.

## Questions to answer

| Area | Ask | Why |
| --- | --- | --- |
| Scope | Which branch, repo, W&B project, k8s context, and prior statuses are in scope? What was checked live? | Prevents stale or cross-branch evidence from driving decisions. |
| Science | What is the best verified test result per dataset, the exact metric contract, the benchmark target, and the gap? Which results are val-only or truncated? | Separates paper-ready wins from internal steering signals. |
| Harvest | Which live or recent runs deserve full-test harvest before shutdown? Which negative results stop whole families? | Spends remaining GPU on paper-facing upside. |
| PR queue | How many PRs are WIP/review/draft/stale? Are WIP PRs labeled with branch + exactly one `student:*`? | Finds invisible work before pods sit idle. |
| Fleet | Which pods are Ready, which have real `train.py`, and which are Running but not training? | Avoids mistaking infrastructure health for useful experiments. |
| Raw logs | Do student/advisor logs show active training, no-PR loops, submitted/done states, setup failures, or sleep/monitor loops? | Explains why work is or is not moving. |
| Advisor | What does `CURRENT_RESEARCH_STATE.md` claim, and what do GitHub/W&B/pods contradict? | Stops stale or validation-heavy steering from consuming the budget. |

## Command patterns

Adjust repo, branch, context, and project names to the request. These are
patterns, not mandatory copy-paste blocks.

```bash
# Status history and headings
rg --files -g 'STATUS*.md' -g 'analysis/STATUS*.md' | sort
rg -n '^#{1,3} ' analysis/STATUS*.md

# PR queue
gh pr list --repo wandb/senpai --base radford --state open --limit 200 \
  --json number,title,state,createdAt,updatedAt,isDraft,headRefName,labels,url

# Detect WIP PRs missing student/branch labels
gh pr list --repo wandb/senpai --base radford --state open --limit 200 \
  --json number,headRefName,labels,createdAt > /tmp/open_prs.json
uv run python - <<'PY'
import json, datetime
prs = json.load(open('/tmp/open_prs.json'))
now = datetime.datetime.now(datetime.timezone.utc)
for p in sorted(prs, key=lambda x: x["number"]):
    labels = sorted(l["name"] for l in p["labels"])
    bad = "status:wip" in labels and (
        "radford" not in labels or not any(x.startswith("student:") for x in labels)
    )
    if bad:
        age = (now - datetime.datetime.fromisoformat(p["createdAt"].replace("Z","+00:00"))).total_seconds()/3600
        intended = p["headRefName"].split("/", 1)[0]
        print(f"#{p['number']} age={age:.1f}h intended={intended} labels={labels} head={p['headRefName']}")
PY

# Fleet overview and real training process sweep
kubectl --context pai-2 get pods -l app=senpai -o wide
for podref in $(kubectl --context pai-2 get pods -l app=senpai,role=student -o name | sort); do
  pod=${podref#pod/}
  student=${pod#senpai-}; student=${student%-*}; student=${student%-*}
  out=$(kubectl --context pai-2 exec "$pod" -- sh -lc "ps -eo pid,etime,comm,args | awk '\$3 ~ /python/ && /train.py/ {print; exit}'" 2>/dev/null || true)
  if [ -n "$out" ]; then
    printf "%-10s PYTRAIN %s\n" "$student" "$(printf "%s" "$out" | sed 's/^[[:space:]]*//; s/[[:space:]][[:space:]]*/ /g' | cut -c1-180)"
  else
    printf "%-10s NO_PYTRAIN\n" "$student"
  fi
done

# Advisor state and raw log locations
gh api 'repos/wandb/senpai/contents/research/CURRENT_RESEARCH_STATE.md?ref=radford' \
  --jq .content | base64 --decode | sed -n '1,220p'
kubectl --context pai-2 exec deploy/senpai-advisor -- sh -lc \
  "find /root/.claude -type f -name '*.jsonl' -printf '%T@ %p %s\n' | sort -nr | head -10"

# Student raw log locations and latest tails
for podref in $(kubectl --context pai-2 get pods -l app=senpai,role=student -o name | sort); do
  pod=${podref#pod/}
  echo "### $pod"
  kubectl --context pai-2 exec "$pod" -- sh -lc \
    "latest=\$(find /root/.claude -type f -name '*.jsonl' -printf '%T@ %p\n' | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,\"\"); print}'); [ -n \"\$latest\" ] && tail -40 \"\$latest\""
done
```

## W&B metric scan pattern

Use the `wandb-primary` skill. Directly fetch known run IDs from PRs, then scan
recent runs only as needed. Keep dataset classification conservative; do not mix
TFP `surface_mse` with AirfRANS `surface_mse`.

Minimum metric keys:

- DrivAerML: `test_primary/surface_rel_l2_pct`
- TandemFoil Paper: `best_test_primary/field_mse`, `test_primary/field_mse`.
- AirfRANS: `full_test/surface_mse`, `full_test/volume_mse`,
  `best_full_test/surface_mse`, `best_full_test/volume_mse`.
- TandemFoil: `best_test_primary/surface_pressure_mae`,
  `test_primary/surface_pressure_mae`.

Always record W&B run name/ID, PR if known, run state, metric key, and caveat.

## Status file template

Write to:

```text
analysis/STATUS_<YYYY-MM-DD-HHMM>_<branch>_<short_topic>.md
```

Use this compact wireframe:

```markdown
# STATUS <YYYY-MM-DD HH:MM TZ> - <branch> <short title>

Collected at `<UTC timestamp>`.

Sources checked: GitHub `<repo>/<branch>`, W&B `<entity/project>`, k8s
`<context>`, raw logs `<yes/no>`, prior statuses `<paths>`.
Changes made during this pass: `<none, or exact labels/pods/branches touched>`.

## Executive read

<5-10 bullets. Lead with test frontier and operational stalls.>

## Test metric frontier

| Dataset | Contract | Best verified test | Target/reference | Gap/read | Source | Caveat |
| --- | --- | ---: | ---: | --- | --- | --- |
| DrivAerML | `test_primary/surface_rel_l2_pct` |  |  |  | W&B `<id>`, PR `#` | full-eval / truncated / final |
| TandemFoil Paper | `test_primary/field_mse` |  |  |  | W&B `<id>`, PR `#` | best-checkpoint / final |
| AirfRANS | `full_test/surface_mse`, `full_test/volume_mse` |  |  |  | W&B `<id>`, PR `#` | pair metric |
| TandemFoil | `test_primary/surface_pressure_mae` |  |  |  | W&B `<id>`, PR `#` | val-selected / final |

## PR queue and label audit

Counts: open `<n>`, WIP `<n>`, review `<n>`, draft `<n>`, older than 6h `<n>`,
missing `student:*` `<n>`, missing branch label `<n>`.

| PR | Age | Student/read | Labels | Risk | Fix |
| --- | ---: | --- | --- | --- | --- |

## Fleet and raw-log health

| Signal | Count/examples | Evidence | Read |
| --- | ---: | --- | --- |
| Pods Running/Ready |  | `kubectl` |  |
| Real `train.py` |  | process sweep |  |
| Running but not training |  | process sweep + `.claude` tail |  |
| Recent W&B runs |  | W&B scan |  |

## Advisor state risks

State what `CURRENT_RESEARCH_STATE.md` says, what GitHub/W&B/pods contradict,
whether validation is over-weighted, and what intervention is needed.

## Harvest / keep / kill

| Priority | Run/PR | Current signal | Missing test metric | Why it matters | Action |
| --- | --- | --- | --- | --- | --- |

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
