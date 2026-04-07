"""Query W&B runs for thorfinn's boundary-id one-hot feature experiment (PR #2118)."""
import os
import sys
import wandb
import pandas as pd
import numpy as np

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")
from wandb_helpers import runs_to_dataframe

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
api = wandb.Api()
path = f"{entity}/{project}"

print(f"Querying W&B project: {path}")
print("=" * 70)

# Key metrics we want
METRIC_KEYS = [
    "test/p_in", "test/p_oodc", "test/p_tan", "test/p_re",
    "val/loss",
    "surface_mae/p_in", "surface_mae/p_oodc", "surface_mae/p_tan", "surface_mae/p_re",
    "p_in", "p_oodc", "p_tan", "p_re",
]

BASELINE = {"p_in": 13.19, "p_oodc": 7.92, "p_tan": 30.05, "p_re": 6.45}

# Search strategies
search_terms = [
    "boundary-onehot",
    "boundary_onehot",
    "boundary-id",
    "boundary_id",
    "phase6/boundary",
    "boundary-one-hot",
    "onehot",
    "one-hot",
    "thorfinn",
]

found_run_ids = set()
all_results = []

def extract_surface_mae(run):
    """Extract surface MAE metrics from a run using multiple possible key patterns."""
    s = run.summary_metrics
    result = {
        "run_id": run.id,
        "run_name": run.name,
        "group": run.config.get("wandb_group", run.group or ""),
        "state": run.state,
        "created_at": run.created_at,
    }

    # Try multiple key patterns for each metric
    for metric_short, candidates in [
        ("p_in",   ["test/p_in", "surface_mae/p_in", "p_in", "mae/p_in"]),
        ("p_oodc", ["test/p_oodc", "surface_mae/p_oodc", "p_oodc", "mae/p_oodc"]),
        ("p_tan",  ["test/p_tan", "surface_mae/p_tan", "p_tan", "mae/p_tan"]),
        ("p_re",   ["test/p_re", "surface_mae/p_re", "p_re", "mae/p_re"]),
        ("val_loss", ["val/loss", "val_loss", "validation/loss"]),
    ]:
        val = None
        for k in candidates:
            v = s.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                val = v
                break
        result[metric_short] = val

    return result


# Strategy 1: Search by group name patterns
print("\n[1] Searching by wandb_group / group filters...")
for term in ["boundary-onehot", "boundary_onehot", "boundary-id", "boundary_id", "thorfinn"]:
    try:
        filters = {"$or": [
            {"config.wandb_group": {"$regex": term}},
            {"group": {"$regex": term}},
        ]}
        runs = api.runs(path, filters=filters)[:50]
        for r in runs:
            if r.id not in found_run_ids:
                found_run_ids.add(r.id)
                result = extract_surface_mae(r)
                all_results.append(result)
                print(f"  Found: {r.id} | {r.name} | group={result['group']} | state={r.state}")
    except Exception as e:
        print(f"  Filter '{term}' error: {e}")

# Strategy 2: Search by run name patterns
print("\n[2] Searching by run name patterns...")
for term in ["boundary", "onehot", "one.hot", "thorfinn"]:
    try:
        filters = {"display_name": {"$regex": term}}
        runs = api.runs(path, filters=filters)[:50]
        for r in runs:
            if r.id not in found_run_ids:
                found_run_ids.add(r.id)
                result = extract_surface_mae(r)
                all_results.append(result)
                print(f"  Found: {r.id} | {r.name} | group={result['group']} | state={r.state}")
    except Exception as e:
        print(f"  Name filter '{term}' error: {e}")

# Strategy 3: Search by tags
print("\n[3] Searching by tags...")
for term in ["thorfinn", "boundary", "pr-2118", "2118"]:
    try:
        filters = {"tags": {"$in": [term]}}
        runs = api.runs(path, filters=filters)[:50]
        for r in runs:
            if r.id not in found_run_ids:
                found_run_ids.add(r.id)
                result = extract_surface_mae(r)
                all_results.append(result)
                print(f"  Found: {r.id} | {r.name} | group={result['group']} | state={r.state}")
    except Exception as e:
        print(f"  Tag filter '{term}' error: {e}")

# Strategy 4: Scan recent runs for thorfinn-related metadata
print("\n[4] Scanning recent 200 runs for thorfinn/boundary metadata...")
try:
    recent_runs = api.runs(path, order="-created_at")[:200]
    for r in recent_runs:
        if r.id in found_run_ids:
            continue
        # Check if any config or name hints at thorfinn or boundary
        cfg = r.config
        name_lower = (r.name or "").lower()
        group_lower = (r.group or "").lower()
        cfg_group = (cfg.get("wandb_group", "") or "").lower()
        cfg_user = (cfg.get("student", "") or cfg.get("user", "") or "").lower()

        keywords = ["boundary", "onehot", "one_hot", "thorfinn", "pr2118", "2118"]
        if any(kw in name_lower or kw in group_lower or kw in cfg_group or kw in cfg_user
               for kw in keywords):
            found_run_ids.add(r.id)
            result = extract_surface_mae(r)
            all_results.append(result)
            print(f"  Found: {r.id} | {r.name} | group={result['group']} | state={r.state}")
except Exception as e:
    print(f"  Recent scan error: {e}")


# Final summary
print("\n" + "=" * 70)
print(f"TOTAL RUNS FOUND: {len(all_results)}")
print("=" * 70)

if not all_results:
    print("No runs found matching thorfinn/boundary-id/one-hot criteria.")
    print("\nTrying broader scan of all runs to show recent group names...")
    try:
        recent = api.runs(path, order="-created_at")[:50]
        groups_seen = set()
        for r in recent:
            g = r.group or r.config.get("wandb_group", "")
            if g and g not in groups_seen:
                groups_seen.add(g)
                print(f"  group={g} | run={r.name} | id={r.id}")
    except Exception as e:
        print(f"  Error: {e}")
else:
    df = pd.DataFrame(all_results)
    df = df.sort_values("created_at", ascending=False)

    print("\nDetailed Results:")
    for _, row in df.iterrows():
        print(f"\n  Run ID:    {row['run_id']}")
        print(f"  Name:      {row['run_name']}")
        print(f"  Group:     {row['group']}")
        print(f"  State:     {row['state']}")
        print(f"  Created:   {row['created_at']}")
        print(f"  Metrics:")
        print(f"    p_in:    {row['p_in']}  (baseline < {BASELINE['p_in']})")
        print(f"    p_oodc:  {row['p_oodc']}  (baseline < {BASELINE['p_oodc']})")
        print(f"    p_tan:   {row['p_tan']}  (baseline < {BASELINE['p_tan']})")
        print(f"    p_re:    {row['p_re']}  (baseline < {BASELINE['p_re']})")
        print(f"    val/loss: {row['val_loss']}")

        # Beat baseline checks
        beats = []
        for k in ["p_in", "p_oodc", "p_tan", "p_re"]:
            v = row[k]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                if v < BASELINE[k]:
                    beats.append(f"{k}({v:.4f}<{BASELINE[k]})")
        if beats:
            print(f"  BEATS BASELINE: {', '.join(beats)}")
        else:
            print(f"  Does not beat baseline (or metrics missing)")

    print("\n--- Summary Table ---")
    cols = ["run_id", "run_name", "group", "state", "p_in", "p_oodc", "p_tan", "p_re", "val_loss"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
