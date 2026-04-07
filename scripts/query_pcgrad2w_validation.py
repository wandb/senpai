"""
Query W&B for PCGrad 2-way 8-seed validation runs.
Group: phase6/pcgrad-2way-validation
Pattern: askeladd/pcgrad2w-pct15-s{42..49}
"""
import os
import sys
import wandb
import pandas as pd
import numpy as np

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "wandb")
project = os.environ.get("WANDB_PROJECT", "senpai")
path = f"{entity}/{project}"

print(f"Querying W&B path: {path}")
print("=" * 70)

# Strategy 1: Query by group name
print("\n--- Strategy 1: Query by group 'phase6/pcgrad-2way-validation' ---")
try:
    runs_by_group = api.runs(
        path,
        filters={"group": "phase6/pcgrad-2way-validation"},
    )
    runs_by_group_list = runs_by_group[:50]
    print(f"Found {len(runs_by_group_list)} runs in group 'phase6/pcgrad-2way-validation'")
    for r in runs_by_group_list:
        print(f"  Run: {r.id} | name: {r.name} | state: {r.state} | group: {r.group}")
except Exception as e:
    print(f"  Error: {e}")

# Strategy 2: Query by run name pattern (seeds 42-49)
print("\n--- Strategy 2: Query by run name pattern 'pcgrad2w-pct15-s4' ---")
seed_runs = []
for seed in range(42, 50):
    run_name = f"pcgrad2w-pct15-s{seed}"
    try:
        runs = api.runs(
            path,
            filters={"display_name": run_name},
        )
        matched = runs[:5]
        if matched:
            for r in matched:
                print(f"  Found: {r.id} | name: {r.name} | state: {r.state} | group: {r.group}")
                seed_runs.append(r)
        else:
            print(f"  No run found for seed {seed} (name: {run_name})")
    except Exception as e:
        print(f"  Error for seed {seed}: {e}")

# Strategy 3: Query by username prefix in name
print("\n--- Strategy 3: Query runs with 'pcgrad2w' in name (broad search) ---")
try:
    runs_broad = api.runs(
        path,
        filters={"display_name": {"$regex": "pcgrad2w"}},
    )
    broad_list = runs_broad[:50]
    print(f"Found {len(broad_list)} runs matching 'pcgrad2w'")
    for r in broad_list:
        print(f"  Run: {r.id} | name: {r.name} | state: {r.state} | group: {r.group}")
        if r not in seed_runs:
            seed_runs.append(r)
except Exception as e:
    print(f"  Error: {e}")

# Merge all found runs (deduplicate by id)
all_runs = {r.id: r for r in seed_runs}
# Also add group results
try:
    for r in runs_by_group_list:
        all_runs[r.id] = r
except:
    pass

print(f"\n\nTotal unique runs found: {len(all_runs)}")
print("=" * 70)

# Surface MAE metrics of interest
mae_keys = [
    "surface_mae/p_in",
    "surface_mae/p_oodc",
    "surface_mae/p_tan",
    "surface_mae/p_re",
    "val/surface_mae/p_in",
    "val/surface_mae/p_oodc",
    "val/surface_mae/p_tan",
    "val/surface_mae/p_re",
    "test/surface_mae/p_in",
    "test/surface_mae/p_oodc",
    "test/surface_mae/p_tan",
    "test/surface_mae/p_re",
]

rows = []
for run_id, run in all_runs.items():
    summary = run.summary_metrics
    config = run.config

    row = {
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "group": run.group,
        "seed": config.get("seed", config.get("SEED", "?")),
    }

    # Try all possible metric key patterns
    found_any = False
    for k in mae_keys:
        val = summary.get(k)
        if val is not None:
            short_key = k.split("/")[-1] if "/" in k else k
            row[short_key] = val
            found_any = True

    # Also try without prefix
    for short in ["p_in", "p_oodc", "p_tan", "p_re"]:
        if short not in row:
            val = summary.get(short)
            if val is not None:
                row[short] = val
                found_any = True

    if not found_any:
        # Print all summary keys to debug
        print(f"\nRun {run.id} ({run.name}) - no MAE metrics found. Available summary keys (sample):")
        all_keys = list(summary.keys())
        mae_related = [k for k in all_keys if "mae" in k.lower() or "surface" in k.lower() or "pressure" in k.lower()]
        print(f"  MAE/surface/pressure keys: {mae_related[:20]}")
        print(f"  All keys (first 30): {all_keys[:30]}")

    rows.append(row)

if rows:
    df = pd.DataFrame(rows)
    print("\n--- Per-run surface MAE metrics ---")
    print(df.to_string(index=False))

    # Compute 8-seed means for numeric columns
    numeric_cols = [c for c in ["p_in", "p_oodc", "p_tan", "p_re"] if c in df.columns]
    if numeric_cols:
        print("\n--- 8-seed mean surface MAE ---")
        means = df[numeric_cols].mean()
        for col, val in means.items():
            print(f"  {col}: {val:.6f}")
        print(f"\n  Overall mean: {means.mean():.6f}")
    else:
        print("\nNo p_in/p_oodc/p_tan/p_re columns found — check metric key names above.")
else:
    print("No runs found to report metrics for.")
