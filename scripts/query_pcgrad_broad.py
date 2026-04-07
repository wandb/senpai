"""
Broad search for PCGrad runs and related validation work.
Also fetches original 4 runs to understand metric naming.
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

# First: Check the original 4 runs to understand metric names
print("\n--- Original 4 PCGrad runs (phase6/pcgrad-3way) ---")
original_ids = ["p9oupnt2", "6sp2uazt", "kov6n0rs", "l308y9lx"]
for rid in original_ids:
    try:
        run = api.run(f"{path}/{rid}")
        summary = run.summary_metrics
        print(f"\nRun {rid} ({run.name}) | state={run.state} | group={run.group}")
        # Find all MAE related keys
        all_keys = list(summary.keys())
        mae_keys = [k for k in all_keys if "mae" in k.lower() or "surface" in k.lower() or "pressure" in k.lower()]
        print(f"  MAE/surface/pressure keys: {mae_keys}")
        # Print values
        for k in mae_keys[:15]:
            print(f"    {k}: {summary.get(k)}")
    except Exception as e:
        print(f"  Error fetching {rid}: {e}")

# Search for any pcgrad runs created recently
print("\n\n--- Broad search: any run with 'pcgrad' in name ---")
try:
    runs = api.runs(
        path,
        filters={"display_name": {"$regex": "pcgrad"}},
        order="-created_at",
    )
    results = runs[:30]
    print(f"Found {len(results)} runs")
    for r in results:
        print(f"  {r.id} | {r.name} | state={r.state} | group={r.group} | created={r.created_at}")
except Exception as e:
    print(f"  Error: {e}")

# Search for any run with 'pcgrad' in tags
print("\n--- Broad search: any run with 'pcgrad' in tags ---")
try:
    runs = api.runs(
        path,
        filters={"tags": {"$in": ["pcgrad", "pcgrad2w", "pcgrad-2way"]}},
        order="-created_at",
    )
    results = runs[:30]
    print(f"Found {len(results)} runs")
    for r in results:
        print(f"  {r.id} | {r.name} | state={r.state} | group={r.group}")
except Exception as e:
    print(f"  Error: {e}")

# Search for 'askeladd' runs created in last few days
print("\n--- Broad search: recent runs by askeladd ---")
try:
    runs = api.runs(
        path,
        filters={"username": "askeladd"},
        order="-created_at",
    )
    results = runs[:20]
    print(f"Found {len(results)} runs by askeladd")
    for r in results:
        print(f"  {r.id} | {r.name} | state={r.state} | group={r.group} | created={r.created_at}")
except Exception as e:
    print(f"  Error: {e}")

# Also search by group pattern
print("\n--- Broad search: groups containing 'pcgrad' ---")
try:
    runs = api.runs(
        path,
        filters={"group": {"$regex": "pcgrad"}},
        order="-created_at",
    )
    results = runs[:30]
    print(f"Found {len(results)} runs in pcgrad groups")
    for r in results:
        print(f"  {r.id} | {r.name} | state={r.state} | group={r.group} | created={r.created_at}")
except Exception as e:
    print(f"  Error: {e}")

# Check phase6 group runs broadly
print("\n--- Broad search: groups containing 'phase6' ---")
try:
    runs = api.runs(
        path,
        filters={"group": {"$regex": "phase6"}},
        order="-created_at",
    )
    results = runs[:50]
    print(f"Found {len(results)} runs in phase6 groups")
    for r in results:
        print(f"  {r.id} | {r.name} | state={r.state} | group={r.group} | created={r.created_at}")
except Exception as e:
    print(f"  Error: {e}")
