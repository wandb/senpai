"""
Fetch per-epoch metrics for TTA comparison runs.
Extracts mae_surf_p_in, mae_surf_p_tan, mae_surf_p_oodc, mae_surf_p_re
at epoch 130 (or closest available) and at the final stopping epoch.
"""

import os
import sys
import wandb
import numpy as np
import pandas as pd

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]

RUN_IDS = {
    "baseline_s42":  "m6p2jskq",
    "baseline_s43":  "u24b9oh9",
    "tta_d0.5_s42":  "6biat14l",
    "tta_d0.5_s43":  "j604eyv0",
    "tta_d1.0_s42":  "j7n5d3wf",
    "tta_d1.0_s43":  "z4kv5ghh",
}

METRICS = [
    "mae_surf_p_in",
    "mae_surf_p_tan",
    "mae_surf_p_oodc",
    "mae_surf_p_re",
]

TARGET_EPOCH = 130

def find_run(run_id):
    """Try to find the run across projects."""
    # Try direct path first with known project
    for project in ["senpai-v1", "phase6/tta-aoa", "phase6", "senpai"]:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            print(f"  Found run {run_id} in project: {project} | name: {run.name}")
            return run
        except Exception:
            pass
    # Try searching all projects
    try:
        runs = api.runs(f"{entity}/senpai-v1", filters={"name": run_id})
        for r in runs[:5]:
            if r.id == run_id:
                print(f"  Found run {run_id} via search in senpai-v1 | name: {r.name}")
                return r
    except Exception:
        pass
    print(f"  WARNING: Could not find run {run_id}")
    return None

def get_metrics_at_epoch(run, target_epoch, metrics):
    """
    Scan history and find values at the target epoch (or closest available).
    Returns (epoch_found, values_dict, final_epoch, final_values_dict)
    """
    # Collect epoch + metrics
    keys = ["epoch"] + metrics

    rows = []
    for row in run.scan_history(keys=keys):
        epoch = row.get("epoch")
        if epoch is None:
            # some runs log by step; try _step
            epoch = row.get("_step")
        if epoch is None:
            continue
        vals = {m: row.get(m) for m in metrics}
        if any(v is not None for v in vals.values()):
            rows.append({"epoch": epoch, **vals})

    if not rows:
        print(f"  WARNING: no history rows with metrics found for {run.id}")
        return None, {}, None, {}

    df = pd.DataFrame(rows).sort_values("epoch")

    # Find closest to target epoch
    epochs = df["epoch"].values
    closest_idx = np.argmin(np.abs(epochs - target_epoch))
    closest_epoch = epochs[closest_idx]
    closest_row = df.iloc[closest_idx]
    at_target = {m: closest_row.get(m) for m in metrics}

    # Find best (minimum) for each metric across all epochs
    # (use the row with lowest overall surface p mae, or just the last row for "final")
    last_row = df.iloc[-1]
    final_epoch = last_row["epoch"]
    final_vals = {}
    for m in metrics:
        col = df[m].dropna()
        if len(col) > 0:
            final_vals[m] = col.min()  # best value achieved
        else:
            final_vals[m] = None

    return closest_epoch, at_target, final_epoch, final_vals

results = {}

for label, run_id in RUN_IDS.items():
    print(f"\nFetching {label} ({run_id})...")
    run = find_run(run_id)
    if run is None:
        results[label] = {
            "run_id": run_id,
            "epoch_130_epoch": None,
            **{f"ep130_{m}": None for m in METRICS},
            "final_epoch": None,
            **{f"best_{m}": None for m in METRICS},
        }
        continue

    closest_epoch, at_target, final_epoch, final_vals = get_metrics_at_epoch(run, TARGET_EPOCH, METRICS)

    row = {
        "run_id": run_id,
        "run_name": run.name,
        "epoch_130_epoch": closest_epoch,
    }
    for m in METRICS:
        row[f"ep130_{m}"] = at_target.get(m)
    row["final_epoch"] = final_epoch
    for m in METRICS:
        row[f"best_{m}"] = final_vals.get(m)

    results[label] = row

    print(f"  Closest epoch to 130: {closest_epoch}, final epoch: {final_epoch}")
    for m in METRICS:
        print(f"    {m}: @ep130={at_target.get(m)}, best={final_vals.get(m)}")

print("\n\n" + "="*80)
print("EPOCH 130 COMPARISON TABLE")
print("="*80)

short_names = {
    "baseline_s42": "Baseline s42",
    "baseline_s43": "Baseline s43",
    "tta_d0.5_s42": "TTA d=0.5 s42",
    "tta_d0.5_s43": "TTA d=0.5 s43",
    "tta_d1.0_s42": "TTA d=1.0 s42",
    "tta_d1.0_s43": "TTA d=1.0 s43",
}

header = f"{'Run':<16} {'Ep':>5} {'p_in':>10} {'p_tan':>10} {'p_oodc':>10} {'p_re':>10}"
print(header)
print("-" * len(header))

for label, row in results.items():
    ep = row.get("epoch_130_epoch")
    ep_str = f"{ep:.0f}" if ep is not None else "N/A"
    vals = [row.get(f"ep130_{m}") for m in METRICS]
    val_strs = [f"{v:.5f}" if v is not None else "  N/A  " for v in vals]
    print(f"{short_names[label]:<16} {ep_str:>5} {val_strs[0]:>10} {val_strs[1]:>10} {val_strs[2]:>10} {val_strs[3]:>10}")

print("\n\n" + "="*80)
print("BEST (FINAL) VALUES TABLE")
print("="*80)

header2 = f"{'Run':<16} {'Ep':>5} {'p_in':>10} {'p_tan':>10} {'p_oodc':>10} {'p_re':>10}"
print(header2)
print("-" * len(header2))

for label, row in results.items():
    ep = row.get("final_epoch")
    ep_str = f"{ep:.0f}" if ep is not None else "N/A"
    vals = [row.get(f"best_{m}") for m in METRICS]
    val_strs = [f"{v:.5f}" if v is not None else "  N/A  " for v in vals]
    print(f"{short_names[label]:<16} {ep_str:>5} {val_strs[0]:>10} {val_strs[1]:>10} {val_strs[2]:>10} {val_strs[3]:>10}")

# Average across seeds
print("\n\n" + "="*80)
print("SEED-AVERAGED AT EPOCH 130")
print("="*80)

groups = {
    "Baseline":   ["baseline_s42",  "baseline_s43"],
    "TTA d=0.5":  ["tta_d0.5_s42", "tta_d0.5_s43"],
    "TTA d=1.0":  ["tta_d1.0_s42", "tta_d1.0_s43"],
}

header3 = f"{'Group':<14} {'p_in':>10} {'p_tan':>10} {'p_oodc':>10} {'p_re':>10}"
print(header3)
print("-" * len(header3))

for gname, labels in groups.items():
    avgs = []
    for m in METRICS:
        vals = [results[l].get(f"ep130_{m}") for l in labels if results[l].get(f"ep130_{m}") is not None]
        avgs.append(np.mean(vals) if vals else None)
    val_strs = [f"{v:.5f}" if v is not None else "  N/A  " for v in avgs]
    print(f"{gname:<14} {val_strs[0]:>10} {val_strs[1]:>10} {val_strs[2]:>10} {val_strs[3]:>10}")

print("\n\n" + "="*80)
print("SEED-AVERAGED BEST VALUES")
print("="*80)

print(header3)
print("-" * len(header3))

for gname, labels in groups.items():
    avgs = []
    for m in METRICS:
        vals = [results[l].get(f"best_{m}") for l in labels if results[l].get(f"best_{m}") is not None]
        avgs.append(np.mean(vals) if vals else None)
    val_strs = [f"{v:.5f}" if v is not None else "  N/A  " for v in avgs]
    print(f"{gname:<14} {val_strs[0]:>10} {val_strs[1]:>10} {val_strs[2]:>10} {val_strs[3]:>10}")
