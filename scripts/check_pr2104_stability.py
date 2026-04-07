"""
Check training stability for the aft_srf runs by inspecting available loss keys.
"""
import os
import sys
import wandb
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

# Focus on the aft_srf runs for stability check
RUN_IDS = {
    "aft_srf_s42":  "cp4ralol",
    "aft_srf_s43":  "w0iccehn",
    "baseline_s42": "rgr1lvbt",
}

for label, run_id in RUN_IDS.items():
    run = api.run(f"{path}/{run_id}")
    print(f"\n=== {label} ({run_id}) — state={run.state} ===")

    # Show summary keys to find loss metric names
    summary_keys = list(run.summary_metrics.keys())
    loss_keys = [k for k in summary_keys if "loss" in k.lower() or "mae" in k.lower()]
    print(f"  Loss-related summary keys: {loss_keys[:20]}")

    # Sample the history to see what keys exist
    sample_rows = list(run.scan_history(max_step=5))
    if sample_rows:
        all_keys = list(sample_rows[0].keys())
        print(f"  History keys (first row sample): {all_keys[:30]}")
    else:
        print("  No history rows found")

    # Check a known train loss key
    for candidate in ["train/loss", "train/total_loss", "total_loss", "loss"]:
        rows = list(run.scan_history(keys=[candidate], max_step=10))
        if rows:
            print(f"  Found key '{candidate}' in history, first few values: {[r.get(candidate) for r in rows[:5]]}")
            # Now get full history for NaN/spike check
            all_vals = [r.get(candidate) for r in run.scan_history(keys=[candidate])]
            arr = np.array([v for v in all_vals if v is not None], dtype=float)
            nan_count = int(np.sum(np.isnan(arr)))
            if len(arr) > 0:
                finite = arr[np.isfinite(arr)]
                med = np.median(finite) if len(finite) > 0 else 0
                spikes = int(np.sum(finite > 10 * med)) if med > 0 else 0
                print(f"    Steps: {len(arr)}, NaNs: {nan_count}, Spikes(>10x med): {spikes}, min: {arr.min():.4f}, final: {arr[-1]:.4f}")
            break
