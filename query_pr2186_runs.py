import os
import sys
import wandb
import numpy as np

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "wandb")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = ["3j7eqs2i", "zvkocnap"]

# Metrics we care about
metric_keys = [
    "val_in_dist/mae_surf_p",
    "val_tandem_transfer/mae_surf_p",
    "val_ood_cond/mae_surf_p",
    "val_ood_re/mae_surf_p",
]

results = {}

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
    except Exception as e:
        print(f"ERROR fetching run {run_id}: {e}")
        continue

    print(f"\n{'='*60}")
    print(f"Run: {run_id}")
    print(f"  Name: {run.name}")
    print(f"  State: {run.state}")
    print(f"  Tags: {run.tags}")
    print(f"  Created: {run.created_at}")

    # Print all summary metrics
    print(f"\n  --- All summary_metrics ---")
    sm = dict(run.summary_metrics)
    for k, v in sorted(sm.items()):
        if "mae" in k.lower() or "p_" in k.lower() or "surf" in k.lower():
            print(f"    {k}: {v}")

    print(f"\n  --- All summary keys (filtered for val_) ---")
    for k, v in sorted(sm.items()):
        if k.startswith("val_"):
            print(f"    {k}: {v}")

    print(f"\n  --- Target metrics ---")
    run_metrics = {}
    for key in metric_keys:
        val = sm.get(key, None)
        if val is None:
            # Try alternate key formats
            alt1 = key.replace("/", ".")
            val = sm.get(alt1, None)
        run_metrics[key] = val
        print(f"    {key}: {val}")

    results[run_id] = run_metrics

# Compute averages
print(f"\n{'='*60}")
print("2-SEED AVERAGES:")
for key in metric_keys:
    vals = [results[r].get(key) for r in run_ids if results.get(r, {}).get(key) is not None]
    if vals:
        avg = np.mean(vals)
        print(f"  {key}: {avg:.4f}  (vals: {[round(v,4) for v in vals]})")
    else:
        print(f"  {key}: N/A")

# Scale to ×1e4 for comparison with student-reported values
print(f"\n{'='*60}")
print("2-SEED AVERAGES (×1e4, for comparison with student-reported values):")
for key in metric_keys:
    vals = [results[r].get(key) for r in run_ids if results.get(r, {}).get(key) is not None]
    if vals:
        avg = np.mean(vals) * 1e4
        print(f"  {key}: {avg:.2f}")
    else:
        print(f"  {key}: N/A")
