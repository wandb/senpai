import wandb
import pandas as pd
import numpy as np
import sys

api = wandb.Api()
project = "wandb-applied-ai-team/senpai-v1"

run_ids = [
    "zosxwjmm",  # PR 2127 seed 42
    "twilqf1x",  # PR 2127 seed 73
    "np3anmah",  # PR 2128 control s42
    "fcri663y",  # PR 2128 control s73
    "7bnlke5o",  # PR 2128 FiLM s42
    "yh5ixlx8",  # PR 2128 FiLM s73
]

labels = {
    "zosxwjmm": "PR#2127 aft-srf-knn / seed42",
    "twilqf1x": "PR#2127 aft-srf-knn / seed73",
    "np3anmah": "PR#2128 control s42",
    "fcri663y": "PR#2128 control s73",
    "7bnlke5o": "PR#2128 FiLM s42",
    "yh5ixlx8": "PR#2128 FiLM s73",
}

surface_keys = ["p_in", "p_oodc", "p_tan", "p_re"]

# Also check EMA variants
ema_variants = [
    "{k}",
    "ema_{k}",
    "best_{k}",
    "val_{k}",
    "val/ema_{k}",
    "val/best_{k}",
    "surface_mae/{k}",
    "ema/surface_mae/{k}",
]

results = []

for run_id in run_ids:
    run = api.run(f"{project}/{run_id}")
    summary = run.summary_metrics

    row = {"run_id": run_id, "label": labels[run_id], "name": run.name, "state": run.state}

    # Print all summary keys for inspection
    print(f"\n--- Run {run_id} ({labels[run_id]}) ---")
    print(f"State: {run.state}")

    # Find relevant keys
    all_keys = list(summary.keys())
    surface_related = [k for k in all_keys if any(sk in k.lower() for sk in ["p_in", "p_oodc", "p_tan", "p_re", "surface", "mae"])]
    print(f"Surface-related keys: {surface_related}")

    # Try to extract best values for each metric
    for metric in surface_keys:
        val = None
        # Try various naming conventions, prefer EMA/best
        for pattern in [f"ema_{metric}", f"best_{metric}", f"val/ema_{metric}", f"val/{metric}", metric]:
            if pattern in summary:
                val = summary[pattern]
                print(f"  {metric}: {val:.6f} (from key '{pattern}')")
                break
        if val is None:
            print(f"  {metric}: NOT FOUND in summary")
        row[metric] = val

    results.append(row)

print("\n\n=== SURFACE MAE SUMMARY TABLE ===\n")
df = pd.DataFrame(results)
display_cols = ["label", "state"] + surface_keys
print(df[display_cols].to_string(index=False))

# Compute mean across seeds for each PR/condition
print("\n\n=== MEAN ACROSS SEEDS ===\n")
groups = {
    "PR#2127 aft-srf-knn (mean s42+s73)": ["zosxwjmm", "twilqf1x"],
    "PR#2128 control (mean s42+s73)": ["np3anmah", "fcri663y"],
    "PR#2128 FiLM (mean s42+s73)": ["7bnlke5o", "yh5ixlx8"],
}
for group_label, ids in groups.items():
    subset = df[df["run_id"].isin(ids)]
    means = subset[surface_keys].mean()
    print(f"{group_label}:")
    for k in surface_keys:
        print(f"  {k}: {means[k]:.6f}")
    print()
