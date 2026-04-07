import os
import sys
import wandb
import numpy as np

api = wandb.Api()

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["ecxe1ti3", "29xvq9bz"]

surface_mae_keys = ["p_in", "p_tan", "p_oodc", "p_re"]

# Also try prefixed versions
prefixes = ["test/", "val/", "surface_mae/", "mae/", ""]

results = {}

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        print(f"\n=== Run: {run_id} ===")
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")
        print(f"  Tags: {run.tags}")

        # Print all summary metrics keys to understand naming
        summary = run.summary_metrics
        print(f"  All summary keys: {sorted(summary.keys())}")

        # Try to find surface MAE metrics
        found_metrics = {}
        for key in sorted(summary.keys()):
            key_lower = key.lower()
            for metric in surface_mae_keys:
                if metric in key_lower:
                    found_metrics[key] = summary.get(key)

        print(f"  Matching metrics: {found_metrics}")

        # Also scan history for these metrics
        history_keys = []
        # Get the last few rows of history to find metric names
        hist_sample = run.history(samples=5)
        if not hist_sample.empty:
            print(f"  History columns: {sorted(hist_sample.columns.tolist())}")
            # Find surface MAE columns
            for col in hist_sample.columns:
                col_lower = col.lower()
                for metric in surface_mae_keys:
                    if metric in col_lower:
                        history_keys.append(col)
            if history_keys:
                print(f"  History surface MAE columns: {history_keys}")
                # Get the best (min) values
                full_hist = run.history(samples=10000, keys=history_keys)
                for col in history_keys:
                    col_data = full_hist[col].dropna()
                    if len(col_data) > 0:
                        print(f"    {col}: min={col_data.min():.4f}, final={col_data.iloc[-1]:.4f}, steps={len(col_data)}")

        results[run_id] = found_metrics

    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        import traceback
        traceback.print_exc()

print("\n\n=== SUMMARY ===")
for run_id, metrics in results.items():
    print(f"\nRun {run_id}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
