import os
import sys
import wandb

api = wandb.Api()

entity = "senpai-wandb"
project = "senpai"
run_ids = ["oy1fu86u", "shkekpq4"]

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        print(f"\n=== Run: {run_id} | Name: {run.name} | State: {run.state} ===")
        summary = run.summary_metrics
        # Print all keys that contain 'surface' or 'mae' or 'p_in' or 'p_oodc' or 'p_tan' or 'p_re'
        relevant = {k: v for k, v in summary.items() if any(
            term in k.lower() for term in ['surface', 'mae', 'p_in', 'p_oodc', 'p_tan', 'p_re', 'best']
        )}
        if relevant:
            print("Relevant metrics:")
            for k, v in sorted(relevant.items()):
                print(f"  {k}: {v}")
        else:
            print("No surface/MAE metrics found. All summary keys:")
            for k, v in sorted(summary.items()):
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
