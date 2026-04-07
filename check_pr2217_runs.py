import wandb
import os

api = wandb.Api()
entity = "senpai-wandb"
project = "senpai"

run_ids = ["t2eoumup", "o44zx3wy"]

for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"\nRun ID: {run_id}")
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")
        print(f"  Created: {run.created_at}")

        # Check summary metrics for surface MAE values
        summary = run.summary_metrics
        print(f"  Summary keys (surface-related): {[k for k in summary.keys() if any(x in k for x in ['p_in', 'p_oodc', 'p_tan', 'p_re', 'mae', 'MAE', 'surface'])]}")

        for key in ['p_in', 'p_oodc', 'p_tan', 'p_re']:
            val = summary.get(key)
            if val is None:
                # try with mae prefix
                val = summary.get(f"mae_{key}")
            print(f"  {key}: {val}")

        # Print all summary keys to find the right metric names
        print(f"  All summary keys: {list(summary.keys())[:50]}")

    except Exception as e:
        print(f"\nRun ID: {run_id} — ERROR: {e}")
