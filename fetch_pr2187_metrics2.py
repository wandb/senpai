import os
import wandb
import numpy as np

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "senpai-cfd")
project = os.environ.get("WANDB_PROJECT", "tandemfoil")

run_ids = ["ey0n53eo", "ejjcn6kq"]

print(f"Fetching runs from {entity}/{project}\n")

for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"\nRun ID: {run_id}  ({run.name})")
        print(f"  State: {run.state}   Group: {run.group}")

        # Dump ALL summary keys
        summary = run.summary_metrics
        print(f"  ALL summary keys ({len(summary)} keys):")
        for k, v in sorted(summary.items()):
            print(f"    {k}: {v}")

    except Exception as e:
        print(f"  ERROR fetching run {run_id}: {e}")

print("\nDone.")
