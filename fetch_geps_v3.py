import os
import wandb
import numpy as np

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "senpai-cfd")
project = os.environ.get("WANDB_PROJECT", "tandemfoil")

run_ids = ["evz9po4m", "fdodi6m3", "ldf4sof7", "d7l91p0x", "etepxvjc"]
labels = {
    "evz9po4m": "GEPS TTA 10-step s73",
    "fdodi6m3": "GEPS TTA 20-step s42",
    "ldf4sof7": "GEPS TTA 20-step s73",
    "d7l91p0x": "DCT freq loss s42 (baseline)",
    "etepxvjc": "DCT freq loss s73 (baseline)",
}

print(f"Fetching runs from {entity}/{project}\n")

for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        summary = run.summary_metrics
        print(f"\nRun ID: {run_id}  ({labels[run_id]}) -- name={run.name}")
        print(f"  State: {run.state}")
        print(f"  created_at={run.created_at}, updated_at={run.updated_at}")
        print(f"  ALL summary keys ({len(summary)} total):")
        for k, v in sorted(summary.items()):
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"Run {run_id}: ERROR - {e}")

print("\nDone.")
