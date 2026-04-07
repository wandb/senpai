import os
import wandb

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
project = os.environ.get("WANDB_PROJECT", "senpai-v1")
run_ids = ["oy1fu86u", "shkekpq4"]

print(f"Querying entity={entity}, project={project}")

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        print(f"\n=== Run: {run_id} | Name: {run.name} | State: {run.state} ===")
        summary = run.summary_metrics
        # Print all keys
        for k, v in sorted(summary.items()):
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nError fetching {run_id} at {path}: {e}")
