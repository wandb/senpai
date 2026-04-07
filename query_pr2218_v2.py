import os
import wandb

api = wandb.Api()

# Try different entity/project combinations
combos = [
    ("senpai-wandb", "senpai"),
    ("senpai-wandb", "cfd_tandemfoil"),
    ("senpai", "senpai"),
    ("senpai", "cfd_tandemfoil"),
]

run_ids = ["oy1fu86u", "shkekpq4"]

for run_id in run_ids:
    print(f"\n--- Searching for run: {run_id} ---")
    found = False
    for entity, project in combos:
        path = f"{entity}/{project}/{run_id}"
        try:
            run = api.run(path)
            print(f"FOUND at {entity}/{project}: name={run.name}, state={run.state}")
            summary = run.summary_metrics
            for k, v in sorted(summary.items()):
                print(f"  {k}: {v}")
            found = True
            break
        except Exception as e:
            print(f"  Not at {entity}/{project}: {e}")
    if not found:
        print(f"  Run {run_id} not found in any combo")
