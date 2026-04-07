"""Verify W&B metrics for PR #2110 runs."""
import os
import sys
sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

import wandb
import pandas as pd

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

api = wandb.Api()

run_ids = {
    "7n8v8nlh": "surframp-40ep-s42",
    "ui8t0t7g": "surframp-40ep-s73",
    "mqu6c2vd": "surframp-80ep-s42",
    "vkrnehh2": "surframp-80ep-s73",
}

# Surface MAE metric keys to check
SURFACE_KEYS = [
    "test/p_in_mae", "test/p_oodc_mae", "test/p_tan_mae", "test/p_re_mae",
    "p_in_mae", "p_oodc_mae", "p_tan_mae", "p_re_mae",
    "val/p_in_mae", "val/p_oodc_mae", "val/p_tan_mae", "val/p_re_mae",
]

rows = []
for run_id, run_name in run_ids.items():
    try:
        run = api.run(f"{path}/{run_id}")
        summary = run.summary_metrics

        # Print all summary keys for inspection
        print(f"\n=== Run: {run_name} ({run_id}) ===")
        print(f"  State: {run.state}")
        print(f"  W&B name: {run.name}")
        print(f"  Group: {run.group}")

        # Find all relevant metric keys
        all_keys = list(summary.keys())
        surface_keys_found = [k for k in all_keys if any(term in k.lower() for term in ["mae", "p_in", "p_oodc", "p_tan", "p_re"])]
        print(f"  Surface-related keys: {surface_keys_found}")

        # Extract best surface MAE values
        row = {"run_id": run_id, "run_name": run_name, "state": run.state, "group": run.group}
        for key in surface_keys_found:
            row[key] = summary.get(key)
        rows.append(row)

        # Print full summary for debugging
        print(f"  Full summary keys: {sorted(all_keys)[:50]}")

    except Exception as e:
        print(f"ERROR fetching {run_id}: {e}")
        rows.append({"run_id": run_id, "run_name": run_name, "error": str(e)})

print("\n\n=== SUMMARY TABLE ===")
df = pd.DataFrame(rows)
print(df.to_string())
