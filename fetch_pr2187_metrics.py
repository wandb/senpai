import os
import wandb
import numpy as np

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "senpai-cfd")
project = os.environ.get("WANDB_PROJECT", "tandemfoil")

run_ids = ["ey0n53eo", "ejjcn6kq"]
metric_keys = ["p_in", "p_oodc", "p_tan", "p_re", "val_loss"]

print(f"Fetching runs from {entity}/{project}\n")
print("=" * 70)

for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"\nRun ID: {run_id}")
        print(f"  Name:   {run.name}")
        print(f"  State:  {run.state}")
        print(f"  Group:  {run.group}")

        # Get summary metrics
        summary = run.summary_metrics
        print(f"  Summary metrics:")
        for k in metric_keys:
            val = summary.get(k, "N/A")
            print(f"    {k}: {val}")

        # Try to get epoch count
        epoch_val = summary.get("epoch", summary.get("_step", "N/A"))
        print(f"    epoch/step: {epoch_val}")

        # Scan history for final values of key metrics
        print(f"  Scanning history for final metric values...")
        hist_keys = metric_keys + ["epoch"]
        rows = list(run.scan_history(keys=hist_keys))
        if rows:
            last = rows[-1]
            print(f"  Final row (last step):")
            for k in hist_keys:
                print(f"    {k}: {last.get(k, 'N/A')}")
            print(f"  Total history rows: {len(rows)}")
        else:
            print("  No history rows found.")

    except Exception as e:
        print(f"  ERROR fetching run {run_id}: {e}")

print("\n" + "=" * 70)
print("Done.")
