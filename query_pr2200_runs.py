"""Query W&B metrics for PR #2200 runs: bpgzy063 and 9x2f1pis"""
import os
import sys
import wandb
import numpy as np

sys.path.insert(0, "/workspace/senpai/skills/wandb-primary/scripts")

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

RUN_IDS = ["bpgzy063", "9x2f1pis"]

SURFACE_MAE_KEYS = [
    "val/surface_mae_p_in",
    "val/surface_mae_p_oodc",
    "val/surface_mae_p_tan",
    "val/surface_mae_p_re",
]

# Also try alternate key formats
ALT_KEYS = [
    "surface_mae_p_in",
    "surface_mae_p_oodc",
    "surface_mae_p_tan",
    "surface_mae_p_re",
    "val_surface_mae_p_in",
    "val_surface_mae_p_oodc",
    "val_surface_mae_p_tan",
    "val_surface_mae_p_re",
]

for run_id in RUN_IDS:
    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"Name: {run.name}")
        print(f"State: {run.state}")

        # Print all summary metrics to see what's available
        summary = dict(run.summary_metrics)
        print(f"\nAll summary keys (filtered for 'mae' or 'surface'):")
        for k, v in sorted(summary.items()):
            if 'mae' in k.lower() or 'surface' in k.lower():
                print(f"  {k}: {v}")

        print(f"\nAll summary keys (filtered for 'p_in' or 'p_oodc' or 'p_tan' or 'p_re'):")
        for k, v in sorted(summary.items()):
            if any(x in k.lower() for x in ['p_in', 'p_oodc', 'p_tan', 'p_re']):
                print(f"  {k}: {v}")

        # Get epoch count from history
        try:
            epoch_data = list(run.scan_history(keys=["epoch"], page_size=1000))
            if epoch_data:
                max_epoch = max(r.get("epoch", 0) for r in epoch_data if r.get("epoch") is not None)
                print(f"\nEpochs completed: {max_epoch + 1} (0-indexed max: {max_epoch})")
            else:
                # Try _step
                step_data = list(run.scan_history(keys=["_step"], page_size=10))
                print(f"\nNo epoch key found. Sample steps: {[r.get('_step') for r in step_data[:5]]}")
                total_steps = summary.get("_step", "unknown")
                print(f"Total steps: {total_steps}")
        except Exception as e:
            print(f"Error getting epoch data: {e}")

        # Try to get best surface MAE from history
        print(f"\nChecking surface MAE from summary:")
        found_any = False
        for key in SURFACE_MAE_KEYS + ALT_KEYS:
            val = summary.get(key)
            if val is not None:
                print(f"  {key}: {val:.6f}")
                found_any = True

        if not found_any:
            print("  No surface MAE keys found in summary. Printing all summary keys:")
            for k, v in sorted(summary.items()):
                if not k.startswith('_') and not k.startswith('system'):
                    print(f"  {k}: {v}")

    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        import traceback
        traceback.print_exc()

print("\nDone.")
