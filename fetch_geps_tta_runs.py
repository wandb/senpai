import os
import sys
import wandb
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = {
    "evz9po4m": "GEPS TTA 10-step s73",
    "fdodi6m3": "GEPS TTA 20-step s42",
    "ldf4sof7": "GEPS TTA 20-step s73",
    "d7l91p0x": "DCT freq loss s42 (baseline)",
    "etepxvjc": "DCT freq loss s73 (baseline)",
}

surface_mae_keys = [
    "val/surface_mae_p_in",
    "val/surface_mae_p_oodc",
    "val/surface_mae_p_tan",
    "val/surface_mae_p_re",
]

# Also try alternate key naming conventions
alt_keys = [
    "surface_mae_p_in", "surface_mae_p_oodc", "surface_mae_p_tan", "surface_mae_p_re",
    "val/p_in", "val/p_oodc", "val/p_tan", "val/p_re",
    "p_in", "p_oodc", "p_tan", "p_re",
]

results = {}

for run_id, label in run_ids.items():
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        summary = run.summary_metrics

        # Print all summary keys to discover naming
        print(f"\n{'='*60}")
        print(f"Run: {run_id} | {label}")
        print(f"  State: {run.state}")
        print(f"  Name: {run.name}")

        # Duration
        created_at = run.created_at
        updated_at = run.updated_at
        print(f"  Created: {created_at}")
        print(f"  Updated: {updated_at}")

        # Epochs
        epoch_val = summary.get("epoch", None)
        epochs_val = summary.get("epochs", None)
        _step = summary.get("_step", None)
        print(f"  epoch in summary: {epoch_val}")
        print(f"  epochs in summary: {epochs_val}")
        print(f"  _step in summary: {_step}")

        # Show all keys containing 'surface' or 'mae' or 'p_in' etc
        print(f"  Relevant summary keys:")
        for k, v in sorted(summary.items()):
            if any(x in k.lower() for x in ["surface", "mae", "p_in", "p_oodc", "p_tan", "p_re", "epoch", "duration", "runtime", "time"]):
                print(f"    {k}: {v}")

        # Try to get best surface MAE metrics from history
        print(f"  Scanning history for surface MAE keys...")

        # Find which keys exist
        sample_history = list(run.scan_history(keys=surface_mae_keys + alt_keys[:4], max_step=10))
        if sample_history:
            available_keys = set()
            for row in sample_history:
                available_keys.update(row.keys())
            print(f"  History keys found: {available_keys}")
        else:
            print(f"  No history rows found with those keys, checking summary for all keys...")
            # Print a subset of summary keys
            all_keys = list(summary.keys())
            print(f"  All summary keys: {all_keys[:50]}")

    except Exception as e:
        print(f"\nRun {run_id}: ERROR - {e}")

print("\n\nDone scanning. Now extracting metrics...")
