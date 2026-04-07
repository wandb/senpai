"""Probe actual summary keys for the boundary-id-onehot runs."""
import os
import wandb
import numpy as np
import json

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
api = wandb.Api()
path = f"{entity}/{project}"

# The two thorfinn boundary-id-onehot runs found
run_ids = ["txl1svzm", "12uns5n9"]

BASELINE = {"p_in": 13.19, "p_oodc": 7.92, "p_tan": 30.05, "p_re": 6.45}

for run_id in run_ids:
    run = api.run(f"{path}/{run_id}")
    print(f"\n{'='*70}")
    print(f"Run: {run.id} | Name: {run.name}")
    print(f"Group: {run.group} | State: {run.state}")
    print(f"Created: {run.created_at}")
    print(f"\n--- All summary_metrics keys ---")

    sm = dict(run.summary_metrics)
    # Print all keys sorted
    for k, v in sorted(sm.items()):
        if isinstance(v, float) and not np.isnan(v):
            print(f"  {k}: {v:.6f}")
        elif isinstance(v, (int, str)):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    print(f"\n--- Config wandb_group / relevant fields ---")
    cfg = run.config
    for k in sorted(cfg.keys()):
        print(f"  {k}: {cfg[k]}")

    # Now look specifically for surface MAE metrics
    print(f"\n--- Surface MAE search ---")
    mae_keys = [k for k in sm.keys() if any(x in k.lower() for x in
                ['mae', 'p_in', 'p_oodc', 'p_tan', 'p_re', 'surface', 'ood', 'tandem', 'reynolds'])]
    if mae_keys:
        for k in sorted(mae_keys):
            print(f"  {k}: {sm[k]}")
    else:
        print("  No surface MAE keys found in summary_metrics")

    # Check history for surface mae metrics
    print(f"\n--- Last history step (all keys) ---")
    try:
        # Get last few steps of history
        hist = list(run.scan_history(keys=None, min_step=0))
        if hist:
            last = hist[-1]
            # Find surface-related keys
            surface_keys = [k for k in last.keys() if any(x in k.lower() for x in
                           ['mae', 'p_in', 'p_oodc', 'p_tan', 'p_re', 'surface', 'ood', 'tandem', 'reynolds', 'test'])]
            if surface_keys:
                print(f"  Surface-related keys in history step {last.get('_step', '?')}:")
                for k in sorted(surface_keys):
                    print(f"    {k}: {last[k]}")
            else:
                print(f"  No surface-related keys in last history step")
                print(f"  Available keys: {sorted([k for k in last.keys() if not k.startswith('_')])[:30]}")
    except Exception as e:
        print(f"  History scan error: {e}")
