"""
Inspect history rows that contain validation metrics to understand epoch and key naming.
"""

import os
import wandb

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]

run = api.run(f"{entity}/senpai-v1/m6p2jskq")

# The val metrics — look for rows containing mae_surf_p keys
PROBE_KEY = "val_in_dist/mae_surf_p"

print(f"=== Scanning for rows with '{PROBE_KEY}' ===")
count = 0
for row in run.scan_history(keys=["epoch", PROBE_KEY, "val_ood_cond/mae_surf_p", "val_ood_re/mae_surf_p", "val_tandem_transfer/mae_surf_p"]):
    if PROBE_KEY in row and row[PROBE_KEY] is not None:
        if count < 5 or count % 20 == 0:
            print(f"  epoch={row.get('epoch')}, step={row.get('_step')}, {PROBE_KEY}={row.get(PROBE_KEY):.4f}")
        count += 1

print(f"\n  Total rows with validation metrics: {count}")
