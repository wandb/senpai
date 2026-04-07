"""
Find where val metrics are stored - check summary best_* and scan_history with different keys.
"""

import os
import wandb

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]

run = api.run(f"{entity}/senpai-v1/m6p2jskq")

# Try scanning with no key filter to find all unique keys
print("=== Scanning full history for unique non-train keys ===")
all_keys = set()
for row in run.scan_history(page_size=100):
    all_keys.update(row.keys())

val_keys = sorted([k for k in all_keys if not k.startswith("_") and not k.startswith("train")])
print("Non-train keys found in history:")
for k in val_keys:
    print(f"  {k}")

print("\n=== Looking at rows that have 'epoch' key ===")
count = 0
for row in run.scan_history(keys=["epoch"]):
    if "epoch" in row and row["epoch"] is not None:
        if count < 5:
            print(f"  step={row.get('_step')}, epoch={row.get('epoch')}, keys={list(row.keys())}")
        count += 1
print(f"  Total rows with 'epoch': {count}")

# Try with "best_epoch" or look for val metrics in the scan differently
print("\n=== Checking 'best_val_loss' key in scan ===")
count = 0
for row in run.scan_history(keys=["best_val_loss", "best_epoch"]):
    if row.get("best_val_loss") is not None:
        if count < 5:
            print(f"  {row}")
        count += 1
print(f"  Total rows with 'best_val_loss': {count}")
