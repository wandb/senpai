"""
Inspect the available keys and summary of a run to understand metric naming.
"""

import os
import wandb
import json

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]

# Just look at one run first
run = api.run(f"{entity}/senpai-v1/m6p2jskq")

print("=== RUN SUMMARY ===")
summary = dict(run.summary)
# Filter to show only numeric/non-system entries
for k, v in sorted(summary.items()):
    if not k.startswith("_") and not k.startswith("gradients/") and not k.startswith("parameters/"):
        print(f"  {k}: {v}")

print("\n=== FIRST 3 HISTORY ROWS (raw) ===")
count = 0
for row in run.scan_history():
    if count >= 3:
        break
    # show all keys in this row
    keys = [k for k in row.keys() if not k.startswith("gradients/") and not k.startswith("parameters/")]
    print(f"\n  Row {count}: {json.dumps({k: row[k] for k in keys}, default=str, indent=4)}")
    count += 1

print("\n=== HISTORY KEYS (sample from first 5 rows) ===")
all_keys = set()
count = 0
for row in run.scan_history():
    if count >= 5:
        break
    all_keys.update(row.keys())
    count += 1

for k in sorted(all_keys):
    if not k.startswith("gradients/") and not k.startswith("parameters/"):
        print(f"  {k}")
