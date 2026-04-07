import wandb
import os

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "wandb")
project = os.environ.get("WANDB_PROJECT", "senpai")

# Probe the first run to see all available summary keys and last-step history keys
run = api.run(f"{entity}/{project}/fctgmn1d")

print("=== SUMMARY KEYS (all) ===")
for k, v in sorted(run.summary_metrics.items()):
    print(f"  {k!r}: {v}")

print()
print("=== CONFIG keys (surface-mae related) ===")
for k, v in sorted(run.config.items()):
    if any(x in k.lower() for x in ["mae", "srf", "foil", "loss", "seed"]):
        print(f"  {k!r}: {v}")

print()
print("=== LAST HISTORY ROW (sample keys) ===")
# Get last row of history
rows = list(run.scan_history(keys=None))
if rows:
    last = rows[-1]
    for k, v in sorted(last.items()):
        if any(x in k.lower() for x in ["mae", "loss", "p_in", "p_tan", "p_oodc", "p_re", "surface", "val"]):
            print(f"  {k!r}: {v}")
