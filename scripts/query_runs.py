import os
import sys
import wandb
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["zapen0x3", "3vuz3adi", "g1uhcorj", "ea551p6b", "fdf1vsi3", "al6opl9g", "jg15oow3", "vzm3s42y"]

metrics = ["surface_mae/p_in", "surface_mae/p_oodc", "surface_mae/p_tan", "surface_mae/p_re"]
short = ["p_in", "p_oodc", "p_tan", "p_re"]

results = {}

for rid in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{rid}")
        sm = run.summary_metrics
        row = {}
        for m, s in zip(metrics, short):
            val = sm.get(m)
            if val is None:
                # Try alternate key patterns
                for key in sm.keys():
                    if s in key and "mae" in key.lower():
                        val = sm[key]
                        break
            row[s] = val
        results[rid] = row
        print(f"{rid}: {row}")
    except Exception as e:
        print(f"{rid}: ERROR - {e}")

# Print all keys for first run to debug if needed
if results:
    first_id = run_ids[0]
    run = api.run(f"{entity}/{project}/{first_id}")
    sm = run.summary_metrics
    surface_keys = [k for k in sm.keys() if "mae" in k.lower() or "surface" in k.lower() or any(x in k for x in ["p_in", "p_oodc", "p_tan", "p_re"])]
    print(f"\nSurface-related keys in {first_id}: {surface_keys}")

# Compute stats
print("\n--- Per-run table ---")
print(f"{'Run':<12} {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8}")
for rid, row in results.items():
    vals = [row.get(s) for s in short]
    formatted = [f"{v:.4f}" if v is not None else "  NaN  " for v in vals]
    print(f"{rid:<12} {formatted[0]:>8} {formatted[1]:>8} {formatted[2]:>8} {formatted[3]:>8}")

print("\n--- Stats ---")
for s in short:
    vals = [results[rid][s] for rid in run_ids if results.get(rid, {}).get(s) is not None]
    if vals:
        arr = np.array(vals)
        print(f"{s}: mean={arr.mean():.4f}, std={arr.std():.4f}, n={len(arr)}")
    else:
        print(f"{s}: no data")
