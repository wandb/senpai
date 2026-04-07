import os
import sys
import wandb
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["tmqq1xlo", "84aff7cq", "23y1pfj5", "yw2djp6d", "afis6090", "c80t1a69", "xcmpfkqs", "75d4hhzm"]

metrics_keys = [
    "best_best_val_in_dist/mae_surf_p",
    "best_best_val_ood_cond/mae_surf_p",
    "best_best_val_tandem_transfer/mae_surf_p",
    "best_best_val_ood_re/mae_surf_p",
]

short_keys = ["p_in", "p_oodc", "p_tan", "p_re"]

rows = []

for rid in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{rid}")
        sm = run.summary_metrics
        seed = run.config.get("seed", run.config.get("random_seed", "N/A"))
        group = run.group
        row = {
            "run_id": rid,
            "name": run.name,
            "seed": seed,
            "group": group,
            "state": run.state,
        }
        for mk, sk in zip(metrics_keys, short_keys):
            val = sm.get(mk, None)
            row[sk] = val
        rows.append(row)
    except Exception as e:
        rows.append({"run_id": rid, "name": "ERROR", "seed": "N/A", "group": "N/A", "state": "ERROR",
                     "p_in": None, "p_oodc": None, "p_tan": None, "p_re": None, "error": str(e)})

# Print per-run table
header = f"{'run_id':<12} {'name':<30} {'seed':<8} {'state':<12} {'p_in':<20} {'p_oodc':<20} {'p_tan':<20} {'p_re':<20}"
print(header)
print("-" * len(header))
for r in rows:
    p_in = f"{r['p_in']}" if r['p_in'] is not None else "MISSING"
    p_oodc = f"{r['p_oodc']}" if r['p_oodc'] is not None else "MISSING"
    p_tan = f"{r['p_tan']}" if r['p_tan'] is not None else "MISSING"
    p_re = f"{r['p_re']}" if r['p_re'] is not None else "MISSING"
    print(f"{r['run_id']:<12} {str(r['name']):<30} {str(r['seed']):<8} {str(r['state']):<12} {p_in:<20} {p_oodc:<20} {p_tan:<20} {p_re:<20}")

print()

# Compute stats
for sk in short_keys:
    vals = [r[sk] for r in rows if r[sk] is not None and not (isinstance(r[sk], float) and np.isnan(r[sk]))]
    nan_count = sum(1 for r in rows if r[sk] is None or (isinstance(r[sk], float) and np.isnan(r[sk])))
    if vals:
        arr = np.array(vals, dtype=float)
        print(f"{sk}: mean={arr.mean()}, std={arr.std()}, n={len(arr)}, missing/NaN={nan_count}")
    else:
        print(f"{sk}: ALL MISSING (n=0)")

print()

# Best and worst seed for p_tan
valid_ptan = [(r['run_id'], r['name'], r['seed'], r['p_tan']) for r in rows if r['p_tan'] is not None and not (isinstance(r['p_tan'], float) and np.isnan(r['p_tan']))]
if valid_ptan:
    best = min(valid_ptan, key=lambda x: x[3])
    worst = max(valid_ptan, key=lambda x: x[3])
    print(f"Best  p_tan: run_id={best[0]}, name={best[1]}, seed={best[2]}, p_tan={best[3]}")
    print(f"Worst p_tan: run_id={worst[0]}, name={worst[1]}, seed={worst[2]}, p_tan={worst[3]}")
