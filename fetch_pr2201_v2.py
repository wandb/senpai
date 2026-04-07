import os
import sys
import wandb
import numpy as np

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
api = wandb.Api()

run_ids = ["r5s4apc5", "uwlgfj78"]
labels = {"r5s4apc5": "seed42", "uwlgfj78": "seed73"}

# The actual keys use the "best_best_*" prefix
metric_map = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

results = {}
for run_id in run_ids:
    run = api.run(f"{entity}/{project}/{run_id}")
    sm = run.summary_metrics
    row = {}
    for short, key in metric_map.items():
        v = sm.get(key)
        row[short] = v
        print(f"  [{run_id} {labels[run_id]}] {short} ({key}): {v}")
    results[run_id] = {"label": labels[run_id], "vals": row}

# Also check if val_tandem_transfer key exists at all
print("\n--- Checking available val_tandem keys ---")
for run_id in run_ids:
    run = api.run(f"{entity}/{project}/{run_id}")
    sm = run.summary_metrics
    tan_keys = [k for k in sm.keys() if "tandem" in k.lower()]
    print(f"[{run_id}] tandem keys: {tan_keys}")

print("\n\n=== FINAL TABLE ===")
print(f"{'Run':<12} {'p_in':<10} {'p_tan':<10} {'p_oodc':<10} {'p_re':<10}")

per_metric = {k: [] for k in metric_map}
for run_id, data in results.items():
    v = data["vals"]
    row = [v.get("p_in"), v.get("p_tan"), v.get("p_oodc"), v.get("p_re")]
    for k, val in v.items():
        if val is not None:
            per_metric[k].append(val)
    strs = [f"{x:.4f}" if x is not None else "N/A" for x in row]
    print(f"{data['label']:<12} {strs[0]:<10} {strs[1]:<10} {strs[2]:<10} {strs[3]:<10}")

avgs = [np.mean(per_metric[k]) if per_metric[k] else None for k in ["p_in","p_tan","p_oodc","p_re"]]
avg_strs = [f"{x:.4f}" if x is not None else "N/A" for x in avgs]
print(f"{'2-seed avg':<12} {avg_strs[0]:<10} {avg_strs[1]:<10} {avg_strs[2]:<10} {avg_strs[3]:<10}")

baseline = [13.205, 28.502, 7.816, 6.453]
b_strs = [f"{x:.4f}" for x in baseline]
print(f"{'Baseline':<12} {b_strs[0]:<10} {b_strs[1]:<10} {b_strs[2]:<10} {b_strs[3]:<10}")

# Beat baseline?
print("\n=== BEAT BASELINE? ===")
keys = ["p_in","p_tan","p_oodc","p_re"]
for k, avg, base in zip(keys, avgs, baseline):
    if avg is not None:
        beat = "YES" if avg < base else "NO"
        diff = avg - base
        print(f"  {k}: avg={avg:.4f}, baseline={base:.4f}, diff={diff:+.4f} => {beat}")
    else:
        print(f"  {k}: N/A")
