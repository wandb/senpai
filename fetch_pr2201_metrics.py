import os
import sys
import wandb
import numpy as np

sys.path.insert(0, "/workspace/senpai/skills/wandb-primary/scripts")

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
api = wandb.Api()

run_ids = ["r5s4apc5", "uwlgfj78"]
labels = {"r5s4apc5": "seed42", "uwlgfj78": "seed73"}

# Metric keys to look for
metric_keys = [
    "mae_surf_p/val_in_dist",
    "mae_surf_p/val_tandem_transfer",
    "mae_surf_p/val_ood_cond",
    "mae_surf_p/val_ood_re",
]

results = {}
for run_id in run_ids:
    run = api.run(f"{entity}/{project}/{run_id}")
    print(f"\n=== Run {run_id} ({labels[run_id]}) ===")
    print(f"  Name: {run.name}")
    print(f"  State: {run.state}")
    
    # Check summary metrics
    sm = run.summary_metrics
    print(f"  Summary keys (sample): {list(sm.keys())[:30]}")
    
    row = {}
    for k in metric_keys:
        v = sm.get(k)
        print(f"  {k}: {v}")
        row[k] = v
    
    # Also try scanning history for best values
    print("  Scanning history for best values...")
    history_data = {k: [] for k in metric_keys}
    for step in run.scan_history(keys=metric_keys):
        for k in metric_keys:
            if k in step and step[k] is not None:
                history_data[k].append(step[k])
    
    print("  Best from history:")
    best_row = {}
    for k in metric_keys:
        vals = history_data[k]
        if vals:
            best_val = min(vals)
            print(f"    {k}: best={best_val:.4f} over {len(vals)} steps")
            best_row[k] = best_val
        else:
            print(f"    {k}: no data")
            best_row[k] = None
    
    results[run_id] = {"label": labels[run_id], "summary": row, "best": best_row}

print("\n\n=== SUMMARY TABLE ===")
metric_short = {
    "mae_surf_p/val_in_dist": "p_in",
    "mae_surf_p/val_tandem_transfer": "p_tan",
    "mae_surf_p/val_ood_cond": "p_oodc",
    "mae_surf_p/val_ood_re": "p_re",
}

print(f"{'Run':<12} {'p_in':<10} {'p_tan':<10} {'p_oodc':<10} {'p_re':<10}")
best_values = {k: [] for k in metric_keys}
for run_id, data in results.items():
    row_vals = []
    for k in metric_keys:
        v = data["best"].get(k)
        if v is None:
            v = data["summary"].get(k)
        row_vals.append(v)
        if v is not None:
            best_values[k].append(v)
    vals_str = [f"{v:.4f}" if v is not None else "N/A" for v in row_vals]
    print(f"{data['label']:<12} {vals_str[0]:<10} {vals_str[1]:<10} {vals_str[2]:<10} {vals_str[3]:<10}")

# 2-seed average
avg_vals = []
for k in metric_keys:
    if best_values[k]:
        avg = np.mean(best_values[k])
        avg_vals.append(avg)
    else:
        avg_vals.append(None)

avg_strs = [f"{v:.4f}" if v is not None else "N/A" for v in avg_vals]
print(f"{'2-seed avg':<12} {avg_strs[0]:<10} {avg_strs[1]:<10} {avg_strs[2]:<10} {avg_strs[3]:<10}")

# Baseline for comparison
baseline = {"p_in": 13.205, "p_tan": 28.502, "p_oodc": 7.816, "p_re": 6.453}
print(f"\n{'Baseline':<12} {baseline['p_in']:<10} {baseline['p_tan']:<10} {baseline['p_oodc']:<10} {baseline['p_re']:<10}")
