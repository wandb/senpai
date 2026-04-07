"""
Fetch per-epoch metrics for TTA comparison runs using correct key names.
Metrics are logged per epoch at val steps, no 'epoch' field — using _step proxy.
The baseline best_epoch=157 from summary, so each val log row = 1 epoch.
"""

import os
import wandb
import numpy as np
import pandas as pd

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]

RUN_IDS = {
    "baseline_s42":  "m6p2jskq",
    "baseline_s43":  "u24b9oh9",
    "tta_d0.5_s42":  "6biat14l",
    "tta_d0.5_s43":  "j604eyv0",
    "tta_d1.0_s42":  "j7n5d3wf",
    "tta_d1.0_s43":  "z4kv5ghh",
}

# Actual metric keys in W&B
# Mapping: friendly name -> W&B key
METRIC_MAP = {
    "p_in":   "val_in_dist/mae_surf_p",
    "p_tan":  "val_tandem_transfer/mae_surf_p",
    "p_oodc": "val_ood_cond/mae_surf_p",
    "p_re":   "val_ood_re/mae_surf_p",
}

WB_KEYS = list(METRIC_MAP.values())
TARGET_STEP = 130  # epoch 130 corresponds to step ~130 in val logs

def get_metrics_for_run(run_id, label):
    run = api.run(f"{entity}/senpai-v1/{run_id}")
    print(f"\n{label} ({run_id}) | name: {run.name}")
    print(f"  Summary best_epoch: {run.summary_metrics.get('best_epoch')}, total_epochs: {run.summary_metrics.get('total_epochs')}")

    # Collect all val rows
    rows = []
    for row in run.scan_history(keys=["global_step"] + WB_KEYS, page_size=500):
        # Check if any val metric is present
        if any(row.get(k) is not None for k in WB_KEYS):
            rows.append(row)

    if not rows:
        print(f"  ERROR: no val history rows found")
        return None

    # Sort by _step proxy (which wandb inserts)
    # Each row should correspond to one epoch
    print(f"  Total val rows: {len(rows)}")
    print(f"  First row keys: {list(rows[0].keys())}")

    # Build a numbered list — treat list index as epoch number (1-indexed)
    # because global_step increments per batch, not per epoch
    # We use sequential index as epoch proxy
    df_rows = []
    for i, row in enumerate(rows):
        epoch_num = i + 1  # 1-indexed epoch
        r = {"epoch": epoch_num}
        for fname, wbkey in METRIC_MAP.items():
            r[fname] = row.get(wbkey)
        df_rows.append(r)

    df = pd.DataFrame(df_rows)

    # Find row at or closest to target step
    epochs = df["epoch"].values
    closest_idx = np.argmin(np.abs(epochs - TARGET_STEP))
    closest_epoch = epochs[closest_idx]
    at_target = {m: df.iloc[closest_idx][m] for m in METRIC_MAP.keys()}

    # Best (minimum) for each metric
    final_epoch = epochs[-1]
    best_vals = {m: df[m].dropna().min() for m in METRIC_MAP.keys()}

    print(f"  Closest epoch to {TARGET_STEP}: {closest_epoch} (of {final_epoch} total)")
    for m in METRIC_MAP.keys():
        print(f"    {m}: @ep{closest_epoch}={at_target[m]}, best={best_vals[m]}")

    return {
        "label": label,
        "run_id": run_id,
        "run_name": run.name,
        "epoch_at_target": closest_epoch,
        "final_epoch": final_epoch,
        **{f"ep130_{m}": at_target[m] for m in METRIC_MAP.keys()},
        **{f"best_{m}": best_vals[m] for m in METRIC_MAP.keys()},
    }

results = {}
for label, run_id in RUN_IDS.items():
    results[label] = get_metrics_for_run(run_id, label)

print("\n\n" + "="*80)
print("EPOCH 130 COMPARISON TABLE")
print("="*80)

short = {
    "baseline_s42": "Baseline s42",
    "baseline_s43": "Baseline s43",
    "tta_d0.5_s42": "TTA d=0.5 s42",
    "tta_d0.5_s43": "TTA d=0.5 s43",
    "tta_d1.0_s42": "TTA d=1.0 s42",
    "tta_d1.0_s43": "TTA d=1.0 s43",
}

header = f"{'Run':<15} {'Ep':>4} {'p_in':>10} {'p_tan':>10} {'p_oodc':>10} {'p_re':>10}"
print(header)
print("-" * len(header))

for label, row in results.items():
    if row is None:
        print(f"{short[label]:<15}  N/A  (no data)")
        continue
    ep = row["epoch_at_target"]
    vals = [row.get(f"ep130_{m}") for m in METRIC_MAP.keys()]
    val_strs = [f"{v:.4f}" if v is not None else "  N/A " for v in vals]
    print(f"{short[label]:<15} {ep:>4} {val_strs[0]:>10} {val_strs[1]:>10} {val_strs[2]:>10} {val_strs[3]:>10}")

print("\n\n" + "="*80)
print("BEST VALUES TABLE (over entire run)")
print("="*80)

header2 = f"{'Run':<15} {'FinalEp':>7} {'p_in':>10} {'p_tan':>10} {'p_oodc':>10} {'p_re':>10}"
print(header2)
print("-" * len(header2))

for label, row in results.items():
    if row is None:
        print(f"{short[label]:<15}  (no data)")
        continue
    ep = row["final_epoch"]
    vals = [row.get(f"best_{m}") for m in METRIC_MAP.keys()]
    val_strs = [f"{v:.4f}" if v is not None else "  N/A " for v in vals]
    print(f"{short[label]:<15} {ep:>7} {val_strs[0]:>10} {val_strs[1]:>10} {val_strs[2]:>10} {val_strs[3]:>10}")

# Seed averages
groups = {
    "Baseline":   ["baseline_s42",  "baseline_s43"],
    "TTA d=0.5":  ["tta_d0.5_s42", "tta_d0.5_s43"],
    "TTA d=1.0":  ["tta_d1.0_s42", "tta_d1.0_s43"],
}

print("\n\n" + "="*80)
print("SEED-AVERAGED AT EPOCH 130")
print("="*80)

header3 = f"{'Group':<12} {'p_in':>10} {'p_tan':>10} {'p_oodc':>10} {'p_re':>10}"
print(header3)
print("-" * len(header3))

for gname, labels in groups.items():
    avgs = []
    for m in METRIC_MAP.keys():
        vals = [results[l][f"ep130_{m}"] for l in labels if results.get(l) and results[l].get(f"ep130_{m}") is not None]
        avgs.append(np.mean(vals) if vals else None)
    val_strs = [f"{v:.4f}" if v is not None else "  N/A " for v in avgs]
    print(f"{gname:<12} {val_strs[0]:>10} {val_strs[1]:>10} {val_strs[2]:>10} {val_strs[3]:>10}")

print("\n\n" + "="*80)
print("SEED-AVERAGED BEST VALUES")
print("="*80)

print(header3)
print("-" * len(header3))

for gname, labels in groups.items():
    avgs = []
    for m in METRIC_MAP.keys():
        vals = [results[l][f"best_{m}"] for l in labels if results.get(l) and results[l].get(f"best_{m}") is not None]
        avgs.append(np.mean(vals) if vals else None)
    val_strs = [f"{v:.4f}" if v is not None else "  N/A " for v in avgs]
    print(f"{gname:<12} {val_strs[0]:>10} {val_strs[1]:>10} {val_strs[2]:>10} {val_strs[3]:>10}")

# Compute deltas vs baseline
print("\n\n" + "="*80)
print("DELTA vs BASELINE (SEED-AVERAGED AT EPOCH 130, lower is better)")
print("="*80)
print("  Positive = worse than baseline, Negative = better than baseline")

header4 = f"{'Group':<12} {'p_in':>12} {'p_tan':>12} {'p_oodc':>12} {'p_re':>12}"
print(header4)
print("-" * len(header4))

base_avgs = {}
for m in METRIC_MAP.keys():
    vals = [results[l][f"ep130_{m}"] for l in groups["Baseline"] if results.get(l) and results[l].get(f"ep130_{m}") is not None]
    base_avgs[m] = np.mean(vals) if vals else None

for gname in ["TTA d=0.5", "TTA d=1.0"]:
    labels = groups[gname]
    deltas = []
    for m in METRIC_MAP.keys():
        vals = [results[l][f"ep130_{m}"] for l in labels if results.get(l) and results[l].get(f"ep130_{m}") is not None]
        avg = np.mean(vals) if vals else None
        if avg is not None and base_avgs.get(m) is not None:
            delta = avg - base_avgs[m]
            pct = 100 * delta / base_avgs[m]
            deltas.append(f"{delta:+.4f} ({pct:+.1f}%)")
        else:
            deltas.append("  N/A ")
    print(f"{gname:<12} {deltas[0]:>12} {deltas[1]:>12} {deltas[2]:>12} {deltas[3]:>12}")
