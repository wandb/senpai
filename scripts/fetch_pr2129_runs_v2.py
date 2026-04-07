import os
import sys
import wandb
import pandas as pd
import math

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

run_ids = ["bnskf03a", "yfy3efjq", "s2txmpe9", "3is100bp"]

# The splits map to:
#   p_in   = best_best_val_in_dist/mae_surf_p
#   p_oodc = best_best_val_ood_cond/mae_surf_p
#   p_tan  = best_best_val_tandem_transfer/mae_surf_p
#   p_re   = best_best_val_ood_re/mae_surf_p

METRIC_MAP = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

def is_nan_or_none(v):
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False

results = []
for run_id in run_ids:
    run = api.run(f"{path}/{run_id}")
    sm = run.summary_metrics
    cfg = run.config

    weight = cfg.get("surf_grad_weight", "N/A")
    seed = cfg.get("seed", "N/A")

    row = {
        "run_id": run_id,
        "name": run.name,
        "weight": weight,
        "seed": seed,
        "state": run.state,
    }

    for label, key in METRIC_MAP.items():
        val = sm.get(key)
        row[label] = val

    results.append(row)

df = pd.DataFrame(results)

print("=== PR #2129 — phase6/surf-grad-aux — Surface MAE Metrics ===\n")
print(f"{'run_id':<12} {'name':<36} {'weight':>8} {'seed':>6} {'p_in':>10} {'p_oodc':>10} {'p_tan':>10} {'p_re':>10} {'state'}")
print("-" * 115)
for _, row in df.iterrows():
    def fmt(v):
        if is_nan_or_none(v):
            return "MISSING"
        return f"{float(v):.6f}"
    print(f"{row['run_id']:<12} {row['name']:<36} {str(row['weight']):>8} {str(row['seed']):>6} "
          f"{fmt(row['p_in']):>10} {fmt(row['p_oodc']):>10} {fmt(row['p_tan']):>10} {fmt(row['p_re']):>10} "
          f"{row['state']}")

print()
print("=== NaN / Missing Check ===")
any_missing = False
for _, row in df.iterrows():
    for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
        if is_nan_or_none(row.get(metric)):
            print(f"  MISSING: run {row['run_id']} metric {metric}")
            any_missing = True
if not any_missing:
    print("  All metrics present and non-NaN.")

# Compute per-weight averages
print()
print("=== Averages by Weight ===")
for w in [0.05, 0.10]:
    sub = df[df["weight"] == w]
    if len(sub) > 0:
        print(f"  weight={w}: p_in={sub['p_in'].mean():.6f}, p_oodc={sub['p_oodc'].mean():.6f}, "
              f"p_tan={sub['p_tan'].mean():.6f}, p_re={sub['p_re'].mean():.6f}")
