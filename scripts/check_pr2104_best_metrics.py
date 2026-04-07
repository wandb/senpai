"""
Pull best_best_ summary metrics for PR #2104 runs.
These are the checkpoint-based best metrics, which are definitive.
"""
import os
import wandb
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

RUN_IDS = {
    "baseline_s42": "rgr1lvbt",
    "baseline_s43": "uw32s6kz",
    "aft_srf_s42":  "cp4ralol",
    "aft_srf_s43":  "w0iccehn",
}

# best_best_ prefix = best checkpoint value
METRIC_MAP = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

print(f"{'Config':<15} {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8}  {'NaN_loss':>8}  State")
print("-" * 80)

results = {}
for label, run_id in RUN_IDS.items():
    run = api.run(f"{path}/{run_id}")
    sm = run.summary_metrics

    metrics = {}
    for alias, key in METRIC_MAP.items():
        v = sm.get(key)
        metrics[alias] = v if v is not None else float("nan")

    # NaN check in train/loss history
    nan_count = 0
    for row in run.scan_history(keys=["train/loss"]):
        v = row.get("train/loss")
        if v is not None and (isinstance(v, float) and np.isnan(v)):
            nan_count += 1

    results[label] = metrics
    p_in   = metrics["p_in"]
    p_oodc = metrics["p_oodc"]
    p_tan  = metrics["p_tan"]
    p_re   = metrics["p_re"]
    print(f"{label:<15} {p_in:>8.4f} {p_oodc:>8.4f} {p_tan:>8.4f} {p_re:>8.4f}  {nan_count:>8}  {run.state}")

print()
print("=== Paired seed averages: aft_srf vs baseline ===")
for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
    base_avg = (results["baseline_s42"][metric] + results["baseline_s43"][metric]) / 2
    srf_avg  = (results["aft_srf_s42"][metric]  + results["aft_srf_s43"][metric])  / 2
    delta    = srf_avg - base_avg
    pct      = 100 * delta / base_avg if base_avg != 0 else float("nan")
    sign     = "BETTER" if delta < 0 else "WORSE"
    print(f"  {metric:<8}: baseline={base_avg:.4f}  aft_srf={srf_avg:.4f}  delta={delta:+.4f} ({pct:+.2f}%)  [{sign}]")

print()
print("=== Average of 2 seeds per config ===")
for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
    base = (results["baseline_s42"][metric] + results["baseline_s43"][metric]) / 2
    srf  = (results["aft_srf_s42"][metric]  + results["aft_srf_s43"][metric])  / 2
    print(f"  {metric}: baseline={base:.4f}, aft_srf={srf:.4f}")
