"""
Check W&B surface MAE metrics for PR #2104 runs.
Run IDs: rgr1lvbt (baseline s42), uw32s6kz (baseline s43),
         cp4ralol (aft_srf s42), w0iccehn (aft_srf s43)
"""
import os
import sys
import wandb
import numpy as np

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

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

METRIC_KEYS = [
    "val_in_dist/mae_surf_p",
    "val_ood_cond/mae_surf_p",
    "val_tandem_transfer/mae_surf_p",
    "val_ood_re/mae_surf_p",
]

ALIASES = {
    "val_in_dist/mae_surf_p":        "p_in",
    "val_ood_cond/mae_surf_p":       "p_oodc",
    "val_tandem_transfer/mae_surf_p":"p_tan",
    "val_ood_re/mae_surf_p":         "p_re",
}

def best_metric(run, key):
    """Return the best (minimum) value of a metric across history."""
    values = []
    for row in run.scan_history(keys=[key]):
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            values.append(v)
    if not values:
        # fall back to summary
        v = run.summary_metrics.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
        return float("nan")
    return min(values)

def check_stability(run):
    """Look for NaN loss or large spikes in train loss."""
    keys = ["train/loss", "loss", "train_loss"]
    losses = []
    for row in run.scan_history(keys=keys):
        for k in keys:
            v = row.get(k)
            if v is not None:
                losses.append(v)
                break
    if not losses:
        return "no_loss_data"
    arr = np.array(losses)
    nan_count = int(np.sum(np.isnan(arr)))
    if nan_count > 0:
        return f"NaN_loss ({nan_count} steps)"
    # check for spikes: any value > 10x median
    finite = arr[np.isfinite(arr)]
    if len(finite) > 10:
        med = np.median(finite)
        spikes = int(np.sum(finite > 10 * med)) if med > 0 else 0
        if spikes > 0:
            return f"spikes ({spikes} steps > 10x median)"
    return "stable"

print(f"{'Config':<15} {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8}  {'Stability'}")
print("-" * 70)

results = {}
for label, run_id in RUN_IDS.items():
    try:
        run = api.run(f"{path}/{run_id}")
        metrics = {ALIASES[k]: best_metric(run, k) for k in METRIC_KEYS}
        stability = check_stability(run)
        results[label] = {**metrics, "stability": stability, "state": run.state}
        p_in   = metrics["p_in"]
        p_oodc = metrics["p_oodc"]
        p_tan  = metrics["p_tan"]
        p_re   = metrics["p_re"]
        print(f"{label:<15} {p_in:>8.4f} {p_oodc:>8.4f} {p_tan:>8.4f} {p_re:>8.4f}  {stability}  [{run.state}]")
    except Exception as e:
        print(f"{label:<15} ERROR: {e}")

print()
print("=== Summary ===")
# Average aft_srf vs baseline
try:
    for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
        base_avg = (results["baseline_s42"][metric] + results["baseline_s43"][metric]) / 2
        srf_avg  = (results["aft_srf_s42"][metric]  + results["aft_srf_s43"][metric])  / 2
        delta    = srf_avg - base_avg
        pct      = 100 * delta / base_avg if base_avg != 0 else float("nan")
        sign     = "BETTER" if delta < 0 else "WORSE"
        print(f"  {metric}: baseline_avg={base_avg:.4f}  aft_srf_avg={srf_avg:.4f}  delta={delta:+.4f} ({pct:+.2f}%)  [{sign}]")
except Exception as e:
    print(f"  Could not compute averages: {e}")
