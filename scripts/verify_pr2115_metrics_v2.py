"""Verify W&B metrics for PR #2115 - Gap/Stagger Perturbation Augmentation.
The actual metric names use the pattern best_best_val_<split>/mae_surf_p etc.
The student labels p_in, p_oodc, p_tan, p_re correspond to:
  p_in   -> in-distribution        -> best_best_val_in_dist/mae_surf_p
  p_oodc -> OOD condition          -> best_best_val_ood_cond/mae_surf_p
  p_tan  -> tandem transfer        -> best_best_val_tandem_transfer/mae_surf_p
  p_re   -> OOD Reynolds           -> best_best_val_ood_re/mae_surf_p
"""
import os
import sys
import wandb
import pandas as pd
import numpy as np

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "wandb")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = [
    ("hszpxxof", 0.02, 42),
    ("weovkf6s", 0.02, 73),
    ("7lpt7n7o", 0.05, 42),
    ("k1oihv6h", 0.05, 73),
    ("x8xxcpt0", 0.10, 42),
    ("vnfp0i7s", 0.10, 73),
]

# Student-reported metrics for comparison
student_reported = {
    "hszpxxof": {"p_in": 13.166, "p_oodc": 7.514, "p_tan": 29.645, "p_re": 6.486},
    "weovkf6s": {"p_in": 12.907, "p_oodc": 7.377, "p_tan": 30.781, "p_re": 6.216},
    "7lpt7n7o": {"p_in": 13.232, "p_oodc": 7.643, "p_tan": 30.162, "p_re": 6.359},
    "k1oihv6h": {"p_in": 13.006, "p_oodc": 7.603, "p_tan": 30.407, "p_re": 6.483},
    "x8xxcpt0": {"p_in": 13.316, "p_oodc": 7.704, "p_tan": 31.113, "p_re": 6.542},
    "vnfp0i7s": {"p_in": 12.815, "p_oodc": 7.559, "p_tan": 30.631, "p_re": 6.359},
}

# Mapping from student's metric names to W&B actual keys
# Based on W&B summary keys observed:
# best_best_val_in_dist/mae_surf_p  -> p_in
# best_best_val_ood_cond/mae_surf_p -> p_oodc
# best_best_val_tandem_transfer/mae_surf_p -> p_tan
# best_best_val_ood_re/mae_surf_p  -> p_re
METRIC_MAP = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

results = []

for run_id, sigma, seed in run_ids:
    path = f"{entity}/{project}/{run_id}"
    print(f"\n{'='*60}")
    print(f"Fetching run: {run_id} (sigma={sigma}, seed={seed})")
    try:
        run = api.run(path)
        state = run.state
        print(f"  State: {state}")
        print(f"  Name: {run.name}")
        print(f"  Group: {run.group}")

        summary = run.summary_metrics

        # Extract mapped metrics
        actuals = {}
        for label, wb_key in METRIC_MAP.items():
            val = summary.get(wb_key)
            actuals[label] = val
            reported = student_reported[run_id][label]
            if val is not None:
                diff = abs(float(val) - reported)
                print(f"  {label}: actual={val:.4f}, reported={reported:.3f}, diff={diff:.4f}")
            else:
                print(f"  {label}: NOT FOUND in summary (key: {wb_key})")

        results.append({
            "run_id": run_id,
            "sigma": sigma,
            "seed": seed,
            "state": state,
            "p_in_actual":   actuals.get("p_in"),
            "p_oodc_actual": actuals.get("p_oodc"),
            "p_tan_actual":  actuals.get("p_tan"),
            "p_re_actual":   actuals.get("p_re"),
            "p_in_reported":   student_reported[run_id]["p_in"],
            "p_oodc_reported": student_reported[run_id]["p_oodc"],
            "p_tan_reported":  student_reported[run_id]["p_tan"],
            "p_re_reported":   student_reported[run_id]["p_re"],
        })

    except Exception as e:
        print(f"  ERROR fetching run {run_id}: {e}")
        results.append({
            "run_id": run_id,
            "sigma": sigma,
            "seed": seed,
            "state": "ERROR",
            "p_in_actual": None, "p_oodc_actual": None,
            "p_tan_actual": None, "p_re_actual": None,
            "p_in_reported":   student_reported[run_id]["p_in"],
            "p_oodc_reported": student_reported[run_id]["p_oodc"],
            "p_tan_reported":  student_reported[run_id]["p_tan"],
            "p_re_reported":   student_reported[run_id]["p_re"],
        })

print("\n\n" + "="*80)
print("VERIFICATION SUMMARY TABLE")
print("="*80)

df = pd.DataFrame(results)

# Print full comparison table
for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
    print(f"\n--- {metric} ---")
    print(f"{'run_id':<12} {'sigma':>6} {'seed':>6} | {'actual':>10} {'reported':>10} {'diff':>10} {'status':>10}")
    print("-" * 65)
    for _, row in df.iterrows():
        actual = row[f"{metric}_actual"]
        reported = row[f"{metric}_reported"]
        if actual is not None:
            diff = float(actual) - reported
            status = "OK" if abs(diff) <= 0.01 else "MISMATCH"
            print(f"{row['run_id']:<12} {row['sigma']:>6.2f} {row['seed']:>6} | {actual:>10.3f} {reported:>10.3f} {diff:>10.3f} {status:>10}")
        else:
            print(f"{row['run_id']:<12} {row['sigma']:>6.2f} {row['seed']:>6} | {'N/A':>10} {reported:>10.3f} {'N/A':>10} {'MISSING':>10}")

print("\n\n" + "="*80)
print("COMPLETION STATUS")
print("="*80)
for _, row in df.iterrows():
    state = row.get("state", "UNKNOWN")
    status = "COMPLETED" if state == "finished" else f"WARNING: state={state}"
    print(f"  Run {row['run_id']} (sigma={row['sigma']}, seed={row['seed']}): {status}")

print("\n\n" + "="*80)
print("OVERALL VERDICT")
print("="*80)
all_match = True
missing_count = 0
for _, row in df.iterrows():
    for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
        actual = row[f"{metric}_actual"]
        reported = row[f"{metric}_reported"]
        if actual is None:
            missing_count += 1
            all_match = False
        elif abs(float(actual) - reported) > 0.01:
            all_match = False

if missing_count > 0:
    print(f"WARNING: {missing_count} metrics could not be retrieved from W&B")
elif all_match:
    print("All metrics verified: student-reported numbers match W&B within tolerance (0.01)")
else:
    print("DISCREPANCIES FOUND: student-reported numbers do NOT match W&B")
