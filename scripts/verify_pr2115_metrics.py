"""Verify W&B metrics for PR #2115 - Gap/Stagger Perturbation Augmentation."""
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

surface_mae_keys = [
    "surface_mae/p_in",
    "surface_mae/p_oodc",
    "surface_mae/p_tan",
    "surface_mae/p_re",
]

# Also check alternative key names
alt_keys = [
    "val/surface_mae/p_in",
    "val/surface_mae/p_oodc",
    "val/surface_mae/p_tan",
    "val/surface_mae/p_re",
    "p_in_mae",
    "p_oodc_mae",
    "p_tan_mae",
    "p_re_mae",
    "mae_p_in",
    "mae_p_oodc",
    "mae_p_tan",
    "mae_p_re",
]

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

        # Print all summary metrics keys to understand structure
        summary = run.summary_metrics
        print(f"  Summary keys (first 40): {list(summary.keys())[:40]}")

        # Try to find surface MAE metrics
        found_metrics = {}

        # Check direct keys
        for key in surface_mae_keys:
            val = summary.get(key)
            if val is not None:
                found_metrics[key] = val

        # Check alternative keys
        for key in alt_keys:
            val = summary.get(key)
            if val is not None:
                found_metrics[key] = val

        # Scan all keys for anything containing p_in, p_oodc, p_tan, p_re
        for key, val in summary.items():
            if any(k in key.lower() for k in ["p_in", "p_oodc", "p_tan", "p_re", "surface"]):
                found_metrics[key] = val

        print(f"  Found surface/pressure metrics: {found_metrics}")

        # Try to normalize to standard names
        p_in = None
        p_oodc = None
        p_tan = None
        p_re = None

        for key, val in found_metrics.items():
            if "p_in" in key:
                p_in = val
            elif "p_oodc" in key:
                p_oodc = val
            elif "p_tan" in key:
                p_tan = val
            elif "p_re" in key:
                p_re = val

        results.append({
            "run_id": run_id,
            "sigma": sigma,
            "seed": seed,
            "state": state,
            "p_in_actual": p_in,
            "p_oodc_actual": p_oodc,
            "p_tan_actual": p_tan,
            "p_re_actual": p_re,
            "p_in_reported": student_reported[run_id]["p_in"],
            "p_oodc_reported": student_reported[run_id]["p_oodc"],
            "p_tan_reported": student_reported[run_id]["p_tan"],
            "p_re_reported": student_reported[run_id]["p_re"],
        })

    except Exception as e:
        print(f"  ERROR fetching run {run_id}: {e}")
        results.append({
            "run_id": run_id,
            "sigma": sigma,
            "seed": seed,
            "state": "ERROR",
            "error": str(e),
            "p_in_actual": None,
            "p_oodc_actual": None,
            "p_tan_actual": None,
            "p_re_actual": None,
            "p_in_reported": student_reported[run_id]["p_in"],
            "p_oodc_reported": student_reported[run_id]["p_oodc"],
            "p_tan_reported": student_reported[run_id]["p_tan"],
            "p_re_reported": student_reported[run_id]["p_re"],
        })

print("\n\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

df = pd.DataFrame(results)
print(df[["run_id", "sigma", "seed", "state",
          "p_in_actual", "p_in_reported",
          "p_oodc_actual", "p_oodc_reported",
          "p_tan_actual", "p_tan_reported",
          "p_re_actual", "p_re_reported"]].to_string(index=False))

print("\nDISCREPANCY CHECK:")
for _, row in df.iterrows():
    discrepancies = []
    for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
        actual = row.get(f"{metric}_actual")
        reported = row.get(f"{metric}_reported")
        if actual is not None and reported is not None:
            diff = abs(float(actual) - float(reported))
            if diff > 0.01:
                discrepancies.append(f"{metric}: actual={actual:.3f}, reported={reported:.3f}, diff={diff:.3f}")
    if discrepancies:
        print(f"  Run {row['run_id']} (sigma={row['sigma']}, seed={row['seed']}): DISCREPANCIES FOUND")
        for d in discrepancies:
            print(f"    {d}")
    else:
        print(f"  Run {row['run_id']} (sigma={row['sigma']}, seed={row['seed']}): OK (all within 0.01)")

# Check completion
print("\nCOMPLETION STATUS:")
for _, row in df.iterrows():
    state = row.get("state", "UNKNOWN")
    status = "COMPLETED" if state == "finished" else f"WARNING: state={state}"
    print(f"  Run {row['run_id']} (sigma={row['sigma']}, seed={row['seed']}): {status}")
