import wandb
import os

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["p9oupnt2", "6sp2uazt", "kov6n0rs", "l308y9lx"]

# The student labels map to these W&B keys:
# p_in   -> best_best_val_in_dist/mae_surf_p
# p_oodc -> best_best_val_ood_cond/mae_surf_p
# p_tan  -> best_best_val_tandem_transfer/mae_surf_p
# p_re   -> best_best_val_ood_re/mae_surf_p

METRIC_MAP = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

# Student reported values for comparison
STUDENT_REPORTED = {
    "p9oupnt2": {"p_in": 13.5, "p_oodc": 7.6, "p_tan": 30.0, "p_re": 6.4, "pct": 0.15, "seed": 42},
    "6sp2uazt": {"p_in": 12.9, "p_oodc": 7.5, "p_tan": 29.4, "p_re": 6.4, "pct": 0.15, "seed": 73},
    "kov6n0rs": {"p_in": 13.5, "p_oodc": 7.6, "p_tan": 30.0, "p_re": 6.4, "pct": 0.10, "seed": 42},
    "l308y9lx": {"p_in": 13.4, "p_oodc": 7.8, "p_tan": 29.0, "p_re": 6.4, "pct": 0.10, "seed": 73},
}

print(f"Querying {entity}/{project} for runs: {run_ids}\n")

results = []
for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        summary = run.summary_metrics

        actual = {}
        for label, wandb_key in METRIC_MAP.items():
            v = summary.get(wandb_key)
            actual[label] = round(float(v) * 1000, 2) if v is not None else None  # convert to match student's scale

        # Also try without multiplying by 1000 first to see the raw value
        actual_raw = {}
        for label, wandb_key in METRIC_MAP.items():
            v = summary.get(wandb_key)
            actual_raw[label] = float(v) if v is not None else None

        reported = STUDENT_REPORTED[run_id]

        print(f"Run: {run_id} ({run.name}), seed={reported['seed']}, pct={reported['pct']}")
        print(f"  Raw W&B values:")
        for label, wandb_key in METRIC_MAP.items():
            raw = actual_raw[label]
            print(f"    {label} ({wandb_key}): {raw}")
        print()

        results.append({
            "run_id": run_id,
            "name": run.name,
            "seed": reported["seed"],
            "pct": reported["pct"],
            "actual_raw": actual_raw,
            "reported": reported,
        })

    except Exception as e:
        print(f"Run {run_id}: ERROR - {e}\n")

# Now determine scale
# Check first run's raw values to understand scale
if results:
    sample_raw = results[0]["actual_raw"]["p_in"]
    sample_reported = results[0]["reported"]["p_in"]
    print(f"\nScale check: raw={sample_raw}, reported={sample_reported}")
    # If raw is ~0.0135 and reported is 13.5, scale factor is 1000
    # If raw is ~13.5 and reported is 13.5, scale factor is 1
    if sample_raw is not None and sample_reported is not None:
        if abs(sample_raw - sample_reported) < 1.0:
            scale = 1.0
        elif abs(sample_raw * 1000 - sample_reported) < 1.0:
            scale = 1000.0
        else:
            scale = 1.0  # unknown, use raw
        print(f"Detected scale factor: {scale}\n")
    else:
        scale = 1.0

print("\n=== ACTUAL W&B VALUES vs STUDENT REPORTED ===")
print(f"{'Run ID':<12} {'Config':<26} {'Metric':<7} {'W&B Actual':>12} {'Student Reported':>17} {'Match?':>8}")
print("-" * 85)

for r in results:
    config_str = f"pct={r['pct']}, s{r['seed']}"
    for label in ["p_in", "p_oodc", "p_tan", "p_re"]:
        raw = r["actual_raw"][label]
        actual_scaled = round(raw * scale, 2) if raw is not None else None
        rep = r["reported"][label]
        if actual_scaled is not None and rep is not None:
            match = "YES" if abs(actual_scaled - rep) < 0.15 else "NO"
        else:
            match = "N/A"
        print(f"{r['run_id']:<12} {config_str:<26} {label:<7} {str(actual_scaled):>12} {str(rep):>17} {match:>8}")
    print()

print("\n=== COMPACT SUMMARY TABLE (W&B actual values) ===")
print(f"{'Run ID':<12} {'Config':<26} {'p_in':>7} {'p_oodc':>8} {'p_tan':>7} {'p_re':>6}")
print("-" * 65)
for r in results:
    config_str = f"pct={r['pct']}, s{r['seed']}"
    vals = {k: (round(v * scale, 2) if v is not None else "N/A") for k, v in r["actual_raw"].items()}
    print(f"{r['run_id']:<12} {config_str:<26} {str(vals['p_in']):>7} {str(vals['p_oodc']):>8} {str(vals['p_tan']):>7} {str(vals['p_re']):>6}")
