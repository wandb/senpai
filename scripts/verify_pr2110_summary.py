"""Clean summary of PR #2110 surface pressure MAE metrics."""
import os
import wandb
import pandas as pd

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

api = wandb.Api()

run_ids = {
    "7n8v8nlh": ("surframp-40ep-s42", "40ep", 42),
    "ui8t0t7g": ("surframp-40ep-s73", "40ep", 73),
    "mqu6c2vd": ("surframp-80ep-s42", "80ep", 42),
    "vkrnehh2": ("surframp-80ep-s73", "80ep", 73),
}

# The four splits that correspond to p_in, p_oodc, p_tan, p_re
# Based on the metric schema:
#  p_in   -> val_in_dist/mae_surf_p
#  p_oodc -> val_ood_cond/mae_surf_p
#  p_tan  -> val_tandem_transfer/mae_surf_p
#  p_re   -> val_ood_re/mae_surf_p
SPLIT_MAP = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

print("=== PR #2110: Progressive Surface Focus Schedule ===")
print("=== W&B Group: phase6/progressive-surface ===\n")

rows = []
for run_id, (run_name, ramp, seed) in run_ids.items():
    run = api.run(f"{path}/{run_id}")
    s = run.summary_metrics
    row = {
        "Run": run_name,
        "W&B ID": run_id,
        "Ramp": ramp,
        "Seed": seed,
        "State": run.state,
        "Best Epoch": s.get("best_epoch", "N/A"),
        "p_in (best)": round(s.get(SPLIT_MAP["p_in"], float("nan")), 3),
        "p_oodc (best)": round(s.get(SPLIT_MAP["p_oodc"], float("nan")), 3),
        "p_tan (best)": round(s.get(SPLIT_MAP["p_tan"], float("nan")), 3),
        "p_re (best)": round(s.get(SPLIT_MAP["p_re"], float("nan")), 3),
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string(index=False))

print("\n=== Comparison vs Student-Reported Metrics ===")
student_reported = {
    "7n8v8nlh": (13.334, 7.927, 30.574, 6.487),
    "ui8t0t7g": (12.912, 7.917, 29.349, 6.638),
    "mqu6c2vd": (13.396, 7.690, 29.716, 6.291),
    "vkrnehh2": (12.854, 8.093, 29.864, 6.347),
}

print(f"\n{'Run':<22} {'Metric':<8} {'Student':>10} {'Verified':>10} {'Match?':>8}")
print("-" * 65)
for r in rows:
    rid = r["W&B ID"]
    sr = student_reported[rid]
    metrics = [
        ("p_in",   r["p_in (best)"],   sr[0]),
        ("p_oodc", r["p_oodc (best)"], sr[1]),
        ("p_tan",  r["p_tan (best)"],  sr[2]),
        ("p_re",   r["p_re (best)"],   sr[3]),
    ]
    for metric, verified, reported in metrics:
        match = "OK" if abs(verified - reported) < 0.01 else "MISMATCH"
        print(f"{r['Run']:<22} {metric:<8} {reported:>10.3f} {verified:>10.3f} {match:>8}")
    print()
