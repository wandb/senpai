import os
import wandb
import pandas as pd

api = wandb.Api()

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["ftrzalka", "fceuhys3", "pfm9qsaj", "iyi2npzd", "opnlewzj", "atize8g5"]

run_names = {
    "ftrzalka": "contrast-w0.01-s42",
    "fceuhys3": "contrast-w0.01-s73",
    "pfm9qsaj": "contrast-w0.05-s42",
    "iyi2npzd": "contrast-w0.05-s73",
    "opnlewzj": "contrast-w0.1-s42",
    "atize8g5": "contrast-w0.1-s73",
}

# Student reported values (p = pressure surface MAE)
# p_in = in-dist, p_oodc = ood-cond, p_tan = tandem, p_re = ood-re
student_reported = {
    "ftrzalka": {"p_in": 13.3, "p_oodc": 7.8, "p_tan": 30.3, "p_re": 6.4},
    "fceuhys3": {"p_in": 13.1, "p_oodc": 7.8, "p_tan": 30.0, "p_re": 6.4},
    "pfm9qsaj": {"p_in": 12.5, "p_oodc": 7.7, "p_tan": 30.6, "p_re": 6.5},
    "iyi2npzd": {"p_in": 13.3, "p_oodc": 8.2, "p_tan": 30.3, "p_re": 6.6},
    "opnlewzj": {"p_in": 12.9, "p_oodc": 7.7, "p_tan": 31.0, "p_re": 6.3},
    "atize8g5": {"p_in": 12.7, "p_oodc": 7.9, "p_tan": 30.3, "p_re": 6.6},
}

# Map from our shorthand to W&B keys
# best_best_val_in_dist/mae_surf_p  -> p_in
# best_best_val_ood_cond/mae_surf_p -> p_oodc
# best_best_val_tandem_transfer/mae_surf_p -> p_tan
# best_best_val_ood_re/mae_surf_p   -> p_re
wandb_key_map = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

rows = []
discrepancies = []

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    summary = run.summary_metrics

    actual = {}
    for key, wk in wandb_key_map.items():
        actual[key] = summary.get(wk)

    reported = student_reported[run_id]
    row = {
        "W&B ID": run_id,
        "Run": run_names[run_id],
        "state": run.state,
        "actual_p_in": round(actual["p_in"], 2) if actual["p_in"] is not None else None,
        "reported_p_in": reported["p_in"],
        "actual_p_oodc": round(actual["p_oodc"], 2) if actual["p_oodc"] is not None else None,
        "reported_p_oodc": reported["p_oodc"],
        "actual_p_tan": round(actual["p_tan"], 2) if actual["p_tan"] is not None else None,
        "reported_p_tan": reported["p_tan"],
        "actual_p_re": round(actual["p_re"], 2) if actual["p_re"] is not None else None,
        "reported_p_re": reported["p_re"],
    }
    rows.append(row)

    for k in ["p_in", "p_oodc", "p_tan", "p_re"]:
        if actual[k] is not None:
            diff = abs(actual[k] - reported[k])
            if diff > 0.15:
                discrepancies.append({
                    "run_id": run_id,
                    "name": run_names[run_id],
                    "metric": k,
                    "reported": reported[k],
                    "actual": round(actual[k], 2),
                    "diff": round(diff, 2),
                })

df = pd.DataFrame(rows)

print("=== ACTUAL vs REPORTED SURFACE PRESSURE MAE ===")
print(f"{'Run':<22} {'p_in (act/rep)':<16} {'p_oodc (act/rep)':<18} {'p_tan (act/rep)':<17} {'p_re (act/rep)':<16}")
print("-" * 90)
for _, r in df.iterrows():
    print(f"{r['Run']:<22} {r['actual_p_in']}/{r['reported_p_in']:<11} "
          f"{r['actual_p_oodc']}/{r['reported_p_oodc']:<13} "
          f"{r['actual_p_tan']}/{r['reported_p_tan']:<11} "
          f"{r['actual_p_re']}/{r['reported_p_re']}")

print()
if discrepancies:
    print("=== DISCREPANCIES (diff > 0.15) ===")
    for d in discrepancies:
        print(f"  {d['name']} ({d['run_id']}): {d['metric']} reported={d['reported']} actual={d['actual']} diff={d['diff']}")
else:
    print("No significant discrepancies (all within 0.15 of reported values).")

# Print best run
print("\n=== BEST RUN BY LOWEST AVERAGE SURFACE PRESSURE MAE ===")
for _, r in df.iterrows():
    vals = [r['actual_p_in'], r['actual_p_oodc'], r['actual_p_tan'], r['actual_p_re']]
    if all(v is not None for v in vals):
        avg = sum(vals) / 4
        print(f"  {r['Run']}: avg={avg:.2f} | p_in={r['actual_p_in']}, p_oodc={r['actual_p_oodc']}, p_tan={r['actual_p_tan']}, p_re={r['actual_p_re']}")
