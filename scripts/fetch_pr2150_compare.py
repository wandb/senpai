import os
import wandb
import pandas as pd

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "senpai-fyi")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = ["cm9uz650", "kj8cvxpw", "js6sm78l", "2o2499gz"]
labels = {
    "cm9uz650": "σ=0.03, seed 42",
    "kj8cvxpw": "σ=0.03, seed 73",
    "js6sm78l": "σ=0.08, seed 42",
    "2o2499gz": "σ=0.08, seed 73",
}

# Mapping: the student calls these p_in / p_oodc / p_tan / p_re
# The actual W&B keys (from inspection) are:
#   p_in   -> best_best_val_in_dist/mae_surf_p
#   p_oodc -> best_best_val_ood_cond/mae_surf_p
#   p_tan  -> best_best_val_tandem_transfer/mae_surf_p
#   p_re   -> best_best_val_ood_re/mae_surf_p

WB_KEYS = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

student_reported = {
    "cm9uz650": {"p_in": 12.9, "p_oodc": 7.8, "p_tan": 28.2, "p_re": 6.6},
    "kj8cvxpw": {"p_in": 13.6, "p_oodc": 8.1, "p_tan": 29.5, "p_re": 6.6},
    "js6sm78l": {"p_in": 12.9, "p_oodc": 7.9, "p_tan": 28.9, "p_re": 6.6},
    "2o2499gz": {"p_in": 13.4, "p_oodc": 7.9, "p_tan": 29.4, "p_re": 6.5},
}

rows = []
for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    summary = run.summary_metrics

    wb = {col: summary.get(key) for col, key in WB_KEYS.items()}
    rep = student_reported[run_id]

    row = {
        "run_id": run_id,
        "label": labels[run_id],
        "state": run.state,
    }
    for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
        row[f"wb_{col}"]  = wb[col]
        row[f"rep_{col}"] = rep[col]
        if wb[col] is not None:
            row[f"diff_{col}"] = abs(wb[col] - rep[col])
        else:
            row[f"diff_{col}"] = None
    rows.append(row)

df = pd.DataFrame(rows)

print("=" * 90)
print("PR #2150 — DSDF2 Sigma Sweep: W&B Actual vs Student Reported")
print("=" * 90)
print("\nSplit: p_in = val_in_dist surface pressure MAE")
print("       p_oodc = val_ood_cond surface pressure MAE")
print("       p_tan  = val_tandem_transfer surface pressure MAE")
print("       p_re   = val_ood_re surface pressure MAE\n")

for _, row in df.iterrows():
    print(f"Run {row['run_id']}  ({row['label']})  state={row['state']}")
    print(f"  {'Metric':<8}  {'W&B Actual':>12}  {'Student Rpt':>12}  {'Delta':>8}  {'Flag'}")
    print(f"  {'-'*60}")
    for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
        wb_val  = row[f"wb_{col}"]
        rep_val = row[f"rep_{col}"]
        diff    = row[f"diff_{col}"]
        flag = ""
        if diff is not None and diff > 0.15:
            flag = "*** DISCREPANCY"
        wb_str  = f"{wb_val:.4f}" if wb_val is not None else "N/A"
        print(f"  {col:<8}  {wb_str:>12}  {rep_val:>12.1f}  {diff if diff is not None else float('nan'):>8.4f}  {flag}")
    print()

print("=" * 90)
print("\nSUMMARY TABLE")
print("=" * 90)
cols_to_show = ["run_id", "label",
                "wb_p_in", "rep_p_in",
                "wb_p_oodc", "rep_p_oodc",
                "wb_p_tan", "rep_p_tan",
                "wb_p_re", "rep_p_re"]
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(df[cols_to_show].to_string(index=False))

print("\n\nFLAGGED DISCREPANCIES (|delta| > 0.15):")
any_flag = False
for _, row in df.iterrows():
    for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
        diff = row[f"diff_{col}"]
        if diff is not None and diff > 0.15:
            print(f"  Run {row['run_id']} ({row['label']}): {col}  W&B={row[f'wb_{col}']:.4f}  reported={row[f'rep_{col}']:.1f}  diff={diff:.4f}")
            any_flag = True
if not any_flag:
    print("  None — all metrics are within 0.15 of reported values.")
