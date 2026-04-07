import os
import wandb
import pandas as pd

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["yrckevun", "fwof1f72", "mh5sy993", "agndk2w9"]
metric_keys = [
    "best_best_val_in_dist/mae_surf_p",
    "best_best_val_ood_cond/mae_surf_p",
    "best_best_val_tandem_transfer/mae_surf_p",
    "best_best_val_ood_re/mae_surf_p",
]

rows = []
for run_id in run_ids:
    run = api.run(f"{entity}/{project}/{run_id}")
    row = {
        "run_id": run_id,
        "name": run.name,
        "group": run.group,
        "state": run.state,
    }
    for key in metric_keys:
        val = run.summary_metrics.get(key)
        row[key] = val
    rows.append(row)

df = pd.DataFrame(rows)
print("\n=== Per-run metrics ===")
print(df.to_string(index=False))

# Check for NaN or missing
print("\n=== NaN / Missing check ===")
for _, r in df.iterrows():
    for key in metric_keys:
        val = r[key]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            print(f"  MISSING/NaN: run={r['run_id']}, metric={key}")
        else:
            print(f"  OK: run={r['run_id']}, metric={key}, value={val}")

# 2-seed means: control = yrckevun, fwof1f72; GSB = mh5sy993, agndk2w9
print("\n=== 2-seed means ===")
control_df = df[df["run_id"].isin(["yrckevun", "fwof1f72"])]
gsb_df = df[df["run_id"].isin(["mh5sy993", "agndk2w9"])]

print("\nControl mean (yrckevun, fwof1f72):")
for key in metric_keys:
    vals = control_df[key].dropna()
    mean_val = vals.mean() if len(vals) > 0 else float("nan")
    print(f"  {key}: {mean_val}")

print("\nGSB mean (mh5sy993, agndk2w9):")
for key in metric_keys:
    vals = gsb_df[key].dropna()
    mean_val = vals.mean() if len(vals) > 0 else float("nan")
    print(f"  {key}: {mean_val}")
