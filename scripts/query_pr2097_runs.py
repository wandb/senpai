import os
import sys
import wandb
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = {
    42: "k1v8bvum",
    43: "11yf2z5i",
    44: "2qwd1irz",
    45: "x97wrsdz",
    46: "e1l5eokg",
    47: "xpfnr80l",
    48: "ds46hwpq",
    49: "ocxi1i4c",
}

METRIC_KEYS = [
    "val_in_dist/mae_surf_p",
    "val_ood_cond/mae_surf_p",
    "val_tandem_transfer/mae_surf_p",
    "val_ood_re/mae_surf_p",
]

METRIC_LABELS = {
    "val_in_dist/mae_surf_p": "p_in",
    "val_ood_cond/mae_surf_p": "p_oodc",
    "val_tandem_transfer/mae_surf_p": "p_tan",
    "val_ood_re/mae_surf_p": "p_re",
}

rows = []
for seed, run_id in run_ids.items():
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        metrics = {label: None for label in METRIC_LABELS.values()}
        for key, label in METRIC_LABELS.items():
            val = run.summary_metrics.get(key)
            if val is None:
                # Try alternate forms
                val = run.summary_metrics.get(key.replace("/", "."))
            metrics[label] = val
        rows.append({
            "seed": seed,
            "run_id": run_id,
            "state": run.state,
            **metrics,
        })
        print(f"Seed {seed} ({run_id}): state={run.state}, p_in={metrics['p_in']}, p_oodc={metrics['p_oodc']}, p_tan={metrics['p_tan']}, p_re={metrics['p_re']}")
    except Exception as e:
        print(f"ERROR fetching seed {seed} ({run_id}): {e}")
        rows.append({"seed": seed, "run_id": run_id, "state": "ERROR", "p_in": None, "p_oodc": None, "p_tan": None, "p_re": None})

df = pd.DataFrame(rows)
print("\n--- Full Table ---")
print(df.to_string(index=False))

numeric_cols = ["p_in", "p_oodc", "p_tan", "p_re"]
means = {}
stds = {}
for col in numeric_cols:
    vals = df[col].dropna().astype(float)
    means[col] = vals.mean()
    stds[col] = vals.std()

print("\n--- 8-Seed Mean / Std ---")
for col in numeric_cols:
    print(f"  {col}: mean={means[col]:.4f}, std={stds[col]:.4f}")

print("\n--- Comparison with student-reported ---")
student = {"p_in": 12.81, "p_oodc": 7.91, "p_tan": 31.10, "p_re": 6.49}
for col in numeric_cols:
    diff = means[col] - student[col]
    print(f"  {col}: computed={means[col]:.4f} vs student={student[col]:.2f}  delta={diff:+.4f}")

# Also print all available summary keys for first run to help with key discovery
first_run_id = list(run_ids.values())[0]
run0 = api.run(f"{entity}/{project}/{first_run_id}")
print(f"\n--- All summary keys for run {first_run_id} ---")
surf_keys = [k for k in run0.summary_metrics.keys() if "surf" in k.lower() or "mae" in k.lower() or "val_" in k]
for k in sorted(surf_keys):
    print(f"  {k}: {run0.summary_metrics[k]}")
