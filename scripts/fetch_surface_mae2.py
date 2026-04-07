import wandb
import pandas as pd
import numpy as np

api = wandb.Api()
project = "wandb-applied-ai-team/senpai-v1"

run_ids = [
    "zosxwjmm",  # PR 2127 seed 42
    "twilqf1x",  # PR 2127 seed 73
    "np3anmah",  # PR 2128 control s42
    "fcri663y",  # PR 2128 control s73
    "7bnlke5o",  # PR 2128 FiLM s42
    "yh5ixlx8",  # PR 2128 FiLM s73
]

labels = {
    "zosxwjmm": "PR#2127 aft-srf-knn / s42",
    "twilqf1x": "PR#2127 aft-srf-knn / s73",
    "np3anmah": "PR#2128 control / s42",
    "fcri663y": "PR#2128 control / s73",
    "7bnlke5o": "PR#2128 FiLM / s42",
    "yh5ixlx8": "PR#2128 FiLM / s73",
}

# Mapping from shorthand to W&B key prefix (using best_ = best checkpoint)
# p_in  -> val_in_dist  -> best_best_val_in_dist/mae_surf_p
# p_oodc -> val_ood_cond -> best_best_val_ood_cond/mae_surf_p
# p_tan  -> val_tandem_transfer -> best_best_val_tandem_transfer/mae_surf_p
# p_re   -> val_ood_re  -> best_best_val_ood_re/mae_surf_p

metric_map = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

# Also check final (non-best) values as fallback
metric_map_final = {
    "p_in":   "val_in_dist/mae_surf_p",
    "p_oodc": "val_ood_cond/mae_surf_p",
    "p_tan":  "val_tandem_transfer/mae_surf_p",
    "p_re":   "val_ood_re/mae_surf_p",
}

results = []

for run_id in run_ids:
    run = api.run(f"{project}/{run_id}")
    summary = run.summary_metrics
    row = {"run_id": run_id, "label": labels[run_id], "state": run.state}

    for metric, key in metric_map.items():
        val = summary.get(key)
        if val is None:
            # fallback to final value
            val = summary.get(metric_map_final[metric])
        row[metric] = val

    results.append(row)

df = pd.DataFrame(results)
surface_keys = ["p_in", "p_oodc", "p_tan", "p_re"]

print("=== PER-RUN SURFACE MAE (pressure, best checkpoint) ===\n")
display_cols = ["label", "state"] + surface_keys
print(df[display_cols].to_string(index=False))

print("\n\n=== MEAN ACROSS SEEDS ===\n")
groups = {
    "PR#2127 aft-srf-knn": ["zosxwjmm", "twilqf1x"],
    "PR#2128 control":     ["np3anmah", "fcri663y"],
    "PR#2128 FiLM":        ["7bnlke5o", "yh5ixlx8"],
}

mean_rows = []
for group_label, ids in groups.items():
    subset = df[df["run_id"].isin(ids)]
    means = subset[surface_keys].mean()
    row = {"group": group_label}
    for k in surface_keys:
        row[k] = f"{means[k]:.6f}" if not pd.isna(means[k]) else "N/A"
    mean_rows.append(row)

mdf = pd.DataFrame(mean_rows)
print(mdf.to_string(index=False))

# Also print raw values for full fidelity
print("\n\n=== RAW VALUES (full precision) ===")
for _, r in df.iterrows():
    print(f"\n{r['label']} [{r['run_id']}]:")
    for k in surface_keys:
        v = r[k]
        print(f"  {k}: {v:.8f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else f"  {k}: N/A")
