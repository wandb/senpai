import os
import wandb

api = wandb.Api()

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["ecxe1ti3", "29xvq9bz"]

# Mapping from the 4 splits to the metric keys in W&B
# p_in  = val_in_dist/mae_surf_p  (best_ prefixed for best epoch)
# p_tan = val_tandem_transfer/mae_surf_p
# p_oodc = val_ood_cond/mae_surf_p
# p_re  = val_ood_re/mae_surf_p

split_map = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

# Also the "current epoch" versions (final epoch values)
split_map_final = {
    "p_in":   "val_in_dist/mae_surf_p",
    "p_tan":  "val_tandem_transfer/mae_surf_p",
    "p_oodc": "val_ood_cond/mae_surf_p",
    "p_re":   "val_ood_re/mae_surf_p",
}

baseline = {
    "p_in": 13.24,
    "p_tan": 30.53,
    "p_oodc": 7.73,
    "p_re": 6.50,
}

print(f"{'Metric':<10} {'Baseline':>10} {'ecxe1ti3 (s42)':>16} {'29xvq9bz (s43)':>16} {'Avg':>10}")
print("-" * 65)

all_results = {}
for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    summary = run.summary_metrics
    all_results[run_id] = {}

    for metric, key in split_map.items():
        val = summary.get(key)
        all_results[run_id][metric] = val

for metric in ["p_in", "p_tan", "p_oodc", "p_re"]:
    v42 = all_results["ecxe1ti3"].get(metric)
    v43 = all_results["29xvq9bz"].get(metric)
    base = baseline[metric]
    avg = (v42 + v43) / 2 if (v42 is not None and v43 is not None) else None
    v42_str = f"{v42:.4f}" if v42 is not None else "N/A"
    v43_str = f"{v43:.4f}" if v43 is not None else "N/A"
    avg_str = f"{avg:.4f}" if avg is not None else "N/A"
    print(f"{metric:<10} {base:>10.2f} {v42_str:>16} {v43_str:>16} {avg_str:>10}")

print()
print("Delta vs baseline (negative = improvement):")
print(f"{'Metric':<10} {'d(s42)':>10} {'d(s43)':>10} {'d(avg)':>10}")
print("-" * 45)
for metric in ["p_in", "p_tan", "p_oodc", "p_re"]:
    v42 = all_results["ecxe1ti3"].get(metric)
    v43 = all_results["29xvq9bz"].get(metric)
    base = baseline[metric]
    avg = (v42 + v43) / 2 if (v42 is not None and v43 is not None) else None
    d42 = (v42 - base) if v42 is not None else None
    d43 = (v43 - base) if v43 is not None else None
    davg = (avg - base) if avg is not None else None
    d42_str = f"{d42:+.4f}" if d42 is not None else "N/A"
    d43_str = f"{d43:+.4f}" if d43 is not None else "N/A"
    davg_str = f"{davg:+.4f}" if davg is not None else "N/A"
    print(f"{metric:<10} {d42_str:>10} {d43_str:>10} {davg_str:>10}")

print()
print("=== Additional context ===")
for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    summary = run.summary_metrics
    name = run.name
    best_epoch = summary.get("best_epoch")
    total_epochs = summary.get("total_epochs")
    best_val_loss = summary.get("best_val_loss")
    re_sigma = summary.get("aug/re_sigma")
    print(f"\n{run_id} ({name}):")
    print(f"  best_epoch={best_epoch}, total_epochs={total_epochs}")
    print(f"  best_val_loss={best_val_loss}")
    print(f"  aug/re_sigma={re_sigma}")
