import os
import wandb
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["zapen0x3", "3vuz3adi", "g1uhcorj", "ea551p6b", "fdf1vsi3", "al6opl9g", "jg15oow3", "vzm3s42y"]

# Based on discovered keys:
# p_in  -> best_best_val_in_dist/mae_surf_p
# p_oodc -> best_best_val_ood_cond/mae_surf_p
# p_tan  -> best_best_val_tandem_transfer/mae_surf_p
# p_re   -> best_best_val_ood_re/mae_surf_p

key_map = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

results = {}

for rid in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{rid}")
        sm = run.summary_metrics
        row = {s: sm.get(k) for s, k in key_map.items()}
        results[rid] = row
    except Exception as e:
        print(f"{rid}: ERROR - {e}")
        results[rid] = {s: None for s in key_map}

print("--- Per-run table ---")
print(f"{'Run':<12} {'p_in':>10} {'p_oodc':>10} {'p_tan':>10} {'p_re':>10}")
for rid in run_ids:
    row = results.get(rid, {})
    vals = [row.get(s) for s in key_map]
    formatted = [f"{v:.4f}" if v is not None else "    NaN   " for v in vals]
    print(f"{rid:<12} {formatted[0]:>10} {formatted[1]:>10} {formatted[2]:>10} {formatted[3]:>10}")

print("\n--- Aggregate stats (mean ± std) ---")
for s in key_map:
    vals = [results[rid][s] for rid in run_ids if results.get(rid, {}).get(s) is not None]
    if vals:
        arr = np.array(vals, dtype=float)
        print(f"{s}: mean={arr.mean():.2f} ± std={arr.std():.2f}  (n={len(arr)})")
    else:
        print(f"{s}: no data")

print("\n--- Student reported ---")
print("p_in=13.24±0.33, p_oodc=7.73±0.22, p_tan=30.53±0.50, p_re=6.50±0.07")
