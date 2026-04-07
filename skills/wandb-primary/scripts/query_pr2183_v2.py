import os
import wandb

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = {
    "vw3jpr15": "Config A w=0.05, seed 42",
    "lo2dayem": "Config A w=0.05, seed 73",
    "avivx0eh": "Config B w=0.1,  seed 42",
    "9afvei72": "Config B w=0.1,  seed 73",
}

# Mapping from split key to readable name matching the baseline convention
# p_in  = in_dist surface p MAE
# p_oodc = ood_cond surface p MAE
# p_tan = tandem_transfer surface p MAE
# p_re  = ood_re surface p MAE
SPLIT_MAP = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

# Also grab final-epoch values for comparison
FINAL_MAP = {
    "p_in_final":   "val_in_dist/mae_surf_p",
    "p_oodc_final": "val_ood_cond/mae_surf_p",
    "p_tan_final":  "val_tandem_transfer/mae_surf_p",
    "p_re_final":   "val_ood_re/mae_surf_p",
}

BASELINE = {
    "p_in":   13.205,
    "p_oodc":  7.816,
    "p_tan":  28.502,
    "p_re":    6.453,
}

print(f"Entity: {entity}, Project: {project}")
print()

rows = []
for run_id, label in run_ids.items():
    run = api.run(f"{entity}/{project}/{run_id}")
    s = run.summary_metrics

    row = {
        "run_id": run_id,
        "label": label,
        "state": run.state,
        "best_epoch": s.get("best_epoch"),
        "total_epochs": s.get("total_epochs"),
        "total_time_min": s.get("total_time_min"),
    }

    for k, wkey in SPLIT_MAP.items():
        row[k] = s.get(wkey)

    for k, wkey in FINAL_MAP.items():
        row[k] = s.get(wkey)

    rows.append(row)

# Print summary table
print("=" * 100)
print(f"{'Run ID':<12} {'Label':<26} {'State':<10} {'BestEp':>7} {'TotEp':>6} | {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8}")
print("=" * 100)
for r in rows:
    p_in   = r['p_in']   or float('nan')
    p_oodc = r['p_oodc'] or float('nan')
    p_tan  = r['p_tan']  or float('nan')
    p_re   = r['p_re']   or float('nan')
    print(f"{r['run_id']:<12} {r['label']:<26} {r['state']:<10} {str(r['best_epoch']):>7} {str(r['total_epochs']):>6} | {p_in:>8.3f} {p_oodc:>8.3f} {p_tan:>8.3f} {p_re:>8.3f}")

print("-" * 100)
print(f"{'BASELINE':^12} {'':<26} {'':<10} {'':>7} {'':>6} | {BASELINE['p_in']:>8.3f} {BASELINE['p_oodc']:>8.3f} {BASELINE['p_tan']:>8.3f} {BASELINE['p_re']:>8.3f}")
print("=" * 100)

# Per-run delta vs baseline
print()
print("Delta vs baseline (negative = improvement):")
for r in rows:
    deltas = {k: (r[k] or float('nan')) - BASELINE[k] for k in ["p_in", "p_oodc", "p_tan", "p_re"]}
    print(f"  {r['run_id']} ({r['label']}): p_in={deltas['p_in']:+.3f}, p_oodc={deltas['p_oodc']:+.3f}, p_tan={deltas['p_tan']:+.3f}, p_re={deltas['p_re']:+.3f}")

# Also check completion status
print()
print("Completion check:")
for r in rows:
    completed = r['state'] == 'finished'
    stopped_early = r.get('best_epoch') is not None and r.get('total_epochs') is not None and r['best_epoch'] < r['total_epochs'] - 1
    print(f"  {r['run_id']}: state={r['state']}, best_epoch={r['best_epoch']}, total_epochs={r['total_epochs']}, time_min={r.get('total_time_min'):.1f}")
