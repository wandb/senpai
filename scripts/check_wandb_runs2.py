import os
import wandb

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "wild-ai")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = ["p9oupnt2", "6sp2uazt", "kov6n0rs", "l308y9lx"]

# Mapping: student label -> (best_ key, val_ key)
# p_in  = in-distribution pressure surface MAE
# p_oodc = ood_cond pressure surface MAE
# p_tan  = tandem_transfer pressure surface MAE
# p_re   = ood_re pressure surface MAE

key_map = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

# Also grab val_ (final epoch) vs best_best_ (best epoch)
val_key_map = {
    "p_in":   "val_in_dist/mae_surf_p",
    "p_oodc": "val_ood_cond/mae_surf_p",
    "p_tan":  "val_tandem_transfer/mae_surf_p",
    "p_re":   "val_ood_re/mae_surf_p",
}

student_reported = {
    "p9oupnt2": {"p_in": 13.5, "p_oodc": 7.6, "p_tan": 30.0, "p_re": 6.4},
    "6sp2uazt": {"p_in": 12.9, "p_oodc": 7.5, "p_tan": 29.4, "p_re": 6.4},
    "kov6n0rs": {"p_in": 13.5, "p_oodc": 7.6, "p_tan": 30.0, "p_re": 6.4},
    "l308y9lx": {"p_in": 13.4, "p_oodc": 7.8, "p_tan": 29.0, "p_re": 6.4},
}

labels = {
    "p9oupnt2": "pcgrad_3way pct=0.15 s42",
    "6sp2uazt": "pcgrad_3way pct=0.15 s73",
    "kov6n0rs": "pcgrad_3way pct=0.10 s42",
    "l308y9lx": "pcgrad_3way pct=0.10 s73",
}

print("=" * 110)
print(f"{'Run ID':<12} {'Config':<28} {'Metric':<8} {'Best(W&B)':>12} {'Final(W&B)':>12} {'Reported':>10} {'Match?':>8}")
print("=" * 110)

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    summary = run.summary_metrics
    config_label = labels[run_id]
    reported = student_reported[run_id]

    for metric_label, best_key in key_map.items():
        val_key = val_key_map[metric_label]
        best_val = summary.get(best_key)
        final_val = summary.get(val_key)
        rep_val = reported[metric_label]

        def fmt(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"

        if isinstance(best_val, (int, float)):
            match = "OK" if abs(best_val - rep_val) < 0.15 else "MISMATCH"
        elif isinstance(final_val, (int, float)):
            match = "OK" if abs(final_val - rep_val) < 0.15 else "MISMATCH"
        else:
            match = "N/A"

        print(f"{run_id:<12} {config_label:<28} {metric_label:<8} {fmt(best_val):>12} {fmt(final_val):>12} {fmt(rep_val):>10} {match:>8}")

    print("-" * 110)

print("\n\n=== RAW BEST METRICS PER RUN ===")
for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    summary = run.summary_metrics
    print(f"\n{run_id} ({labels[run_id]}) — state={run.state}")
    print(f"  best_best_val_in_dist/mae_surf_p       = {summary.get('best_best_val_in_dist/mae_surf_p')}")
    print(f"  best_best_val_ood_cond/mae_surf_p      = {summary.get('best_best_val_ood_cond/mae_surf_p')}")
    print(f"  best_best_val_tandem_transfer/mae_surf_p = {summary.get('best_best_val_tandem_transfer/mae_surf_p')}")
    print(f"  best_best_val_ood_re/mae_surf_p        = {summary.get('best_best_val_ood_re/mae_surf_p')}")
    print(f"  val_in_dist/mae_surf_p                 = {summary.get('val_in_dist/mae_surf_p')}")
    print(f"  val_ood_cond/mae_surf_p                = {summary.get('val_ood_cond/mae_surf_p')}")
    print(f"  val_tandem_transfer/mae_surf_p         = {summary.get('val_tandem_transfer/mae_surf_p')}")
    print(f"  val_ood_re/mae_surf_p                  = {summary.get('val_ood_re/mae_surf_p')}")
