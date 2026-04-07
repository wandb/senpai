import wandb

api = wandb.Api()
entity = "wandb-applied-ai-team"
project = "senpai-v1"

run_ids = ["rqpfsjey", "v56f12x9"]

# Actual W&B key → reported label mapping
# best_best_val_in_dist    → p_in
# best_best_val_ood_cond   → p_oodc
# best_best_val_tandem_transfer → p_tan
# best_best_val_ood_re     → p_re
key_map = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

reported = {
    "rqpfsjey": {"p_in": 12.334, "p_oodc": 7.680, "p_tan": 28.422, "p_re": 6.077},
    "v56f12x9": {"p_in": 12.130, "p_oodc": 7.761, "p_tan": 28.025, "p_re": 5.995},
}

seed_labels = {
    "rqpfsjey": "seed 42",
    "v56f12x9": "seed 73",
}

print("PR #2218 v2 — LE Coordinate Frame, chord-normalized")
print("Metric key mapping: p_in=in_dist, p_oodc=ood_cond, p_tan=tandem_transfer, p_re=ood_re")
print()

verified = {}
discrepancies = []

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    rep = reported[run_id]
    run_verified = {}

    print(f"Run {run_id} ({seed_labels[run_id]}) — {run.name} [{run.state}]")
    print(f"  {'Metric':<8} {'W&B Key':<45} {'Reported':>10} {'W&B':>12} {'Diff':>8} {'Status'}")
    print(f"  {'-'*90}")

    for metric, wb_key in key_map.items():
        wb_val = run.summary_metrics.get(wb_key)
        rep_val = rep[metric]
        run_verified[metric] = wb_val

        if wb_val is not None:
            diff = abs(rep_val - wb_val)
            status = "DISCREPANCY" if diff > 0.1 else "OK"
            if diff > 0.1:
                discrepancies.append({
                    "run_id": run_id, "seed": seed_labels[run_id],
                    "metric": metric, "reported": rep_val, "wandb": wb_val, "diff": diff
                })
            print(f"  {metric:<8} {wb_key:<45} {rep_val:>10.3f} {wb_val:>12.6f} {diff:>8.3f}  {status}")
        else:
            print(f"  {metric:<8} {wb_key:<45} {rep_val:>10.3f} {'NOT FOUND':>12} {'N/A':>8}  MISSING")

    verified[run_id] = run_verified
    print()

print("\n" + "="*60)
print("VERIFIED METRICS (from W&B)")
print("="*60)
print(f"{'Run':<12} {'Seed':<10} {'p_in':>10} {'p_oodc':>10} {'p_tan':>10} {'p_re':>10}")
print("-"*60)
for run_id in run_ids:
    v = verified[run_id]
    s = seed_labels[run_id]
    vals = [v.get(m) for m in ["p_in", "p_oodc", "p_tan", "p_re"]]
    strs = [f"{x:.3f}" if x is not None else "N/A" for x in vals]
    print(f"{run_id:<12} {s:<10} {strs[0]:>10} {strs[1]:>10} {strs[2]:>10} {strs[3]:>10}")

if discrepancies:
    print(f"\n*** {len(discrepancies)} DISCREPANCIES > 0.1 FOUND ***")
    for d in discrepancies:
        print(f"  {d['run_id']} ({d['seed']}) {d['metric']}: reported={d['reported']:.3f}, wandb={d['wandb']:.3f}, diff={d['diff']:.3f}")
else:
    print("\nNo discrepancies > 0.1 found. All reported values match W&B within tolerance.")

print("\nDone.")
