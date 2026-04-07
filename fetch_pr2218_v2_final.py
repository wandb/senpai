import wandb
import os

api = wandb.Api()
entity = "wandb-applied-ai-team"
project = "senpai-v1"

run_ids = ["rqpfsjey", "v56f12x9"]
target_keys = [
    "best_best_val_p_in/mae_surf_p",
    "best_best_val_p_oodc/mae_surf_p",
    "best_best_val_p_tan/mae_surf_p",
    "best_best_val_p_re/mae_surf_p",
]

reported = {
    "rqpfsjey": {"p_in": 12.334, "p_oodc": 7.680, "p_tan": 28.422, "p_re": 6.077},
    "v56f12x9": {"p_in": 12.130, "p_oodc": 7.761, "p_tan": 28.025, "p_re": 5.995},
}

seed_labels = {
    "rqpfsjey": "seed 42",
    "v56f12x9": "seed 73",
}

verified = {}

for run_id in run_ids:
    print(f"\n{'='*60}")
    print(f"Run: {run_id} ({seed_labels[run_id]})")
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")

        # Print all best_best_val* keys
        print(f"\n  All best_best_val* keys in summary:")
        bb_found = {}
        for k, v in run.summary_metrics.items():
            if "best_best_val" in k and "mae_surf_p" in k:
                print(f"    {k}: {v}")
                bb_found[k] = v

        if not bb_found:
            print("    (none found — checking all summary keys)")
            for k, v in run.summary_metrics.items():
                if "mae" in k.lower() or "surf" in k.lower():
                    print(f"    {k}: {v}")

        # Discrepancy check
        print(f"\n  Discrepancy check (reported vs W&B):")
        key_map = {
            "p_in":   "best_best_val_p_in/mae_surf_p",
            "p_oodc": "best_best_val_p_oodc/mae_surf_p",
            "p_tan":  "best_best_val_p_tan/mae_surf_p",
            "p_re":   "best_best_val_p_re/mae_surf_p",
        }
        rep = reported[run_id]
        run_verified = {}
        for metric, wb_key in key_map.items():
            rep_val = rep[metric]
            wb_val = run.summary_metrics.get(wb_key)
            run_verified[metric] = wb_val
            if wb_val is not None:
                diff = abs(rep_val - wb_val)
                flag = "  *** DISCREPANCY > 0.1 ***" if diff > 0.1 else "  OK"
                print(f"    {metric}: reported={rep_val:.3f}, wandb={wb_val:.3f}, diff={diff:.3f}{flag}")
            else:
                print(f"    {metric}: reported={rep_val:.3f}, wandb=NOT FOUND")
        verified[run_id] = run_verified

    except Exception as e:
        print(f"  ERROR: {e}")

print("\n\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Run':<12} {'Metric':<10} {'Reported':>10} {'W&B':>10} {'Diff':>8} {'Status'}")
print("-"*60)
for run_id in run_ids:
    rep = reported[run_id]
    ver = verified.get(run_id, {})
    key_map = {"p_in": "p_in", "p_oodc": "p_oodc", "p_tan": "p_tan", "p_re": "p_re"}
    for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
        rep_val = rep[metric]
        wb_val = ver.get(metric)
        if wb_val is not None:
            diff = abs(rep_val - wb_val)
            status = "DISCREPANCY" if diff > 0.1 else "OK"
            print(f"{run_id:<12} {metric:<10} {rep_val:>10.3f} {wb_val:>10.3f} {diff:>8.3f}  {status}")
        else:
            print(f"{run_id:<12} {metric:<10} {rep_val:>10.3f} {'N/A':>10} {'N/A':>8}  NOT FOUND")

print("\nDone.")
