import wandb
import os

api = wandb.Api()
entity = "senpai-wandb"
project = "senpai"

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

for run_id in run_ids:
    print(f"\n{'='*60}")
    print(f"Run: {run_id}")
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")

        # Check summary_metrics for the target keys
        print(f"\n  Summary metrics (target keys):")
        found = {}
        for key in target_keys:
            val = run.summary_metrics.get(key)
            print(f"    {key}: {val}")
            if val is not None:
                # extract the suffix: p_in, p_oodc, p_tan, p_re
                suffix = key.split("best_best_val_")[1].split("/")[0]  # e.g. p_in
                found[suffix] = val

        # Also scan all summary keys for anything matching best_best_val
        print(f"\n  All best_best_val* keys in summary:")
        for k, v in run.summary_metrics.items():
            if "best_best_val" in k:
                print(f"    {k}: {v}")

        # Compare with reported values
        print(f"\n  Discrepancy check (reported vs W&B):")
        rep = reported[run_id]
        key_map = {"p_in": "best_best_val_p_in/mae_surf_p",
                   "p_oodc": "best_best_val_p_oodc/mae_surf_p",
                   "p_tan": "best_best_val_p_tan/mae_surf_p",
                   "p_re": "best_best_val_p_re/mae_surf_p"}
        for metric, wb_key in key_map.items():
            rep_val = rep[metric]
            wb_val = run.summary_metrics.get(wb_key)
            if wb_val is not None:
                diff = abs(rep_val - wb_val)
                flag = " *** DISCREPANCY > 0.1 ***" if diff > 0.1 else ""
                print(f"    {metric}: reported={rep_val:.3f}, wandb={wb_val:.3f}, diff={diff:.3f}{flag}")
            else:
                print(f"    {metric}: reported={rep_val:.3f}, wandb=NOT FOUND")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone.")
