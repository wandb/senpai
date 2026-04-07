import os
import sys
sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

import wandb
import pandas as pd

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["obn1wfja", "52irfwwg"]

metrics_of_interest = [
    "val_tandem_transfer/mae_surf_p",
    "val_in_dist/mae_surf_p",
    "val_ood_cond/mae_surf_p",
    "val_ood_re/mae_surf_p",
]

for run_id in run_ids:
    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"  Name:    {run.name}")
        print(f"  State:   {run.state}")
        print(f"  Created: {run.created_at}")

        # Runtime
        duration = run.summary_metrics.get("_runtime", None)
        if duration:
            print(f"  Runtime: {duration/3600:.2f} hours ({duration:.0f} seconds)")
        else:
            print(f"  Runtime: not available")

        print(f"\n  Summary metrics (final values):")
        for m in metrics_of_interest:
            val = run.summary_metrics.get(m, "NOT FOUND")
            print(f"    {m}: {val}")

        # Also check best_ variants
        print(f"\n  Best metrics (if tracked):")
        for m in metrics_of_interest:
            best_key = m.replace("val_", "best_val_")
            val = run.summary_metrics.get(best_key, "NOT FOUND")
            if val != "NOT FOUND":
                print(f"    {best_key}: {val}")

        # Dump all summary keys containing mae_surf_p
        print(f"\n  All summary keys with 'mae_surf_p':")
        for k, v in sorted(run.summary_metrics.items()):
            if "mae_surf_p" in k.lower():
                print(f"    {k}: {v}")

        # Also check for any p_tan, p_in keys
        print(f"\n  All summary keys with 'p_tan' or 'p_in' or 'p_oodc' or 'p_re':")
        for k, v in sorted(run.summary_metrics.items()):
            kl = k.lower()
            if any(x in kl for x in ["p_tan", "p_in", "p_oodc", "p_re"]):
                print(f"    {k}: {v}")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone.")
