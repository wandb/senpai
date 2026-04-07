import wandb
import sys

api = wandb.Api()
entity = "wandb-applied-ai-team"
project = "senpai-v1"
path = f"{entity}/{project}"

target_runs = ["cuehr0b0", "nqkaw1cf"]

metric_keys = [
    "val/best_p_in_mae",
    "val/best_p_oodc_mae",
    "val/best_p_tan_mae",
    "val/best_p_re_mae",
]

for run_id in target_runs:
    run = api.run(f"{path}/{run_id}")
    print(f"\n=== Run: {run_id} ({run.name}) state={run.state} ===")

    # Check all summary keys that match our interest
    summary = run.summary._json_dict if hasattr(run.summary, '_json_dict') else dict(run.summary)
    matched = {k: v for k, v in summary.items() if 'mae' in k.lower() or 'p_in' in k.lower() or 'p_oo' in k.lower() or 'p_tan' in k.lower() or 'p_re' in k.lower()}
    print("  Summary MAE keys:", matched)

    # Also sample last few history rows to find key names
    hist = list(run.scan_history(keys=metric_keys, page_size=1000))
    if hist:
        # Get max across all steps
        import numpy as np
        for k in metric_keys:
            vals = [r[k] for r in hist if k in r and r[k] is not None]
            if vals:
                print(f"  {k}: min={min(vals):.4f}, last={vals[-1]:.4f}, n={len(vals)}")
            else:
                print(f"  {k}: no data")
    else:
        print("  No history with those keys. Checking actual key names...")
        # Sample a few rows to find key names
        sample = list(run.scan_history(page_size=5))
        if sample:
            keys_in_hist = set()
            for row in sample:
                keys_in_hist.update(row.keys())
            mae_keys = [k for k in keys_in_hist if 'mae' in k.lower()]
            print("  MAE keys in history:", sorted(mae_keys))
        else:
            print("  No history rows found.")
