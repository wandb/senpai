import wandb
import sys

api = wandb.Api()
entity = "wandb-applied-ai-team"
project = "senpai-v1"
path = f"{entity}/{project}"

target_runs = ["cuehr0b0", "nqkaw1cf"]

for run_id in target_runs:
    run = api.run(f"{path}/{run_id}")
    print(f"\n=== Run: {run_id} ({run.name}) state={run.state} ===")

    # Print all summary keys with 'mae' or 'p_' in them
    summary = run.summary._json_dict if hasattr(run.summary, '_json_dict') else {}
    print("  All summary keys:", sorted(summary.keys())[:30])
    matched = {k: v for k, v in summary.items() if 'mae' in k.lower()}
    print("  MAE in summary:", matched)

    # Scan a tiny slice of history to find key names
    sample_rows = list(run.scan_history(page_size=3))
    if sample_rows:
        all_hist_keys = set()
        for row in sample_rows:
            all_hist_keys.update(row.keys())
        mae_hist_keys = sorted(k for k in all_hist_keys if 'mae' in k.lower())
        print("  MAE keys in history:", mae_hist_keys)
    else:
        print("  No history rows.")
