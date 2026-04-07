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

    # Print summary
    summary = dict(run.summary)
    # Show all keys
    print(f"  Summary key count: {len(summary)}")
    # Find MAE-related
    mae_keys = {k: v for k, v in summary.items() if 'mae' in k.lower()}
    print("  MAE keys in summary:", mae_keys)
    # Show a sample of all keys
    all_keys = sorted(summary.keys())
    print("  Sample keys:", all_keys[:20])
    # Look for p_ keys specifically
    p_keys = {k: v for k, v in summary.items() if k.startswith('val/')}
    print("  val/ keys in summary:", dict(list(p_keys.items())[:20]))
