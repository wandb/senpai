"""Quick verify W&B metrics for PR #2202 runs."""
import os
import sys
import wandb

api = wandb.Api(timeout=30)

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

for run_id in ["dq0blopc", "a0d4jbyz"]:
    print(f"\n=== Run: {run_id} ===")
    try:
        run = api.run(f"{path}/{run_id}")
        print(f"Name: {run.name}")
        print(f"State: {run.state}")
        print(f"Config (seed): {run.config.get('seed', 'N/A')}")
        # Show all summary metrics
        summary = dict(run.summary_metrics)
        # Filter for surface/MAE metrics
        mae_summary = {k: v for k, v in summary.items()
                       if any(x in k.lower() for x in ["mae", "p_in", "p_tan", "p_oodc", "p_re", "surface"])}
        print(f"MAE summary metrics:")
        for k, v in sorted(mae_summary.items()):
            print(f"  {k}: {v}")
        if not mae_summary:
            print(f"  (no MAE metrics in summary — all keys: {list(summary.keys())[:30]})")
    except Exception as e:
        print(f"Error: {e}")

print("\nDone.")
