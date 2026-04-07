import wandb
import os

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["cor600he", "5pfgo0bx"]
metric_keys = ["p_in", "p_oodc", "p_tan", "p_re", "val/loss"]

print(f"Querying W&B: {entity}/{project}")
print("=" * 60)

results = {}
for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        summary = run.summary_metrics
        print(f"\nRun: {run_id}")
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")
        print(f"  Created: {run.created_at}")

        metrics = {}
        for key in metric_keys:
            val = summary.get(key)
            if val is not None:
                metrics[key] = val
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: NOT FOUND")

        # Also check for any surface MAE keys with different names
        all_keys = [k for k in summary.keys() if any(x in k.lower() for x in ["p_in", "p_oodc", "p_tan", "p_re", "surface", "mae", "pressure"])]
        if all_keys:
            print(f"  Related keys: {all_keys}")

        results[run_id] = metrics
    except Exception as e:
        print(f"\nRun {run_id}: ERROR - {e}")

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Metric':<15} {'cor600he (s42)':>18} {'5pfgo0bx (s73)':>18} {'2-seed avg':>15}")
print("-" * 70)

for key in metric_keys:
    v42 = results.get("cor600he", {}).get(key)
    v73 = results.get("5pfgo0bx", {}).get(key)
    if v42 is not None and v73 is not None:
        avg = (v42 + v73) / 2
        print(f"{key:<15} {v42:>18.4f} {v73:>18.4f} {avg:>15.4f}")
    elif v42 is not None:
        print(f"{key:<15} {v42:>18.4f} {'N/A':>18} {'N/A':>15}")
    elif v73 is not None:
        print(f"{key:<15} {'N/A':>18} {v73:>18.4f} {'N/A':>15}")
    else:
        print(f"{key:<15} {'N/A':>18} {'N/A':>18} {'N/A':>15}")

print("\nBaseline (PR #2207): p_in=12.490, p_oodc=7.618, p_tan=28.521, p_re=6.411")
