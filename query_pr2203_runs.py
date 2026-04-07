import wandb
import os

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "senpai-advisory")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = ["kylbayco", "fhlwq5hr"]

for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"\n=== Run: {run_id} ===")
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")
        print(f"  Tags: {run.tags}")

        summary = run.summary_metrics

        # Look for surface MAE metrics
        surface_keys = [k for k in summary.keys() if 'surface' in k.lower() or 'p_in' in k or 'p_oodc' in k or 'p_tan' in k or 'p_re' in k or 'mae' in k.lower()]
        print(f"\n  Surface MAE keys found: {surface_keys}")

        for k in surface_keys:
            print(f"    {k}: {summary[k]}")

        # Also try common key patterns
        for key in ['surface_mae/p_in', 'surface_mae/p_oodc', 'surface_mae/p_tan', 'surface_mae/p_re',
                    'p_in', 'p_oodc', 'p_tan', 'p_re',
                    'val/surface_mae/p_in', 'val/surface_mae/p_oodc', 'val/surface_mae/p_tan', 'val/surface_mae/p_re']:
            val = summary.get(key)
            if val is not None:
                print(f"    [direct] {key}: {val}")

        # Print all summary keys for inspection
        print(f"\n  All summary keys:")
        for k, v in sorted(summary.items()):
            if not k.startswith('_'):
                print(f"    {k}: {v}")

    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
