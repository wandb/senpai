import os
import sys
import wandb

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = {
    "vw3jpr15": "Config A w=0.05, seed 42",
    "lo2dayem": "Config A w=0.05, seed 73",
    "avivx0eh": "Config B w=0.1, seed 42",
    "9afvei72": "Config B w=0.1, seed 73",
}

SURFACE_MAE_KEYS = [
    "test/surface_mae_p_in",
    "test/surface_mae_p_tan",
    "test/surface_mae_p_oodc",
    "test/surface_mae_p_re",
    "surface_mae_p_in",
    "surface_mae_p_tan",
    "surface_mae_p_oodc",
    "surface_mae_p_re",
    "p_in",
    "p_tan",
    "p_oodc",
    "p_re",
]

print(f"Entity: {entity}, Project: {project}\n")

for run_id, label in run_ids.items():
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        summary = run.summary_metrics

        print(f"=== Run {run_id} | {label} ===")
        print(f"  State: {run.state}")
        print(f"  Name: {run.name}")

        # Print all summary keys to understand the schema
        print(f"  All summary keys: {sorted(summary.keys())}")

        # Try known metric keys
        found = {}
        for key in SURFACE_MAE_KEYS:
            val = summary.get(key)
            if val is not None:
                found[key] = val

        if found:
            print(f"  Found metrics: {found}")
        else:
            print("  WARNING: No surface MAE keys found in summary — check key names above")

        print()
    except Exception as e:
        print(f"=== Run {run_id} | {label} ===")
        print(f"  ERROR: {e}")
        print()
