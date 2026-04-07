import os
import sys
import wandb
import json

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["d7l91p0x", "j9btfx09"]

for run_id in run_ids:
    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    print('='*60)
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"Name: {run.name}")
        print(f"State: {run.state}")

        # Get all summary metrics
        summary = run.summary_metrics

        # Find pressure surface MAE metrics
        print("\n--- Pressure Surface MAE metrics ---")
        pressure_keys = [k for k in summary.keys() if 'mae_surf_p' in k or ('mae' in k and 'surf' in k)]
        pressure_keys.sort()
        for k in pressure_keys:
            print(f"  {k}: {summary.get(k)}")

        # Also check best_* metrics
        print("\n--- Best_* metrics ---")
        best_keys = [k for k in summary.keys() if k.startswith('best_')]
        best_keys.sort()
        for k in best_keys:
            print(f"  {k}: {summary.get(k)}")

        # All summary keys for completeness
        print("\n--- All summary keys (filtered for relevant) ---")
        relevant_keys = [k for k in summary.keys() if any(x in k for x in ['p_in', 'p_oodc', 'p_tan', 'p_re', 'mae', 'epoch', 'step'])]
        relevant_keys.sort()
        for k in relevant_keys:
            print(f"  {k}: {summary.get(k)}")

        # Epochs
        print(f"\n--- Epoch info ---")
        epoch_val = summary.get('epoch', 'N/A')
        print(f"  epoch (summary): {epoch_val}")

        # Check config for flags
        print("\n--- Config flags ---")
        config = run.config
        flags_to_check = ['gap_stagger_spatial_bias', 'pcgrad_3way', 'pcgrad', 'seed']
        for flag in flags_to_check:
            val = config.get(flag, 'NOT FOUND')
            print(f"  {flag}: {val}")

        # Print all config keys that might be relevant
        print("\n--- All config keys (filtered) ---")
        relevant_config = {k: v for k, v in config.items() if any(x in k.lower() for x in ['grad', 'bias', 'spatial', 'gap', 'stagger', 'seed', 'epoch', 'lr'])}
        for k, v in sorted(relevant_config.items()):
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"ERROR: {e}")

print("\n\nDone.")
