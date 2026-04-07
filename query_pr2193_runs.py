import wandb
import os
import json

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["y0mce10q", "nv7ahjp4"]
labels = {"y0mce10q": "Seed 42", "nv7ahjp4": "Seed 73"}

metric_keys = [
    "best/val_in_dist/mae_surf_p",
    "best/val_tandem_transfer/mae_surf_p",
    "best/val_ood_cond/mae_surf_p",
    "best/val_ood_re/mae_surf_p",
    # fallback keys without 'best/' prefix
    "val_in_dist/mae_surf_p",
    "val_tandem_transfer/mae_surf_p",
    "val_ood_cond/mae_surf_p",
    "val_ood_re/mae_surf_p",
]

for run_id in run_ids:
    print(f"\n{'='*60}")
    print(f"Run ID: {run_id} ({labels[run_id]})")
    print(f"{'='*60}")
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"Name: {run.name}")
        print(f"State: {run.state}")

        summary = run.summary_metrics
        print(f"\n--- Summary metrics (all keys with 'mae' or 'surf') ---")
        found_keys = {k: v for k, v in summary.items() if 'mae' in k.lower() or 'surf' in k.lower()}
        for k, v in sorted(found_keys.items()):
            print(f"  {k}: {v}")

        print(f"\n--- Targeted metric keys ---")
        for key in metric_keys:
            val = summary.get(key, "NOT FOUND")
            print(f"  {key}: {val}")

        # Check epochs
        epoch_val = summary.get("epoch", summary.get("_step", "N/A"))
        trainer_epoch = summary.get("trainer/global_step", "N/A")
        print(f"\n--- Training progress ---")
        print(f"  epoch (summary): {epoch_val}")
        print(f"  trainer/global_step: {trainer_epoch}")

        # Check VRAM
        print(f"\n--- VRAM / system metrics ---")
        vram_keys = {k: v for k, v in summary.items() if 'vram' in k.lower() or 'memory' in k.lower() or 'gpu' in k.lower()}
        if vram_keys:
            for k, v in sorted(vram_keys.items()):
                print(f"  {k}: {v}")
        else:
            print("  No VRAM/GPU keys in summary")

        # Print all summary keys to help with debugging
        print(f"\n--- All summary keys (first 60) ---")
        all_keys = sorted(summary.keys())
        for k in all_keys[:60]:
            print(f"  {k}: {summary.get(k)}")

    except Exception as e:
        print(f"ERROR fetching run {run_id}: {e}")
