import os
import wandb

api = wandb.Api(timeout=30)
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["evz9po4m", "fdodi6m3", "ldf4sof7", "d7l91p0x", "etepxvjc"]
labels = {
    "evz9po4m": "GEPS TTA 10-step s73",
    "fdodi6m3": "GEPS TTA 20-step s42",
    "ldf4sof7": "GEPS TTA 20-step s73",
    "d7l91p0x": "DCT freq loss s42 (baseline)",
    "etepxvjc": "DCT freq loss s73 (baseline)",
}

for run_id in run_ids:
    print(f"\n{'='*60}")
    print(f"Run: {run_id} | {labels[run_id]}")
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        summary = dict(run.summary_metrics)
        print(f"  state={run.state}, name={run.name}")
        print(f"  created_at={run.created_at}, updated_at={run.updated_at}")
        # Print all keys
        for k in sorted(summary.keys()):
            print(f"  {k}: {summary[k]}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone.")
