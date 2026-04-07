import os
import wandb
import numpy as np

# Use WANDB_ENTITY / WANDB_PROJECT from env
entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
project = os.environ.get("WANDB_PROJECT", "senpai-v1")

print(f"Entity: {entity}, Project: {project}")

# Disable login verification
os.environ["WANDB_SILENT"] = "true"

api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))

run_ids = ["evz9po4m", "fdodi6m3", "ldf4sof7", "d7l91p0x", "etepxvjc"]
labels = {
    "evz9po4m": "GEPS TTA 10-step s73",
    "fdodi6m3": "GEPS TTA 20-step s42",
    "ldf4sof7": "GEPS TTA 20-step s73",
    "d7l91p0x": "DCT freq loss s42 (baseline)",
    "etepxvjc": "DCT freq loss s73 (baseline)",
}

WB_KEYS = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
    "epoch":  "epoch",
    "_step":  "_step",
    "runtime": "_runtime",
}

print(f"Fetching runs from {entity}/{project}\n")

rows = []
for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        summary = run.summary_metrics

        data = {
            "run_id": run_id,
            "label": labels[run_id],
            "state": run.state,
            "name": run.name,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
        }

        for key, wb_key in WB_KEYS.items():
            data[key] = summary.get(wb_key)

        rows.append(data)

        # Also print all summary keys for inspection
        print(f"Run {run_id} ({labels[run_id]}):")
        print(f"  state={run.state}, name={run.name}")
        print(f"  created_at={run.created_at}")
        mae_keys = {k: v for k, v in summary.items() if "mae" in k.lower() or "epoch" in k.lower() or "runtime" in k.lower() or "_step" in k.lower()}
        print(f"  MAE/epoch keys: {mae_keys}")
        print()

    except Exception as e:
        print(f"Run {run_id}: ERROR - {e}")
        import traceback; traceback.print_exc()

print("\nSummary:")
for r in rows:
    print(f"  {r['run_id']} | {r['label'][:30]:<30} | state={r['state']} | epoch={r['epoch']} | p_in={r['p_in']} | p_oodc={r['p_oodc']} | p_tan={r['p_tan']} | p_re={r['p_re']} | runtime={r['runtime']}")

print("\nDone.")
