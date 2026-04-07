import os
import wandb
import pandas as pd

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["swsmtceu", "ewrhonfl", "d6x7l5cg", "ivs2xoxz"]

metric_keys = [
    "best_best_val_in_dist/mae_surf_p",
    "best_best_val_ood_cond/mae_surf_p",
    "best_best_val_tandem_transfer/mae_surf_p",
    "best_best_val_ood_re/mae_surf_p",
]

rows = []
for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    sm = run.summary_metrics

    p_in   = sm.get("best_best_val_in_dist/mae_surf_p")
    p_oodc = sm.get("best_best_val_ood_cond/mae_surf_p")
    p_tan  = sm.get("best_best_val_tandem_transfer/mae_surf_p")
    p_re   = sm.get("best_best_val_ood_re/mae_surf_p")

    # Epoch / step info
    cfg_epochs = run.config.get("epochs", run.config.get("num_epochs", run.config.get("SENPAI_MAX_EPOCHS", "N/A")))
    step = sm.get("_step", "N/A")

    rows.append({
        "run_id": run_id,
        "name": run.name,
        "p_in": p_in,
        "p_oodc": p_oodc,
        "p_tan": p_tan,
        "p_re": p_re,
        "final_step": step,
        "cfg_epochs": cfg_epochs,
        "state": run.state,
    })

df = pd.DataFrame(rows)
pd.set_option("display.float_format", "{:.6f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df.to_string(index=False))

print("\n--- Mean across runs ---")
for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
    vals = [r[col] for r in rows if r[col] is not None]
    if vals:
        print(f"  {col}: {sum(vals)/len(vals):.6f}")
