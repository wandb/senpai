import os
import sys
import wandb
import pandas as pd

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

api = wandb.Api()

entity  = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path    = f"{entity}/{project}"

RUNS = {
    "baseline-s42":      "tqlbfz9y",
    "baseline-s73":      "dck4ur8w",
    "ohem-w1.5-p75-s42": "4x372o82",
    "ohem-w1.5-p75-s66": "c8v1a5l0",
    "ohem-w1.5-p75-s73": "6db353lp",
    "ohem-w2.0-p75-s42": "nzdg07dy",
    "ohem-w2.0-p75-s73": "0k97472w",
    "ohem-w1.5-p90-s42": "dng4d3lt",
}

METRIC_KEYS = [
    "val_in_dist/mae_surf_p",
    "val_ood_cond/mae_surf_p",
    "val_tandem_transfer/mae_surf_p",
    "val_ood_re/mae_surf_p",
]

rows = []
for name, run_id in RUNS.items():
    try:
        run = api.run(f"{path}/{run_id}")
        sm = run.summary_metrics
        row = {
            "run_name": name,
            "run_id":   run_id,
            "state":    run.state,
        }
        for k in METRIC_KEYS:
            val = sm.get(k)
            row[k] = round(val, 5) if val is not None else None
        rows.append(row)
        print(f"  OK  {name} ({run_id}): state={run.state}")
    except Exception as e:
        print(f"  ERR {name} ({run_id}): {e}")
        rows.append({"run_name": name, "run_id": run_id, "state": "ERROR", **{k: None for k in METRIC_KEYS}})

df = pd.DataFrame(rows)
df = df.rename(columns={
    "val_in_dist/mae_surf_p":       "p_in",
    "val_ood_cond/mae_surf_p":      "p_oodc",
    "val_tandem_transfer/mae_surf_p": "p_tan",
    "val_ood_re/mae_surf_p":        "p_re",
})

print("\n=== W&B confirmed surface MAE metrics (PR #2101 — OHEM) ===\n")
pd.set_option("display.float_format", "{:.5f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(df[["run_name", "run_id", "state", "p_in", "p_oodc", "p_tan", "p_re"]].to_string(index=False))
