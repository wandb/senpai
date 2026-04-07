import os
import sys
import wandb
import pandas as pd

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["wwil2gdr", "bfa65aup", "z3b8tdfy", "p5pljk4j", "nicjx1g0", "a4jaduno"]

metric_keys = [
    "best_best_val_in_dist/mae_surf_p",
    "best_best_val_ood_cond/mae_surf_p",
    "best_best_val_tandem_transfer/mae_surf_p",
    "best_best_val_ood_re/mae_surf_p",
]

config_key = "aug_gap_stagger_sigma"

rows = []
for rid in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{rid}")
        sm = run.summary_metrics
        cfg = run.config

        row = {
            "run_id": rid,
            "name": run.name,
            "group": cfg.get("wandb_group", ""),
            "p_in":   sm.get("best_best_val_in_dist/mae_surf_p"),
            "p_oodc": sm.get("best_best_val_ood_cond/mae_surf_p"),
            "p_tan":  sm.get("best_best_val_tandem_transfer/mae_surf_p"),
            "p_re":   sm.get("best_best_val_ood_re/mae_surf_p"),
            "aug_gap_stagger_sigma_present": config_key in cfg,
            "aug_gap_stagger_sigma_value":   cfg.get(config_key, "NOT_FOUND"),
        }
        rows.append(row)
        print(f"OK: {rid} / {run.name}")
    except Exception as e:
        print(f"ERROR fetching {rid}: {e}")
        rows.append({"run_id": rid, "error": str(e)})

df = pd.DataFrame(rows)
print("\n=== PER-RUN METRICS ===")
cols = ["run_id", "name", "group", "p_in", "p_oodc", "p_tan", "p_re", "aug_gap_stagger_sigma_present", "aug_gap_stagger_sigma_value"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))

print("\n=== NaN / MISSING CHECK ===")
for _, row in df.iterrows():
    for m in ["p_in", "p_oodc", "p_tan", "p_re"]:
        val = row.get(m)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            print(f"  MISSING/NaN: {row['run_id']} -> {m}")
print("  (done checking)")

print("\n=== GROUP MEANS ===")
# We'll group by run name patterns or group field
# Print group col and name to understand groupings
print(df[["run_id", "name", "group"]].to_string(index=False))
