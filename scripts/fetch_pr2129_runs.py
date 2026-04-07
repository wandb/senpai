import os
import sys
import wandb
import pandas as pd

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")
from wandb_helpers import runs_to_dataframe

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

run_ids = ["bnskf03a", "yfy3efjq", "s2txmpe9", "3is100bp"]

results = []
for run_id in run_ids:
    try:
        run = api.run(f"{path}/{run_id}")
        sm = run.summary_metrics

        # Try various metric name patterns
        def get_metric(sm, *keys):
            for k in keys:
                v = sm.get(k)
                if v is not None:
                    return v
            return None

        p_in = get_metric(sm,
            "val/mae_p_in", "mae_p_in", "val/surface_mae_p_in",
            "p_in_mae", "surface_mae/p_in", "val/p_in_mae")
        p_oodc = get_metric(sm,
            "val/mae_p_oodc", "mae_p_oodc", "val/surface_mae_p_oodc",
            "p_oodc_mae", "surface_mae/p_oodc", "val/p_oodc_mae")
        p_tan = get_metric(sm,
            "val/mae_p_tan", "mae_p_tan", "val/surface_mae_p_tan",
            "p_tan_mae", "surface_mae/p_tan", "val/p_tan_mae")
        p_re = get_metric(sm,
            "val/mae_p_re", "mae_p_re", "val/surface_mae_p_re",
            "p_re_mae", "surface_mae/p_re", "val/p_re_mae")

        # Get config for weight and seed
        cfg = run.config
        weight = cfg.get("aux_weight", cfg.get("surf_grad_weight", cfg.get("weight", "N/A")))
        seed = cfg.get("seed", cfg.get("random_seed", "N/A"))
        group = run.group

        results.append({
            "run_id": run_id,
            "name": run.name,
            "group": group,
            "weight": weight,
            "seed": seed,
            "p_in": p_in,
            "p_oodc": p_oodc,
            "p_tan": p_tan,
            "p_re": p_re,
            "state": run.state,
        })

        print(f"\n=== Run {run_id} ({run.name}) ===")
        print(f"  State: {run.state}")
        print(f"  Group: {group}")
        print(f"  Config weight keys: {[k for k in cfg.keys() if 'weight' in k.lower() or 'aux' in k.lower() or 'grad' in k.lower()]}")
        print(f"  Config seed keys: {[k for k in cfg.keys() if 'seed' in k.lower()]}")
        print(f"  weight={weight}, seed={seed}")
        print(f"  p_in={p_in}, p_oodc={p_oodc}, p_tan={p_tan}, p_re={p_re}")

        # Print all summary metric keys for inspection
        all_keys = sorted(sm.keys())
        surface_keys = [k for k in all_keys if any(x in k.lower() for x in ['mae', 'surface', 'p_in', 'p_oodc', 'p_tan', 'p_re'])]
        print(f"  Surface-related summary keys: {surface_keys[:30]}")

    except Exception as e:
        print(f"ERROR for run {run_id}: {e}")
        results.append({"run_id": run_id, "error": str(e)})

print("\n\n=== SUMMARY TABLE ===")
df = pd.DataFrame(results)
print(df[["run_id", "name", "weight", "seed", "p_in", "p_oodc", "p_tan", "p_re", "state"]].to_string(index=False))

# Check for NaN/missing
print("\n=== NaN / Missing Check ===")
for _, row in df.iterrows():
    for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
        val = row.get(metric)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            print(f"  MISSING: run {row['run_id']} metric {metric}")
print("Done.")
