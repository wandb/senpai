import os
import sys
import wandb
import pandas as pd

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "wandb")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = ["7uz2d0ol", "1v86awoi", "dj0elhj2", "rgs2r8jj"]

surface_mae_keys = [
    "surface_mae/p_in",
    "surface_mae/p_oodc",
    "surface_mae/p_tan",
    "surface_mae/p_re",
]

results = []
for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        row = {
            "run_id": run_id,
            "name": run.name,
            "state": run.state,
            "group": run.group,
        }
        # Try summary_metrics first
        sm = run.summary_metrics
        for key in surface_mae_keys:
            short = key.split("/")[-1]
            val = sm.get(key, sm.get(short, None))
            row[short] = val

        # If any are None, try scanning history for the best values
        missing = [k for k in surface_mae_keys if row.get(k.split("/")[-1]) is None]
        if missing:
            # Try alternate key formats
            alt_keys = []
            for k in missing:
                short = k.split("/")[-1]
                alt_keys.extend([k, short, f"val_{short}", f"best_{short}"])
            # Scan last portion of history
            history_rows = list(run.scan_history(keys=alt_keys))
            if history_rows:
                for k in missing:
                    short = k.split("/")[-1]
                    vals = [r.get(k, r.get(short)) for r in history_rows if r.get(k, r.get(short)) is not None]
                    if vals:
                        row[short] = vals[-1]  # final value
                        row[f"{short}_min"] = min(vals)

        # Also check all summary keys for any surface_mae pattern
        all_surface = {k: v for k, v in sm.items() if "surface" in k.lower() or "mae" in k.lower() or "p_in" in k or "p_oodc" in k or "p_tan" in k or "p_re" in k}
        row["all_surface_keys"] = str(all_surface)

        results.append(row)
    except Exception as e:
        results.append({"run_id": run_id, "error": str(e)})

df = pd.DataFrame(results)
print("\n=== Surface MAE Metrics for PR #2108 runs ===\n")
display_cols = ["run_id", "name", "state", "group", "p_in", "p_oodc", "p_tan", "p_re"]
available = [c for c in display_cols if c in df.columns]
print(df[available].to_string(index=False))

print("\n=== All surface-related summary keys ===")
for _, row in df.iterrows():
    print(f"\nRun {row['run_id']} ({row.get('name', 'N/A')}):")
    print(f"  {row.get('all_surface_keys', 'N/A')}")

# Also check for best/min values if found
min_cols = [c for c in df.columns if c.endswith("_min")]
if min_cols:
    print("\n=== Min (best) values from history scan ===")
    print(df[["run_id"] + min_cols].to_string(index=False))
