import os
import sys
sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

import wandb

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "senpai-lab")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = ["e7lucein", "vuxgzmui"]

for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"\n=== Run: {run_id} ===")
        print(f"  Name:  {run.name}")
        print(f"  State: {run.state}")
        print(f"  Group: {run.group}")

        # Print all summary metrics
        summary = dict(run.summary_metrics)
        print(f"  All summary keys ({len(summary)}):")
        # Filter for pressure/surface metrics
        pressure_keys = [k for k in summary if any(x in k.lower() for x in
                         ["p_in", "p_tan", "p_oodc", "p_re", "surface", "mae", "best", "pressure"])]
        if pressure_keys:
            print("  --- Pressure/Surface metrics ---")
            for k in sorted(pressure_keys):
                print(f"    {k}: {summary[k]}")
        else:
            print("  (no pressure/surface keys found, printing all)")
            for k in sorted(summary.keys())[:50]:
                print(f"    {k}: {summary[k]}")
    except Exception as e:
        print(f"Error fetching {run_id}: {e}")
        # Try alternate entity
        try:
            for ent in ["senpai-lab", "senpai", "askeladd"]:
                try:
                    run = api.run(f"{ent}/{project}/{run_id}")
                    print(f"  Found under entity: {ent}")
                    summary = dict(run.summary_metrics)
                    pressure_keys = [k for k in summary if any(x in k.lower() for x in
                                     ["p_in", "p_tan", "p_oodc", "p_re", "surface", "mae", "best"])]
                    for k in sorted(pressure_keys):
                        print(f"    {k}: {summary[k]}")
                    break
                except Exception:
                    continue
        except Exception as e2:
            print(f"  Also failed alternate entities: {e2}")
