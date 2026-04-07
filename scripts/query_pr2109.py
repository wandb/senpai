import os
import sys
import wandb
import pandas as pd

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

api = wandb.Api()

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["ftrzalka", "fceuhys3", "pfm9qsaj", "iyi2npzd", "opnlewzj", "atize8g5"]

run_names = {
    "ftrzalka": "contrast-w0.01-s42",
    "fceuhys3": "contrast-w0.01-s73",
    "pfm9qsaj": "contrast-w0.05-s42",
    "iyi2npzd": "contrast-w0.05-s73",
    "opnlewzj": "contrast-w0.1-s42",
    "atize8g5": "contrast-w0.1-s73",
}

student_reported = {
    "ftrzalka": {"p_in": 13.3, "p_oodc": 7.8, "p_tan": 30.3, "p_re": 6.4},
    "fceuhys3": {"p_in": 13.1, "p_oodc": 7.8, "p_tan": 30.0, "p_re": 6.4},
    "pfm9qsaj": {"p_in": 12.5, "p_oodc": 7.7, "p_tan": 30.6, "p_re": 6.5},
    "iyi2npzd": {"p_in": 13.3, "p_oodc": 8.2, "p_tan": 30.3, "p_re": 6.6},
    "opnlewzj": {"p_in": 12.9, "p_oodc": 7.7, "p_tan": 31.0, "p_re": 6.3},
    "atize8g5": {"p_in": 12.7, "p_oodc": 7.9, "p_tan": 30.3, "p_re": 6.6},
}

surface_keys = ["p_in", "p_oodc", "p_tan", "p_re"]
# Also try alternative naming conventions
alt_keys = [
    "val/mae_p_in", "val/mae_p_oodc", "val/mae_p_tan", "val/mae_p_re",
    "mae_p_in", "mae_p_oodc", "mae_p_tan", "mae_p_re",
    "surface_mae_p_in", "surface_mae_p_oodc", "surface_mae_p_tan", "surface_mae_p_re",
]

rows = []
discrepancies = []

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        summary = run.summary_metrics

        print(f"\n=== Run: {run_id} ({run_names[run_id]}) ===")
        print(f"  State: {run.state}")
        print(f"  Name: {run.name}")
        print(f"  Group: {run.group}")

        # Print all summary keys to find the right metric names
        all_keys = list(summary.keys()) if summary else []
        print(f"  All summary keys (sample): {sorted(all_keys)[:40]}")

        # Try to find surface MAE metrics
        actual = {}
        for k in surface_keys:
            val = summary.get(k)
            if val is None:
                # try alternatives
                for alt in alt_keys:
                    if k.replace("p_", "") in alt:
                        val = summary.get(alt)
                        if val is not None:
                            print(f"  Found {k} as {alt}: {val}")
                            break
            actual[k] = val

        print(f"  Actual surface MAE: p_in={actual['p_in']}, p_oodc={actual['p_oodc']}, p_tan={actual['p_tan']}, p_re={actual['p_re']}")

        reported = student_reported[run_id]
        row = {
            "run_id": run_id,
            "name": run_names[run_id],
            "state": run.state,
        }
        for k in surface_keys:
            row[f"actual_{k}"] = actual[k]
            row[f"reported_{k}"] = reported[k]
            if actual[k] is not None:
                diff = abs(actual[k] - reported[k])
                row[f"diff_{k}"] = round(diff, 2)
                if diff > 0.15:
                    discrepancies.append({
                        "run_id": run_id,
                        "name": run_names[run_id],
                        "metric": k,
                        "reported": reported[k],
                        "actual": round(actual[k], 2),
                        "diff": round(diff, 2),
                    })
        rows.append(row)
    except Exception as e:
        print(f"  ERROR fetching {run_id}: {e}")
        rows.append({"run_id": run_id, "name": run_names[run_id], "error": str(e)})

print("\n\n=== SUMMARY TABLE ===")
df = pd.DataFrame(rows)
print(df.to_string(index=False))

if discrepancies:
    print("\n=== DISCREPANCIES (diff > 0.15) ===")
    print(pd.DataFrame(discrepancies).to_string(index=False))
else:
    print("\nNo significant discrepancies found (all within 0.15 of reported values).")
