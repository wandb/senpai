import os
import wandb
import pandas as pd

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["7ofuolg3", "zt31115v", "ibywi5rr", "4jd564uq"]
labels = {
    "7ofuolg3": "anneal→50%, seed 42",
    "zt31115v": "anneal→50%, seed 73",
    "ibywi5rr": "anneal→0%, seed 42",
    "4jd564uq": "anneal→0%, seed 73",
}

srf_keys = [
    "best_srf_mae_p_in",
    "best_srf_mae_p_oodc",
    "best_srf_mae_p_tan",
    "best_srf_mae_p_re",
]

rows = []
for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        sm = run.summary_metrics
        # Print all keys that contain 'srf' or 'mae' (case insensitive) to discover metric names
        relevant_keys = {k: v for k, v in sm.items() if "srf" in k.lower() or ("mae" in k.lower() and "p_" in k.lower())}
        print(f"\nRun {run_id} ({labels[run_id]}) — state: {run.state}")
        print(f"  Relevant summary keys: {relevant_keys}")
        row = {"run_id": run_id, "label": labels[run_id], "state": run.state}
        for k in srf_keys:
            row[k] = sm.get(k, None)
        rows.append(row)
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        rows.append({"run_id": run_id, "label": labels[run_id], "state": "ERROR", **{k: None for k in srf_keys}})

df = pd.DataFrame(rows)
print("\n\n=== Surface MAE Results Table ===")
print(df[["run_id", "label", "state"] + srf_keys].to_string(index=False))

# Also compare with student-reported values
student_reported = {
    "7ofuolg3": {"p_in": 12.878, "p_oodc": 7.838, "p_tan": 28.443, "p_re": 6.341},
    "zt31115v": {"p_in": 13.294, "p_oodc": 7.857, "p_tan": 29.348, "p_re": 6.380},
    "ibywi5rr": {"p_in": 12.739, "p_oodc": 7.993, "p_tan": 29.278, "p_re": 6.533},
    "4jd564uq": {"p_in": 13.796, "p_oodc": 7.817, "p_tan": 29.125, "p_re": 6.439},
}

print("\n=== Comparison: Actual vs Student-Reported ===")
for row in rows:
    rid = row["run_id"]
    reported = student_reported.get(rid, {})
    print(f"\n{rid} ({row['label']}):")
    for metric, actual_key in zip(["p_in", "p_oodc", "p_tan", "p_re"], srf_keys):
        actual = row.get(actual_key)
        rep = reported.get(metric)
        match = "MATCH" if actual is not None and abs(actual - rep) < 0.001 else "MISMATCH"
        print(f"  {metric}: actual={actual}, reported={rep}  [{match}]")
