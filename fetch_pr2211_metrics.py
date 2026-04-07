import wandb
import sys

api = wandb.Api()
entity = "wandb-applied-ai-team"
project = "senpai-v1"
path = f"{entity}/{project}"

target_runs = ["cuehr0b0", "nqkaw1cf"]
metric_keys = [
    "val/best_p_in_mae",
    "val/best_p_oodc_mae",
    "val/best_p_tan_mae",
    "val/best_p_re_mae",
]

print("=== Specific runs ===")
for run_id in target_runs:
    try:
        run = api.run(f"{path}/{run_id}")
        state = run.state
        summary = run.summary_metrics
        epochs = summary.get("epoch", summary.get("_step", "?"))
        metrics = {k: summary.get(k, "N/A") for k in metric_keys}
        print(f"Run: {run_id}")
        print(f"  state={state}, epochs={epochs}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Run {run_id}: ERROR - {e}")

print("\n=== Group: round12/surface-dp-ds ===")
try:
    runs = api.runs(path, filters={"group": "round12/surface-dp-ds"})
    found = 0
    for run in runs:
        found += 1
        state = run.state
        summary = run.summary_metrics
        epochs = summary.get("epoch", summary.get("_step", "?"))
        metrics = {k: summary.get(k, "N/A") for k in metric_keys}
        print(f"Run: {run.id} ({run.name})")
        print(f"  state={state}, epochs={epochs}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    if found == 0:
        print("No runs found in group.")
except Exception as e:
    print(f"Group query ERROR: {e}")
