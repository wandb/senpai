import wandb
import os

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
project = os.environ.get("WANDB_PROJECT", "senpai-v1")
path = f"{entity}/{project}"

run_ids = ["z94hfr0m", "i3p6sfhs", "xyvlpjc3", "dte1drat", "17l5t3we", "srn2nh08"]

metric_keys = [
    "val/best_p_in_mae",
    "val/best_p_oodc_mae",
    "val/best_p_tan_mae",
    "val/best_p_re_mae",
]

print("=== Individual Run Metrics ===")
for run_id in run_ids:
    try:
        run = api.run(f"{path}/{run_id}")
        sm = run.summary_metrics
        epochs = sm.get("epoch", sm.get("trainer/global_step", "N/A"))
        print(f"\nRun: {run_id}")
        print(f"  Name:   {run.name}")
        print(f"  State:  {run.state}")
        print(f"  Group:  {run.group}")
        print(f"  Epoch:  {epochs}")
        for k in metric_keys:
            v = sm.get(k, "MISSING")
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nRun {run_id}: ERROR - {e}")

print("\n\n=== Group Search: round7/curvature-loss-weight ===")
try:
    group_runs = api.runs(
        path,
        filters={"group": "round7/curvature-loss-weight"}
    )
    found = list(group_runs[:50])
    print(f"Found {len(found)} runs in group")
    for run in found:
        sm = run.summary_metrics
        epochs = sm.get("epoch", sm.get("trainer/global_step", "N/A"))
        print(f"\nRun: {run.id}")
        print(f"  Name:   {run.name}")
        print(f"  State:  {run.state}")
        print(f"  Epoch:  {epochs}")
        for k in metric_keys:
            v = sm.get(k, "MISSING")
            print(f"  {k}: {v}")
except Exception as e:
    print(f"Group search ERROR: {e}")
