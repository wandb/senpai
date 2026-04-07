import os
import wandb

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["7ofuolg3", "zt31115v", "ibywi5rr", "4jd564uq"]

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    sm = run.summary_metrics
    print(f"\n=== Run {run_id} — all summary keys ===")
    for k, v in sorted(sm.items()):
        print(f"  {k}: {v}")
