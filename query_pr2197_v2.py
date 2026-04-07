import wandb
import os

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
project = os.environ.get("WANDB_PROJECT", "senpai-v1")
path = f"{entity}/{project}"

# Inspect the actual summary keys for one run
run = api.run(f"{path}/z94hfr0m")
sm = run.summary_metrics
print("=== All summary_metrics keys for z94hfr0m ===")
for k, v in sorted(sm.items()):
    if "mae" in k.lower() or "p_in" in k.lower() or "p_oodc" in k.lower() or "p_tan" in k.lower() or "p_re" in k.lower() or "val" in k.lower() or "surface" in k.lower() or "epoch" in k.lower() or "step" in k.lower():
        print(f"  {k}: {v}")

print("\n=== All summary keys (full list) ===")
for k in sorted(sm.keys()):
    print(f"  {k}: {sm[k]}")
