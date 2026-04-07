"""Fetch W&B metrics for PR #2107 runs using the correct key naming."""
import os
import wandb

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
project = os.environ.get("WANDB_PROJECT", "senpai-v1")

# The key naming convention uses best_best_val_*
# p_in   = best_best_val_in_dist/mae_surf_p     (in-distribution surface pressure)
# p_oodc = best_best_val_ood_cond/mae_surf_p    (OOD condition surface pressure)
# p_tan  = best_best_val_ood_tan/mae_surf_p     (OOD tandem surface pressure)
# p_re   = best_best_val_ood_re/mae_surf_p      (OOD Reynolds surface pressure)

# Let's first inspect one run to confirm all available metric keys
run = api.run(f"{entity}/{project}/00lod6uk")
sm = run.summary_metrics
all_keys = sorted([k for k in sm.keys() if not k.startswith("_")])
print("=== All summary metric keys for run 00lod6uk ===")
for k in all_keys:
    print(f"  {k}: {sm.get(k)}")

print("\n\n=== Pressure surface MAE mapping ===")
# Check what keys exist for 'tan' (tandem OOD)
tan_keys = [k for k in all_keys if 'tan' in k.lower() or 'ood' in k.lower()]
print("OOD/tandem-related keys:")
for k in tan_keys:
    print(f"  {k}: {sm.get(k)}")
