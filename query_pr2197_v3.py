import wandb
import os

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
project = os.environ.get("WANDB_PROJECT", "senpai-v1")
path = f"{entity}/{project}"

# Actual metric keys based on inspection:
# p_in  = best_best_val_in_dist/mae_surf_p
# p_oodc = best_best_val_ood_cond/mae_surf_p
# p_tan  = best_best_val_tandem_transfer/mae_surf_p
# p_re   = best_best_val_ood_re/mae_surf_p

def get_metrics(run):
    sm = run.summary_metrics
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "group": run.group,
        "best_epoch": sm.get("best_epoch", "N/A"),
        "total_epochs": sm.get("total_epochs", "N/A"),
        "p_in":   sm.get("best_best_val_in_dist/mae_surf_p", "MISSING"),
        "p_oodc": sm.get("best_best_val_ood_cond/mae_surf_p", "MISSING"),
        "p_tan":  sm.get("best_best_val_tandem_transfer/mae_surf_p", "MISSING"),
        "p_re":   sm.get("best_best_val_ood_re/mae_surf_p", "MISSING"),
    }

run_ids = ["z94hfr0m", "i3p6sfhs", "xyvlpjc3", "dte1drat", "17l5t3we", "srn2nh08"]

print("=== PR #2197 Run Metrics (exact from W&B) ===\n")
print(f"{'Run ID':<12} {'Name':<42} {'State':<10} {'Epochs':<10} {'p_in':>10} {'p_oodc':>10} {'p_tan':>10} {'p_re':>10}")
print("-" * 120)
for run_id in run_ids:
    try:
        run = api.run(f"{path}/{run_id}")
        m = get_metrics(run)
        print(f"{m['id']:<12} {m['name']:<42} {m['state']:<10} {str(m['best_epoch'])+'/'+str(m['total_epochs']):<10} {str(m['p_in']):>10} {str(m['p_oodc']):>10} {str(m['p_tan']):>10} {str(m['p_re']):>10}")
    except Exception as e:
        print(f"{run_id:<12} ERROR: {e}")

print("\n\n=== Group: round7/curvature-loss-weight (all runs) ===\n")
try:
    group_runs = api.runs(path, filters={"group": "round7/curvature-loss-weight"})
    found = list(group_runs[:50])
    print(f"Total runs in group: {len(found)}\n")
    print(f"{'Run ID':<12} {'Name':<42} {'State':<10} {'Epochs':<10} {'p_in':>10} {'p_oodc':>10} {'p_tan':>10} {'p_re':>10}")
    print("-" * 120)
    for run in found:
        m = get_metrics(run)
        print(f"{m['id']:<12} {m['name']:<42} {m['state']:<10} {str(m['best_epoch'])+'/'+str(m['total_epochs']):<10} {str(m['p_in']):>10} {str(m['p_oodc']):>10} {str(m['p_tan']):>10} {str(m['p_re']):>10}")
except Exception as e:
    print(f"Group search ERROR: {e}")
