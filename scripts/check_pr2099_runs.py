import os
import sys
import wandb
import numpy as np

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")
from wandb_helpers import diagnose_run

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

run_ids = ["s93o0li2", "fovffeeg", "6dt5ofgn", "lqq102pu"]

# Surface MAE metric keys to look for
surface_mae_keys = [
    "surface_mae/p_in",
    "surface_mae/p_oodc",
    "surface_mae/p_tan",
    "surface_mae/p_re",
    # alternate naming patterns
    "val/surface_mae/p_in",
    "val/surface_mae/p_oodc",
    "val/surface_mae/p_tan",
    "val/surface_mae/p_re",
    "p_in",
    "p_oodc",
    "p_tan",
    "p_re",
]

print("=" * 80)
print(f"Checking W&B runs for PR #2099 (DropPath experiment)")
print(f"Entity/Project: {path}")
print("=" * 80)

results = []
for run_id in run_ids:
    full_path = f"{path}/{run_id}"
    print(f"\n--- Run: {run_id} ---")
    try:
        run = api.run(full_path)
        print(f"  Name:   {run.name}")
        print(f"  State:  {run.state}")
        print(f"  Tags:   {run.tags}")

        # Config
        cfg = run.config
        droppath = cfg.get("drop_path_rate", cfg.get("droppath", cfg.get("drop_path", "N/A")))
        seed = cfg.get("seed", "N/A")
        print(f"  Config drop_path_rate: {droppath}")
        print(f"  Config seed: {seed}")

        # Summary metrics
        sm = run.summary_metrics
        print(f"  Summary keys (sample): {list(sm.keys())[:30]}")

        # Look for surface MAE in summary
        found_metrics = {}
        for k in sm.keys():
            if any(key_part in k.lower() for key_part in ["p_in", "p_oodc", "p_tan", "p_re", "surface_mae", "mae"]):
                found_metrics[k] = sm[k]

        print(f"  Relevant summary metrics:")
        for k, v in found_metrics.items():
            print(f"    {k}: {v}")

        # Check for NaN or issues
        nan_keys = [k for k, v in sm.items() if isinstance(v, float) and np.isnan(v)]
        if nan_keys:
            print(f"  WARNING - NaN metrics found: {nan_keys}")

        # Check training stability via history
        history_keys = ["train/loss", "val/loss", "loss", "train_loss", "val_loss"]
        for hk in history_keys:
            try:
                vals = [r[hk] for r in run.scan_history(keys=[hk], page_size=500) if hk in r]
                if vals:
                    arr = np.array(vals)
                    nan_count = np.isnan(arr).sum()
                    print(f"  Loss ({hk}): steps={len(arr)}, min={np.nanmin(arr):.6f}, final={arr[-1]:.6f}, NaNs={nan_count}")
                    # Check for spikes (>2x rolling mean)
                    if len(arr) > 10:
                        rolling = np.convolve(arr, np.ones(10)/10, mode='valid')
                        spikes = np.where(arr[9:] > 2 * rolling)[0]
                        if len(spikes) > 0:
                            print(f"  WARNING - Potential loss spikes at steps: {spikes[:5]}")
                    break
            except Exception:
                continue

        results.append({
            "run_id": run_id,
            "name": run.name,
            "state": run.state,
            "drop_path_rate": droppath,
            "seed": seed,
            "metrics": found_metrics,
        })

    except Exception as e:
        print(f"  ERROR fetching run {run_id}: {e}")
        results.append({"run_id": run_id, "error": str(e)})

# Final summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Run ID':<12} {'Name':<30} {'State':<12} {'DropPath':<10} {'Seed':<6}")
print("-" * 80)
for r in results:
    if "error" not in r:
        print(f"{r['run_id']:<12} {str(r['name']):<30} {r['state']:<12} {str(r['drop_path_rate']):<10} {str(r['seed']):<6}")
        for k, v in r['metrics'].items():
            print(f"  {k}: {v}")

print("\nDone.")
