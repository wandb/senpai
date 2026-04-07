import os
import wandb
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["x2mkegnk", "pvjzi2ov", "q5bprc0t", "i4144o3d"]
labels = {
    "x2mkegnk": "Config A (0.06→0.02), s42",
    "pvjzi2ov":  "Config A (0.06→0.02), s73",
    "q5bprc0t":  "Config B (0.04→0.02), s42",
    "i4144o3d":  "Config B (0.04→0.02), s73",
}

ema_keys = [
    "ema_surface_mae_p_in",
    "ema_surface_mae_p_oodc",
    "ema_surface_mae_p_tan",
    "ema_surface_mae_p_re",
]

# Also try alternate naming conventions
alt_keys = [
    "val/ema_surface_mae_p_in",
    "val/ema_surface_mae_p_oodc",
    "val/ema_surface_mae_p_tan",
    "val/ema_surface_mae_p_re",
]

print(f"{'Run':<12} {'Label':<30} {'p_in':>10} {'p_oodc':>10} {'p_tan':>10} {'p_re':>10}")
print("-" * 82)

for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        sm = run.summary_metrics

        # Try to find the right keys
        results = {}
        for suffix in ["p_in", "p_oodc", "p_tan", "p_re"]:
            val = None
            for prefix in ["ema_surface_mae_", "val/ema_surface_mae_", "surface_mae_", "val/surface_mae_"]:
                key = prefix + suffix
                if key in sm:
                    val = sm[key]
                    break
            results[suffix] = val

        # If none found, print all summary keys to help debug
        if all(v is None for v in results.values()):
            print(f"\nRun {run_id} summary keys (sample):")
            keys = list(sm.keys())[:60]
            for k in keys:
                if "mae" in k.lower() or "surface" in k.lower() or "ema" in k.lower():
                    print(f"  {k}: {sm[k]}")

        label = labels.get(run_id, run_id)
        p_in   = f"{results['p_in']:.4f}"   if results['p_in']   is not None else "N/A"
        p_oodc = f"{results['p_oodc']:.4f}" if results['p_oodc'] is not None else "N/A"
        p_tan  = f"{results['p_tan']:.4f}"  if results['p_tan']  is not None else "N/A"
        p_re   = f"{results['p_re']:.4f}"   if results['p_re']   is not None else "N/A"
        print(f"{run_id:<12} {label:<30} {p_in:>10} {p_oodc:>10} {p_tan:>10} {p_re:>10}")

    except Exception as e:
        print(f"{run_id:<12} ERROR: {e}")
