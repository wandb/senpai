import wandb
import os
import math

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "senpai-fyi")
project = os.environ.get("WANDB_PROJECT", "senpai")

# PR #2174 Attention Temperature Curriculum runs
run_ids = ["nvyisrfa", "zjx1jmy9", "9uii4cn2", "k4k8ko01"]
run_labels = {
    "nvyisrfa": "Config A (2.0->0.3, 80ep) seed42",
    "zjx1jmy9": "Config A (2.0->0.3, 80ep) seed73",
    "9uii4cn2": "Config B (1.5->0.5, 60ep) seed42",
    "k4k8ko01": "Config B (1.5->0.5, 60ep) seed73",
}

# Try EMA metric keys first, then fallback
ema_keys = [
    "ema_surface_mae_p_in",
    "ema_surface_mae_p_oodc",
    "ema_surface_mae_p_tan",
    "ema_surface_mae_p_re",
]
best_keys = [
    "best_best_val_in_dist/mae_surf_p",
    "best_best_val_ood_cond/mae_surf_p",
    "best_best_val_tandem_transfer/mae_surf_p",
    "best_best_val_ood_re/mae_surf_p",
]
short_names = ["p_in", "p_oodc", "p_tan", "p_re"]

def fmt(v):
    if v is None:
        return "MISSING"
    try:
        if math.isnan(float(v)):
            return "NaN"
        return f"{float(v):.4f}"
    except Exception:
        return str(v)

print("=" * 100)
print("W&B Metric Verification: PR #2174 (Attention Temperature Curriculum)")
print("=" * 100)

# First, inspect what keys exist in summary
print("\nInspecting summary keys for first run (nvyisrfa):")
try:
    run = api.run(f"{entity}/{project}/nvyisrfa")
    summary = run.summary_metrics
    ema_found = {k: v for k, v in summary.items() if "ema" in k.lower() and "mae" in k.lower()}
    best_found = {k: v for k, v in summary.items() if "mae_surf" in k.lower()}
    surface_found = {k: v for k, v in summary.items() if "surface" in k.lower()}
    print(f"  EMA+MAE keys: {list(ema_found.keys())[:10]}")
    print(f"  mae_surf keys: {list(best_found.keys())[:10]}")
    print(f"  surface keys: {list(surface_found.keys())[:10]}")
    if not ema_found and not best_found and not surface_found:
        print("  None found. Showing all MAE-related keys:")
        mae_all = {k: v for k, v in summary.items() if "mae" in k.lower()}
        for k, v in sorted(mae_all.items()):
            print(f"    {k}: {v}")
except Exception as e:
    print(f"  Error: {e}")

print()
print(f"{'Run ID':<12} {'Label':<40} {'p_in':>10} {'p_oodc':>10} {'p_tan':>10} {'p_re':>10} {'state':>10}")
print("-" * 100)

for rid in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{rid}")
        summary = run.summary_metrics

        # Try EMA keys first
        ema_vals = [summary.get(k) for k in ema_keys]
        best_vals = [summary.get(k) for k in best_keys]

        # Use whichever has values
        if any(v is not None for v in ema_vals):
            vals = ema_vals
            key_type = "EMA"
        elif any(v is not None for v in best_vals):
            vals = best_vals
            key_type = "best"
        else:
            vals = [None, None, None, None]
            key_type = "NONE"

        label = run_labels.get(rid, rid)
        print(f"{rid:<12} {label:<40} {fmt(vals[0]):>10} {fmt(vals[1]):>10} {fmt(vals[2]):>10} {fmt(vals[3]):>10} {run.state:>10}  [{key_type}]")
    except Exception as e:
        print(f"{rid:<12} ERROR: {e}")

print()
print("Student-reported (for comparison):")
student = {
    "nvyisrfa": {"p_tan": 29.7, "label": "Config A seed42"},
    "zjx1jmy9": {"p_tan": 29.8, "label": "Config A seed73"},
    "9uii4cn2": {"p_tan": 29.3, "label": "Config B seed42"},
    "k4k8ko01": {"p_tan": 29.4, "label": "Config B seed73"},
}
for rid, d in student.items():
    print(f"  {rid} ({d['label']}): p_tan={d['p_tan']}")
print("  Baseline: p_tan=28.60")
