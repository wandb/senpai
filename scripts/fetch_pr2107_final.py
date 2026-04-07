"""Fetch W&B metrics for PR #2107 runs with correct key mapping.

Mapping:
  p_in   = val_in_dist/mae_surf_p         (in-distribution surface pressure MAE)
  p_oodc = val_ood_cond/mae_surf_p        (OOD condition surface pressure MAE)
  p_tan  = val_tandem_transfer/mae_surf_p (tandem transfer OOD surface pressure MAE)
  p_re   = val_ood_re/mae_surf_p          (OOD Reynolds surface pressure MAE)
"""
import os
import wandb
import numpy as np

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
project = os.environ.get("WANDB_PROJECT", "senpai-v1")

P_IN_KEY   = "val_in_dist/mae_surf_p"
P_OODC_KEY = "val_ood_cond/mae_surf_p"
P_TAN_KEY  = "val_tandem_transfer/mae_surf_p"
P_RE_KEY   = "val_ood_re/mae_surf_p"
VAL_LOSS   = "val/loss"

run_ids = {
    "00lod6uk": "v1-replace s42",
    "3llpj5yj": "v1-replace s73",
    "7e0dma73": "dual-frame s42",
    "w8cyqceg": "dual-frame s73",
    "bklq38ec": "v2-4seed s42",
    "qlkaovuv": "v2-4seed s73",
    "bw5ny846": "v2-4seed s44",
    "74cxcgue": "v2-4seed s45",
}

BASELINE = {"p_in": 13.19, "p_oodc": 7.92, "p_tan": 30.05, "p_re": 6.45}

def fmt(v):
    return f"{v:7.4f}" if v is not None else "   N/A "

def delta(v, b):
    if v is None: return "  N/A "
    pct = (v - b) / b * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"

results = {}

print(f"{'Run ID':<12} {'Label':<20} {'p_in':>8} {'p_oodc':>8} {'p_tan':>9} {'p_re':>8}  {'val_loss':>9}  state")
print("=" * 95)

for run_id, label in run_ids.items():
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        sm = run.summary_metrics
        p_in   = sm.get(P_IN_KEY)
        p_oodc = sm.get(P_OODC_KEY)
        p_tan  = sm.get(P_TAN_KEY)
        p_re   = sm.get(P_RE_KEY)
        vl     = sm.get(VAL_LOSS) or sm.get("best_val_loss")
        results[run_id] = {"label": label, "p_in": p_in, "p_oodc": p_oodc,
                           "p_tan": p_tan, "p_re": p_re, "val_loss": vl,
                           "state": run.state}
        print(f"{run_id:<12} {label:<20} {fmt(p_in)} {fmt(p_oodc)} {fmt(p_tan)} {fmt(p_re)}  {fmt(vl)}  {run.state}")
    except Exception as e:
        print(f"{run_id:<12} {label:<20} ERROR: {e}")
        results[run_id] = None

# Summary tables
print("\n\n=== v1 (coord-replace, 2 seeds) vs advisor-provided baseline ===")
print(f"{'Metric':<10} {'Baseline':>10} {'v1 s42':>10} {'v1 s73':>10} {'vs BL (s42)':>12} {'vs BL (s73)':>12}")
print("-" * 60)
for metric, bkey, rkeys in [
    ("p_in",   "p_in",   ("00lod6uk", "3llpj5yj")),
    ("p_oodc", "p_oodc", ("00lod6uk", "3llpj5yj")),
    ("p_tan",  "p_tan",  ("00lod6uk", "3llpj5yj")),
    ("p_re",   "p_re",   ("00lod6uk", "3llpj5yj")),
]:
    bl = BASELINE[bkey]
    v1s42 = results.get(rkeys[0], {}) or {}
    v1s73 = results.get(rkeys[1], {}) or {}
    r42 = v1s42.get(metric)
    r73 = v1s73.get(metric)
    print(f"{metric:<10} {bl:>10.4f} {fmt(r42)} {fmt(r73)} {delta(r42,bl):>12} {delta(r73,bl):>12}")

print("\n\n=== v2 dual-frame (4-seed) individual results ===")
v2_run_ids = ["bklq38ec", "qlkaovuv", "bw5ny846", "74cxcgue"]
print(f"{'Metric':<10} {'Baseline':>10} {'s42':>10} {'s73':>10} {'s44':>10} {'s45':>10} {'4-seed mean':>12} {'vs BL':>8}")
print("-" * 80)
for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
    bl = BASELINE[metric]
    vals = []
    row = f"{metric:<10} {bl:>10.4f}"
    for rid in v2_run_ids:
        r = results.get(rid) or {}
        v = r.get(metric)
        vals.append(v)
        row += f" {fmt(v)}"
    valid = [v for v in vals if v is not None]
    mean = np.mean(valid) if valid else None
    row += f"  {fmt(mean)} {delta(mean, bl):>8}"
    print(row)

print("\n\n=== Best seeds across all runs (v1 + dual initial + v2 4-seed) ===")
all_run_data = {k: v for k, v in results.items() if v is not None}
for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
    best_val = None
    best_id = None
    bl = BASELINE[metric]
    for rid, r in all_run_data.items():
        v = r.get(metric)
        if v is not None and (best_val is None or v < best_val):
            best_val = v
            best_id = rid
    print(f"{metric}: best={fmt(best_val)} (run {best_id}, {all_run_data[best_id]['label']})  baseline={bl:.4f}  delta={delta(best_val, bl)}")

print("\n\n=== KEY: Does ANY single metric beat the given baseline targets? ===")
print(f"  Targets: p_in<{BASELINE['p_in']}, p_oodc<{BASELINE['p_oodc']}, p_tan<{BASELINE['p_tan']}, p_re<{BASELINE['p_re']}")
for metric in ["p_in", "p_oodc", "p_tan", "p_re"]:
    bl = BASELINE[metric]
    beats = [(rid, r.get(metric)) for rid, r in all_run_data.items()
             if r.get(metric) is not None and r.get(metric) < bl]
    if beats:
        print(f"  {metric}: BEATS baseline — " + ", ".join(f"{rid}({r[1]:.4f})" for rid, r in beats))
    else:
        print(f"  {metric}: does NOT beat baseline target {bl}")
