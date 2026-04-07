#!/usr/bin/env python3
"""Fetch surface MAE metrics (p_in, p_oodc, p_tan, p_re) for PR #2166 runs."""
import os
import wandb
import pandas as pd
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = [
    "bmjgaeqp",  # w=0.1, s42
    "e7iu2ix3",  # w=0.1, s73
    "bfau5gzb",  # w=0.1, s17
    "mmwix7z6",  # w=0.1, s31
    "mvfbf0jh",  # w=0.1, s55
    "0m0fg9ut",  # w=0.1, s88
    "wvtqymw7",  # w=0.05, s42
    "i9as9n60",  # w=0.05, s73
]

seed_map = {
    "bmjgaeqp": (0.1, 42),
    "e7iu2ix3": (0.1, 73),
    "bfau5gzb": (0.1, 17),
    "mmwix7z6": (0.1, 31),
    "mvfbf0jh": (0.1, 55),
    "0m0fg9ut": (0.1, 88),
    "wvtqymw7": (0.05, 42),
    "i9as9n60": (0.05, 73),
}

# Metric keys mapping (from summary scan):
# p_in   -> best_best_val_in_dist/mae_surf_p
# p_oodc -> best_best_val_ood_cond/mae_surf_p
# p_tan  -> best_best_val_tandem_transfer/mae_surf_p
# p_re   -> best_best_val_ood_re/mae_surf_p

path = f"{entity}/{project}"
rows = []

for run_id in run_ids:
    try:
        run = api.run(f"{path}/{run_id}")
        weight, seed = seed_map[run_id]
        s = run.summary_metrics

        p_in   = s.get("best_best_val_in_dist/mae_surf_p")
        p_oodc = s.get("best_best_val_ood_cond/mae_surf_p")
        p_tan  = s.get("best_best_val_tandem_transfer/mae_surf_p")
        p_re   = s.get("best_best_val_ood_re/mae_surf_p")

        # Also check EMA-specific keys if present
        ema_p_in   = s.get("ema_best_val_in_dist/mae_surf_p",
                    s.get("best_ema_val_in_dist/mae_surf_p"))
        ema_p_oodc = s.get("ema_best_val_ood_cond/mae_surf_p",
                    s.get("best_ema_val_ood_cond/mae_surf_p"))
        ema_p_tan  = s.get("ema_best_val_tandem_transfer/mae_surf_p",
                    s.get("best_ema_val_tandem_transfer/mae_surf_p"))
        ema_p_re   = s.get("ema_best_val_ood_re/mae_surf_p",
                    s.get("best_ema_val_ood_re/mae_surf_p"))

        rows.append({
            "run_id": run_id,
            "weight": weight,
            "seed": seed,
            "state": run.state,
            "p_in": p_in,
            "p_oodc": p_oodc,
            "p_tan": p_tan,
            "p_re": p_re,
            "ema_p_in": ema_p_in,
            "ema_p_oodc": ema_p_oodc,
            "ema_p_tan": ema_p_tan,
            "ema_p_re": ema_p_re,
        })
    except Exception as e:
        print(f"ERROR for {run_id}: {e}")
        rows.append({
            "run_id": run_id,
            "weight": seed_map[run_id][0],
            "seed": seed_map[run_id][1],
            "state": "ERROR",
            "p_in": None, "p_oodc": None, "p_tan": None, "p_re": None,
            "ema_p_in": None, "ema_p_oodc": None, "ema_p_tan": None, "ema_p_re": None,
        })

df = pd.DataFrame(rows)

print("\n=== PR #2166 — dp/dn=0 Physics Loss — Surface MAE (best_best_ keys) ===\n")
print(df[["run_id", "weight", "seed", "state", "p_in", "p_oodc", "p_tan", "p_re"]].to_string(index=False))

# Check if EMA keys are populated
ema_cols = ["ema_p_in", "ema_p_oodc", "ema_p_tan", "ema_p_re"]
if df[ema_cols].notna().any().any():
    print("\n=== EMA Surface MAE keys ===\n")
    print(df[["run_id", "weight", "seed"] + ema_cols].to_string(index=False))
else:
    print("\n(No separate EMA keys found — 'best_best_' keys ARE the EMA best values)")

# Group stats
print("\n=== Group Stats ===\n")
for w, grp in df.groupby("weight"):
    print(f"  weight={w} ({len(grp)} seeds):")
    for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
        vals = grp[col].dropna()
        if len(vals):
            print(f"    {col}: mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}, std={vals.std():.4f}")

# Composite
df["composite"] = df[["p_in", "p_oodc", "p_tan", "p_re"]].mean(axis=1)
print("\n=== With Composite (mean of 4 surf p metrics) ===\n")
print(df[["run_id", "weight", "seed", "p_in", "p_oodc", "p_tan", "p_re", "composite"]].sort_values("composite").to_string(index=False))

print("\n=== Best run per group ===")
for w, grp in df.groupby("weight"):
    best = grp.loc[grp["composite"].idxmin()]
    print(f"  w={w}: best run={best['run_id']} (s={best['seed']}) composite={best['composite']:.4f}")

print("\n=== Overall best ===")
best_overall = df.loc[df["composite"].idxmin()]
print(f"  {best_overall['run_id']} (w={best_overall['weight']}, s={best_overall['seed']}): composite={best_overall['composite']:.4f}")
print(f"  p_in={best_overall['p_in']:.4f}, p_oodc={best_overall['p_oodc']:.4f}, p_tan={best_overall['p_tan']:.4f}, p_re={best_overall['p_re']:.4f}")
