"""Verify W&B metrics for PR #2202 runs: dq0blopc (seed 42) and a0d4jbyz (seed 73)."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "skills/wandb-primary/scripts")

import wandb

api = wandb.Api()

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

run_ids = {
    "dq0blopc": "seed 42",
    "a0d4jbyz": "seed 73",
}

# Student-reported best values for reference
reported = {
    "dq0blopc": {"p_in": 13.9, "p_tan": 28.3, "p_oodc": 8.3, "p_re": 6.7, "best_epoch": 141},
    "a0d4jbyz": {"p_in": 13.9, "p_tan": 29.9, "p_oodc": 8.0, "p_re": 6.9, "best_epoch": 141},
}

# Surface MAE metric keys to look for (denormalized)
# Common naming conventions used in cfd_tandemfoil
SURFACE_KEYS_CANDIDATES = [
    # denormalized surface MAE
    "val/p_in_mae_denorm", "val/p_tan_mae_denorm", "val/p_oodc_mae_denorm", "val/p_re_mae_denorm",
    "val_p_in_mae_denorm", "val_p_tan_mae_denorm", "val_p_oodc_mae_denorm", "val_p_re_mae_denorm",
    "p_in_mae_denorm", "p_tan_mae_denorm", "p_oodc_mae_denorm", "p_re_mae_denorm",
    # normalized surface MAE
    "val/p_in_mae", "val/p_tan_mae", "val/p_oodc_mae", "val/p_re_mae",
    "val_p_in_mae", "val_p_tan_mae", "val_p_oodc_mae", "val_p_re_mae",
    "p_in_mae", "p_tan_mae", "p_oodc_mae", "p_re_mae",
    # surface MAE aggregate
    "val/surface_mae", "surface_mae", "val_surface_mae",
    # epoch
    "epoch",
]

for run_id, label in run_ids.items():
    print(f"\n{'='*70}")
    print(f"Run: {run_id} ({label})")
    print(f"{'='*70}")

    try:
        run = api.run(f"{path}/{run_id}")
    except Exception as e:
        print(f"  ERROR fetching run: {e}")
        continue

    print(f"  Name:   {run.name}")
    print(f"  State:  {run.state}")
    print(f"  Tags:   {run.tags}")
    print(f"  Config keys (sample): {list(run.config.keys())[:20]}")

    # Print relevant config fields
    for k in ["seed", "model", "lr", "epochs", "batch_size"]:
        if k in run.config:
            print(f"  config.{k}: {run.config[k]}")

    # Check summary metrics
    print(f"\n  --- Summary metrics ---")
    summary = run.summary_metrics
    for k, v in sorted(summary.items()):
        if any(x in k.lower() for x in ["mae", "surface", "p_in", "p_tan", "p_oodc", "p_re"]):
            print(f"    {k}: {v}")

    # Discover actual metric keys in history
    print(f"\n  --- Discovering history keys ---")
    # Grab a small sample to see what keys exist
    sample_rows = list(run.scan_history(page_size=10))
    if sample_rows:
        all_keys = set()
        for row in sample_rows:
            all_keys.update(row.keys())
        mae_keys = sorted([k for k in all_keys if any(x in k.lower() for x in ["mae", "surface", "p_in", "p_tan", "p_oodc", "p_re"])])
        print(f"  Found MAE-related keys: {mae_keys}")
        epoch_keys = sorted([k for k in all_keys if "epoch" in k.lower() or k == "_step"])
        print(f"  Found epoch/step keys: {epoch_keys}")
    else:
        print("  No history rows found")
        mae_keys = []

    if not mae_keys:
        print("  No MAE keys found in history — checking summary only")
        continue

    # Load full history for relevant keys
    keys_to_fetch = mae_keys + ["epoch", "_step"]
    keys_to_fetch = list(set(keys_to_fetch))

    print(f"\n  --- Loading full history ---")
    rows = []
    for row in run.scan_history(keys=keys_to_fetch):
        rows.append(row)

    if not rows:
        print("  No history data found")
        continue

    df = pd.DataFrame(rows)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Find the p_in, p_tan, p_oodc, p_re denormalized keys
    # Try to identify which keys correspond to the four surface metrics
    def find_key(df_cols, *candidates):
        for c in candidates:
            if c in df_cols:
                return c
        return None

    p_in_key = find_key(df.columns,
        "val/p_in_mae_denorm", "val_p_in_mae_denorm", "p_in_mae_denorm",
        "val/p_in_mae", "val_p_in_mae", "p_in_mae")
    p_tan_key = find_key(df.columns,
        "val/p_tan_mae_denorm", "val_p_tan_mae_denorm", "p_tan_mae_denorm",
        "val/p_tan_mae", "val_p_tan_mae", "p_tan_mae")
    p_oodc_key = find_key(df.columns,
        "val/p_oodc_mae_denorm", "val_p_oodc_mae_denorm", "p_oodc_mae_denorm",
        "val/p_oodc_mae", "val_p_oodc_mae", "p_oodc_mae")
    p_re_key = find_key(df.columns,
        "val/p_re_mae_denorm", "val_p_re_mae_denorm", "p_re_mae_denorm",
        "val/p_re_mae", "val_p_re_mae", "p_re_mae")

    print(f"\n  Identified keys:")
    print(f"    p_in  -> {p_in_key}")
    print(f"    p_tan -> {p_tan_key}")
    print(f"    p_oodc-> {p_oodc_key}")
    print(f"    p_re  -> {p_re_key}")

    metric_keys = [k for k in [p_in_key, p_tan_key, p_oodc_key, p_re_key] if k is not None]
    if not metric_keys:
        print("  Could not identify surface MAE keys")
        continue

    # Drop NaN rows for these metrics
    df_clean = df.dropna(subset=metric_keys).copy()
    print(f"  Rows with all surface metrics: {len(df_clean)}")

    if df_clean.empty:
        print("  No complete rows with surface metrics")
        continue

    # Compute mean surface MAE across the four metrics
    df_clean["mean_surface_mae"] = df_clean[metric_keys].mean(axis=1)

    # Find best row (minimum mean surface MAE)
    best_idx = df_clean["mean_surface_mae"].idxmin()
    best_row = df_clean.loc[best_idx]

    # Determine epoch
    epoch_col = find_key(df_clean.columns, "epoch", "_step")
    best_epoch = best_row.get(epoch_col, "N/A") if epoch_col else "N/A"

    print(f"\n  --- Best epoch results ---")
    print(f"  Best epoch/step ({epoch_col}): {best_epoch}")
    if p_in_key:
        print(f"  p_in  MAE: {best_row[p_in_key]:.4f}  (reported: {reported[run_id]['p_in']})")
    if p_tan_key:
        print(f"  p_tan MAE: {best_row[p_tan_key]:.4f}  (reported: {reported[run_id]['p_tan']})")
    if p_oodc_key:
        print(f"  p_oodc MAE: {best_row[p_oodc_key]:.4f}  (reported: {reported[run_id]['p_oodc']})")
    if p_re_key:
        print(f"  p_re  MAE: {best_row[p_re_key]:.4f}  (reported: {reported[run_id]['p_re']})")
    print(f"  Mean surface MAE: {best_row['mean_surface_mae']:.4f}")

    # Check if metrics match student report
    print(f"\n  --- Match check (reported epoch 141) ---")
    rep = reported[run_id]

    # Also check epoch 141 specifically if different from best
    if epoch_col and int(best_epoch) != rep["best_epoch"]:
        epoch_141_rows = df_clean[df_clean[epoch_col].between(rep["best_epoch"] - 0.5, rep["best_epoch"] + 0.5)]
        if not epoch_141_rows.empty:
            row_141 = epoch_141_rows.iloc[0]
            print(f"  Epoch {rep['best_epoch']} metrics:")
            if p_in_key:   print(f"    p_in  MAE: {row_141[p_in_key]:.4f}")
            if p_tan_key:  print(f"    p_tan MAE: {row_141[p_tan_key]:.4f}")
            if p_oodc_key: print(f"    p_oodc MAE: {row_141[p_oodc_key]:.4f}")
            if p_re_key:   print(f"    p_re  MAE: {row_141[p_re_key]:.4f}")
        else:
            print(f"  Epoch {rep['best_epoch']} not found in history")

    # Show progression summary
    print(f"\n  --- Training curve summary ---")
    if epoch_col and len(df_clean) > 1:
        # Sample at quarters
        n = len(df_clean)
        checkpoints = [0, n//4, n//2, 3*n//4, n-1]
        checkpoints = sorted(set(checkpoints))
        for i in checkpoints:
            r = df_clean.iloc[i]
            ep = r.get(epoch_col, i)
            mm = r["mean_surface_mae"]
            print(f"    epoch~{ep:.0f}: mean_surface_mae={mm:.4f}")

print("\n\nDone.")
