"""Check best epoch for PR #2202 runs."""
import os
import wandb
import numpy as np

api = wandb.Api(timeout=30)
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

# The surface pressure metrics we care about
surf_p_keys = [
    "val_in_dist/mae_surf_p",
    "val_ood_cond/mae_surf_p",
    "val_ood_re/mae_surf_p",
    "val_tandem_transfer/mae_surf_p",
]

# Metric naming in history uses / or some other separator — try to discover
for run_id, seed_label in [("dq0blopc", "seed42"), ("a0d4jbyz", "seed73")]:
    print(f"\n=== {run_id} ({seed_label}) ===")
    run = api.run(f"{path}/{run_id}")

    # Sample first few rows to see key names
    sample = list(run.scan_history(page_size=5))
    if not sample:
        print("  No history")
        continue

    all_keys = set()
    for row in sample:
        all_keys.update(row.keys())

    p_keys = sorted([k for k in all_keys if "mae_surf_p" in k or "surf_p" in k.lower()])
    epoch_keys = [k for k in all_keys if k in ("epoch", "_step", "_runtime")]
    print(f"  Surface-p keys found: {p_keys}")
    print(f"  Epoch/step keys found: {epoch_keys}")

    fetch_keys = p_keys + ["epoch", "_step"]
    fetch_keys = list(set(fetch_keys))

    rows = []
    for row in run.scan_history(keys=fetch_keys):
        rows.append(row)
    print(f"  Total history rows: {len(rows)}")

    if not rows:
        continue

    import pandas as pd
    df = pd.DataFrame(rows)

    # Identify best epoch by mean surface p across all splits
    available_p_keys = [k for k in p_keys if k in df.columns]
    if not available_p_keys:
        print("  No surface-p columns found in history")
        continue

    df_clean = df.dropna(subset=available_p_keys).copy()
    df_clean["mean_surf_p"] = df_clean[available_p_keys].mean(axis=1)
    best_idx = df_clean["mean_surf_p"].idxmin()
    best_row = df_clean.loc[best_idx]

    epoch_col = "epoch" if "epoch" in df_clean.columns else "_step"
    best_ep = best_row.get(epoch_col, "N/A")

    print(f"\n  Best epoch ({epoch_col}): {best_ep}")
    print(f"  Mean surf_p MAE at best: {best_row['mean_surf_p']:.4f}")
    for k in available_p_keys:
        print(f"    {k}: {best_row[k]:.4f}")

    # Show last 10 epochs
    print(f"\n  Last 10 epochs:")
    tail = df_clean.tail(10)
    for _, r in tail.iterrows():
        ep = r.get(epoch_col, "?")
        mm = r["mean_surf_p"]
        print(f"    epoch={ep:.0f}: mean_surf_p={mm:.4f}")

print("\nDone.")
