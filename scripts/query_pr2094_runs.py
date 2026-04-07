import os
import sys
import wandb
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

run_ids = {
    "Baseline s42": "3mim1mhi",
    "Baseline s43": "gk16mcse",
    "SWAD s42":     "1ee50z25",
    "SWAD s43":     "hbm6rfcg",
    "SWAD s44":     "hsvhokae",
    "SWAD s45":     "r632qi5f",
    "SWAD s66":     "86sd67n7",
    "SWAD s67":     "tm0513wp",
}

METRICS = [
    "val_in_dist/mae_surf_p",
    "val_ood_cond/mae_surf_p",
    "val_tandem_transfer/mae_surf_p",
    "val_ood_re/mae_surf_p",
]

SWAD_METRICS = [
    "swad_checkpoint_count",
    "swad/val_in_dist/mae_surf_p",
    "swad/val_ood_cond/mae_surf_p",
    "swad/val_tandem_transfer/mae_surf_p",
    "swad/val_ood_re/mae_surf_p",
    "val_loss",
]

print("=" * 100)
print(f"{'Label':<16} {'Run ID':<12} {'State':<10} {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8}")
print("=" * 100)

results = {}
for label, run_id in run_ids.items():
    try:
        run = api.run(f"{path}/{run_id}")
        state = run.state
        sm = run.summary_metrics

        p_in   = sm.get("val_in_dist/mae_surf_p", float("nan"))
        p_oodc = sm.get("val_ood_cond/mae_surf_p", float("nan"))
        p_tan  = sm.get("val_tandem_transfer/mae_surf_p", float("nan"))
        p_re   = sm.get("val_ood_re/mae_surf_p", float("nan"))

        def fmt(v):
            return f"{v:.4f}" if v == v else "NaN"

        print(f"{label:<16} {run_id:<12} {state:<10} {fmt(p_in):>8} {fmt(p_oodc):>8} {fmt(p_tan):>8} {fmt(p_re):>8}")
        results[label] = {
            "run_id": run_id,
            "state": state,
            "p_in": p_in,
            "p_oodc": p_oodc,
            "p_tan": p_tan,
            "p_re": p_re,
            "summary": sm,
        }
    except Exception as e:
        print(f"{label:<16} {run_id:<12} ERROR: {e}")
        results[label] = {"run_id": run_id, "error": str(e)}

print()
print("=" * 100)
print("SWAD-SPECIFIC METRICS (summary)")
print("=" * 100)

for label, run_id in run_ids.items():
    if "SWAD" not in label:
        continue
    r = results.get(label, {})
    if "error" in r:
        continue
    sm = r.get("summary", {})
    print(f"\n{label} ({run_id}):")
    for m in SWAD_METRICS:
        v = sm.get(m, "NOT_FOUND")
        print(f"  {m}: {v}")
    # Also look for any swad-related keys
    swad_keys = [k for k in sm.keys() if "swad" in k.lower()]
    if swad_keys:
        print(f"  All SWAD keys in summary: {swad_keys}")
    else:
        print(f"  No SWAD keys found in summary")

print()
print("=" * 100)
print("CHECKING HISTORY FOR SWAD CHECKPOINT EVENTS")
print("=" * 100)

for label, run_id in run_ids.items():
    if "SWAD" not in label:
        continue
    r = results.get(label, {})
    if "error" in r:
        continue
    try:
        run = api.run(f"{path}/{run_id}")
        # Scan history for swad_checkpoint_count or val_loss
        keys_to_check = ["swad_checkpoint_count", "val_loss", "val_in_dist/mae_surf_p"]
        history_rows = list(run.scan_history(keys=keys_to_check))
        print(f"\n{label} ({run_id}): {len(history_rows)} history rows with those keys")
        if history_rows:
            # Check if swad_checkpoint_count ever appears
            swad_rows = [row for row in history_rows if row.get("swad_checkpoint_count") is not None]
            val_loss_rows = [row for row in history_rows if row.get("val_loss") is not None]
            print(f"  Rows with swad_checkpoint_count: {len(swad_rows)}")
            if swad_rows:
                counts = [r["swad_checkpoint_count"] for r in swad_rows]
                print(f"  swad_checkpoint_count values: min={min(counts)}, max={max(counts)}, last={counts[-1]}")
            print(f"  Rows with val_loss: {len(val_loss_rows)}")
            if val_loss_rows:
                losses = [r["val_loss"] for r in val_loss_rows]
                print(f"  val_loss: min={min(losses):.6f}, final={losses[-1]:.6f}, count={len(losses)}")
            # Show first and last val_in_dist rows
            val_rows = [row for row in history_rows if row.get("val_in_dist/mae_surf_p") is not None]
            print(f"  Rows with val_in_dist/mae_surf_p: {len(val_rows)}")
            if val_rows:
                vals = [r["val_in_dist/mae_surf_p"] for r in val_rows]
                print(f"  val_in_dist/mae_surf_p: min={min(vals):.6f}, final={vals[-1]:.6f}")
    except Exception as e:
        print(f"{label} ({run_id}): ERROR: {e}")
