"""Verify W&B metrics for PR #2195 inter-foil distance feature experiment."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

import wandb

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "senpai-cfd")
project = os.environ.get("WANDB_PROJECT", "senpai")

# Run IDs to verify
run_ids = {
    "gf94dd2t": "seed 42",
    "x5l2mf4g": "seed 73",
}

# Student-reported values for comparison
student_reported = {
    "gf94dd2t": {"p_in": 12.9, "p_tan": 28.7, "p_oodc": 7.7, "p_re": 6.3},
    "x5l2mf4g": {"p_in": 12.9, "p_tan": 29.7, "p_oodc": 7.9, "p_re": 6.4},
}

# Metric keys to look for (surface MAE metrics)
metric_keys = [
    "val/surface_mae/p_in",
    "val/surface_mae/p_oodc",
    "val/surface_mae/p_tan",
    "val/surface_mae/p_re",
    # Also try without val/ prefix
    "surface_mae/p_in",
    "surface_mae/p_oodc",
    "surface_mae/p_tan",
    "surface_mae/p_re",
    # Also try short keys
    "p_in",
    "p_oodc",
    "p_tan",
    "p_re",
]

print(f"Querying W&B entity={entity}, project={project}")
print("=" * 70)

results = {}

for run_id, seed_label in run_ids.items():
    print(f"\n--- Run {run_id} ({seed_label}) ---")
    try:
        path = f"{entity}/{project}/{run_id}"
        run = api.run(path)
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")
        print(f"  Group: {run.group}")
        print(f"  Tags: {run.tags}")

        # Check summary metrics
        summary = run.summary_metrics
        print(f"\n  Summary metrics keys (sample): {list(summary.keys())[:30]}")

        # Try to find our surface MAE metrics
        found_metrics = {}
        for key in summary.keys():
            key_lower = key.lower()
            if any(m in key_lower for m in ["p_in", "p_oodc", "p_tan", "p_re", "surface_mae"]):
                found_metrics[key] = summary[key]

        if found_metrics:
            print(f"\n  Found surface MAE-related metrics:")
            for k, v in sorted(found_metrics.items()):
                print(f"    {k}: {v}")
        else:
            print("  No surface MAE metrics found in summary — checking all summary keys...")
            for k, v in sorted(summary.items()):
                print(f"    {k}: {v}")

        # Check epoch count
        epoch_val = summary.get("epoch") or summary.get("trainer/global_step") or summary.get("_step")
        print(f"\n  Epoch/step in summary: {epoch_val}")

        # Scan history for the metric keys to find best values
        print(f"\n  Scanning history for surface MAE metrics...")

        # First, find what surface-related keys exist in history
        hist_sample = list(run.scan_history(keys=None, min_step=0, max_step=5))
        if hist_sample:
            hist_keys = list(hist_sample[0].keys())
            surface_hist_keys = [k for k in hist_keys if any(m in k.lower() for m in ["p_in", "p_oodc", "p_tan", "p_re", "surface_mae", "surface"])]
            print(f"  Surface-related history keys: {surface_hist_keys}")
        else:
            surface_hist_keys = []
            print("  No history rows found.")

        run_result = {
            "run_id": run_id,
            "seed": seed_label,
            "state": run.state,
            "group": run.group,
        }

        # Try to extract best values from history
        if surface_hist_keys:
            all_history = []
            for row in run.scan_history(keys=surface_hist_keys):
                all_history.append(row)

            if all_history:
                df_hist = pd.DataFrame(all_history)
                print(f"\n  History rows: {len(df_hist)}")
                print(f"  History columns: {list(df_hist.columns)}")

                for key in surface_hist_keys:
                    if key in df_hist.columns:
                        col_data = df_hist[key].dropna()
                        if len(col_data) > 0:
                            best_val = col_data.min()  # surface MAE — lower is better
                            print(f"  Best {key}: {best_val:.4f} (over {len(col_data)} non-NaN steps)")
                            run_result[key] = best_val

            # Also check epochs
            epoch_keys = ["epoch", "trainer/epoch", "_step"]
            epoch_history = []
            for row in run.scan_history(keys=epoch_keys):
                epoch_history.append(row)
            if epoch_history:
                df_ep = pd.DataFrame(epoch_history)
                for ek in epoch_keys:
                    if ek in df_ep.columns:
                        max_epoch = df_ep[ek].max()
                        print(f"  Max {ek}: {max_epoch}")
                        run_result[f"max_{ek}"] = max_epoch

        results[run_id] = run_result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[run_id] = {"run_id": run_id, "error": str(e)}

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

# Build comparison table
rows = []
for run_id, seed_label in run_ids.items():
    res = results.get(run_id, {})
    reported = student_reported[run_id]

    # Try various key patterns for the 4 metrics
    def find_metric(res, metric_name):
        """Try multiple key patterns to find a metric."""
        patterns = [
            f"val/surface_mae/{metric_name}",
            f"surface_mae/{metric_name}",
            metric_name,
            f"best_val/surface_mae/{metric_name}",
        ]
        for p in patterns:
            if p in res:
                return res[p]
        # Also try case-insensitive partial match
        for k, v in res.items():
            if metric_name in k.lower() and isinstance(v, (int, float)):
                return v
        return None

    p_in_actual = find_metric(res, "p_in")
    p_oodc_actual = find_metric(res, "p_oodc")
    p_tan_actual = find_metric(res, "p_tan")
    p_re_actual = find_metric(res, "p_re")

    rows.append({
        "run_id": run_id,
        "seed": seed_label,
        "state": res.get("state", "unknown"),
        "group": res.get("group", "unknown"),
        "p_in (reported)": reported["p_in"],
        "p_in (actual)": f"{p_in_actual:.4f}" if p_in_actual is not None else "N/A",
        "p_oodc (reported)": reported["p_oodc"],
        "p_oodc (actual)": f"{p_oodc_actual:.4f}" if p_oodc_actual is not None else "N/A",
        "p_tan (reported)": reported["p_tan"],
        "p_tan (actual)": f"{p_tan_actual:.4f}" if p_tan_actual is not None else "N/A",
        "p_re (reported)": reported["p_re"],
        "p_re (actual)": f"{p_re_actual:.4f}" if p_re_actual is not None else "N/A",
    })

df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False))

print("\nDone.")
