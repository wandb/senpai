import os
import sys
import wandb
import pandas as pd

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "senpai-fyi")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = ["cm9uz650", "kj8cvxpw", "js6sm78l", "2o2499gz"]
labels = {
    "cm9uz650": "σ=0.03, seed 42",
    "kj8cvxpw": "σ=0.03, seed 73",
    "js6sm78l": "σ=0.08, seed 42",
    "2o2499gz": "σ=0.08, seed 73",
}

student_reported = {
    "cm9uz650": {"p_in": 12.9, "p_oodc": 7.8, "p_tan": 28.2, "p_re": 6.6},
    "kj8cvxpw": {"p_in": 13.6, "p_oodc": 8.1, "p_tan": 29.5, "p_re": 6.6},
    "js6sm78l": {"p_in": 12.9, "p_oodc": 7.9, "p_tan": 28.9, "p_re": 6.6},
    "2o2499gz": {"p_in": 13.4, "p_oodc": 7.9, "p_tan": 29.4, "p_re": 6.5},
}

# Surface MAE metric name candidates
METRIC_CANDIDATES = [
    "best_surface_p_in_mae",
    "best_surface_p_oodc_mae",
    "best_surface_p_tan_mae",
    "best_surface_p_re_mae",
    # alternate naming
    "surface_p_in_mae",
    "surface_p_oodc_mae",
    "surface_p_tan_mae",
    "surface_p_re_mae",
    # more candidates
    "val/surface_p_in_mae",
    "val/surface_p_oodc_mae",
    "val/surface_p_tan_mae",
    "val/surface_p_re_mae",
]

print(f"Fetching runs from {entity}/{project}\n")

rows = []
for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        summary = run.summary_metrics

        # Print all summary keys for inspection
        print(f"\n{'='*60}")
        print(f"Run: {run_id} ({labels[run_id]})")
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")

        # Find surface MAE keys in summary
        surface_keys = {k: v for k, v in summary.items()
                        if "surface" in k.lower() or "mae" in k.lower() or "p_in" in k.lower()
                        or "p_oodc" in k.lower() or "p_tan" in k.lower() or "p_re" in k.lower()}

        print(f"  Surface/MAE summary keys found: {len(surface_keys)}")
        for k, v in sorted(surface_keys.items()):
            print(f"    {k}: {v}")

        # Try to extract the 4 key metrics
        def get_metric(summary, *candidates):
            for c in candidates:
                if c in summary and summary[c] is not None:
                    return summary[c]
            return None

        p_in = get_metric(summary,
            "best_surface_p_in_mae", "surface_p_in_mae",
            "val/surface_p_in_mae", "p_in_mae")
        p_oodc = get_metric(summary,
            "best_surface_p_oodc_mae", "surface_p_oodc_mae",
            "val/surface_p_oodc_mae", "p_oodc_mae")
        p_tan = get_metric(summary,
            "best_surface_p_tan_mae", "surface_p_tan_mae",
            "val/surface_p_tan_mae", "p_tan_mae")
        p_re = get_metric(summary,
            "best_surface_p_re_mae", "surface_p_re_mae",
            "val/surface_p_re_mae", "p_re_mae")

        rows.append({
            "run_id": run_id,
            "label": labels[run_id],
            "state": run.state,
            "wandb_p_in": p_in,
            "wandb_p_oodc": p_oodc,
            "wandb_p_tan": p_tan,
            "wandb_p_re": p_re,
            "reported_p_in": student_reported[run_id]["p_in"],
            "reported_p_oodc": student_reported[run_id]["p_oodc"],
            "reported_p_tan": student_reported[run_id]["p_tan"],
            "reported_p_re": student_reported[run_id]["p_re"],
        })

    except Exception as e:
        print(f"  ERROR fetching {run_id}: {e}")
        rows.append({
            "run_id": run_id,
            "label": labels[run_id],
            "state": "ERROR",
            "wandb_p_in": None, "wandb_p_oodc": None,
            "wandb_p_tan": None, "wandb_p_re": None,
            "reported_p_in": student_reported[run_id]["p_in"],
            "reported_p_oodc": student_reported[run_id]["p_oodc"],
            "reported_p_tan": student_reported[run_id]["p_tan"],
            "reported_p_re": student_reported[run_id]["p_re"],
        })

print("\n\n" + "="*80)
print("COMPARISON TABLE: W&B Actual vs Student Reported")
print("="*80)

df = pd.DataFrame(rows)
print(df[["run_id", "label", "state",
          "wandb_p_in", "reported_p_in",
          "wandb_p_oodc", "reported_p_oodc",
          "wandb_p_tan", "reported_p_tan",
          "wandb_p_re", "reported_p_re"]].to_string(index=False))

print("\n\nDISCREPANCY ANALYSIS:")
for _, row in df.iterrows():
    discrepancies = []
    for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
        w = row[f"wandb_{col}"]
        r = row[f"reported_{col}"]
        if w is not None and r is not None:
            diff = abs(float(w) - float(r))
            if diff > 0.15:  # flag if > 0.15 difference
                discrepancies.append(f"{col}: W&B={w:.2f} vs reported={r:.1f} (diff={diff:.2f})")
    if discrepancies:
        print(f"\nRun {row['run_id']} ({row['label']}):")
        for d in discrepancies:
            print(f"  *** DISCREPANCY: {d}")
    else:
        w_vals = [row[f"wandb_{c}"] for c in ["p_in","p_oodc","p_tan","p_re"]]
        if all(v is not None for v in w_vals):
            print(f"\nRun {row['run_id']} ({row['label']}): OK - all metrics within tolerance")
        else:
            print(f"\nRun {row['run_id']} ({row['label']}): UNABLE TO VERIFY (metrics not found in W&B summary)")
