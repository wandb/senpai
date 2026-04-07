"""
Final analysis: PCGrad 2-way validation runs.
No new 8-seed runs found for phase6/pcgrad-2way-validation group.
Analyze original 4 runs from phase6/pcgrad-3way for metric naming and
check if any askeladd runs in the broader phase6 space match the 2-way pattern.
"""
import os
import wandb
import pandas as pd
import numpy as np

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "wandb")
project = os.environ.get("WANDB_PROJECT", "senpai")
path = f"{entity}/{project}"

# The original 4 PCGrad runs - map them to their actual experiment type
# Note: run names are pcgrad3w-pct15 (3-way, 15%) and pcgrad3w-pct10 (3-way, 10%)
# We need pcgrad2w (2-way) runs which do NOT exist yet

original_ids = {
    "p9oupnt2": "pcgrad3w-pct15-s42",
    "6sp2uazt": "pcgrad3w-pct15-s73",
    "kov6n0rs": "pcgrad3w-pct10-s42",
    "l308y9lx": "pcgrad3w-pct10-s73",
}

print("=" * 70)
print("PCGrad ORIGINAL 4 RUNS (phase6/pcgrad-3way)")
print("These are 3-WAY PCGrad runs, NOT 2-way runs.")
print("=" * 70)

# The metric keys for these runs
# p_in = val_in_dist/mae_surf_p
# p_oodc = val_ood_cond/mae_surf_p
# p_tan = val_tandem_transfer/mae_surf_p
# p_re = val_ood_re/mae_surf_p

metric_map = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

rows = []
for rid, name in original_ids.items():
    run = api.run(f"{path}/{rid}")
    summary = run.summary_metrics
    row = {"run_id": rid, "name": name, "group": run.group, "state": run.state}
    for short, full_key in metric_map.items():
        row[short] = summary.get(full_key)
    rows.append(row)

df = pd.DataFrame(rows)
print("\nPer-run surface pressure MAE (best_best / best across training):")
print(df.to_string(index=False))

# Split by experiment variant
df_pct15 = df[df["name"].str.contains("pct15")]
df_pct10 = df[df["name"].str.contains("pct10")]

print("\n--- pct15 (15% gradient conflict threshold) mean ---")
for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
    print(f"  {col}: {df_pct15[col].mean():.4f}")

print("\n--- pct10 (10% gradient conflict threshold) mean ---")
for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
    print(f"  {col}: {df_pct10[col].mean():.4f}")

print("\n--- ALL 4 RUNS combined mean ---")
for col in ["p_in", "p_oodc", "p_tan", "p_re"]:
    print(f"  {col}: {df[col].mean():.4f}")

# Also check for any askeladd runs that look like 2-way
print("\n\n" + "=" * 70)
print("Checking for any 2-way specific askeladd runs...")
print("=" * 70)

# Check all phase6 runs by date - looking for anything created after pcgrad-3way runs
# The 3-way runs were created at 2026-04-04T12:51:00Z
# If askeladd ran new 2-way seeds, they should be after that

try:
    all_phase6 = api.runs(
        path,
        filters={"group": {"$regex": "phase6"}},
        order="-created_at",
    )
    phase6_list = all_phase6[:100]
    askeladd_runs = [r for r in phase6_list if r.name.startswith("askeladd/")]
    print(f"Found {len(askeladd_runs)} askeladd runs in phase6 groups:")
    for r in askeladd_runs:
        print(f"  {r.id} | {r.name} | state={r.state} | group={r.group} | created={r.created_at}")
except Exception as e:
    print(f"  Error: {e}")
