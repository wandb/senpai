import os
import wandb
import sys

api = wandb.Api()

entity = os.environ.get("WANDB_ENTITY", "wild-ai")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = ["p9oupnt2", "6sp2uazt", "kov6n0rs", "l308y9lx"]

surface_mae_keys = [
    "surface_mae/p_in",
    "surface_mae/p_oodc",
    "surface_mae/p_tan",
    "surface_mae/p_re",
]

results = []

for run_id in run_ids:
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        summary = run.summary_metrics

        # Print all summary keys for debugging
        print(f"\n=== Run: {run_id} | Name: {run.name} | State: {run.state} ===")
        print("All summary keys:", [k for k in summary.keys() if "mae" in k.lower() or "p_in" in k.lower() or "p_re" in k.lower() or "surface" in k.lower()])

        row = {"run_id": run_id, "name": run.name, "state": run.state}
        for key in surface_mae_keys:
            val = summary.get(key)
            row[key] = val
            print(f"  {key}: {val}")

        # Also try alternate naming patterns
        alt_keys = [k for k in summary.keys() if any(x in k for x in ["p_in", "p_oodc", "p_tan", "p_re", "surface_mae"])]
        if alt_keys:
            print("  Alternate matching keys found:")
            for k in alt_keys:
                print(f"    {k}: {summary.get(k)}")
                row[k] = summary.get(k)

        results.append(row)

    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        results.append({"run_id": run_id, "error": str(e)})

print("\n\n=== SUMMARY TABLE ===")
print(f"{'Run ID':<12} {'Name':<35} {'State':<10} {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8}")
print("-" * 100)
for row in results:
    if "error" in row:
        print(f"{row['run_id']:<12} ERROR: {row['error']}")
    else:
        p_in   = row.get("surface_mae/p_in", "N/A")
        p_oodc = row.get("surface_mae/p_oodc", "N/A")
        p_tan  = row.get("surface_mae/p_tan", "N/A")
        p_re   = row.get("surface_mae/p_re", "N/A")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

        print(f"{row['run_id']:<12} {row.get('name','?'):<35} {row.get('state','?'):<10} {fmt(p_in):>8} {fmt(p_oodc):>8} {fmt(p_tan):>8} {fmt(p_re):>8}")
