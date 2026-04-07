import wandb
import os

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = ["p9oupnt2", "6sp2uazt", "kov6n0rs", "l308y9lx"]

print(f"Querying {entity}/{project} for runs: {run_ids}\n")

results = []
for run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        summary = run.summary_metrics

        # Collect all keys that look like surface MAE metrics
        mae_keys = [k for k in summary.keys() if any(x in k.lower() for x in ["p_in", "p_oodc", "p_tan", "p_re", "surface_mae", "mae"])]

        # Try common naming patterns
        def get_metric(keys_to_try):
            for k in keys_to_try:
                v = summary.get(k)
                if v is not None:
                    return round(float(v), 4), k
            return None, None

        p_in_val, p_in_key = get_metric(["p_in", "surface_mae/p_in", "val/p_in", "best/p_in", "test/p_in"])
        p_oodc_val, p_oodc_key = get_metric(["p_oodc", "surface_mae/p_oodc", "val/p_oodc", "best/p_oodc", "test/p_oodc"])
        p_tan_val, p_tan_key = get_metric(["p_tan", "surface_mae/p_tan", "val/p_tan", "best/p_tan", "test/p_tan"])
        p_re_val, p_re_key = get_metric(["p_re", "surface_mae/p_re", "val/p_re", "best/p_re", "test/p_re"])

        print(f"Run ID: {run_id}")
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")
        print(f"  Config keys: {list(run.config.keys())[:20]}")
        print(f"  All MAE-related keys found: {mae_keys}")
        print(f"  p_in  ({p_in_key}): {p_in_val}")
        print(f"  p_oodc({p_oodc_key}): {p_oodc_val}")
        print(f"  p_tan ({p_tan_key}): {p_tan_val}")
        print(f"  p_re  ({p_re_key}): {p_re_val}")

        # Also check for config
        pcgrad_pct = run.config.get("pcgrad_pct", run.config.get("pcgrad_conflict_pct", run.config.get("conflict_pct", "N/A")))
        seed = run.config.get("seed", "N/A")
        print(f"  pcgrad_pct: {pcgrad_pct}, seed: {seed}")
        print()

        results.append({
            "run_id": run_id,
            "name": run.name,
            "state": run.state,
            "p_in": p_in_val,
            "p_oodc": p_oodc_val,
            "p_tan": p_tan_val,
            "p_re": p_re_val,
            "pcgrad_pct": pcgrad_pct,
            "seed": seed,
            "all_mae_keys": mae_keys,
        })

    except Exception as e:
        print(f"Run {run_id}: ERROR - {e}\n")
        results.append({"run_id": run_id, "error": str(e)})

print("\n=== SUMMARY TABLE ===")
print(f"{'Run ID':<12} {'Name':<30} {'p_in':>7} {'p_oodc':>8} {'p_tan':>7} {'p_re':>6} {'pct':>5} {'seed':>6}")
print("-" * 90)
for r in results:
    if "error" not in r:
        print(f"{r['run_id']:<12} {r['name']:<30} {str(r['p_in']):>7} {str(r['p_oodc']):>8} {str(r['p_tan']):>7} {str(r['p_re']):>6} {str(r['pcgrad_pct']):>5} {str(r['seed']):>6}")
    else:
        print(f"{r['run_id']:<12} ERROR: {r['error']}")
