import os
import wandb

api = wandb.Api()
entity = "senpai-wandb"
project = "senpai"

target_ids = ["k4xt5z1x", "dh2taztr", "sa1xdfb7", "xvdyqpd7"]

reported = {
    "k4xt5z1x": dict(label="PR #2222 seed=42", p_in=13.151, p_oodc=7.796, p_tan=29.225, p_re=6.438),
    "dh2taztr": dict(label="PR #2222 seed=73", p_in=12.193, p_oodc=7.691, p_tan=28.164, p_re=6.581),
    "sa1xdfb7": dict(label="PR #2221 seed=42", p_in=13.037, p_oodc=7.551, p_tan=29.656, p_re=6.482),
    "xvdyqpd7": dict(label="PR #2221 seed=73", p_in=12.166, p_oodc=8.045, p_tan=27.714, p_re=6.351),
}

surface_keys = ["p_in", "p_oodc", "p_tan", "p_re"]

# Search by run name filter
runs = api.runs(
    f"{entity}/{project}",
    filters={"$or": [{"name": rid} for rid in target_ids]},
    per_page=20,
)

found_runs = {}
for r in runs:
    if r.id in target_ids or r.name in target_ids:
        found_runs[r.name if r.name in target_ids else r.id] = r

print(f"Found {len(found_runs)} runs: {list(found_runs.keys())}")

# If not found by name filter, search by ID
if len(found_runs) < len(target_ids):
    missing = [rid for rid in target_ids if rid not in found_runs]
    print(f"Trying ID-based search for: {missing}")
    for rid in missing:
        try:
            r = api.run(f"{entity}/{project}/{rid}")
            found_runs[rid] = r
            print(f"  Found by ID: {rid}")
        except Exception as e:
            print(f"  Not found by ID: {rid} -- {e}")

def get_surface_metrics(run):
    summary = run.summary_metrics
    # Print all keys for debug
    all_keys = list(summary.keys())
    surf_keys = [k for k in all_keys if "surf" in k.lower() or "mae" in k.lower()]

    metrics = {}
    for sk in surface_keys:
        # Try various patterns
        patterns = [
            f"best_best_val_{sk}/mae_surf_p",
            f"best_val_{sk}/mae_surf_p",
            f"val_{sk}/mae_surf_p",
            f"best_{sk}_mae_surf_p",
            f"{sk}_mae_surf_p",
        ]
        for pat in patterns:
            v = summary.get(pat)
            if v is not None:
                metrics[sk] = v
                break
        if sk not in metrics:
            # Partial match
            for k in all_keys:
                if sk in k and "surf" in k and "mae" in k:
                    metrics[sk] = summary[k]
                    break
    return metrics, surf_keys

print("\n=== Actual vs Reported Metrics ===\n")
for rid in target_ids:
    run = found_runs.get(rid)
    if run is None:
        print(f"{rid}: NOT FOUND IN W&B")
        continue

    rep = reported[rid]
    metrics, surf_key_list = get_surface_metrics(run)

    print(f"Run {rid} ({rep['label']}) — state={run.state}, name={run.name}")
    print(f"  Surface MAE keys in summary: {surf_key_list[:10]}")
    print(f"  {'Metric':<8} {'Reported':>10} {'Actual':>10} {'Delta':>10} {'Flag'}")
    print(f"  {'-'*55}")

    for sk in surface_keys:
        actual = metrics.get(sk)
        rep_val = rep[sk]
        if actual is not None:
            delta = actual - rep_val
            flag = " *** >0.1" if abs(delta) > 0.1 else ""
            print(f"  {sk:<8} {rep_val:>10.3f} {actual:>10.3f} {delta:>+10.3f}{flag}")
        else:
            print(f"  {sk:<8} {rep_val:>10.3f} {'N/A':>10}")
    print()
