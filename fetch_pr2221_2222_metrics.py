import os
import sys
import wandb

api = wandb.Api()
entity = "senpai-wandb"
project = "senpai"

run_ids = {
    "k4xt5z1x": "PR #2222 seed=42",
    "dh2taztr": "PR #2222 seed=73",
    "sa1xdfb7": "PR #2221 seed=42",
    "xvdyqpd7": "PR #2221 seed=73",
}

reported = {
    "k4xt5z1x": dict(p_in=13.151, p_oodc=7.796, p_tan=29.225, p_re=6.438),
    "dh2taztr": dict(p_in=12.193, p_oodc=7.691, p_tan=28.164, p_re=6.581),
    "sa1xdfb7": dict(p_in=13.037, p_oodc=7.551, p_tan=29.656, p_re=6.482),
    "xvdyqpd7": dict(p_in=12.166, p_oodc=8.045, p_tan=27.714, p_re=6.351),
}

surface_keys = ["p_in", "p_oodc", "p_tan", "p_re"]

# Mapping from short key to possible wandb summary key patterns
key_patterns = {
    "p_in":   ["best_best_val_p_in/mae_surf_p",   "best_val_p_in/mae_surf_p",   "val_p_in/mae_surf_p"],
    "p_oodc": ["best_best_val_p_oodc/mae_surf_p", "best_val_p_oodc/mae_surf_p", "val_p_oodc/mae_surf_p"],
    "p_tan":  ["best_best_val_p_tan/mae_surf_p",  "best_val_p_tan/mae_surf_p",  "val_p_tan/mae_surf_p"],
    "p_re":   ["best_best_val_p_re/mae_surf_p",   "best_val_p_re/mae_surf_p",   "val_p_re/mae_surf_p"],
}

results = {}

for run_id, label in run_ids.items():
    run = api.run(f"{entity}/{project}/{run_id}")
    summary = run.summary_metrics

    # Print all keys containing "mae_surf" to understand naming
    mae_keys = {k: v for k, v in summary.items() if "mae_surf" in k.lower() or "surf" in k.lower()}

    found = {}
    for short_key, patterns in key_patterns.items():
        for pat in patterns:
            val = summary.get(pat)
            if val is not None:
                found[short_key] = val
                break
        if short_key not in found:
            # Try scanning all keys for partial match
            for k, v in summary.items():
                k_lower = k.lower()
                if short_key in k_lower and "mae_surf" in k_lower:
                    found[short_key] = v
                    break

    results[run_id] = {
        "label": label,
        "name": run.name,
        "state": run.state,
        "found": found,
        "mae_surf_keys": mae_keys,
    }

# Print results
print("\n=== W&B Actual Metrics vs Reported ===\n")
header = f"{'Run ID':<12} {'Label':<20} {'Metric':<8} {'Reported':>10} {'Actual':>10} {'Delta':>10} {'Flag'}"
print(header)
print("-" * 80)

for run_id, data in results.items():
    label = data["label"]
    found = data["found"]
    rep = reported[run_id]

    for sk in surface_keys:
        actual = found.get(sk)
        rep_val = rep.get(sk)
        if actual is not None:
            delta = actual - rep_val
            flag = " *** DISCREPANCY" if abs(delta) > 0.1 else ""
            print(f"{run_id:<12} {label:<20} {sk:<8} {rep_val:>10.3f} {actual:>10.3f} {delta:>+10.3f}{flag}")
        else:
            print(f"{run_id:<12} {label:<20} {sk:<8} {rep_val:>10.3f} {'N/A':>10} {'?':>10}")
    print()

# Also print all mae_surf keys found for reference
print("\n=== Raw mae_surf summary keys found ===")
for run_id, data in results.items():
    print(f"\n{run_id} ({data['label']}):")
    for k, v in sorted(data["mae_surf_keys"].items()):
        print(f"  {k}: {v}")
