import wandb

api = wandb.Api()
entity = "wandb-applied-ai-team"
project = "senpai-v1"
path = f"{entity}/{project}"

target_ids = ["k4xt5z1x", "dh2taztr", "sa1xdfb7", "xvdyqpd7"]

# The reported metrics use shorthand: p_in, p_oodc, p_tan, p_re
# The W&B keys use: in_dist, ood_cond, ood_tang (tangential), ood_re
# Mapping reported shorthand -> W&B split name
split_map = {
    "p_in":   "in_dist",
    "p_oodc": "ood_cond",
    "p_tan":  "ood_tang",
    "p_re":   "ood_re",
}

reported = {
    "k4xt5z1x": dict(label="PR #2222 (mHC Residuals) seed=42", p_in=13.151, p_oodc=7.796, p_tan=29.225, p_re=6.438),
    "dh2taztr": dict(label="PR #2222 (mHC Residuals) seed=73", p_in=12.193, p_oodc=7.691, p_tan=28.164, p_re=6.581),
    "sa1xdfb7": dict(label="PR #2221 (Wake Angle) seed=42",    p_in=13.037, p_oodc=7.551, p_tan=29.656, p_re=6.482),
    "xvdyqpd7": dict(label="PR #2221 (Wake Angle) seed=73",    p_in=12.166, p_oodc=8.045, p_tan=27.714, p_re=6.351),
}

surface_keys = ["p_in", "p_oodc", "p_tan", "p_re"]

def get_surface_p_metrics(run):
    summary = run.summary_metrics
    all_keys = list(summary.keys())

    metrics = {}
    key_used = {}

    for sk, split in split_map.items():
        # Try best_best first, then best_val
        patterns = [
            f"best_best_val_{split}/mae_surf_p",
            f"best_val_{split}/mae_surf_p",
            f"val_{split}/mae_surf_p",
            f"val/{split}/mae_surf_p",
        ]
        for pat in patterns:
            v = summary.get(pat)
            if v is not None:
                metrics[sk] = v
                key_used[sk] = pat
                break

        if sk not in metrics:
            # Broader search
            for k in all_keys:
                kl = k.lower()
                if split in kl and "surf_p" in kl and "mae" in kl:
                    metrics[sk] = summary[k]
                    key_used[sk] = k
                    break

    return metrics, key_used

print("=" * 95)
print("W&B Actual vs Reported Surface MAE (pressure) — PR #2221 and PR #2222")
print("=" * 95)
print()

all_results = {}

for rid in target_ids:
    run = api.run(f"{path}/{rid}")
    rep = reported[rid]
    metrics, key_used = get_surface_p_metrics(run)
    all_results[rid] = metrics

    print(f"Run {rid} | {rep['label']} | state={run.state}")
    print(f"  {'Metric':<8} {'Reported':>10} {'Actual':>10} {'Delta':>10}  W&B Key")
    print(f"  {'-'*80}")
    for sk in surface_keys:
        rep_val = rep[sk]
        actual = metrics.get(sk)
        ku = key_used.get(sk, "?")
        if actual is not None:
            delta = actual - rep_val
            flag = "  *** >0.1 DISCREPANCY" if abs(delta) > 0.1 else ""
            print(f"  {sk:<8} {rep_val:>10.3f} {actual:>10.3f} {delta:>+10.3f}  {ku}{flag}")
        else:
            print(f"  {sk:<8} {rep_val:>10.3f} {'N/A':>10}  (key not found)")
    print()

# Print all available mae_surf_p keys for one run to confirm key names
run0 = api.run(f"{path}/k4xt5z1x")
surf_p_keys = {k: v for k, v in run0.summary_metrics.items() if "surf_p" in k.lower()}
print("Reference — all mae_surf_p keys in k4xt5z1x summary:")
for k, v in sorted(surf_p_keys.items()):
    print(f"  {k}: {v}")
