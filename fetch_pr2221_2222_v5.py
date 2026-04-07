import wandb

api = wandb.Api()
entity = "wandb-applied-ai-team"
project = "senpai-v1"
path = f"{entity}/{project}"

target_ids = ["k4xt5z1x", "dh2taztr", "sa1xdfb7", "xvdyqpd7"]

# Corrected mapping: p_tan -> tandem_transfer
split_map = {
    "p_in":   "in_dist",
    "p_oodc": "ood_cond",
    "p_tan":  "tandem_transfer",
    "p_re":   "ood_re",
}

reported = {
    "k4xt5z1x": dict(label="PR #2222 (mHC Residuals) seed=42", p_in=13.151, p_oodc=7.796, p_tan=29.225, p_re=6.438),
    "dh2taztr": dict(label="PR #2222 (mHC Residuals) seed=73", p_in=12.193, p_oodc=7.691, p_tan=28.164, p_re=6.581),
    "sa1xdfb7": dict(label="PR #2221 (Wake Angle) seed=42",    p_in=13.037, p_oodc=7.551, p_tan=29.656, p_re=6.482),
    "xvdyqpd7": dict(label="PR #2221 (Wake Angle) seed=73",    p_in=12.166, p_oodc=8.045, p_tan=27.714, p_re=6.351),
}

surface_keys = ["p_in", "p_oodc", "p_tan", "p_re"]

def get_metrics(run):
    summary = run.summary_metrics
    metrics = {}
    for sk, split in split_map.items():
        key = f"best_best_val_{split}/mae_surf_p"
        v = summary.get(key)
        if v is not None:
            metrics[sk] = v
    return metrics

print("=" * 90)
print("Reported vs Actual W&B Surface Pressure MAE — PR #2221 and PR #2222")
print("Key: best_best_val_<split>/mae_surf_p")
print("=" * 90)

discrepancies = []

for rid in target_ids:
    run = api.run(f"{path}/{rid}")
    rep = reported[rid]
    metrics = get_metrics(run)

    print(f"\n{rid} | {rep['label']}")
    print(f"  {'Metric':<8} {'Reported':>10} {'Actual (W&B)':>14} {'Delta':>10}  Note")
    print(f"  {'-'*65}")
    for sk in surface_keys:
        rep_val = rep[sk]
        actual = metrics.get(sk)
        if actual is not None:
            delta = actual - rep_val
            note = "*** DISCREPANCY > 0.1" if abs(delta) > 0.1 else "OK"
            if abs(delta) > 0.1:
                discrepancies.append((rid, sk, rep_val, actual, delta))
            print(f"  {sk:<8} {rep_val:>10.3f} {actual:>14.3f} {delta:>+10.4f}  {note}")
        else:
            print(f"  {sk:<8} {rep_val:>10.3f} {'N/A':>14}")

print()
print("=" * 90)
if discrepancies:
    print(f"DISCREPANCIES FOUND ({len(discrepancies)}):")
    for rid, sk, rep_val, actual, delta in discrepancies:
        print(f"  {rid} {sk}: reported={rep_val:.3f}, actual={actual:.3f}, delta={delta:+.4f}")
else:
    print("NO discrepancies > 0.1 found. All reported values match W&B.")
print("=" * 90)
