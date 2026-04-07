import wandb

api = wandb.Api()
entity = "wandb-applied-ai-team"
project = "senpai-v1"
path = f"{entity}/{project}"

target_ids = ["k4xt5z1x", "dh2taztr", "sa1xdfb7", "xvdyqpd7"]

reported = {
    "k4xt5z1x": dict(label="PR #2222 seed=42", p_in=13.151, p_oodc=7.796, p_tan=29.225, p_re=6.438),
    "dh2taztr": dict(label="PR #2222 seed=73", p_in=12.193, p_oodc=7.691, p_tan=28.164, p_re=6.581),
    "sa1xdfb7": dict(label="PR #2221 seed=42", p_in=13.037, p_oodc=7.551, p_tan=29.656, p_re=6.482),
    "xvdyqpd7": dict(label="PR #2221 seed=73", p_in=12.166, p_oodc=8.045, p_tan=27.714, p_re=6.351),
}

surface_keys = ["p_in", "p_oodc", "p_tan", "p_re"]

def get_surface_metrics(run):
    summary = run.summary_metrics
    all_keys = list(summary.keys())

    metrics = {}
    for sk in surface_keys:
        # Try common naming patterns (most specific first)
        patterns = [
            f"best_best_val_{sk}/mae_surf_p",
            f"best_val_{sk}/mae_surf_p",
            f"val/{sk}/mae_surf_p",
            f"val/best_{sk}_mae",
            f"val/best_{sk}_mae_surf",
            f"val/best_{sk}_surf_mae",
        ]
        for pat in patterns:
            v = summary.get(pat)
            if v is not None:
                metrics[sk] = (v, pat)
                break
        if sk not in metrics:
            # Broad partial match: must contain sk, "surf", and "mae"
            for k in all_keys:
                kl = k.lower()
                if sk in kl and "surf" in kl and "mae" in kl:
                    metrics[sk] = (summary[k], k)
                    break

    # Also collect all surf/mae keys for reference
    surf_keys_found = {k: summary[k] for k in all_keys if "surf" in k.lower() and "mae" in k.lower()}
    return metrics, surf_keys_found

print("=" * 90)
print("W&B Actual Metrics vs Reported — PR #2221 and PR #2222")
print("=" * 90)

for rid in target_ids:
    try:
        run = api.run(f"{path}/{rid}")
    except Exception as e:
        print(f"\n{rid}: NOT FOUND — {e}")
        continue

    rep = reported[rid]
    metrics, surf_keys_found = get_surface_metrics(run)

    print(f"\nRun {rid} ({rep['label']}) | state={run.state} | name={run.name}")
    if surf_keys_found:
        print(f"  Mae-surf keys in summary: {list(surf_keys_found.keys())[:8]}")
    else:
        print("  !! No mae_surf keys found in summary")
        # Print sample of all keys
        sm = run.summary_metrics
        print(f"  Sample summary keys: {list(sm.keys())[:15]}")

    print(f"  {'Metric':<8} {'Reported':>10} {'Actual':>10} {'Delta':>10}  Key used")
    print(f"  {'-'*70}")
    for sk in surface_keys:
        rep_val = rep[sk]
        if sk in metrics:
            actual, key_used = metrics[sk]
            delta = actual - rep_val
            flag = " *** DISCREPANCY" if abs(delta) > 0.1 else ""
            print(f"  {sk:<8} {rep_val:>10.3f} {actual:>10.3f} {delta:>+10.3f}  {key_used}{flag}")
        else:
            print(f"  {sk:<8} {rep_val:>10.3f} {'N/A':>10}  (not found)")

print("\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)
print(f"{'Run':<12} {'PR/Seed':<20} {'p_in R':>8} {'p_in A':>8} {'p_oodc R':>9} {'p_oodc A':>9} {'p_tan R':>8} {'p_tan A':>8} {'p_re R':>7} {'p_re A':>7}")
print("-" * 100)

for rid in target_ids:
    try:
        run = api.run(f"{path}/{rid}")
    except:
        continue
    rep = reported[rid]
    metrics, _ = get_surface_metrics(run)

    def fmt(sk):
        if sk in metrics:
            return f"{metrics[sk][0]:>8.3f}"
        return f"{'N/A':>8}"

    print(f"{rid:<12} {rep['label']:<20} {rep['p_in']:>8.3f} {fmt('p_in')} {rep['p_oodc']:>9.3f} {fmt('p_oodc')} {rep['p_tan']:>8.3f} {fmt('p_tan')} {rep['p_re']:>7.3f} {fmt('p_re')}")
