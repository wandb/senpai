"""Verify PR #2202 metrics using summary only (faster, avoids scan_history hang)."""
import os
import wandb

api = wandb.Api(timeout=60)
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

reported = {
    "dq0blopc": {"label": "seed42", "p_in": 13.9, "p_oodc": 8.3, "p_re": 6.7, "p_tan": 28.3, "best_epoch": 141},
    "a0d4jbyz": {"label": "seed73", "p_in": 13.9, "p_oodc": 8.0, "p_re": 6.9, "p_tan": 29.9, "best_epoch": 141},
}

# Mappings: student label -> W&B split
# p_in = val_in_dist/mae_surf_p
# p_oodc = val_ood_cond/mae_surf_p
# p_re = val_ood_re/mae_surf_p
# p_tan = val_tandem_transfer/mae_surf_p
METRIC_MAP = {
    "p_in": "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_re": "best_best_val_ood_re/mae_surf_p",
    "p_tan": "best_best_val_tandem_transfer/mae_surf_p",
}

print("=" * 72)
print("PR #2202 Fore-Aft Cross-Attention SRF — Metrics Verification")
print("=" * 72)

for run_id, rep in reported.items():
    print(f"\n--- Run {run_id} ({rep['label']}) ---")
    run = api.run(f"{path}/{run_id}")
    sm = run.summary_metrics

    print(f"  Name:  {run.name}")
    print(f"  State: {run.state}")

    # Best epoch from summary
    best_epoch_key = "best_epoch"
    actual_best_epoch = sm.get(best_epoch_key, sm.get("best_step", "N/A"))
    print(f"  Best epoch (summary): {actual_best_epoch}  (student reported: {rep['best_epoch']})")

    print(f"\n  {'Metric':<12} {'W&B Value':>12} {'Reported':>10} {'Match?':>8} {'Delta':>8}")
    print(f"  {'-'*52}")

    all_match = True
    for metric_label, wandb_key in METRIC_MAP.items():
        wandb_val = sm.get(wandb_key)
        rep_val = rep[metric_label]
        if wandb_val is None:
            print(f"  {metric_label:<12} {'N/A':>12} {rep_val:>10.1f} {'???':>8}")
            continue
        delta = wandb_val - rep_val
        # Allow ±0.1 rounding tolerance
        match = abs(delta) <= 0.1
        if not match:
            all_match = False
        match_str = "YES" if match else "NO ***"
        print(f"  {metric_label:<12} {wandb_val:>12.4f} {rep_val:>10.1f} {match_str:>8} {delta:>+8.4f}")

    print(f"\n  Overall match: {'YES' if all_match else 'NO - DISCREPANCY FOUND'}")

    # Also check the final (non-best) epoch values for comparison
    print(f"\n  Final epoch (last logged) surface-p values:")
    for label, key_prefix in [("p_in", "val_in_dist"), ("p_oodc", "val_ood_cond"),
                               ("p_re", "val_ood_re"), ("p_tan", "val_tandem_transfer")]:
        final_key = f"{key_prefix}/mae_surf_p"
        final_val = sm.get(final_key)
        best_key = f"best_{final_key}"
        best_val = sm.get(best_key)
        if final_val is not None and best_val is not None:
            print(f"    {label}: final={final_val:.4f}, best={best_val:.4f}, diff={final_val - best_val:+.4f}")

print("\nDone.")
