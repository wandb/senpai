import wandb
import pandas as pd

api = wandb.Api()

entity = "wandb-applied-ai-team"
project = "senpai-v1"

# Mapping: student-reported shorthand -> W&B metric keys
# From the summary keys we see:
#   p_in   = val_in_dist/mae_surf_p  (in-distribution surface pressure MAE)
#   p_oodc = val_ood_cond/mae_surf_p  (OOD condition surface pressure MAE)
#   p_tan  = val_tandem_transfer/mae_surf_p  (tandem transfer surface pressure MAE)
#   p_re   = val_ood_re/mae_surf_p   (OOD Re surface pressure MAE)
#
# We also check best_* variants to use the best checkpoint, not just final step.

run_ids = [
    ("thssa2ru", "Baseline s42",       {"p_in": 13.2, "p_tan": 31.4, "p_oodc": 7.7, "p_re": 6.3}),
    ("4lj8ch8s", "Baseline s43",       {"p_in": 13.3, "p_tan": 30.9, "p_oodc": 7.6, "p_re": 6.6}),
    ("qq24kqqv", "Distill a=0.6 s42", {"p_in": 14.4, "p_tan": 31.6, "p_oodc": 8.0, "p_re": 6.6}),
    ("o2ir8mc6", "Distill a=0.6 s43", {"p_in": 14.3, "p_tan": 32.2, "p_oodc": 8.3, "p_re": 6.8}),
    ("fjqz35fa", "Distill a=0.7 s42", {"p_in": 14.1, "p_tan": 30.8, "p_oodc": 7.7, "p_re": 6.5}),
    ("m5smb3w6", "Distill a=0.7 s43", {"p_in": 13.6, "p_tan": 30.3, "p_oodc": 8.0, "p_re": 6.5}),
    ("whoha5za", "Distill a=0.8 s42", {"p_in": 13.2, "p_tan": 30.5, "p_oodc": 7.8, "p_re": 6.5}),
    ("j2pc23lk", "Distill a=0.8 s43", {"p_in": 13.2, "p_tan": 30.4, "p_oodc": 7.8, "p_re": 6.6}),
]

def get_metric(sm, *candidates):
    for c in candidates:
        v = sm.get(c)
        if v is not None:
            return round(float(v), 2)
    return None

rows = []
for run_id, label, student_reported in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        sm = run.summary_metrics

        # best_epoch from summary; total_epochs is final epoch count
        best_epoch = sm.get("best_epoch")
        total_epochs = sm.get("total_epochs")
        state = run.state

        # Extract using the best_* keys (best checkpoint) and also final step values
        # Student's p_in = in-distribution surface pressure
        p_in   = get_metric(sm,
                    "best_best_val_in_dist/mae_surf_p",
                    "val_in_dist/mae_surf_p")
        # Student's p_tan = tandem transfer surface pressure
        p_tan  = get_metric(sm,
                    "best_best_val_tandem_transfer/mae_surf_p",
                    "val_tandem_transfer/mae_surf_p")
        # Student's p_oodc = OOD condition surface pressure
        p_oodc = get_metric(sm,
                    "best_best_val_ood_cond/mae_surf_p",
                    "val_ood_cond/mae_surf_p")
        # Student's p_re = OOD Reynolds number surface pressure
        p_re   = get_metric(sm,
                    "best_best_val_ood_re/mae_surf_p",
                    "val_ood_re/mae_surf_p")

        row = {
            "run_id": run_id,
            "label": label,
            "state": state,
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "actual_p_in":   p_in,
            "actual_p_tan":  p_tan,
            "actual_p_oodc": p_oodc,
            "actual_p_re":   p_re,
            "rep_p_in":   student_reported["p_in"],
            "rep_p_tan":  student_reported["p_tan"],
            "rep_p_oodc": student_reported["p_oodc"],
            "rep_p_re":   student_reported["p_re"],
        }
        rows.append(row)

    except Exception as e:
        rows.append({"run_id": run_id, "label": label, "error": str(e)})

print("=" * 80)
print("PR #2090 — Knowledge Distillation — W&B Metrics Verification")
print("=" * 80)

all_discrepancies = []

for r in rows:
    if "error" in r and r.get("error"):
        print(f"\n{r['run_id']} ({r.get('label','?')}): ERROR — {r['error']}")
        continue

    print(f"\n{'─'*60}")
    print(f"Run: {r['run_id']}  |  {r['label']}")
    print(f"  State: {r['state']}  |  best_epoch={r['best_epoch']}  |  total_epochs={r['total_epochs']}")
    print(f"  {'Metric':<8} {'Actual':>8} {'Reported':>10} {'Diff':>8} {'Flag':>6}")
    print(f"  {'------':<8} {'------':>8} {'--------':>10} {'----':>8} {'----':>6}")

    run_discrepancies = []
    for metric in ["p_in", "p_tan", "p_oodc", "p_re"]:
        actual   = r.get(f"actual_{metric}")
        reported = r.get(f"rep_{metric}")
        if actual is not None and reported is not None:
            diff = actual - reported
            flag = "*** DIFF" if abs(diff) > 0.1 else "OK"
            print(f"  {metric:<8} {actual:>8.2f} {reported:>10.1f} {diff:>+8.2f} {flag:>8}")
            if abs(diff) > 0.1:
                run_discrepancies.append(
                    f"{r['run_id']} ({r['label']}): {metric} actual={actual} vs reported={reported} (diff={diff:+.2f})"
                )
        else:
            print(f"  {metric:<8} {'N/A':>8} {reported:>10.1f} {'N/A':>8} {'?':>8}")
    all_discrepancies.extend(run_discrepancies)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
if all_discrepancies:
    print(f"\nDiscrepancies found (>0.1 difference):")
    for d in all_discrepancies:
        print(f"  - {d}")
else:
    print("\nNo significant discrepancies — all reported metrics match W&B within 0.1.")

# Also print a clean comparison table
print(f"\n{'─'*80}")
print("Compact comparison table:")
print(f"{'Run ID':<12} {'Label':<22} {'ep':>5} {'act_pin':>8} {'rep_pin':>8} {'act_tan':>8} {'rep_tan':>8} {'act_oodc':>9} {'rep_oodc':>9} {'act_re':>7} {'rep_re':>7}")
for r in rows:
    if "error" in r and r.get("error"):
        continue
    print(f"{r['run_id']:<12} {r['label']:<22} {str(r['best_epoch']):>5} "
          f"{str(r['actual_p_in']):>8} {str(r['rep_p_in']):>8} "
          f"{str(r['actual_p_tan']):>8} {str(r['rep_p_tan']):>8} "
          f"{str(r['actual_p_oodc']):>9} {str(r['rep_p_oodc']):>9} "
          f"{str(r['actual_p_re']):>7} {str(r['rep_p_re']):>7}")
