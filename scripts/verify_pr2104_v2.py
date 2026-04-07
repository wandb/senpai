import wandb
import numpy as np
import os

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "wandb")
project = os.environ.get("WANDB_PROJECT", "senpai")

run_ids = [
    ("42", "fctgmn1d"),
    ("43", "rc40fpuu"),
    ("44", "ygqo9rom"),
    ("45", "r5uxnp4b"),
    ("46", "yxhjfisl"),
    ("47", "qrbprrli"),
    ("48", "9whdgscd"),
    ("49", "ekdcwekr"),
]

# Metric mapping: student label -> W&B key
# p_in   = in-distribution pressure surface MAE
# p_tan  = tandem transfer pressure surface MAE
# p_oodc = OOD condition pressure surface MAE
# p_re   = OOD Re pressure surface MAE
METRIC_MAP = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
    "val_loss": "best_val_loss",
}

reported = {
    "42": {"p_in": 13.4, "p_tan": 29.8, "p_oodc": 7.7, "p_re": 6.4, "val_loss": 0.3866},
    "43": {"p_in": 13.2, "p_tan": 29.3, "p_oodc": 7.8, "p_re": 6.4, "val_loss": 0.3859},
    "44": {"p_in": 13.6, "p_tan": 30.2, "p_oodc": 8.1, "p_re": 6.5, "val_loss": 0.3931},
    "45": {"p_in": 12.9, "p_tan": 30.5, "p_oodc": 8.1, "p_re": 6.5, "val_loss": 0.3863},
    "46": {"p_in": 13.4, "p_tan": 30.2, "p_oodc": 7.8, "p_re": 6.4, "val_loss": 0.3891},
    "47": {"p_in": 12.6, "p_tan": 30.4, "p_oodc": 8.2, "p_re": 6.6, "val_loss": 0.3898},
    "48": {"p_in": 12.9, "p_tan": 29.9, "p_oodc": 7.8, "p_re": 6.4, "val_loss": 0.3852},
    "49": {"p_in": 13.5, "p_tan": 30.1, "p_oodc": 7.9, "p_re": 6.4, "val_loss": 0.3913},
}

all_vals = {k: [] for k in METRIC_MAP}
discrepancies = []

header = f"{'Seed':<6} {'Run':<10} {'State':<10} {'aft_foil_srf':<14} {'p_in':>7} {'p_tan':>7} {'p_oodc':>7} {'p_re':>7} {'val_loss':>9}  Issues"
print(header)
print("-" * len(header) + "-" * 20)

for seed, run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        state = run.state
        cfg = run.config
        sm = run.summary_metrics

        aft_foil_srf = cfg.get("aft_foil_srf", "MISSING")
        actual_seed  = cfg.get("seed", "?")

        vals = {}
        for label, key in METRIC_MAP.items():
            vals[label] = sm.get(key)

        issues = []
        if state != "finished":
            issues.append(f"state={state}")
        if aft_foil_srf is not True:
            issues.append(f"aft_foil_srf={aft_foil_srf}")
        if str(actual_seed) != seed:
            issues.append(f"seed_mismatch:cfg_seed={actual_seed}")

        rep = reported[seed]
        tols = {"p_in": 0.2, "p_tan": 0.2, "p_oodc": 0.2, "p_re": 0.2, "val_loss": 0.002}
        for k in ["p_in", "p_tan", "p_oodc", "p_re"]:
            v = vals[k]
            if v is None:
                issues.append(f"{k}_MISSING")
            elif abs(v - rep[k]) > tols[k]:
                issues.append(f"{k}:{v:.2f}!={rep[k]}")
        v = vals["val_loss"]
        if v is None:
            issues.append("val_loss_MISSING")
        elif abs(v - rep["val_loss"]) > tols["val_loss"]:
            issues.append(f"val_loss:{v:.4f}!={rep['val_loss']:.4f}")

        for k, lst in all_vals.items():
            if vals[k] is not None:
                lst.append(vals[k])

        def fmt(v):
            if v is None: return "  NaN"
            return f"{v:7.2f}" if v > 1 else f"{v:7.4f}"

        print(f"{seed:<6} {run_id:<10} {state:<10} {str(aft_foil_srf):<14} "
              f"{fmt(vals['p_in'])} {fmt(vals['p_tan'])} {fmt(vals['p_oodc'])} {fmt(vals['p_re'])} "
              f"{vals['val_loss']:9.6f}  "
              f"{'OK' if not issues else ', '.join(issues)}")

        if issues:
            discrepancies.append((seed, run_id, issues))

    except Exception as e:
        print(f"{seed:<6} {run_id:<10} ERROR: {e}")
        discrepancies.append((seed, run_id, [str(e)]))

print()
print("=== 8-SEED AGGREGATES (from W&B) ===")
computed = {}
for label in ["p_in", "p_tan", "p_oodc", "p_re", "val_loss"]:
    arr = np.array(all_vals[label])
    m, s = arr.mean(), arr.std()
    computed[label] = (m, s)
    print(f"  {label}: mean={m:.4f} ± {s:.4f}  (n={len(arr)})")

print()
print("=== CLAIMED vs COMPUTED ===")
claimed = {
    "p_in":   (13.19, 0.33),
    "p_tan":  (30.05, 0.36),
    "p_oodc": (7.92,  0.17),
    "p_re":   (6.45,  0.07),
}
all_stats_match = True
for k, (cm, cs) in claimed.items():
    gm, gs = computed[k]
    mean_ok = abs(gm - cm) < 0.02
    std_ok  = abs(gs - cs) < 0.05
    status = "OK" if (mean_ok and std_ok) else "MISMATCH"
    if status != "OK":
        all_stats_match = False
    print(f"  {k:<8}: claimed={cm:.2f}±{cs:.2f}  computed={gm:.2f}±{gs:.2f}  [{status}]")

print()
if discrepancies:
    print(f"PER-RUN DISCREPANCIES ({len(discrepancies)} runs):")
    for s, r, iss in discrepancies:
        print(f"  seed={s} run={r}: {iss}")
else:
    print("No per-run discrepancies.")

all_runs_ok = not discrepancies
print()
print("=== FINAL VERDICT ===")
if all_runs_ok and all_stats_match:
    print("VERIFIED: All 8 runs match reported metrics and 8-seed stats confirmed.")
else:
    parts = []
    if not all_runs_ok:
        parts.append(f"{len(discrepancies)} per-run discrepancies")
    if not all_stats_match:
        parts.append("aggregate stats mismatch")
    print(f"NOT VERIFIED: {'; '.join(parts)}")
