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

metric_keys = [
    "surface_mae/p_in",
    "surface_mae/p_tan",
    "surface_mae/p_oodc",
    "surface_mae/p_re",
    "val/loss",
]

all_p_in, all_p_tan, all_p_oodc, all_p_re, all_val_loss = [], [], [], [], []
discrepancies = []

print(f"{'Seed':<6} {'State':<12} {'aft_foil_srf':<14} {'p_in':>8} {'p_tan':>8} {'p_oodc':>8} {'p_re':>8} {'val/loss':>10} {'Issues'}")
print("-" * 100)

for seed, run_id in run_ids:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        state = run.state
        cfg = run.config
        aft_foil_srf = cfg.get("aft_foil_srf", cfg.get("use_aft_foil_srf", "NOT_FOUND"))
        sm = run.summary_metrics

        p_in   = sm.get("surface_mae/p_in")
        p_tan  = sm.get("surface_mae/p_tan")
        p_oodc = sm.get("surface_mae/p_oodc")
        p_re   = sm.get("surface_mae/p_re")
        val_loss = sm.get("val/loss")

        issues = []
        if state not in ("finished",):
            issues.append(f"state={state}")
        if not aft_foil_srf and aft_foil_srf != "NOT_FOUND":
            issues.append("aft_foil_srf=False")
        if aft_foil_srf == "NOT_FOUND":
            issues.append("aft_foil_srf_flag_missing")

        rep = reported[seed]
        def chk(name, actual, expected, tol=0.15):
            if actual is None:
                return f"{name}_MISSING"
            if abs(actual - expected) > tol:
                return f"{name}:{actual:.2f}!={expected}"
            return None

        for chk_res in [
            chk("p_in",   p_in,   rep["p_in"]),
            chk("p_tan",  p_tan,  rep["p_tan"]),
            chk("p_oodc", p_oodc, rep["p_oodc"]),
            chk("p_re",   p_re,   rep["p_re"]),
            chk("val_loss", val_loss, rep["val_loss"], tol=0.002),
        ]:
            if chk_res:
                issues.append(chk_res)

        if p_in   is not None: all_p_in.append(p_in)
        if p_tan  is not None: all_p_tan.append(p_tan)
        if p_oodc is not None: all_p_oodc.append(p_oodc)
        if p_re   is not None: all_p_re.append(p_re)
        if val_loss is not None: all_val_loss.append(val_loss)

        print(f"{seed:<6} {state:<12} {str(aft_foil_srf):<14} {p_in or 'NaN':>8} {p_tan or 'NaN':>8} {p_oodc or 'NaN':>8} {p_re or 'NaN':>8} {val_loss or 'NaN':>10} {', '.join(issues) if issues else 'OK'}")

        if issues:
            discrepancies.append((seed, run_id, issues))

    except Exception as e:
        print(f"{seed:<6} ERROR: {e}")
        discrepancies.append((seed, run_id, [str(e)]))

print()
print("=== 8-SEED AGGREGATES ===")
def stats(arr, name):
    a = np.array(arr)
    print(f"  {name}: mean={a.mean():.4f}, std={a.std():.4f}, n={len(a)}")
    return a.mean(), a.std()

m_pin,   s_pin   = stats(all_p_in,   "p_in  ")
m_ptan,  s_ptan  = stats(all_p_tan,  "p_tan ")
m_poodc, s_poodc = stats(all_p_oodc, "p_oodc")
m_pre,   s_pre   = stats(all_p_re,   "p_re  ")
m_vl,    s_vl    = stats(all_val_loss,"val/loss")

print()
print("=== CLAIMED vs COMPUTED ===")
claimed = {
    "p_in":   (13.19, 0.33),
    "p_tan":  (30.05, 0.36),
    "p_oodc": (7.92,  0.17),
    "p_re":   (6.45,  0.07),
}
computed = {
    "p_in":   (m_pin,   s_pin),
    "p_tan":  (m_ptan,  s_ptan),
    "p_oodc": (m_poodc, s_poodc),
    "p_re":   (m_pre,   s_pre),
}
all_match = True
for k, (cm, cs) in claimed.items():
    gm, gs = computed[k]
    mean_ok = abs(gm - cm) < 0.02
    std_ok  = abs(gs - cs) < 0.05
    status = "OK" if (mean_ok and std_ok) else "MISMATCH"
    if status != "OK":
        all_match = False
    print(f"  {k}: claimed={cm:.2f}±{cs:.2f}, computed={gm:.2f}±{gs:.2f}  [{status}]")

print()
if discrepancies:
    print(f"DISCREPANCIES FOUND ({len(discrepancies)} runs):")
    for seed, rid, issues in discrepancies:
        print(f"  seed={seed} run={rid}: {issues}")
else:
    print("No per-run discrepancies.")

print()
print("VERDICT:", "VERIFIED" if (all_match and not discrepancies) else "NOT VERIFIED")
