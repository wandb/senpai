import sys
sys.path.insert(0, "/workspace/senpai/.claude/skills/wandb-primary/scripts")

import wandb
import os

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = {
    "baseline-3L-96s-s42": "u00xzh8z",
    "baseline-3L-96s-s73": "3y9kazhf",
    "deeper-5L-96s-s42": "z8xu1h7q",
    "deeper-5L-96s-s73": "u1zdhi63",
    "wider-3L-160s-s42": "mkxyc0mf",
    "wider-3L-160s-s73": "fzabfqmz",
    "deep-wide-4L-128s-s42": "i4o4g2qm",
    "deep-wide-4L-128s-s73": "910al9e9",
}

metric_keys = [
    "val_in_dist/mae_surf_p",
    "val_ood_cond/mae_surf_p",
    "val_tandem_transfer/mae_surf_p",
    "val_ood_re/mae_surf_p",
]

print(f"{'Config':<28} {'Run ID':<12} {'State':<12} {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8}")
print("-" * 96)

results = {}
for config, run_id in run_ids.items():
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        state = run.state
        summary = run.summary_metrics

        p_in   = summary.get("val_in_dist/mae_surf_p")
        p_oodc = summary.get("val_ood_cond/mae_surf_p")
        p_tan  = summary.get("val_tandem_transfer/mae_surf_p")
        p_re   = summary.get("val_ood_re/mae_surf_p")

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  NaN  "

        print(f"{config:<28} {run_id:<12} {state:<12} {fmt(p_in):>8} {fmt(p_oodc):>8} {fmt(p_tan):>8} {fmt(p_re):>8}")

        results[config] = {
            "run_id": run_id,
            "state": state,
            "p_in": p_in,
            "p_oodc": p_oodc,
            "p_tan": p_tan,
            "p_re": p_re,
        }

        # Check for training issues
        if state not in ("finished", "crashed"):
            pass  # will note below
        if any(v is None for v in [p_in, p_oodc, p_tan, p_re]):
            print(f"  *** WARNING: Missing metrics for {config} ({run_id})")

    except Exception as e:
        print(f"{config:<28} {run_id:<12} ERROR: {e}")
        results[config] = {"run_id": run_id, "state": "ERROR", "error": str(e)}

print()
print("=== GROUPED AVERAGES (by architecture) ===")
import statistics

groups = {
    "baseline-3L-96s":    ["baseline-3L-96s-s42", "baseline-3L-96s-s73"],
    "deeper-5L-96s":      ["deeper-5L-96s-s42", "deeper-5L-96s-s73"],
    "wider-3L-160s":      ["wider-3L-160s-s42", "wider-3L-160s-s73"],
    "deep-wide-4L-128s":  ["deep-wide-4L-128s-s42", "deep-wide-4L-128s-s73"],
}

print(f"{'Group':<24} {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8}")
print("-" * 60)
for group, configs in groups.items():
    vals = {k: [] for k in ["p_in", "p_oodc", "p_tan", "p_re"]}
    for c in configs:
        r = results.get(c, {})
        for k in vals:
            v = r.get(k)
            if v is not None:
                vals[k].append(v)

    def avg(lst):
        return f"{statistics.mean(lst):.4f}" if lst else "  NaN  "

    print(f"{group:<24} {avg(vals['p_in']):>8} {avg(vals['p_oodc']):>8} {avg(vals['p_tan']):>8} {avg(vals['p_re']):>8}")

print()
print("=== RUN STATES ===")
for config, r in results.items():
    state = r.get("state", "unknown")
    if state != "finished":
        print(f"  {config} ({r['run_id']}): state={state}")
    else:
        print(f"  {config} ({r['run_id']}): finished OK")
