"""Fetch W&B metrics for PR #2107 runs."""
import os
import sys
import wandb

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
project = os.environ.get("WANDB_PROJECT", "senpai-v1")

# Run IDs from PR comments
run_ids = {
    # v1 (coord-replace)
    "00lod6uk": "v1 s42",
    "3llpj5yj": "v1 s73",
    # dual-frame initial
    "7e0dma73": "dual s42 (initial)",
    "w8cyqceg": "dual s73 (initial)",
    # 4-seed v2 final
    "bklq38ec": "v2-4seed s42",
    "qlkaovuv": "v2-4seed s73",
    "bw5ny846": "v2-4seed s44",
    "74cxcgue": "v2-4seed s45",
}

surface_keys = ["test/p_in", "test/p_oodc", "test/p_tan", "test/p_re",
                "surface/p_in", "surface/p_oodc", "surface/p_tan", "surface/p_re",
                "val/loss", "val/p_in", "val/p_oodc", "val/p_tan", "val/p_re"]

print(f"Entity: {entity}, Project: {project}\n")
print(f"{'Run ID':<12} {'Label':<22} {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8} {'val_loss':>10}")
print("-" * 90)

for run_id, label in run_ids.items():
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        sm = run.summary_metrics

        # Try multiple key naming conventions
        p_in = (sm.get("test/p_in") or sm.get("surface/p_in") or
                sm.get("val/p_in") or sm.get("p_in"))
        p_oodc = (sm.get("test/p_oodc") or sm.get("surface/p_oodc") or
                  sm.get("val/p_oodc") or sm.get("p_oodc"))
        p_tan = (sm.get("test/p_tan") or sm.get("surface/p_tan") or
                 sm.get("val/p_tan") or sm.get("p_tan"))
        p_re = (sm.get("test/p_re") or sm.get("surface/p_re") or
                sm.get("val/p_re") or sm.get("p_re"))
        val_loss = sm.get("val/loss") or sm.get("val_loss") or sm.get("loss")

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A  "

        print(f"{run_id:<12} {label:<22} {fmt(p_in):>8} {fmt(p_oodc):>8} {fmt(p_tan):>8} {fmt(p_re):>8} {fmt(val_loss):>10}")
        print(f"  -> run.name={run.name}, state={run.state}, group={run.group}")

        # Print all summary keys if metrics not found
        if p_in is None:
            keys = [k for k in sm.keys() if not k.startswith("_") and
                    any(x in k.lower() for x in ["mae", "loss", "p_in", "p_oodc", "p_tan", "p_re", "surface"])]
            print(f"  -> Available metric keys: {sorted(keys)[:20]}")

    except Exception as e:
        print(f"{run_id:<12} {label:<22} ERROR: {e}")

print("\n--- Checking W&B groups ---")
for group_name in ["phase6/aft-foil-local-frame", "phase6/aft-foil-local-frame-v2"]:
    try:
        runs = api.runs(f"{entity}/{project}",
                       filters={"group": group_name},
                       order="-created_at")
        group_runs = runs[:20]
        print(f"\nGroup: {group_name} — {len(group_runs)} runs found")
        for r in group_runs:
            sm = r.summary_metrics
            p_in = sm.get("test/p_in") or sm.get("surface/p_in") or sm.get("val/p_in") or sm.get("p_in")
            p_oodc = sm.get("test/p_oodc") or sm.get("surface/p_oodc") or sm.get("val/p_oodc") or sm.get("p_oodc")
            p_tan = sm.get("test/p_tan") or sm.get("surface/p_tan") or sm.get("val/p_tan") or sm.get("p_tan")
            p_re = sm.get("test/p_re") or sm.get("surface/p_re") or sm.get("val/p_re") or sm.get("p_re")
            def fmt(v): return f"{v:.4f}" if v is not None else "N/A"
            print(f"  {r.id:<12} {r.name:<35} p_in={fmt(p_in)} p_oodc={fmt(p_oodc)} p_tan={fmt(p_tan)} p_re={fmt(p_re)} state={r.state}")
    except Exception as e:
        print(f"Error fetching group {group_name}: {e}")
