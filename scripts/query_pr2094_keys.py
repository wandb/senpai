import os
import wandb

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

# Check one SWAD run and one baseline run for key structure
check_runs = {
    "Baseline s42": "3mim1mhi",
    "SWAD s42": "1ee50z25",
}

for label, run_id in check_runs.items():
    run = api.run(f"{path}/{run_id}")
    print(f"\n{'='*60}")
    print(f"{label} ({run_id}) — state: {run.state}")

    # Summary keys
    sm = run.summary_metrics
    print(f"\nSummary keys ({len(sm)} total):")
    for k in sorted(sm.keys()):
        v = sm[k]
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Config keys relevant to SWAD
    cfg = run.config
    swad_cfg = {k: v for k, v in cfg.items() if "swad" in k.lower()}
    print(f"\nSWAD config keys: {swad_cfg}")

    # Scan history without key filter to see what columns exist
    rows = list(run.scan_history(page_size=5))[:5]
    if rows:
        print(f"\nFirst history row keys: {sorted(rows[0].keys())}")
