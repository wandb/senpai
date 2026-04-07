import os
import wandb

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

run_ids = {
    "Baseline s42": "3mim1mhi",
    "Baseline s43": "gk16mcse",
    "SWAD s42":     "1ee50z25",
    "SWAD s43":     "hbm6rfcg",
    "SWAD s44":     "hsvhokae",
    "SWAD s45":     "r632qi5f",
    "SWAD s66":     "86sd67n7",
    "SWAD s67":     "tm0513wp",
}

BEST_KEYS = {
    "p_in":   "best_best_val_in_dist/mae_surf_p",
    "p_oodc": "best_best_val_ood_cond/mae_surf_p",
    "p_tan":  "best_best_val_tandem_transfer/mae_surf_p",
    "p_re":   "best_best_val_ood_re/mae_surf_p",
}

print("=" * 110)
print(f"{'Label':<16} {'Run ID':<12} {'State':<10} {'Epochs':>7} {'BestEp':>7} {'p_in':>8} {'p_oodc':>8} {'p_tan':>8} {'p_re':>8} {'SWAD?':>6}")
print("=" * 110)

for label, run_id in run_ids.items():
    run = api.run(f"{path}/{run_id}")
    sm = run.summary_metrics
    cfg = run.config

    total_epochs = sm.get("total_epochs", "?")
    best_epoch = sm.get("best_epoch", "?")
    swad_enabled = cfg.get("swad", False)

    p_in   = sm.get(BEST_KEYS["p_in"],   float("nan"))
    p_oodc = sm.get(BEST_KEYS["p_oodc"], float("nan"))
    p_tan  = sm.get(BEST_KEYS["p_tan"],  float("nan"))
    p_re   = sm.get(BEST_KEYS["p_re"],   float("nan"))

    def fmt(v):
        return f"{v:.4f}" if v == v else "NaN"

    print(f"{label:<16} {run_id:<12} {run.state:<10} {str(total_epochs):>7} {str(best_epoch):>7} {fmt(p_in):>8} {fmt(p_oodc):>8} {fmt(p_tan):>8} {fmt(p_re):>8} {str(swad_enabled):>6}")

print()
print("Note: Metrics are from best_best_val_*/mae_surf_p (best checkpoint, not final epoch)")
print()
print("=" * 110)
print("SWAD implementation verification")
print("=" * 110)
print()
print("Looking for SWAD-specific keys in SWAD run summaries...")
for label, run_id in run_ids.items():
    if "SWAD" not in label:
        continue
    run = api.run(f"{path}/{run_id}")
    sm = run.summary_metrics
    cfg = run.config

    swad_keys_cfg = {k: v for k, v in cfg.items() if "swad" in k.lower() or "swa" in k.lower()}
    swad_keys_sum = {k: v for k, v in sm.items() if "swad" in k.lower() or "swa" in k.lower()}

    print(f"\n{label} ({run_id}):")
    print(f"  Config SWAD/SWA keys: {swad_keys_cfg}")
    print(f"  Summary SWAD/SWA keys: {swad_keys_sum if swad_keys_sum else 'NONE'}")
    print(f"  best_epoch={sm.get('best_epoch')}, total_epochs={sm.get('total_epochs')}, best_val_loss={sm.get('best_val_loss', 'N/A'):.6f}")
