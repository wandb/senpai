# TandemFoilSet Baseline

## Current Best Result

No experiments completed yet on `icml-appendix-charlie-pai2-r2`. Round 1 sweeps are in progress.

## Vanilla Transolver Defaults

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- Optimizer: AdamW, Scheduler: CosineAnnealingLR (T_max=epochs)
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, `act=gelu`, `dropout=0.0`)
- Loss: MSE normalized space, `vol_loss + surf_weight * surf_loss`

**Primary metric**: `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 val splits)

## Baseline Metrics

No validated baseline metrics yet — will be established after Round 1 results.
