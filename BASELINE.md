# TandemFoilSet Baseline

## Current Best

| Metric | Value | PR | Notes |
|--------|-------|-----|-------|
| val_avg/mae_surf_p | — | — | No experiments run yet |
| test_avg/mae_surf_p | — | — | No experiments run yet |

## Baseline Configuration

The starting baseline is the default Transolver in `target/train.py`:

- **Model**: Transolver (space_dim=2, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (T_max=epochs)
- **Loss**: vol_loss + surf_weight * surf_loss (surf_weight=10.0)
- **Batch size**: 4
- **Epochs**: 50 (SENPAI_TIMEOUT_MINUTES=30 default)
- **No grad clipping, no EMA**

## Notes

This is a fresh research track targeting the TandemFoilSet benchmark. The primary metric is
`val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 validation splits.
Lower is better.
