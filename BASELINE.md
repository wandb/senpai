# Baseline Metrics

## Current Baseline (Phase 3 Final — 2026-03-26)

| Metric | Value |
|--------|-------|
| val/loss | 0.3994 |
| p_in (in-distribution) | 13.0 |
| p_oodc (OOD conditions) | 8.7 |
| p_tan (tandem transfer) | 33.2 |
| p_re (OOD Reynolds) | 24.6 |

### Reproduce

```bash
python train.py --agent <name> --wandb_name "<name>/baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999
```

### History

| Date | val/loss | p_in | p_oodc | p_tan | p_re | PR | Notes |
|------|----------|------|--------|-------|------|----|-------|
| 2026-03-26 | 0.3994 | 13.0 | 8.7 | 33.2 | 24.6 | Phase 3 final | 7 merges over 87 experiments |
