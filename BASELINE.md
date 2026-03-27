# Baseline Metrics

## Current Baseline (Phase 4 — 2026-03-27, cosine_T_max=180)

| Metric | Mean (4 seeds) | Std | Best single | Worst single |
|--------|---------------|-----|-------------|--------------|
| val/loss | 0.4016 | 0.001 | 0.4002 | 0.4028 |
| p_in | 13.3 | 0.2 | 13.0 | 13.4 |
| p_oodc | 8.3 | 0.2 | 8.1 | 8.5 |
| p_tan | 33.1 | 0.4 | 32.6 | 33.4 |
| p_re | 24.7 | 0.2 | 24.5 | 24.9 |

**Improvement from Phase 3 final:** val/loss -0.8%, p_in -2.2%, p_oodc -4.6%, p_tan -1.2%. All confirmed across 4 seeds.

### Reproduce

```bash
python train.py --agent <name> --wandb_name "<name>/baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 180
```

### History

| Date | val/loss | p_in | p_oodc | p_tan | p_re | PR | Notes |
|------|----------|------|--------|-------|------|----|-------|
| 2026-03-27 | 0.4016±0.001 | 13.3±0.2 | 8.3±0.2 | 33.1±0.4 | 24.7±0.2 | #1845 | **Phase 4: cosine_T_max=180** (4-seed validated) |
| 2026-03-27 | 0.405±0.004 | 13.6±0.5 | 8.7±0.3 | 33.5±0.6 | 24.7±0.2 | #1836 | Previous baseline (wd=5e-5 ~no-op, wide variance) |
| 2026-03-26 | 0.3994 | 13.0 | 8.7 | 33.2 | 24.6 | Phase 3 final | 7 merges over 87 experiments |
