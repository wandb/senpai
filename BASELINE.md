# Baseline Metrics

## Current Baseline (Phase 4 — 2026-03-28, Pressure-First Deep)

| Metric | Mean (4 seeds) | Std | Best single | Worst single |
|--------|---------------|-----|-------------|--------------|
| val/loss | 0.401 | 0.005 | 0.395 | 0.406 |
| p_in | 12.95 | 0.3 | 12.6 | 13.4 |
| p_oodc | 8.40 | 0.4 | 8.0 | 8.9 |
| p_tan | 33.8 | 0.5 | 33.3 | 34.4 |
| p_re | 24.7 | 0.2 | 24.4 | 24.9 |

**Phase 4 improvements merged:** T_max=180 (#1845) + disable_pcgrad (#1846) + pressure_first + pressure_deep (#1867). Memory: ~36.2 GB.

### Reproduce

```bash
python train.py --agent <name> --wandb_name "<name>/baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 180 --disable_pcgrad \
  --pressure_first --pressure_deep
```

### History

| Date | val/loss | p_in | p_oodc | p_tan | p_re | PR | Notes |
|------|----------|------|--------|-------|------|----|-------|
| 2026-03-28 | 0.401±0.005 | 12.95±0.3 | 8.40±0.4 | 33.8±0.5 | 24.7±0.2 | #1867 | **Phase 4: + pressure_first + pressure_deep** (p_in -4.8%) |
| 2026-03-27 | 0.403±0.003 | 13.6±0.4 | 8.6±0.1 | 33.1±0.6 | 24.7±0.1 | #1846 | Phase 4: + disable_pcgrad (18% memory reduction) |
| 2026-03-27 | 0.4016±0.001 | 13.3±0.2 | 8.3±0.2 | 33.1±0.4 | 24.7±0.2 | #1845 | Phase 4: cosine_T_max=180 (4-seed validated) |
| 2026-03-27 | 0.405±0.004 | 13.6±0.5 | 8.7±0.3 | 33.5±0.6 | 24.7±0.2 | #1836 | Previous baseline (wd=5e-5 ~no-op) |
| 2026-03-26 | 0.3994 | 13.0 | 8.7 | 33.2 | 24.6 | Phase 3 final | 7 merges over 87 experiments |
