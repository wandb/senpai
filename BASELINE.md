# Baseline Metrics

## Current Baseline (Phase 5 — 2026-03-29, Residual Prediction)

| Metric | Mean (4 seeds) | Std | Best single | Worst single |
|--------|---------------|-----|-------------|--------------|
| val/loss | 0.396 | 0.003 | 0.392 | 0.399 |
| p_in | 12.93 | 0.22 | 12.6 | 13.2 |
| p_oodc | 7.98 | 0.19 | 7.7 | 8.2 |
| p_tan | 32.93 | 0.29 | 32.5 | 33.3 |
| p_re | 24.53 | 0.08 | 24.4 | 24.6 |

**Phase 5 improvement merged:** + residual_prediction (#1927). Predicts correction to freestream instead of full field. p_oodc -4.7%, p_tan -1.9%, val/loss -1.8%. Memory: ~36.2 GB.
**W&B group:** `phase5/residual-prediction` (runs umj5afev, 21pomc9o, arfuy4np, v3f0q6sk).

### Reproduce

```bash
python train.py --agent <name> --wandb_name "<name>/baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 180 --disable_pcgrad \
  --pressure_first --pressure_deep --residual_prediction
```

### History

| Date | val/loss | p_in | p_oodc | p_tan | p_re | PR | Notes |
|------|----------|------|--------|-------|------|----|-------|
| 2026-03-29 | 0.396±0.003 | 12.93±0.22 | 7.98±0.19 | 32.93±0.29 | 24.53±0.08 | #1927 | **Phase 5: + residual_prediction** (p_oodc -4.7%, p_tan -1.9%) |
| 2026-03-28 | 0.404±0.004 | 13.33±0.58 | 8.37±0.22 | 33.57±0.44 | 24.58±0.13 | #1911 | 8-seed characterization |
| 2026-03-28 | 0.401±0.005 | 12.95±0.3 | 8.40±0.4 | 33.8±0.5 | 24.7±0.2 | #1867 | **Phase 4: + pressure_first + pressure_deep** (p_in -4.8%) |
| 2026-03-27 | 0.403±0.003 | 13.6±0.4 | 8.6±0.1 | 33.1±0.6 | 24.7±0.1 | #1846 | Phase 4: + disable_pcgrad (18% memory reduction) |
| 2026-03-27 | 0.4016±0.001 | 13.3±0.2 | 8.3±0.2 | 33.1±0.4 | 24.7±0.2 | #1845 | Phase 4: cosine_T_max=180 (4-seed validated) |
| 2026-03-27 | 0.405±0.004 | 13.6±0.5 | 8.7±0.3 | 33.5±0.6 | 24.7±0.2 | #1836 | Previous baseline (wd=5e-5 ~no-op) |
| 2026-03-26 | 0.3994 | 13.0 | 8.7 | 33.2 | 24.6 | Phase 3 final | 7 merges over 87 experiments |
