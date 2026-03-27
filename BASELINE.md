# Baseline Metrics

## Current Baseline (Phase 4 — 2026-03-27, T_max=180 + disable_pcgrad)

| Metric | Mean (4 seeds) | Std | Best single | Worst single |
|--------|---------------|-----|-------------|--------------|
| val/loss | 0.403 | 0.003 | 0.399 | 0.406 |
| p_in | 13.6 | 0.4 | 13.2 | 14.1 |
| p_oodc | 8.6 | 0.1 | 8.4 | 8.7 |
| p_tan | 33.1 | 0.6 | 32.6 | 33.9 |
| p_re | 24.7 | 0.1 | 24.5 | 24.8 |

**Phase 4 improvements merged:** T_max=180 (#1845) + disable_pcgrad (#1846). Memory: 34.4 GB (down from 42 GB).

### Reproduce

```bash
python train.py --agent <name> --wandb_name "<name>/baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 180 --disable_pcgrad
```

### History

| Date | val/loss | p_in | p_oodc | p_tan | p_re | PR | Notes |
|------|----------|------|--------|-------|------|----|-------|
| 2026-03-27 | 0.403±0.003 | 13.6±0.4 | 8.6±0.1 | 33.1±0.6 | 24.7±0.1 | #1846 | **Phase 4: + disable_pcgrad** (18% memory reduction) |
| 2026-03-27 | 0.4016±0.001 | 13.3±0.2 | 8.3±0.2 | 33.1±0.4 | 24.7±0.2 | #1845 | Phase 4: cosine_T_max=180 (4-seed validated) |
| 2026-03-27 | 0.405±0.004 | 13.6±0.5 | 8.7±0.3 | 33.5±0.6 | 24.7±0.2 | #1836 | Previous baseline (wd=5e-5 ~no-op) |
| 2026-03-26 | 0.3994 | 13.0 | 8.7 | 33.2 | 24.6 | Phase 3 final | 7 merges over 87 experiments |
