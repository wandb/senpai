# Baseline Metrics

## Current Baseline (Phase 4 — 2026-03-27, multi-seed validated)

| Metric | Mean (4 seeds) | Std | Best single | Worst single |
|--------|---------------|-----|-------------|--------------|
| val/loss | 0.405 | 0.004 | 0.399 | 0.413 |
| p_in | 13.6 | 0.5 | 13.0 | 14.2 |
| p_oodc | 8.7 | 0.3 | 8.4 | 9.0 |
| p_tan | 33.5 | 0.6 | 31.9 | 34.4 |
| p_re | 24.7 | 0.2 | 24.5 | 24.9 |

**NOTE:** PR #1836 (WD sweep) showed that wd=5e-5 is effectively a no-op with Lion at lr=2e-4 (0.05% total decay). The PR #1833 result (p_tan=31.9) was a lucky seed. Multi-seed validation shows p_tan mean=33.5±0.6. To claim improvement, experiments need p_tan < 31 or val/loss < 0.395 across 3+ seeds.

### Reproduce

```bash
python train.py --agent <name> --wandb_name "<name>/baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5
```

### History

| Date | val/loss | p_in | p_oodc | p_tan | p_re | PR | Notes |
|------|----------|------|--------|-------|------|----|-------|
| 2026-03-27 | 0.405±0.004 | 13.6±0.5 | 8.7±0.3 | 33.5±0.6 | 24.7±0.2 | #1836 | Multi-seed validation (4 seeds), wd=5e-5 ~no-op |
| 2026-03-27 | 0.3985 | 13.3 | 8.4 | 31.9 | 24.7 | #1833 | Phase 4 R1: weight_decay=5e-5 (lucky seed) |
| 2026-03-26 | 0.3994 | 13.0 | 8.7 | 33.2 | 24.6 | Phase 3 final | 7 merges over 87 experiments |
