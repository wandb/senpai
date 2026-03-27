# Baseline Metrics

## Current Baseline (Phase 4 R1 — 2026-03-27)

| Metric | Value |
|--------|-------|
| val/loss | 0.3985 |
| p_in (in-distribution) | 13.3 |
| p_oodc (OOD conditions) | 8.4 |
| p_tan (tandem transfer) | 31.9 |
| p_re (OOD Reynolds) | 24.7 |

W&B run: [ace002qd](https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/ace002qd)

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
| 2026-03-27 | 0.3985 | 13.3 | 8.4 | 31.9 | 24.7 | #1833 | Phase 4 R1: weight_decay=5e-5 |
| 2026-03-26 | 0.3994 | 13.0 | 8.7 | 33.2 | 24.6 | Phase 3 final | 7 merges over 87 experiments |
