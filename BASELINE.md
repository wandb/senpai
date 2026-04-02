# Baseline Metrics

## Current Baseline (Phase 5 — 2026-04-02, Faster LR Decay T_max=160)

| Metric | Single seed (s42) | Phase 5 prior | Change |
|--------|-------------------|---------------|--------|
| val/loss | **0.3761** | 0.383 | -1.8% |
| p_in | **12.5** | 12.95 | -3.5% |
| p_oodc | **8.2** | 8.31 | -1.3% |
| p_tan | **29.8** | 30.01 | -0.7% |
| p_re | **6.5** | 6.70 | -3.0% |

**PR #2003** — cosine_T_max 180→160. Single seed, 155 epochs, 38.0 GB VRAM.
W&B run: `9ysz96ll`

**Reproduce:**
```bash
cd cfd_tandemfoil && python train.py --agent edward --wandb_name "baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 160 --disable_pcgrad \
  --pressure_first --pressure_deep --residual_prediction \
  --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3
```

⚠️ **Note:** Single-seed result. Multi-seed validation recommended to confirm robustness.

---

## Previous Baseline (Phase 5 — 2026-03-29, Residual Prediction + Surface Refinement)

| Metric | Mean (8 seeds) | Std | Best single | Worst single |
|--------|---------------|-----|-------------|--------------|
| val/loss | 0.383 | 0.003 | 0.378 | 0.389 |
| p_in | 12.95 | 0.30 | 12.1 | 13.4 |
| p_oodc | 8.31 | 0.18 | 8.1 | 8.7 |
| p_tan | 30.01 | 0.52 | 29.2 | 31.0 |
| p_re | 6.70 | 0.10 | 6.5 | 6.8 |

**Phase 5 improvements merged:**
1. residual_prediction (#1927) — predict correction to freestream. p_oodc -4.7%, p_tan -1.9%.
2. surface_refine (#1935) — dedicated surface-only refinement MLP. p_re -72.7%, p_tan -8.9%, val/loss -3.3%. Manually verified (no target leakage).

Memory: ~38.0 GB. W&B group: `phase5/surface-refine-8seed`.

### Reproduce

```bash
python train.py --agent <name> --wandb_name "<name>/baseline" \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 180 --disable_pcgrad \
  --pressure_first --pressure_deep --residual_prediction \
  --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3
```

### History

| Date | val/loss | p_in | p_oodc | p_tan | p_re | PR | Notes |
|------|----------|------|--------|-------|------|----|-------|
| 2026-03-29 | 0.383±0.003 | 12.95±0.30 | 8.31±0.18 | 30.01±0.52 | 6.70±0.10 | #1935 | **Phase 5: + surface_refine** (p_re -72.7%, p_tan -8.9%) |
| 2026-03-29 | 0.396±0.003 | 12.93±0.22 | 7.98±0.19 | 32.93±0.29 | 24.53±0.08 | #1927 | **Phase 5: + residual_prediction** (p_oodc -4.7%, p_tan -1.9%) |
| 2026-03-28 | 0.404±0.004 | 13.33±0.58 | 8.37±0.22 | 33.57±0.44 | 24.58±0.13 | #1911 | 8-seed characterization |
| 2026-03-28 | 0.401±0.005 | 12.95±0.3 | 8.40±0.4 | 33.8±0.5 | 24.7±0.2 | #1867 | **Phase 4: + pressure_first + pressure_deep** (p_in -4.8%) |
| 2026-03-27 | 0.403±0.003 | 13.6±0.4 | 8.6±0.1 | 33.1±0.6 | 24.7±0.1 | #1846 | Phase 4: + disable_pcgrad (18% memory reduction) |
| 2026-03-27 | 0.4016±0.001 | 13.3±0.2 | 8.3±0.2 | 33.1±0.4 | 24.7±0.2 | #1845 | Phase 4: cosine_T_max=180 (4-seed validated) |
| 2026-03-27 | 0.405±0.004 | 13.6±0.5 | 8.7±0.3 | 33.5±0.6 | 24.7±0.2 | #1836 | Previous baseline (wd=5e-5 ~no-op) |
| 2026-03-26 | 0.3994 | 13.0 | 8.7 | 33.2 | 24.6 | Phase 3 final | 7 merges over 87 experiments |
