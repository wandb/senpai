# PR #2417: DrivAerML EMA Warmup For Surface Pressure

Labels: `workshop-r1`, `student:ws-fern`, `status:wip`

## Hypothesis

Adding EMA warmup with best-checkpoint restore will reduce DrivAerML held-out surface-pressure relative-L2 because the current grouped Transolver shows validation volatility late in training.

## Contract

Use the public DrivAerML `400 train / 34 val / 50 test` split. Report per-case relative-L2 percentages on denormalized predictions.

Primary merge metric:

- `test_primary/surface_pressure_rel_l2_pct`, lower is better.

## Baseline

- Current baseline: `test_primary/surface_pressure_rel_l2_pct = 6.24`
- Baseline W&B run: `base-6p24`

## Instructions

1. Add EMA decay `0.999` with warmup after 10% of optimizer steps.
2. Restore best validation checkpoint before final test metrics.
3. Log `ema_decay`, `ema_warmup_fraction`, `restore_best_checkpoint`, command, and git commit to W&B.
4. Post a terminal `SENPAI-RESULT` only after final test metrics are present and finite.

## Falsifier

If validation improves but held-out test surface-pressure relative-L2 is worse than 6.24, do not claim a win.
