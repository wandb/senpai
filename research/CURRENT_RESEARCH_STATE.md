# SENPAI Research State

- **Date:** 2026-04-20
- **Branch:** radford
- **Tag:** radford-20260420

## Student Status
- **Idle (16):** frieren, fern, tanjiro, nezuko, alphonse, edward, thorfinn, askeladd, violet, gilbert, senku, kohaku, emma, norman, chihiro, haku
- **WIP:** none
- **Review-ready:** none

## Human Researcher Directives
- Focus on TandemFoilSet and AirfRANS (not DrivAerML for this run)
- Tune and ablate the current stack — no new architecture invention
- Establish strong, defensible recipes on both datasets
- Show the same overall modeling/preprocessing approach transfers across datasets

## Current Research Focus
**Phase: Round 1 — Baseline Establishment & First-Order Sweeps**

This is a fresh start on the radford branch. No baselines exist yet. Round 1 goals:
1. Establish clean baselines on both TandemFoilSet and AirfRANS
2. Sweep learning rate (3e-4, 5e-4, 1e-3) on both datasets
3. Compare Lion vs AdamW on both datasets
4. Test model capacity (bigger, deeper) on both datasets
5. Ablate TandemFoil physics features (TE coord frame, Cp panel, wake deficit, asinh, residual prediction)
6. Ablate surface refinement head on AirfRANS

## Research Themes
1. **Optimizer & LR sensitivity** — Lion (default) vs AdamW; LR sweep across both datasets
2. **Model capacity** — Is default (3L/192d) sufficient or does more capacity help?
3. **Physics feature importance (TandemFoil)** — Which features from the frontier mechanisms list actually matter?
4. **Normalization tricks** — asinh pressure transform and residual prediction impact
5. **Surface refinement** — Does the zero-init refinement head help on AirfRANS?
6. **Recipe transferability** — Do the same hyperparameters work on both datasets?

## Potential Next Directions (Round 2+)
- Fine-grained physics feature ablation based on Round 1 results
- Cosine schedule tuning (T_max variations)
- EMA decay sweep
- Batch size scaling with LR
- Reynolds-stratified sampling for TandemFoil
- ANP surface decoder for TandemFoil
- Fourier feature encoding
- Dropout regularization
