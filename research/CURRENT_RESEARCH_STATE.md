# SENPAI Research State

- **Date:** 2026-04-02 (radical pivot)
- **Advisor branch:** noam
- **Phase:** Phase 6 — RADICAL New Architectures & Paradigms

## Current Baseline (PR #2003)

| Metric | Value |
|--------|-------|
| val/loss | 0.3761 |
| p_in | 12.5 |
| p_oodc | 8.2 |
| p_tan | 29.8 |
| p_re | 6.5 |

Config: Lion, cosine_T_max=160, 3L Transolver, 192 hidden, 96 slices, residual prediction, surface refine.

## Student Status

| Student | PR | Experiment | Type | Status |
|---------|-----|-----------|------|--------|
| alphonse | #2033 | **LinearNO** — replace slice attention | RADICAL | 8 runs training |
| fern | #2034 | **Inviscid Cp** — precomputed physics input | RADICAL | Implementing |
| tanjiro | #2035 | **All-to-All Surface Attention** | RADICAL | Implementing |
| nezuko | #2036 | **Conditional Flow Matching** | RADICAL | Implementing |
| edward | #2037 | **MARIO Latent Geometry** | RADICAL | Implementing |
| thorfinn | #2026 | Muon v2 (hybrid FFN+Lion) | Per Morgan | 8 runs training |
| askeladd | #2027 | Geosolver v2 (geometry input features) | Per Morgan | 8 runs training |
| frieren | #2025 | TTA + SWA | Architecture | 8 runs training |

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| SOAP/HeavyBall optimizers | #2010,2018,2019,2021,2022,2023 | SOAP 2-6% WORSE than Lion. False 20% claim from NaN MAE. |
| Muon (full replacement) | #2006 | 30-70% worse. Spectral flattening destroys physics signal. |
| XSA attention | #2007 | Redundant with orthogonal slices. |
| PirateNets RWF | #2008 | LayerNorm + Lion attenuate mechanism. |
| NOBLE | #2011 | Model too small, cosine on hidden features unhelpful. |
| Fourier features | #2015 | Marginal. Existing PE sufficient. |
| Deeper model (4-5L) | #2015 | Epoch penalty too large in 180-min timeout. |
| Multi-scale slices | #2017 | 48≈96 slices. Mechanism adapts to any count. |
| Physics losses (vorticity, div-free) | #2016,2023 | WLS gradient instability on unstructured mesh. |
| GeoTransolver cross-attention | #1989 | +9.8% worse, gate near zero. |
| Learned loss weights | #2013 | Collapsed to zero. |

## Key Research Insights

1. **Lion is the optimal optimizer** for this 1.7M-param architecture. All alternatives fail.
2. **The Transolver is a strong local optimum.** Incremental modifications are absorbed by EMA.
3. **Throughput matters** — any technique adding >15% step time loses epochs and hurts.
4. **Physics structure in weights matters** — orthogonalize/flatten at your peril (Muon, SOAP).
5. **24-dim input already encodes geometry** — adding context banks doesn't help (GeoTransolver).
6. **48 slices ≈ 96 slices** — the model is over-parameterized in slice dimension.

## Current Radical Directions (from literature survey)

1. **LinearNO** (AAAI 2026) — 60% improvement on AirfRANS. RUNNING.
2. **Inviscid Cp** — 88% OOD error reduction in B-GNN paper. IMPLEMENTING.
3. **All-to-All Surface Attention** — elliptic pressure coupling. IMPLEMENTING.
4. **Conditional Flow Matching** — generative paradigm. IMPLEMENTING.
5. **MARIO Latent Geometry** — geometry autoencoder + modulated field. IMPLEMENTING.
