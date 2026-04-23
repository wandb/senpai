# Bold Research Ideas — 2026-04-23 14:00
## Driven by noam branch intelligence + human directive to "think bigger"

The noam branch accumulated ~100 merged PRs with winning techniques. Many are ALREADY PORTED to radford's train.py but UNUSED. This is the highest-priority action: activate these features.

## Available in radford's train.py but NOT USED

| Feature | Flag | Noam Impact | Applicable To |
|---------|------|-------------|---------------|
| ANP cross-attention decoder | `--anp-srf` | **-58.9% TF p_tan, -70% p_in** | TF, TFP |
| Asinh pressure normalization | `--asinh-pressure --asinh-scale 0.75` | **p_oodc -8%, p_re -4.5%** | ALL |
| Residual prediction | `--residual-prediction` | Learned correction to freestream | ALL |
| Panel method Cp | `--enable-cp-panel --enable-cp-panel-tandem-only --cp-panel-scale 0.1` | Physics input feature | TF, TFP |
| TE coordinate frame | `--enable-te-coord-frame` | 6 TE-relative channels | TF, TFP |
| Wake deficit feature | `--enable-wake-deficit` | Gap-normalized fore-TE offset | TF, TFP |
| Wake angle feature | `--enable-wake-angle` | atan2 polar direction | TF, TFP |
| Vortex panel velocity | `--enable-vortex-panel-velocity --vortex-panel-scale 0.1 --vortex-panel-n 64` | Biot-Savart velocity | TF, TFP |
| Re-stratified sampling | `--re-stratified-sampling` | OOD Re robustness | TF, TFP, AF |
| Lookahead optimizer | `--use-lookahead` (default True!) | k=5 slow weights | ALL |
| torch.compile | `--compile-model` (default True!) | Throughput | ALL |
| 96 slices | `--model-slices 96` (default!) | Optimal slice count | ALL |
| 3 heads (64d/head at 192d) | `--model-heads 3` (default!) | Optimal for 192d | TF, TFP |
| Surface refinement head | `--surface-refine` (default True!) | Surface post-processing | ALL |
| Pressure prior addition | `--enable-pressure-prior-addition` | Physics prior | TF, TFP? |

## Key config mismatches (radford vs noam defaults)

| Parameter | Radford TF/TFP | Noam optimal | Impact |
|-----------|----------------|--------------|--------|
| T_max | 10 | **150** | 15x longer schedule — massive underfitting! |
| Heads | 8 | **3** | 64d/head vs 24d/head at 192d |
| Slices | ? | **96** | Swept 48/64/96/128, 96 is optimal |
| Lookahead | off | **on** | Slow-weight averaging |
| compile | off | **on** | More epochs in time budget |
| EMA decay | 0.999 | **0.9999** | Smoother averaging |

## Priority 1: Full noam stack on each dataset

### P1-A: TandemFoil "Full Noam Stack"
### P1-B: TandemFoil Paper "Full Noam Stack"
### P1-C: DrivAerML "Noam training dynamics" (asinh + residual-prediction + compile + slices)
### P1-D: AirfRANS "Noam training dynamics" (same applicable features)

## Priority 2: Individual high-impact ablations
### P2-A: ANP decoder alone on TF
### P2-B: T_max=150 alone on TF
### P2-C: Asinh pressure alone on DM
### P2-D: Residual prediction alone on DM
### P2-E: Lion optimizer on DM (noam's final optimizer)

## Priority 3: Code changes needed (students port from noam)
1. vol_loss_scale (-15.9% impact)
2. PCGrad 3-way (gradient surgery)
3. Attention temperature annealing (-11%)
4. DomainLayerNorm
5. GLU preprocess MLP
6. High-p-clamp

## Confirmed dead ends from noam
- Manifold mixup, slice dropout, multi-exit ensemble, Y-flip aug, polar velocity targets
- Surface pressure smoothness loss, LNO bottleneck, SOAP optimizer, stochastic depth
