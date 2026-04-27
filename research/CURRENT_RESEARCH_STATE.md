# SENPAI Research State
- 2026-04-27 (initial)
- No directives from human researcher team yet
- Baseline: No established val_avg/mae_surf_p yet — Round 1 sweeps will establish the vanilla anchor

## Current Research Focus

Round 1 is a comprehensive hyperparameter and architecture ablation of the vanilla Transolver on TandemFoilSet. The objective is to establish the baseline and identify which knobs matter most before moving to deeper architectural exploration.

**Vanilla anchor** (target/train.py defaults):
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- Optimizer: AdamW, Scheduler: CosineAnnealingLR (T_max=epochs)
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, `act=gelu`, `dropout=0.0`)
- Loss: MSE normalized space, `vol_loss + surf_weight * surf_loss`

**Primary metric**: `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 val splits). Lower is better.

**8 Round 1 experiments (being assigned):**

| Student | Hypothesis |
|---------|-----------|
| alphonse | LR sweep: {1e-4, 3e-4, 5e-4, 1e-3} |
| askeladd | surf_weight sweep: {5, 10, 20, 50} |
| edward | n_hidden width: {64, 128, 192, 256} |
| fern | n_layers depth: {3, 4, 5, 6} |
| frieren | slice_num: {32, 48, 64, 96} |
| nezuko | Asinh target transform to tame high-Re extremes |
| tanjiro | Per-channel pressure loss upweighting |
| thorfinn | Dropout regularization: {0.05, 0.1, 0.2} for OOD splits |

## Potential Next Research Directions

After Round 1 results arrive:

### High-priority (likely high impact)
1. **Compound best config**: Take best hyperparameter from each Round 1 winner and combine.
2. **Gradient clipping**: Clips at {0.1, 0.5, 1.0} — high-Re pressure extremes can cause gradient spikes.
3. **mlp_ratio sweep**: {1, 2, 4, 8} — MLP expansion factor.
4. **LR warmup + cosine**: 5-epoch linear warmup before cosine decay.

### Architecture-level (medium priority)
5. **n_head sweep**: {2, 4, 8} — attention head count.
6. **Pre-LN vs Post-LN**: Layer normalization order in transformer blocks.
7. **Larger model compound**: n_hidden=192, n_layers=6, n_head=6 compound.

### Feature engineering / physics-informed approaches
8. **Wall-distance feature**: Add signed distance to nearest surface as extra input.
9. **Local Re-scaled features**: Normalize local velocity by local Re for cross-regime invariance.
10. **Fourier position encoding**: Replace raw (x,z) with sinusoidal position encoding.

### Loss formulation
11. **Sobolev loss**: Add gradient consistency penalty |∇pred - ∇true|.
12. **Focal-style surface loss**: Upweight samples with highest current surface error.
13. **Huber loss**: More robust to outliers from high-Re extremes than MSE.
