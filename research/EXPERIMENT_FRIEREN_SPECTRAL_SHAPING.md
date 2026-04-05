# Experiment: Spectral Shaping of GatedMLP Activations

## Hypothesis
Depthwise 1D conv (kernel=3) in feature dim after GatedMLP gate, initialized to Gaussian blur.
Filters spectral junk from activations before slice scatter — the most info-dense bottleneck.
576 extra params per TransolverBlock. Inspired by Fanaskov & Oseledets ICLR 2025.

## Expected impact
-1 to -3% p_tan via cleaner feature aggregation.
