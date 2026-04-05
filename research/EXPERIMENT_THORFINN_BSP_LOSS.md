# Experiment: Binned Spectral Power (BSP) Loss on Arc-Length Surface Pressure

## Hypothesis
Neural networks exhibit spectral bias — they learn low-frequency components faster and
more accurately than high-frequency ones (Rahaman et al. ICML 2019). Surface pressure on
the NACA6416 fore-foil has high-frequency components at leading edge stagnation and wake
interaction zones — exactly the features that dominate p_tan error.

Apply 1D DFT along arc-length surface coordinate, bin frequency coefficients,
and apply higher loss weights to mid/high frequency bins.

## Expected impact
-2 to -5% p_tan. Inspired by Koh & Kim JCP 2026 (arXiv:2502.00472).
