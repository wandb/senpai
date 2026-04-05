# Experiment: Smaller SRF Head (h=128, h=96)

## Hypothesis
Wider SRF failed (h=256/384 overfits). If adding capacity hurts, maybe h=192 isn't optimal.
Smaller SRF = implicit regularization = potentially better OOD transfer for NACA6416.
This tests the inverse of frieren's experiment.

## Expected impact
-0.5 to -2% p_tan if overfitting hypothesis is correct.
