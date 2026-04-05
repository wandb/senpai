# Experiment: Low-Rank Foil-1 Geometry Adapter

## Hypothesis
Foil-1 DSDF channels carry critical geometry signal (confirmed by dropout failure #2156).
The model has no dedicated pathway to route this geometry info into slice attention.
A tiny MLP on per-sample DSDF statistics → additive bias on slice logits provides this pathway.
Zero-initialized so baseline is preserved at init. GEPS/AdaIN-inspired.

## Expected impact
-1 to -4% p_tan via better OOD geometry routing.
