# Experiment: Slice Number Sweep (96 → 128, 144)

## Hypothesis
The Transolver uses slice_num=96 attention "slices" for routing point features.
The Gap/Stagger Spatial Bias (GSB) mechanism modulates routing per-sample via spatial biases.
More slices = more routing paths = finer-grained specialization for tandem geometry.

## Configs
- A: slice_num=128 (+33% routing paths)
- B: slice_num=144 (+50% routing paths)

## Expected impact
-1 to -3% p_tan via more expressive GSB routing.
