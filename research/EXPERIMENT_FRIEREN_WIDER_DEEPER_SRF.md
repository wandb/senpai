# Experiment: Wider/Deeper Surface Refinement MLP

## Hypothesis
The surface_refine MLP (hidden=192, layers=3) is the key correction mechanism for surface pressure.
More capacity in this head could capture finer-grained pressure patterns, especially on NACA6416.

## Configs
- A: surface_refine_hidden=256, surface_refine_layers=4
- B: surface_refine_hidden=384, surface_refine_layers=3

## Expected impact
-1 to -3% p_tan via better surface correction fidelity.
