# Experiment: Panel-Method Inviscid Cp as Physics-Informed Input Features

## Hypothesis
Add pre-computed inviscid pressure coefficient (Cp) from a vortex panel solver as an additional
input feature. The model then learns the viscous correction residual rather than predicting
from scratch. Panel method natively captures tandem fore-aft interaction.
B-GNN showed 88% OOD error reduction with this approach.

## Expected impact
High — potentially -5 to -15% p_tan. This is a paradigm shift from everything else tried.
