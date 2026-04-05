# Experiment: Attention Temperature Curriculum

## Hypothesis
The slice attention temperature is initialized at 0.5 and learns freely. High temp = broad routing
(many slices get signal), low temp = sharp specialized routing. Scheduling from high→low forces
broad exploration early, specialized exploitation late. Zero new parameters.

## Expected impact
-1 to -3% p_tan via better attention routing dynamics.
