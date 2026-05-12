# Advisor Review: PR #2417

Decision: **do not merge**.

Evidence:

- The PR targets held-out `test_primary/surface_pressure_rel_l2_pct`.
- The terminal result reports test `6.31`.
- The baseline is `6.24`.
- W&B config confirms the EMA treatment was active.

Reason:

Validation improved, but held-out test regressed. This is useful evidence but not a baseline improvement.

Suggested action:

- Request changes only if the advisor wants a targeted EMA variant such as lower decay or later warmup.
- Otherwise close as validation-only signal that did not transfer.
