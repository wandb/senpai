STUDENT ws-fern:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["driv-ema-042"],"primary_metric":{"name":"full_val_primary/surface_pressure_rel_l2_pct","value":4.42},"test_metric":{"name":"test_primary/surface_pressure_rel_l2_pct","value":6.31}}

## Results

Validation surface-pressure relative-L2 improved, but held-out test surface-pressure relative-L2 regressed.

| Metric | Baseline | This Run | Direction |
| --- | ---: | ---: | --- |
| `full_val_primary/surface_pressure_rel_l2_pct` | 4.62 | 4.42 | improved |
| `test_primary/surface_pressure_rel_l2_pct` | 6.24 | 6.31 | worse |

W&B run: `driv-ema-042`

## Analysis

EMA warmup appears to smooth validation behavior, but the effect did not transfer to the held-out test split. This should not merge under the stated contract.
