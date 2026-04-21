# data

Shared helpers for the multi-dataset ICML 2026 target.

## Layout

- `split_utils.py` — shared manifest helpers used by the dataset-local split scripts
- `smoke_test_datasets.py` — PVC-backed smoke test for the shared ICML dataset loaders

The actual dataset pipelines live under their benchmark directories:

- `../tandemfoil/data/`
- `../airfrans/data/`
- `../drivaerml/data/`

Run the real-data smoke test from the problem root:

```bash
cd target/icml2026
python data/smoke_test_datasets.py
```
