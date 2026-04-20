# drivaerml/data

Repo-local DrivAerML benchmark assets for the ICML 2026 target.

## Files

- `prepare_drivaerml.py` — packaged-array loader for the processed PVC dataset
- `split_drivaerml.py` — split regeneration from the processed PVC manifests
- `split_manifest_drivaerml.json` — committed repo-owned DrivAerML split

## Split contract

The committed split follows the packaged public processed manifest on the PVC and
keeps the surface-first public benchmark contract that matches the packaged data.

## Running

```bash
cd target/icml2026
python drivaerml/data/split_drivaerml.py
python train.py --dataset drivaerml --model reference_transolver
```
