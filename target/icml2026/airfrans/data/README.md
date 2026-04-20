# airfrans/data

Repo-local AirfRANS benchmark assets for the ICML 2026 target.

## Files

- `prepare_airfrans.py` — raw VTK loader and dataset builder
- `split_airfrans.py` — split regeneration from the official AirfRANS manifest
- `split_manifest_airfrans.json` — committed repo-owned AirfRANS split
- `vtk_xml.py` — minimal VTK XML reader used by the loader

## Split contract

The split follows the official AirfRANS task lists. Validation is carved from the
tail of each official training list so that train/val reconstruct the published
training order exactly.

## Running

```bash
cd target/icml2026
python airfrans/data/split_airfrans.py
python train.py --dataset airfrans --model reference_transolver
```
