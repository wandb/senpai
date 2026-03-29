"""Simple model soup: uniform average of checkpoint state dicts.

Usage:
    python make_soup.py models/model-xxx models/model-yyy ... --output models/model-soup/checkpoint.pt
"""
import argparse
import torch
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+", help="Checkpoint directories")
parser.add_argument("--output", default="models/model-soup/checkpoint.pt")
args = parser.parse_args()

# Load all state dicts
state_dicts = []
for ckpt_dir in args.checkpoints:
    path = Path(ckpt_dir) / "checkpoint.pt"
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    state_dicts.append(sd)
    print(f"Loaded {path} ({len(sd)} keys)")

# Uniform average
print(f"\nAveraging {len(state_dicts)} checkpoints...")
avg_sd = {}
for key in state_dicts[0]:
    tensors = [sd[key].float() for sd in state_dicts]
    avg_sd[key] = sum(tensors) / len(tensors)

# Save
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(avg_sd, out_path)

# Also copy config from first checkpoint
import shutil
config_src = Path(args.checkpoints[0]) / "config.yaml"
config_dst = out_path.parent / "config.yaml"
if config_src.exists():
    shutil.copy2(config_src, config_dst)
    print(f"Copied config from {config_src}")

print(f"Saved uniform soup to {out_path}")
print(f"To evaluate: load this checkpoint into train.py's model and run validation")
