#!/usr/bin/env python3
"""Fetch surface MAE metrics for PR #2166 runs."""
import os
import sys
import wandb
import pandas as pd
import numpy as np

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

run_ids = [
    "bmjgaeqp",  # w=0.1, s42
    "e7iu2ix3",  # w=0.1, s73
    "bfau5gzb",  # w=0.1, s17
    "mmwix7z6",  # w=0.1, s31
    "mvfbf0jh",  # w=0.1, s55
    "0m0fg9ut",  # w=0.1, s88
    "wvtqymw7",  # w=0.05, s42
    "i9as9n60",  # w=0.05, s73
]

# Seed info from user
seed_map = {
    "bmjgaeqp": (0.1, 42),
    "e7iu2ix3": (0.1, 73),
    "bfau5gzb": (0.1, 17),
    "mmwix7z6": (0.1, 31),
    "mvfbf0jh": (0.1, 55),
    "0m0fg9ut": (0.1, 88),
    "wvtqymw7": (0.05, 42),
    "i9as9n60": (0.05, 73),
}

rows = []
path = f"{entity}/{project}"

print(f"Fetching runs from {path}...")

for run_id in run_ids:
    try:
        run = api.run(f"{path}/{run_id}")
        weight, seed = seed_map[run_id]

        summary = run.summary_metrics

        # Print all summary keys to see what's available
        print(f"\n--- Run {run_id} (w={weight}, s={seed}) ---")
        print(f"  State: {run.state}")
        print(f"  Name: {run.name}")

        # Look for surface MAE metrics - try different possible key names
        relevant_keys = {k: v for k, v in summary.items()
                        if any(term in k.lower() for term in
                               ['surface', 'mae', 'p_in', 'p_oodc', 'p_tan', 'p_re', 'ema'])}

        print(f"  Relevant summary keys:")
        for k, v in sorted(relevant_keys.items()):
            print(f"    {k}: {v}")

        # Also print all keys for inspection
        print(f"  All summary keys: {sorted(summary.keys())}")

    except Exception as e:
        print(f"  ERROR for {run_id}: {e}")

print("\n\nDone with initial scan. Now looking for specific metric keys...")
