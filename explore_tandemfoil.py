import torch
import os
import sys

BASE = "/mnt/new-pvc/datasets/tandemfoil/"

FILES = [
    "cruise_Re500_aoa0_ive_Part1.pickle",
    "cruise_Re500_aoa0_ive_Part2.pickle",
    "cruise_Re500_aoa0_ive_Part3.pickle",
    "cruise_Re500_aoa0_mgn_Part1.pickle",
    "cruise_Re500_aoa0_mgn_Part2.pickle",
    "cruise_Re500_aoa0_mgn_Part3.pickle",
    "cruise_Re500_aoa5_ive_Part1.pickle",
    "cruise_Re500_aoa5_ive_Part2.pickle",
    "cruise_Re500_aoa5_ive_Part3.pickle",
    "cruise_Re500_aoa5_mgn_Part1.pickle",
    "cruise_Re500_aoa5_mgn_Part2.pickle",
    "cruise_Re500_aoa5_mgn_Part3.pickle",
]

def describe_value(v, indent="  "):
    if isinstance(v, torch.Tensor):
        return f"Tensor shape={tuple(v.shape)} dtype={v.dtype}"
    elif isinstance(v, dict):
        lines = ["{"]
        for kk, vv in v.items():
            lines.append(f"  {kk}: {describe_value(vv)}")
        lines.append("}")
        return "\n".join(lines)
    elif isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)})"
    else:
        return repr(v)

def tensor_stats(v, name):
    if isinstance(v, torch.Tensor) and v.numel() > 0:
        f = v.float()
        print(f"  {name}: min={f.min().item():.6g}  max={f.max().item():.6g}  mean={f.mean().item():.6g}  std={f.std().item():.6g}")

def explore_file(fname):
    path = os.path.join(BASE, fname)
    print(f"\n{'='*80}")
    print(f"FILE: {fname}")
    print(f"{'='*80}")

    print("Loading...", flush=True)
    data = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Type: {type(data)}")
    n = len(data)
    print(f"\n1. Number of samples: {n}")

    sample = data[0]
    print(f"\n2. sample[0] type: {type(sample)}")

    # Keys and shapes
    print("\n--- sample[0] KEYS AND SHAPES ---")
    if isinstance(sample, dict):
        keys = list(sample.keys())
    elif hasattr(sample, '__dict__'):
        keys = list(vars(sample).keys())
    elif hasattr(sample, 'keys'):
        keys = list(sample.keys())
    else:
        print(f"  Not a dict-like: {type(sample)}")
        # Try torch_geometric Data
        try:
            keys = sample.keys()
            print(f"  torch_geometric keys: {list(keys)}")
        except:
            pass
        keys = []

    # Handle torch_geometric Data objects
    try:
        from torch_geometric.data import Data, HeteroData
        if isinstance(sample, (Data, HeteroData)):
            print("  (torch_geometric Data object)")
            for k in sample.keys():
                v = sample[k]
                print(f"  {k}: {describe_value(v)}")
            keys = list(sample.keys())
        elif isinstance(sample, dict):
            for k, v in sample.items():
                print(f"  {k}: {describe_value(v)}")
    except ImportError:
        if isinstance(sample, dict):
            for k, v in sample.items():
                print(f"  {k}: {describe_value(v)}")

    # 3. Tensor statistics for sample[0]
    print("\n3. TENSOR STATISTICS for sample[0]:")
    try:
        from torch_geometric.data import Data, HeteroData
        if isinstance(sample, (Data, HeteroData)):
            for k in sample.keys():
                v = sample[k]
                if isinstance(v, torch.Tensor):
                    tensor_stats(v, k)
        elif isinstance(sample, dict):
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    tensor_stats(v, k)
    except ImportError:
        if isinstance(sample, dict):
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    tensor_stats(v, k)

    # Helper to get field from sample
    def get_field(s, k):
        try:
            from torch_geometric.data import Data, HeteroData
            if isinstance(s, (Data, HeteroData)):
                return s[k] if k in s.keys() else None
        except ImportError:
            pass
        if isinstance(s, dict):
            return s.get(k, None)
        return getattr(s, k, None)

    # 4. Unique values for boundary, zoneID
    print("\n4. UNIQUE VALUES:")
    for field in ["boundary", "zoneID"]:
        v = get_field(sample, field)
        if v is not None and isinstance(v, torch.Tensor):
            unique_vals = torch.unique(v)
            print(f"  {field}: unique={unique_vals.tolist()}")
        elif v is not None:
            print(f"  {field}: {v}")
        else:
            print(f"  {field}: NOT FOUND in sample[0]")

    # 5. flowState dict
    print("\n5. FLOWSTATE DICT:")
    fs = get_field(sample, "flowState")
    if fs is not None:
        if isinstance(fs, dict):
            for k, v in fs.items():
                print(f"  {k}: {describe_value(v)}")
        else:
            print(f"  flowState = {fs}")
    else:
        print("  flowState: NOT FOUND")
        # Check for flowstate (lowercase)
        fs2 = get_field(sample, "flowstate")
        if fs2 is not None:
            print(f"  (found 'flowstate' lowercase): {fs2}")

    # 6. NACA values for first 5 samples
    print("\n6. NACA VALUES (first 5 samples):")
    for i in range(min(5, n)):
        s = data[i]
        for naca_key in ["naca", "NACA", "naca_foil", "airfoil"]:
            v = get_field(s, naca_key)
            if v is not None:
                print(f"  sample[{i}] {naca_key}: {v}")
                break
        else:
            # Try flowState
            fs = get_field(s, "flowState")
            if fs and isinstance(fs, dict):
                found = False
                for nk in ["naca", "NACA", "naca_foil", "airfoil"]:
                    if nk in fs:
                        print(f"  sample[{i}] flowState[{nk}]: {fs[nk]}")
                        found = True
                        break
                if not found:
                    print(f"  sample[{i}] NACA: not found directly (flowState keys: {list(fs.keys())})")
            else:
                print(f"  sample[{i}] NACA: not found")

    # 7. AoA values for first 5 samples
    print("\n7. AoA VALUES (first 5 samples):")
    for i in range(min(5, n)):
        s = data[i]
        for aoa_key in ["aoa", "AoA", "angle_of_attack", "alpha"]:
            v = get_field(s, aoa_key)
            if v is not None:
                print(f"  sample[{i}] {aoa_key}: {v}")
                break
        else:
            fs = get_field(s, "flowState")
            if fs and isinstance(fs, dict):
                found = False
                for ak in ["aoa", "AoA", "angle_of_attack", "alpha"]:
                    if ak in fs:
                        print(f"  sample[{i}] flowState[{ak}]: {fs[ak]}")
                        found = True
                        break
                if not found:
                    print(f"  sample[{i}] AoA: not found in flowState")
            else:
                print(f"  sample[{i}] AoA: not found")

    # 8. gap, stagger for first 5 samples
    print("\n8. GAP / STAGGER VALUES (first 5 samples):")
    for i in range(min(5, n)):
        s = data[i]
        parts = []
        for gk in ["gap", "stagger"]:
            v = get_field(s, gk)
            if v is not None:
                parts.append(f"{gk}={v}")
            else:
                fs = get_field(s, "flowState")
                if fs and isinstance(fs, dict) and gk in fs:
                    parts.append(f"flowState[{gk}]={fs[gk]}")
        if parts:
            print(f"  sample[{i}]: {', '.join(parts)}")
        else:
            print(f"  sample[{i}]: gap/stagger not found")

    # 9. Node count variation
    print("\n9. NODE COUNT VARIATION across all samples:")
    node_counts = []
    for i in range(n):
        s = data[i]
        # Try x, pos, node_feat
        for nk in ["x", "pos", "node_feat", "nodes"]:
            v = get_field(s, nk)
            if v is not None and isinstance(v, torch.Tensor):
                node_counts.append(v.shape[0])
                break
        else:
            # Try num_nodes
            v = get_field(s, "num_nodes")
            if v is not None:
                node_counts.append(int(v))
    if node_counts:
        import statistics
        print(f"  min={min(node_counts)}  max={max(node_counts)}  mean={statistics.mean(node_counts):.1f}  (from {len(node_counts)} samples)")
    else:
        print("  Could not determine node counts")

    # 10. Keys containing "y_est" or "model"
    print("\n10. KEYS containing 'y_est' or 'model':")
    try:
        from torch_geometric.data import Data, HeteroData
        if isinstance(sample, (Data, HeteroData)):
            all_keys = list(sample.keys())
        elif isinstance(sample, dict):
            all_keys = list(sample.keys())
        else:
            all_keys = []
    except ImportError:
        all_keys = list(sample.keys()) if isinstance(sample, dict) else []

    found_special = [k for k in all_keys if "y_est" in k.lower() or "model" in k.lower()]
    if found_special:
        print(f"  Found: {found_special}")
    else:
        print("  None found")

    del data
    print(f"\nDone with {fname}")

# Load and explore just the first file in detail, then collect summary stats for all
explore_file(FILES[0])
