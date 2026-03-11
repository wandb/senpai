import torch
import os

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

def get_all_keys(s):
    try:
        from torch_geometric.data import Data, HeteroData
        if isinstance(s, (Data, HeteroData)):
            return list(s.keys())
    except ImportError:
        pass
    if isinstance(s, dict):
        return list(s.keys())
    return []

def summarize_file(fname):
    path = os.path.join(BASE, fname)
    print(f"\n{'='*80}")
    print(f"FILE: {fname}")
    print(f"{'='*80}")
    print("Loading...", flush=True)
    data = torch.load(path, map_location='cpu', weights_only=False)
    n = len(data)
    print(f"Samples: {n}")

    sample = data[0]
    all_keys = get_all_keys(sample)
    print(f"Keys: {all_keys}")

    # flowState full details
    fs = get_field(sample, "flowState")
    print(f"\nflowState:")
    if isinstance(fs, dict):
        for k, v in fs.items():
            if hasattr(v, '__iter__') and not isinstance(v, str):
                try:
                    print(f"  {k}: {list(v)}")
                except:
                    print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")

    # NACA, AoA for first 5
    print("\nFirst 5 samples - NACA, AoA, gap/stagger, af_pos:")
    for i in range(min(5, n)):
        s = data[i]
        naca = get_field(s, "NACA")
        aoa  = get_field(s, "AoA")
        gap  = get_field(s, "gap")
        stagger = get_field(s, "stagger")
        af_pos = get_field(s, "af_pos")
        fs_i = get_field(s, "flowState")
        # Also check flowState
        if gap is None and isinstance(fs_i, dict): gap = fs_i.get("gap", None)
        if stagger is None and isinstance(fs_i, dict): stagger = fs_i.get("stagger", None)
        print(f"  [{i}] NACA={naca}  AoA={aoa}  gap={gap}  stagger={stagger}  af_pos={af_pos.tolist() if af_pos is not None else None}")

    # Unique boundary / zoneID
    boundary = get_field(sample, "boundary")
    zoneID   = get_field(sample, "zoneID")
    if boundary is not None and isinstance(boundary, torch.Tensor):
        print(f"\nboundary unique: {torch.unique(boundary).tolist()}")
    if zoneID is not None and isinstance(zoneID, torch.Tensor):
        print(f"zoneID unique: {torch.unique(zoneID).tolist()}")

    # Tensor stats
    print(f"\nTensor stats for sample[0]:")
    for k in all_keys:
        v = get_field(sample, k)
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            f = v.float()
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}  min={f.min().item():.6g}  max={f.max().item():.6g}  mean={f.mean().item():.6g}  std={f.std().item():.6g}")

    # Node count variation
    node_counts = []
    for i in range(n):
        s = data[i]
        for nk in ["x", "pos", "node_feat", "nodes"]:
            v = get_field(s, nk)
            if v is not None and isinstance(v, torch.Tensor):
                node_counts.append(v.shape[0])
                break
    if node_counts:
        import statistics
        print(f"\nNode counts: min={min(node_counts)}  max={max(node_counts)}  mean={statistics.mean(node_counts):.1f}")

    # y_est / model keys
    model_keys = [k for k in all_keys if "y_est" in k.lower() or "model" in k.lower()]
    print(f"\nModel/y_est keys: {model_keys}")

    del data
    return {
        "fname": fname,
        "n_samples": n,
        "keys": all_keys,
        "model_keys": model_keys,
        "node_min": min(node_counts) if node_counts else None,
        "node_max": max(node_counts) if node_counts else None,
    }

results = []
for f in FILES:
    r = summarize_file(f)
    results.append(r)

print("\n\n" + "="*80)
print("CROSS-FILE SUMMARY")
print("="*80)
for r in results:
    print(f"{r['fname']:55s}  n={r['n_samples']}  nodes=[{r['node_min']},{r['node_max']}]")
    print(f"  model_keys: {r['model_keys']}")

# Compare aoa0 vs aoa5
print("\n--- aoa0 vs aoa5 key differences ---")
aoa0_keys = set(results[0]['keys'])
aoa5_keys = set(results[6]['keys'])
print(f"  In aoa0 not aoa5: {aoa0_keys - aoa5_keys}")
print(f"  In aoa5 not aoa0: {aoa5_keys - aoa0_keys}")

# Compare ive vs mgn
print("\n--- ive vs mgn key differences ---")
ive_keys = set(results[0]['keys'])
mgn_keys = set(results[3]['keys'])
print(f"  In ive not mgn: {ive_keys - mgn_keys}")
print(f"  In mgn not ive: {mgn_keys - ive_keys}")
