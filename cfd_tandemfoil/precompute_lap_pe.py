"""Pre-compute Laplacian eigenvector positional encodings for all mesh samples."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data.prepare_multi import load_pickle


def compute_laplacian_pe(edge_index: np.ndarray, n_nodes: int, k: int = 16) -> np.ndarray:
    """Compute k smallest non-trivial Laplacian eigenvectors.

    Args:
        edge_index: (2, E) array of [src, dst] pairs
        n_nodes: total number of nodes
        k: number of eigenvectors to compute
    Returns:
        (n_nodes, k) array of eigenvectors (absolute value for sign invariance)
    """
    row = edge_index[0]
    col = edge_index[1]
    data = np.ones(len(row), dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    A = A + A.T
    A.data = np.minimum(A.data, 1.0)  # binarize
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - A

    try:
        # Shift-invert mode (sigma=0) is much faster for smallest eigenvalues
        eigenvalues, eigenvectors = spla.eigsh(L, k=k + 1, sigma=1e-5, which='LM', tol=1e-3)
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx[1:k + 1]]  # skip trivial
        eigenvectors = np.abs(eigenvectors)  # sign invariance
    except Exception as e:
        print(f"  eigsh failed ({e}), returning zeros")
        eigenvectors = np.zeros((n_nodes, k), dtype=np.float32)

    return eigenvectors.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=16, help='Number of eigenvectors')
    parser.add_argument('--cache_dir', type=str, default='cache/lap_pe/', help='Output directory')
    parser.add_argument('--manifest', type=str, default='data/split_manifest.json')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    # Build global index mapping (file_idx, local_idx) → global_idx
    global_idx = 0
    total_samples = 0
    t_start = time.time()

    for fi, pkl_path in enumerate(manifest['pickle_paths']):
        print(f"\nLoading {pkl_path}...")
        data = load_pickle(Path(pkl_path))
        print(f"  {len(data)} samples")

        for si, sample in enumerate(data):
            out_path = cache_dir / f"lap_pe_{global_idx:05d}.pt"
            if out_path.exists():
                global_idx += 1
                total_samples += 1
                continue

            n_nodes = sample.pos.shape[0]
            edge_index = sample.edge_index.numpy()

            t0 = time.time()
            lap_pe = compute_laplacian_pe(edge_index, n_nodes, k=args.k)
            dt = time.time() - t0

            torch.save(torch.from_numpy(lap_pe), out_path)
            total_samples += 1

            if total_samples % 50 == 0 or total_samples <= 5:
                elapsed = time.time() - t_start
                rate = total_samples / elapsed
                print(f"  [{total_samples}] nodes={n_nodes}, k={args.k}, time={dt:.2f}s, rate={rate:.1f} samples/s")

            global_idx += 1

        del data

    elapsed = time.time() - t_start
    print(f"\nDone: {total_samples} samples in {elapsed:.0f}s ({total_samples/elapsed:.1f} samples/s)")
    print(f"Cached to: {cache_dir}")


if __name__ == '__main__':
    main()
