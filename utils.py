"""Visualization utilities for flow field prediction."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import torch
from pathlib import Path

FIELD_NAMES = ["Ux", "Uy", "p"]
OUT_DIR = Path("plots")


def _make_triangulation(px, py, surf_pos):
    """Create Delaunay triangulation and mask triangles inside airfoil bodies."""
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path
    triang = tri.Triangulation(px, py)
    triangles = triang.triangles
    p = np.column_stack([px, py])
    centroids = p[triangles].mean(axis=1)
    # Use convex hull of surface points to mask triangles inside the airfoil
    hull = ConvexHull(surf_pos)
    hull_path = Path(surf_pos[hull.vertices])
    triang.set_mask(hull_path.contains_points(centroids))
    return triang


def plot_samples(dataset, indices=None, n_samples=4, prefix="data_sample"):
    """Plot ground truth flow fields for raw dataset samples."""
    OUT_DIR.mkdir(exist_ok=True)
    saved = []
    if indices is None:
        indices = list(range(min(n_samples, len(dataset))))

    for i, idx in enumerate(indices):
        x, y_true, is_surface = dataset[idx]
        pos = x[:, :2].numpy()
        y_np = y_true.numpy()
        is_surf_np = is_surface.numpy()
        surf_pos = pos[is_surf_np]

        x_lo, x_hi = -1.0, 2.0
        y_lo = max(0.0, pos[:, 1].min())
        y_hi = min(surf_pos[:, 1].mean() + 3.0, pos[:, 1].max())
        near = (
            (pos[:, 0] >= x_lo) & (pos[:, 0] <= x_hi) &
            (pos[:, 1] >= y_lo) & (pos[:, 1] <= y_hi)
        )

        px, py = pos[near, 0], pos[near, 1]
        triang = _make_triangulation(px, py, surf_pos)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Sample {idx}", fontsize=14)

        for col, (name, ch) in enumerate(zip(FIELD_NAMES, range(3))):
            vals = y_np[near, ch]
            vmin, vmax = vals.min(), vals.max()
            ax = axes[col]
            c = ax.tripcolor(triang, vals, shading="flat", vmin=vmin, vmax=vmax, cmap="RdBu_r")
            ax.set_title(name)
            fig.colorbar(c, ax=ax)
            ax.set_aspect("equal")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.plot(surf_pos[:, 0], surf_pos[:, 1], "k.", markersize=0.3)

        plt.tight_layout()
        path = OUT_DIR / f"{prefix}_{idx}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"  Saved {path}")

    return saved


def visualize(model, val_ds, stats, device, n_samples=4):
    """Generate flow field comparison plots for validation samples."""
    OUT_DIR.mkdir(exist_ok=True)
    model.eval()
    saved = []

    indices = list(range(min(n_samples, len(val_ds))))
    for sample_idx in indices:
        x, y_true, is_surface = val_ds[sample_idx]

        with torch.no_grad():
            x_dev = x.unsqueeze(0).to(device)
            x_norm = (x_dev - stats["x_mean"]) / stats["x_std"]
            pred_norm = model({"x": x_norm})["preds"]
            y_pred = (pred_norm * stats["y_std"] + stats["y_mean"]).squeeze(0).cpu()

        pos = x[:, :2].numpy()
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        is_surf_np = is_surface.numpy()

        # Zoom to near-airfoil region
        surf_pos = pos[is_surf_np]
        x_lo, x_hi = -1.0, 2.0
        y_lo = max(0.0, pos[:, 1].min())
        y_hi = min(surf_pos[:, 1].mean() + 3.0, pos[:, 1].max())
        near = (
            (pos[:, 0] >= x_lo) & (pos[:, 0] <= x_hi) &
            (pos[:, 1] >= y_lo) & (pos[:, 1] <= y_hi)
        )

        px, py = pos[near, 0], pos[near, 1]
        triang = _make_triangulation(px, py, surf_pos)

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle(f"Validation sample {sample_idx}", fontsize=14)

        for row, (name, ch) in enumerate(zip(FIELD_NAMES, range(3))):
            gt = y_true_np[near, ch]
            pr = y_pred_np[near, ch]
            err = gt - pr
            vmin, vmax = gt.min(), gt.max()

            ax_gt = axes[row, 0]
            ax_pr = axes[row, 1]
            ax_err = axes[row, 2]

            c0 = ax_gt.tripcolor(triang, gt, shading="flat", vmin=vmin, vmax=vmax, cmap="RdBu_r")
            ax_gt.set_title(f"{name} — Ground Truth")
            fig.colorbar(c0, ax=ax_gt)

            c1 = ax_pr.tripcolor(triang, pr, shading="flat", vmin=vmin, vmax=vmax, cmap="RdBu_r")
            ax_pr.set_title(f"{name} — Predicted")
            fig.colorbar(c1, ax=ax_pr)

            err_max = max(abs(err.min()), abs(err.max()), 1e-6)
            c2 = ax_err.tripcolor(triang, err, shading="flat", vmin=-err_max, vmax=err_max, cmap="RdBu_r")
            ax_err.set_title(f"{name} — Error")
            fig.colorbar(c2, ax=ax_err)

            for ax in [ax_gt, ax_pr, ax_err]:
                ax.set_aspect("equal")
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
                ax.plot(surf_pos[:, 0], surf_pos[:, 1], "k.", markersize=0.3)

        plt.tight_layout()
        path = OUT_DIR / f"val_sample_{sample_idx}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"  Saved {path}")

    return saved
