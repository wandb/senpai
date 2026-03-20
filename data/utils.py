# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Visualization utilities for flow field prediction."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

# Precomputed per-channel normalization stats over full datasets.
# Use these instead of recomputing at train time.
dataset_stats = {
    # 899 samples, ~77M points
    "raceCar_single_randomFields": {
        "x_mean": torch.tensor([0.5040773749351501, 0.9807566404342651, -0.008257666602730751, 0.2679096460342407, 3.593095302581787, 3.5915610790252686, 3.565711498260498, 3.6399688720703125, 3.7195773124694824, 3.420386791229248, 3.2609429359436035, 3.2011666297912598, 0.01843155547976494, 14.517987251281738, -0.08600194752216339, 0.6088866591453552, 0.552564799785614, 0.5195903778076172]),
        "x_std": torch.tensor([1.3417450189590454, 1.1432934999465942, 1.1652204990386963, 1.0989903211593628, 1.5283128023147583, 2.045794725418091, 2.1078484058380127, 2.0708839893341064, 1.4339309930801392, 2.141476631164551, 2.1777710914611816, 2.198330879211426, 0.13450588285923004, 0.826269268989563, 0.050274644047021866, 0.2574135661125183, 0.22478650510311127, 0.1952073723077774]),
        "y_mean": torch.tensor([32.02717208862305, -1.0690499544143677, -222.42221069335938]),
        "y_std": torch.tensor([25.00892448425293, 11.726093292236328, 961.1544799804688]),
    },
}

OUT_DIR = Path("plots")
POINT_SIZE = 0.5  # scatter marker size
QUIVER_STRIDE = 50  # plot every Nth point as arrow


def _add_quiver(ax, px, py, ux, uy, stride=QUIVER_STRIDE):
    """Add velocity arrows to an axes, subsampled for clarity."""
    ax.quiver(px[::stride], py[::stride], ux[::stride], uy[::stride],
              angles="xy", scale_units="xy", scale=150, width=0.002,
              color="k", alpha=0.4, headwidth=3)


def _setup_ax(ax, x_lo, x_hi, y_lo, y_hi, surf_pos):
    """Common axis setup."""
    ax.set_aspect("equal")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.plot(surf_pos[:, 0], surf_pos[:, 1], "k.", markersize=0.3)


def _get_view_bounds(pos, surf_pos):
    """Compute standard view bounds."""
    x_lo, x_hi = -1.0, 2.0
    y_lo = max(0.0, pos[:, 1].min())
    y_hi = min(surf_pos[:, 1].mean() + 3.0, pos[:, 1].max())
    near = (
        (pos[:, 0] >= x_lo) & (pos[:, 0] <= x_hi) &
        (pos[:, 1] >= y_lo) & (pos[:, 1] <= y_hi)
    )
    return x_lo, x_hi, y_lo, y_hi, near


def _scatter_field(ax, fig, px, py, vals, cmap="viridis", vmin=None, vmax=None):
    """Plot a scalar field as a scatter plot."""
    sc = ax.scatter(px, py, c=vals, s=POINT_SIZE, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors="none", rasterized=True)
    fig.colorbar(sc, ax=ax)
    return sc


def plot_samples(dataset, indices=None, n_samples=4, prefix="data_sample", out_dir=None):
    """Plot ground truth flow fields: velocity magnitude + arrows, and pressure."""
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    if indices is None:
        indices = list(range(min(n_samples, len(dataset))))

    for i, idx in enumerate(indices):
        x, y_true, is_surface = dataset[idx]
        pos = x[:, :2].numpy()
        y_np = y_true.numpy()
        is_surf_np = is_surface.numpy()
        surf_pos = pos[is_surf_np]

        x_lo, x_hi, y_lo, y_hi, near = _get_view_bounds(pos, surf_pos)
        px, py = pos[near, 0], pos[near, 1]

        ux, uy, p_field = y_np[near, 0], y_np[near, 1], y_np[near, 2]
        vmag = np.sqrt(ux**2 + uy**2)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Sample {idx}", fontsize=14)

        # Velocity magnitude + arrows
        _scatter_field(axes[0], fig, px, py, vmag, cmap="viridis")
        axes[0].set_title("|U| (velocity magnitude)")
        _add_quiver(axes[0], px, py, ux, uy)
        _setup_ax(axes[0], x_lo, x_hi, y_lo, y_hi, surf_pos)

        # Pressure
        _scatter_field(axes[1], fig, px, py, p_field, cmap="RdBu_r")
        axes[1].set_title("p (pressure)")
        _setup_ax(axes[1], x_lo, x_hi, y_lo, y_hi, surf_pos)

        plt.tight_layout()
        path = out_dir / f"{prefix}_{idx}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved.append(path)
        print(f"  Saved {path}")

    return saved


def visualize(samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], out_dir=None):
    """Generate flow field comparison plots: velocity (magnitude+arrows) and pressure.

    Layout: 2 rows (velocity, pressure) x 3 cols (GT, Predicted, Error).

    Args:
        samples: list of (pos, y_true, y_pred, is_surface) tuples, all CPU tensors.
                 pos: (N, 2), y_true/y_pred: (N, 3), is_surface: (N,) bool.
    """
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for sample_idx, (pos_t, y_true, y_pred, is_surface) in enumerate(samples):
        pos = pos_t.numpy()
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        is_surf_np = is_surface.numpy()

        surf_pos = pos[is_surf_np]
        x_lo, x_hi, y_lo, y_hi, near = _get_view_bounds(pos, surf_pos)
        px, py = pos[near, 0], pos[near, 1]

        gt_ux, gt_uy, gt_p = y_true_np[near, 0], y_true_np[near, 1], y_true_np[near, 2]
        pr_ux, pr_uy, pr_p = y_pred_np[near, 0], y_pred_np[near, 1], y_pred_np[near, 2]
        gt_vmag = np.sqrt(gt_ux**2 + gt_uy**2)
        pr_vmag = np.sqrt(pr_ux**2 + pr_uy**2)
        err_vmag = gt_vmag - pr_vmag
        err_p = gt_p - pr_p

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f"Validation sample {sample_idx}", fontsize=14)

        # Row 0: Velocity magnitude
        vmin_v, vmax_v = gt_vmag.min(), gt_vmag.max()
        _scatter_field(axes[0, 0], fig, px, py, gt_vmag, cmap="viridis", vmin=vmin_v, vmax=vmax_v)
        axes[0, 0].set_title("|U| — Ground Truth")
        _add_quiver(axes[0, 0], px, py, gt_ux, gt_uy)

        _scatter_field(axes[0, 1], fig, px, py, pr_vmag, cmap="viridis", vmin=vmin_v, vmax=vmax_v)
        axes[0, 1].set_title("|U| — Predicted")
        _add_quiver(axes[0, 1], px, py, pr_ux, pr_uy)

        err_v_max = max(abs(err_vmag.min()), abs(err_vmag.max()), 1e-6)
        _scatter_field(axes[0, 2], fig, px, py, err_vmag, cmap="RdBu_r", vmin=-err_v_max, vmax=err_v_max)
        axes[0, 2].set_title("|U| — Error")

        # Row 1: Pressure
        vmin_p, vmax_p = gt_p.min(), gt_p.max()
        _scatter_field(axes[1, 0], fig, px, py, gt_p, cmap="RdBu_r", vmin=vmin_p, vmax=vmax_p)
        axes[1, 0].set_title("p — Ground Truth")

        _scatter_field(axes[1, 1], fig, px, py, pr_p, cmap="RdBu_r", vmin=vmin_p, vmax=vmax_p)
        axes[1, 1].set_title("p — Predicted")

        err_p_max = max(abs(err_p.min()), abs(err_p.max()), 1e-6)
        _scatter_field(axes[1, 2], fig, px, py, err_p, cmap="RdBu_r", vmin=-err_p_max, vmax=err_p_max)
        axes[1, 2].set_title("p — Error")

        for ax in axes.flat:
            _setup_ax(ax, x_lo, x_hi, y_lo, y_hi, surf_pos)

        plt.tight_layout()
        path = out_dir / f"val_sample_{sample_idx}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved.append(path)
        print(f"  Saved {path}")

    return saved
