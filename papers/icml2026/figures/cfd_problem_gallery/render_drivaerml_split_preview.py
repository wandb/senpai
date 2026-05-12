"""Render a local DrivAerML split surface/volume preview.

The input is a reduced local cache with dense front-half surface points and a
rear center-volume slice. It is intentionally independent of the pai2 PVC so we
can iterate on the figure design offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "drivaerml_front_surface_rear_volume_preview.npz"
GALLERY_DATA = HERE / "data" / "cfd_problem_gallery_samples.npz"
DEFAULT_OUTPUT = HERE / "drivaerml_front_surface_rear_volume_preview.png"


def _project(points: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = np.deg2rad(-18.0)
    pitch = np.deg2rad(18.0)
    p = points - center
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x1 = cy * p[:, 0] - sy * p[:, 1]
    y1 = sy * p[:, 0] + cy * p[:, 1]
    z1 = p[:, 2]
    y2 = cp * y1 - sp * z1
    z2 = sp * y1 + cp * z1
    return -x1, z2, y2


def render(data_path: Path, output_path: Path) -> None:
    data = np.load(data_path)
    front = data["front_xyz"].astype(np.float32)
    rear = data["rear_xyz"].astype(np.float32)
    speed = data["rear_speed"].astype(np.float32)
    mid_x = float(data["mid_x"])
    car_width = float(data["maxs"][1] - data["mins"][1]) if "maxs" in data and "mins" in data else 2.0
    split_x = mid_x
    strip_half_width = 0.25 * car_width

    rear_mask = (rear[:, 0] >= split_x) & (rear[:, 0] <= 8.8) & (np.abs(rear[:, 1]) <= strip_half_width)
    rear = rear[rear_mask]
    speed = speed[rear_mask]

    front = front[front[:, 0] <= split_x]
    rear_surface = None
    if GALLERY_DATA.exists():
        gallery_data = np.load(GALLERY_DATA)
        rear_surface = gallery_data["drivaer_surface_xyz"].astype(np.float32)
        rear_surface = rear_surface[rear_surface[:, 0] >= split_x]

    center = np.array([2.9, 0.0, 0.48], dtype=np.float32)
    front_x, front_y, front_depth = _project(front, center)
    rear_x, rear_y, rear_depth = _project(rear, center)
    if rear_surface is not None:
        rear_surface_x, rear_surface_y, rear_surface_depth = _project(rear_surface, center)
        rear_surface_order = np.argsort(rear_surface_depth)
        rear_surface_x = rear_surface_x[rear_surface_order]
        rear_surface_y = rear_surface_y[rear_surface_order]

    front_order = np.argsort(front_depth)
    rear_order = np.argsort(rear_depth)
    front_x = front_x[front_order]
    front_y = front_y[front_order]
    front_depth = front_depth[front_order]
    rear_x = rear_x[rear_order]
    rear_y = rear_y[rear_order]
    speed = speed[rear_order]

    shade = (front_depth - np.percentile(front_depth, 2)) / (
        np.percentile(front_depth, 98) - np.percentile(front_depth, 2)
    )
    height = (front_y - np.percentile(front_y, 2)) / (
        np.percentile(front_y, 98) - np.percentile(front_y, 2)
    )
    shade = np.clip(0.30 + 0.40 * shade + 0.24 * height, 0.14, 0.88)
    front_colors = np.column_stack(
        [shade * 0.80, shade * 0.86, shade * 0.96, np.full_like(shade, 0.76)]
    )

    vmin, vmax = np.percentile(speed, [2.0, 98.0])
    fig = plt.figure(figsize=(11.0, 5.1), dpi=260)
    ax = fig.add_axes([0.015, 0.06, 0.96, 0.86])
    ax.set_facecolor("white")

    if rear_surface is not None:
        ax.scatter(
            rear_surface_x,
            rear_surface_y,
            color=(0.32, 0.37, 0.44, 0.18),
            s=0.045,
            linewidths=0,
            rasterized=True,
        )

    ax.scatter(
        rear_x,
        rear_y,
        c=speed,
        s=0.022,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        alpha=0.64,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        front_x,
        front_y,
        c=front_colors,
        s=0.018,
        linewidths=0,
        rasterized=True,
    )

    ax.set_aspect("equal")
    x_min = min(np.percentile(front_x, 0.1), np.percentile(rear_x, 0.1)) - 0.35
    x_max = max(np.percentile(front_x, 99.9), np.percentile(rear_x, 99.9)) + 0.35
    y_min = min(np.percentile(front_y, 0.2), np.percentile(rear_y, 0.2)) - 0.35
    y_max = max(np.percentile(front_y, 99.8), np.percentile(rear_y, 99.8)) + 0.38
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.data, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
