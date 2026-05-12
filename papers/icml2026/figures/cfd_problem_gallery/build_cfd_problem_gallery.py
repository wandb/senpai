"""Build the CFD problem gallery figure for the ICML paper.

The default mode reads a small reduced data cache and renders the paper figure.
Use ``--extract-from-pvc`` from a pod with ``/mnt/new-pvc`` mounted to rebuild
that cache from the original DrivAerML, AirfRANS, and TandemFoilSet artifacts.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "cfd_problem_gallery_samples.npz"
DEFAULT_DRIVAER_RENDER = HERE / "data" / "divaerml_render3.png"
DEFAULT_PNG = HERE / "cfd_problem_gallery.png"
DEFAULT_PDF = HERE / "cfd_problem_gallery.pdf"

RNG_SEED = 20260507
PANEL_TITLE_STYLE = {"fontsize": 8.5, "pad": 1.5, "weight": "bold"}
SERIF_STACK = [
    "Times New Roman",
    "Times",
    "Latin Modern Roman",
    "CMU Serif",
    "Computer Modern Serif",
    "serif",
]


def _resolve_possible_munged_symlink(path: Path) -> Path:
    if path.exists():
        return path
    if path.is_symlink():
        target = os.readlink(path)
        if target.startswith("/rsyncd-munged/"):
            unmunged = "/" + target.removeprefix("/rsyncd-munged/").lstrip("/")
            candidate = Path(unmunged)
            if candidate.exists():
                return candidate
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = path.parent / target_path
        if target_path.exists():
            return target_path
    raise FileNotFoundError(path)


def _load_npy(path: Path) -> np.ndarray:
    return np.load(_resolve_possible_munged_symlink(path), mmap_mode="r")


def _sample_rows(rng: np.random.Generator, n_rows: int, count: int) -> np.ndarray:
    count = min(count, n_rows)
    return np.sort(rng.choice(n_rows, size=count, replace=False))


AIRFRANS_X_SAMPLES = np.linspace(0.015, 0.985, 96, dtype=np.float32)


def _normalize_airfoil_chord(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy[:, :2], dtype=np.float32)
    xmin = float(xy[:, 0].min())
    xmax = float(xy[:, 0].max())
    scale = max(xmax - xmin, 1e-6)
    x = (xy[:, 0] - xmin) / scale

    leading = int(np.argmin(xy[:, 0]))
    trailing = int(np.argmax(xy[:, 0]))
    y_le = float(xy[leading, 1])
    y_te = float(xy[trailing, 1])
    chord_y = y_le + (y_te - y_le) * x
    y = (xy[:, 1] - chord_y) / scale
    return np.column_stack([x, y]).astype(np.float32)


def _airfoil_envelope(
    xy: np.ndarray,
    samples: np.ndarray = AIRFRANS_X_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return upper/lower envelopes without relying on raw VTP point order."""

    normalized = _normalize_airfoil_chord(xy)
    x = normalized[:, 0]
    y = normalized[:, 1]
    step = float(samples[1] - samples[0])
    upper = np.empty_like(samples)
    lower = np.empty_like(samples)
    for idx, sample_x in enumerate(samples):
        local = np.abs(x - sample_x) <= step * 0.72
        if local.sum() < 4:
            nearest = np.argpartition(np.abs(x - sample_x), min(12, len(x) - 1))[:12]
            values = y[nearest]
        else:
            values = y[local]
        upper[idx] = np.percentile(values, 96)
        lower[idx] = np.percentile(values, 4)
    return samples, upper, lower


def _airfoil_shape_feature(xy: np.ndarray) -> np.ndarray:
    _, upper, lower = _airfoil_envelope(xy)
    thickness = upper - lower
    camber = 0.5 * (upper + lower)
    return np.concatenate([thickness * 1.4, camber * 2.8]).astype(np.float32)


def _extract_drivaerml(pvc_root: Path, rng: np.random.Generator) -> dict[str, np.ndarray]:
    case_dir = pvc_root / "Processed" / "drivaerml_processed" / "run_1"

    surface_xyz = _load_npy(case_dir / "surface_xyz.npy")
    surface_cp = _load_npy(case_dir / "surface_cp.npy")
    surface_wss = _load_npy(case_dir / "surface_wallshearstress.npy")
    surface_idx = _sample_rows(rng, surface_xyz.shape[0], 85000)

    volume_xyz = _load_npy(case_dir / "volume_xyz.npy")
    volume_velocity = _load_npy(case_dir / "volume_velocity.npy")
    probe_idx = _sample_rows(rng, volume_xyz.shape[0], 650000)
    probe_xyz = np.asarray(volume_xyz[probe_idx], dtype=np.float32)
    probe_velocity = np.asarray(volume_velocity[probe_idx], dtype=np.float32)

    # DrivAer length is the x-axis. Keep the downstream near-wake region and a
    # narrow enough span that the wake samples remain legible in the small panel.
    wake_mask = (
        (probe_xyz[:, 0] > 2.7)
        & (probe_xyz[:, 0] < 13.5)
        & (np.abs(probe_xyz[:, 1]) < 2.8)
        & (probe_xyz[:, 2] > -0.25)
        & (probe_xyz[:, 2] < 2.5)
    )
    wake_xyz = probe_xyz[wake_mask]
    wake_velocity = probe_velocity[wake_mask]
    if wake_xyz.shape[0] > 36000:
        keep = _sample_rows(rng, wake_xyz.shape[0], 36000)
        wake_xyz = wake_xyz[keep]
        wake_velocity = wake_velocity[keep]

    return {
        "drivaer_surface_xyz": np.asarray(surface_xyz[surface_idx], dtype=np.float32),
        "drivaer_surface_cp": np.asarray(surface_cp[surface_idx]).reshape(-1).astype(np.float32),
        "drivaer_surface_wss_mag": np.linalg.norm(
            np.asarray(surface_wss[surface_idx], dtype=np.float32), axis=1
        ),
        "drivaer_wake_xyz": wake_xyz.astype(np.float32),
        "drivaer_wake_speed": np.linalg.norm(wake_velocity, axis=1).astype(np.float32),
    }


def _import_airfrans_reader():
    candidates = [
        Path.cwd(),
        Path("/tmp/icml2026"),
        Path("/workspace/senpai/target/icml2026"),
        Path("/Users/mmcguire/ML/icml2026"),
        Path("/Users/mmcguire/ML/senpai/target/icml2026"),
    ]
    for candidate in candidates:
        if (candidate / "airfrans" / "data" / "vtk_xml.py").exists():
            sys.path.insert(0, str(candidate))
            break
    from airfrans.data.vtk_xml import read_vtk_xml

    return read_vtk_xml


def _extract_airfrans(pvc_root: Path) -> dict[str, np.ndarray]:
    read_vtk_xml = _import_airfrans_reader()
    dataset_root = pvc_root / "datasets" / "airfrans" / "Dataset"
    case_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if len(case_dirs) < 9:
        raise ValueError(f"Expected at least 9 AirfRANS cases in {dataset_root}")

    cases: list[tuple[Path, np.ndarray, np.ndarray, float, float]] = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        mesh = read_vtk_xml(case_dir / f"{case_id}_aerofoil.vtp", point_arrays=["Normals"])
        xy = np.asarray(mesh.points[:, :2], dtype=np.float32)
        feature = _airfoil_shape_feature(xy)
        _, upper, lower = _airfoil_envelope(xy)
        thickness = upper - lower
        camber = 0.5 * (upper + lower)
        cases.append((case_dir, xy, feature, float(thickness.max()), float(camber.mean())))

    features = np.stack([case[2] for case in cases])
    features = (features - features.mean(axis=0, keepdims=True)) / (
        features.std(axis=0, keepdims=True) + 1e-6
    )
    u, singular_values, _ = np.linalg.svd(features, full_matrices=False)
    pcs = u[:, :16] * singular_values[:16]
    selected = [int(np.argmax(np.linalg.norm(pcs - pcs.mean(axis=0), axis=1)))]
    while len(selected) < 9:
        distance_to_selection = np.min(
            np.linalg.norm(pcs[:, None, :] - pcs[np.array(selected)][None, :, :], axis=2),
            axis=1,
        )
        distance_to_selection[selected] = -1.0
        selected.append(int(np.argmax(distance_to_selection)))

    selected_array = np.array(selected, dtype=int)
    by_second_pc = selected_array[np.argsort(pcs[selected_array, 1])]
    ordered: list[int] = []
    for row in [by_second_pc[6:9], by_second_pc[3:6], by_second_pc[0:3]]:
        ordered.extend(row[np.argsort(pcs[row, 0])].tolist())

    data: dict[str, np.ndarray] = {}
    for idx, case_idx in enumerate(ordered):
        case_dir, xy, _, max_thickness, mean_camber = cases[case_idx]
        case_id = case_dir.name
        data[f"airfrans_{idx}_xy"] = xy
        data[f"airfrans_{idx}_case"] = np.array(case_id)
        data[f"airfrans_{idx}_max_thickness"] = np.array(max_thickness, dtype=np.float32)
        data[f"airfrans_{idx}_mean_camber"] = np.array(mean_camber, dtype=np.float32)
    return data


def _extract_tandemfoil(pvc_root: Path, rng: np.random.Generator) -> dict[str, np.ndarray]:
    import torch

    split_root = pvc_root / "datasets" / "tandemfoil" / "splits_v2"

    def foil_xy(path: Path) -> np.ndarray:
        sample = torch.load(path, map_location="cpu", weights_only=True)
        x = sample["x"].numpy()
        is_surface = sample["is_surface"].numpy().astype(bool)
        # ``is_surface`` also includes far-field/domain boundaries. For the
        # actual foil bodies, the surface-aware offset features are zero.
        is_foil = is_surface & (np.linalg.norm(x[:, 2:4], axis=1) < 0.01)
        return x[is_foil, :2].astype(np.float32)

    def geometry_feature(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        center = 0.5 * (lo + hi)
        width = max(float(hi[0] - lo[0]), 1e-6)
        height = max(float(hi[1] - lo[1]), 1e-6)
        scale = max(width, height)
        normalized = (xy - center) / scale
        q = np.array([0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0])
        x_quantiles = np.quantile(normalized[:, 0], q)
        y_quantiles = np.quantile(normalized[:, 1], q)
        hist, _, _ = np.histogram2d(
            normalized[:, 0],
            normalized[:, 1],
            bins=[np.linspace(-0.55, 0.55, 17), np.linspace(-0.55, 0.55, 13)],
        )
        hist = hist.reshape(-1).astype(np.float32)
        hist /= hist.sum() + 1e-6
        stats = np.array(
            [width, height, width / height, height / width, xy.shape[0] / 3000.0],
            dtype=np.float32,
        )
        return np.concatenate([stats, x_quantiles, y_quantiles, hist]).astype(np.float32)

    records: list[tuple[str, Path, np.ndarray, np.ndarray]] = []
    for split_name in ["val_geom_camber_rc", "val_geom_camber_cruise"]:
        for path in sorted((split_root / split_name).glob("*.pt")):
            xy = foil_xy(path)
            records.append((split_name, path, xy, geometry_feature(xy)))

    features = np.stack([record[3] for record in records])
    features = (features - features.mean(axis=0, keepdims=True)) / (
        features.std(axis=0, keepdims=True) + 1e-6
    )
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    selected = []
    for split_name in ["val_geom_camber_rc", "val_geom_camber_cruise"]:
        split_indices = [idx for idx, record in enumerate(records) if record[0] == split_name]
        split_distances = distances[np.ix_(split_indices, split_indices)]
        first, second = np.unravel_index(int(np.argmax(split_distances)), split_distances.shape)
        selected.extend([split_indices[first], split_indices[second]])

    data: dict[str, np.ndarray] = {}
    for idx, record_idx in enumerate(selected):
        split_name, path, xy, _ = records[record_idx]
        label = "race-car" if split_name.endswith("_rc") else "cruise"
        data[f"tandem_{idx}_surface_xy"] = xy
        data[f"tandem_{idx}_label"] = np.array(label)
        data[f"tandem_{idx}_split"] = np.array(split_name)
        data[f"tandem_{idx}_case"] = np.array(path.stem)
    return data


def extract_from_pvc(pvc_root: Path, output: Path) -> None:
    rng = np.random.default_rng(RNG_SEED)
    data = {}
    data.update(_extract_drivaerml(pvc_root, rng))
    data.update(_extract_airfrans(pvc_root))
    data.update(_extract_tandemfoil(pvc_root, rng))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(f"Wrote {output}")


def _robust_limits(values: np.ndarray, low: float = 1.0, high: float = 99.0, pad: float = 0.06) -> tuple[float, float]:
    lo, hi = np.percentile(values, [low, high])
    span = max(float(hi - lo), 1e-6)
    return float(lo - pad * span), float(hi + pad * span)


def _style_2d_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _connected_components_2d(xy: np.ndarray, eps: float = 0.008) -> np.ndarray:
    parent = np.arange(xy.shape[0])

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return int(idx)

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    lower = xy.min(axis=0)
    cells = np.floor((xy - lower) / eps).astype(int)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, cell in enumerate(cells):
        buckets.setdefault((int(cell[0]), int(cell[1])), []).append(idx)

    eps2 = eps * eps
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for cell, indices in buckets.items():
        for dx, dy in offsets:
            for neighbor in buckets.get((cell[0] + dx, cell[1] + dy), []):
                for idx in indices:
                    if neighbor <= idx:
                        continue
                    if float(np.sum((xy[idx] - xy[neighbor]) ** 2)) <= eps2:
                        union(idx, neighbor)

    roots = np.array([find(idx) for idx in range(xy.shape[0])])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(int)


def _plot_drivaer(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    surface = data["drivaer_surface_xyz"]
    cp = data["drivaer_surface_cp"]
    wake = data["drivaer_wake_xyz"]
    wake_speed = data["drivaer_wake_speed"]

    cp_lo, cp_hi = np.percentile(cp, [2, 98])
    speed_lo, speed_hi = np.percentile(wake_speed, [5, 95])
    ax.scatter(
        wake[:, 0],
        wake[:, 1],
        wake[:, 2],
        c=wake_speed,
        cmap="YlGnBu",
        vmin=speed_lo,
        vmax=speed_hi,
        s=0.32,
        alpha=0.18,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        surface[:, 0],
        surface[:, 1],
        surface[:, 2],
        c=cp,
        cmap="coolwarm",
        vmin=cp_lo,
        vmax=cp_hi,
        s=0.15,
        alpha=0.92,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.view_init(elev=20, azim=-57)
    ax.set_xlim(-1.0, 10.8)
    ax.set_ylim(-2.2, 2.2)
    ax.set_zlim(-0.35, 1.75)
    ax.set_box_aspect((4.4, 2.0, 1.25))
    ax.set_axis_off()
    ax.set_title("(a) DrivAerML", **PANEL_TITLE_STYLE)
    ax.text2D(0.03, 0.05, "3D surface pressure + wake nodes", transform=ax.transAxes, fontsize=5.6)


def _crop_drivaer_render(image: np.ndarray) -> np.ndarray:
    """Trim the uniform render background while ignoring the orientation triad."""

    rgb = image[..., :3]
    height, width = rgb.shape[:2]
    background = np.median(rgb[:40, :40].reshape(-1, 3), axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background.astype(np.float32), axis=2)
    if rgb.dtype.kind in {"u", "i"}:
        threshold = 28.0
    else:
        threshold = 28.0 / 255.0
    foreground = distance > threshold

    # The lower-left axis triad is useful in the raw render but distracts at
    # paper scale, so crop from the main car foreground above it.
    foreground[int(height * 0.80) :, :] = False
    ys, xs = np.nonzero(foreground)
    if xs.size == 0:
        return image

    pad_x = int(0.055 * width)
    pad_y = int(0.045 * height)
    top_pad_y = int(0.17 * height)
    x0 = max(int(xs.min()) - pad_x, 0)
    x1 = min(int(xs.max()) + pad_x, width)
    y0 = max(int(ys.min()) - pad_y - top_pad_y, 0)
    y1 = min(int(ys.max()) + pad_y, height)
    return image[y0:y1, x0:x1]


def _plot_drivaer_render(ax: plt.Axes, render_path: Path) -> None:
    image = plt.imread(render_path)
    image = _crop_drivaer_render(image)
    ax.imshow(image)
    height, width = image.shape[:2]
    margin = 0.045 * max(width, height)
    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(height + margin, -margin)
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("(a) DrivAerML", **PANEL_TITLE_STYLE)


def _plot_airfrans(fig: plt.Figure, spec, data: np.lib.npyio.NpzFile) -> None:
    sub = spec.subgridspec(
        5,
        3,
        height_ratios=[0.34, 1.0, 1.0, 1.0, 0.34],
        wspace=0.16,
        hspace=0.04,
    )
    envelopes = [_airfoil_envelope(data[f"airfrans_{idx}_xy"]) for idx in range(9)]
    y_min = min(float(lower.min()) for _, _, lower in envelopes)
    y_max = max(float(upper.max()) for _, upper, _ in envelopes)
    y_mid = 0.5 * (y_min + y_max)
    y_span = max(y_max - y_min, 0.24)
    y_limits = (y_mid - 0.58 * y_span, y_mid + 0.58 * y_span)

    for idx, (samples, upper, lower) in enumerate(envelopes):
        ax = fig.add_subplot(sub[idx // 3 + 1, idx % 3])
        outline_x = np.concatenate(
            [np.array([0.0], dtype=np.float32), samples, np.array([1.0], dtype=np.float32), samples[::-1]]
        )
        outline_y = np.concatenate(
            [np.array([0.0], dtype=np.float32), upper, np.array([0.0], dtype=np.float32), lower[::-1]]
        )
        outline_x = np.append(outline_x, np.float32(0.0))
        outline_y = np.append(outline_y, np.float32(0.0))
        ax.plot(outline_x, outline_y, color="black", lw=0.62, solid_joinstyle="round", zorder=2)
        ax.set_xlim(-0.16, 1.16)
        y_center = 0.5 * (y_limits[0] + y_limits[1])
        y_half_span = 0.70 * (y_limits[1] - y_limits[0])
        ax.set_ylim(y_center - y_half_span, y_center + y_half_span)
        _style_2d_axis(ax)
        ax.set_aspect(2.45, adjustable="box")
    title_ax = fig.add_subplot(spec, frame_on=False)
    title_ax.set_xticks([])
    title_ax.set_yticks([])
    title_ax.set_title("(b) AirfRANS", **PANEL_TITLE_STYLE)


def _tandem_components(surf_xy: np.ndarray) -> list[np.ndarray]:
    labels = _connected_components_2d(surf_xy)
    components = []
    for label in range(int(labels.max()) + 1):
        component = surf_xy[labels == label]
        if component.shape[0] < 30:
            continue
        components.append(component)
    return sorted(components, key=lambda component: float(component[:, 0].mean()))


def _plot_tandem_case_box(ax: plt.Axes, surf_xy: np.ndarray) -> None:
    center = 0.5 * (surf_xy.min(axis=0) + surf_xy.max(axis=0))
    scale = max(float(np.ptp(surf_xy[:, 0])), float(np.ptp(surf_xy[:, 1])), 1e-6)
    for component in _tandem_components(surf_xy)[:2]:
        component = (component - center) / scale
        ax.plot(
            component[:, 0],
            component[:, 1],
            color="black",
            lw=0.54,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    ax.set_xlim(-0.58, 0.58)
    ax.set_ylim(-0.58, 0.58)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#c7c7c7")
        spine.set_linewidth(0.36)


def _plot_tandem(fig: plt.Figure, spec, data: np.lib.npyio.NpzFile) -> None:
    sub = spec.subgridspec(2, 2, wspace=0.08, hspace=0.06)
    for idx in range(4):
        row = idx // 2
        ax = fig.add_subplot(sub[row, idx % 2])
        _plot_tandem_case_box(ax, data[f"tandem_{idx}_surface_xy"])
        ax.set_anchor("S" if row == 0 else "N")

    title_ax = fig.add_subplot(spec, frame_on=False)
    title_ax.set_xticks([])
    title_ax.set_yticks([])
    title_ax.set_title("(c) TandemFoilSet", **PANEL_TITLE_STYLE)


def build_figure(data_path: Path, png_path: Path, pdf_path: Path, drivaer_render_path: Path) -> None:
    data = np.load(data_path, allow_pickle=False)

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update(
        {
            "font.family": SERIF_STACK,
            "font.serif": SERIF_STACK[:-1],
            "font.size": 7,
            "axes.linewidth": 0.5,
            "savefig.pad_inches": 0.01,
        }
    )
    fig = plt.figure(figsize=(7.15, 2.78), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.23, 1.0, 1.05], wspace=0.06)

    ax_drivaer = fig.add_subplot(grid[0, 0])
    _plot_drivaer_render(ax_drivaer, drivaer_render_path)
    _plot_airfrans(fig, grid[0, 1], data)
    _plot_tandem(fig, grid[0, 2], data)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=420, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--drivaer-render", type=Path, default=DEFAULT_DRIVAER_RENDER)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--extract-from-pvc", action="store_true")
    parser.add_argument("--pvc-root", type=Path, default=Path("/mnt/new-pvc"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.extract_from_pvc:
        extract_from_pvc(args.pvc_root, args.data)
    else:
        build_figure(args.data, args.png, args.pdf, args.drivaer_render)


if __name__ == "__main__":
    main()
