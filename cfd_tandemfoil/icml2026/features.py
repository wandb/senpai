# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import hashlib
import math

import torch

from cfd_tandemfoil.icml2026.contracts import CaseSample


TANDEM_IS_SURFACE_IDX = 12
TANDEM_LOG_RE_IDX = 13
TANDEM_AOA0_IDX = 14
TANDEM_GAP_IDX = 22


def stable_hash32(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def append_fourier_features(x: torch.Tensor, space_dim: int, freqs: tuple[float, ...]) -> torch.Tensor:
    coords = x[:, :space_dim]
    pieces = [x]
    for freq in freqs:
        pieces.append(torch.sin(coords * freq))
        pieces.append(torch.cos(coords * freq))
    return torch.cat(pieces, dim=1)


def _compute_tandem_cp_panel(x: torch.Tensor) -> torch.Tensor:
    pos = x[:, :2]
    saf = x[:, 2:4]
    is_surface = x[:, TANDEM_IS_SURFACE_IDX] > 0.5
    aoa = float(x[0, TANDEM_AOA0_IDX].item())
    if not torch.any(is_surface):
        return torch.zeros(len(x), 1, dtype=x.dtype)

    saf_norm = saf.norm(dim=1)
    fore_surface = is_surface & (saf_norm <= 5e-3)
    aft_surface = is_surface & ~fore_surface
    x_coords = pos[:, 0]
    y_coords = pos[:, 1]

    def _foil_t(mask: torch.Tensor) -> torch.Tensor:
        if not torch.any(mask):
            return torch.full_like(x_coords, 0.5)
        x_masked = x_coords[mask]
        chord_min = x_masked.min()
        chord_max = x_masked.max()
        chord = torch.clamp(chord_max - chord_min, min=1e-6)
        t = ((x_coords - chord_min) / chord).clamp(0.02, 0.98)
        return t

    t_fore = _foil_t(fore_surface)
    t_aft = _foil_t(aft_surface)
    t = torch.where(aft_surface, t_aft, t_fore)

    fore_y_ref = y_coords[fore_surface].mean() if torch.any(fore_surface) else torch.tensor(0.0, dtype=x.dtype)
    aft_y_ref = y_coords[aft_surface].mean() if torch.any(aft_surface) else fore_y_ref
    y_ref = torch.where(aft_surface, aft_y_ref, fore_y_ref)
    sign = torch.sign(y_coords - y_ref)
    denom = torch.sqrt(t * (1.0 - t)).clamp(min=1e-4)
    cp = -sign * 2.0 * math.sin(abs(aoa)) / denom
    cp = torch.where(is_surface, cp.clamp(-4.0, 2.0), torch.zeros_like(cp))
    return cp.unsqueeze(1)


def _compute_tandem_wake_deficit(x: torch.Tensor, include_angle: bool = False) -> torch.Tensor:
    pos = x[:, :2]
    saf = x[:, 2:4]
    is_surface = x[:, TANDEM_IS_SURFACE_IDX] > 0.5
    gap = float(x[0, TANDEM_GAP_IDX].item())
    saf_norm = saf.norm(dim=1)
    fore_surface = is_surface & (saf_norm <= 5e-3)
    aft_surface = is_surface & ~fore_surface
    if not torch.any(fore_surface):
        width = 3 if include_angle else 2
        return torch.zeros(len(x), width, dtype=x.dtype)

    fore_nodes = pos[fore_surface]
    te_idx = torch.argmax(fore_nodes[:, 0])
    te = fore_nodes[te_idx]
    gap_safe = max(abs(gap), 0.05)
    dx = (pos[:, 0] - te[0]) / gap_safe
    dy = (pos[:, 1] - te[1]) / gap_safe
    tandem_scale = 1.0 if torch.any(aft_surface) else 0.0
    pieces = [dx * tandem_scale, dy * tandem_scale]
    if include_angle:
        pieces.append(torch.atan2(dy, dx + 1e-8) / math.pi * tandem_scale)
    return torch.stack(pieces, dim=1)


def _compute_airfrans_cp_panel(x: torch.Tensor) -> torch.Tensor:
    pos = x[:, :2]
    freestream = x[:, 2:4]
    is_surface = x[:, -1] > 0.5
    if not torch.any(is_surface):
        return torch.zeros(len(x), 1, dtype=x.dtype)
    aoa = math.atan2(float(freestream[0, 1].item()), float(freestream[0, 0].item()))
    x_coords = pos[:, 0]
    y_coords = pos[:, 1]
    surf_x = x_coords[is_surface]
    chord_min = surf_x.min()
    chord_max = surf_x.max()
    chord = torch.clamp(chord_max - chord_min, min=1e-6)
    t = ((x_coords - chord_min) / chord).clamp(0.02, 0.98)
    y_ref = y_coords[is_surface].mean()
    sign = torch.sign(y_coords - y_ref)
    denom = torch.sqrt(t * (1.0 - t)).clamp(min=1e-4)
    cp = -sign * 2.0 * math.sin(abs(aoa)) / denom
    cp = torch.where(is_surface, cp.clamp(-4.0, 2.0), torch.zeros_like(cp))
    return cp.unsqueeze(1)


def augment_case_sample(
    sample: CaseSample,
    *,
    enable_fourier: bool = False,
    fourier_freqs: tuple[float, ...] = (0.5, 2.0, 8.0, 32.0),
    enable_cp_panel: bool = False,
    enable_wake_deficit: bool = False,
    enable_wake_angle: bool = False,
) -> CaseSample:
    surface_x = sample.surface_x
    volume_x = sample.volume_x
    full_x = torch.cat([surface_x, volume_x], dim=0) if volume_x is not None else surface_x

    appended_full: list[torch.Tensor] = []
    if enable_fourier:
        full_x = append_fourier_features(full_x, sample.space_dim, fourier_freqs)
    if enable_cp_panel:
        if sample.dataset_name == "tandemfoilset":
            appended_full.append(_compute_tandem_cp_panel(full_x))
        elif sample.dataset_name == "airfrans":
            appended_full.append(_compute_airfrans_cp_panel(full_x))
    if enable_wake_deficit and sample.dataset_name == "tandemfoilset":
        appended_full.append(_compute_tandem_wake_deficit(full_x, include_angle=enable_wake_angle))
    if appended_full:
        full_x = torch.cat([full_x, *appended_full], dim=1)

    n_surface = sample.surface_x.shape[0]
    surface_x = full_x[:n_surface]
    volume_x = full_x[n_surface:] if sample.volume_x is not None else None
    return CaseSample(
        case_id=sample.case_id,
        dataset_name=sample.dataset_name,
        space_dim=sample.space_dim,
        surface_x=surface_x,
        surface_y=sample.surface_y,
        volume_x=volume_x,
        volume_y=sample.volume_y,
        metadata=dict(sample.metadata),
    )
