# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import hashlib
import math

import torch

from core.contracts import CaseSample


TANDEM_IS_SURFACE_IDX = 12
TANDEM_LOG_RE_IDX = 13
TANDEM_AOA0_IDX = 14
TANDEM_GAP_IDX = 22


def stable_hash32(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def append_fourier_features(
    x: torch.Tensor,
    space_dim: int,
    freqs: tuple[float, ...],
) -> torch.Tensor:
    coords = x[:, :space_dim]
    pieces = [x]
    for freq in freqs:
        pieces.append(torch.sin(coords * freq))
        pieces.append(torch.cos(coords * freq))
    return torch.cat(pieces, dim=1)


def append_batched_fourier_features(
    x: torch.Tensor,
    *,
    freqs: tuple[float, ...] = (0.5, 2.0, 8.0, 32.0),
) -> torch.Tensor:
    raw_xy = x[..., :2]
    xy_min = raw_xy.amin(dim=1, keepdim=True)
    xy_max = raw_xy.amax(dim=1, keepdim=True)
    xy_norm = (raw_xy - xy_min) / (xy_max - xy_min + 1e-8)
    freq_tensor = x.new_tensor(freqs)
    xy_scaled = xy_norm.unsqueeze(-1) * freq_tensor
    fourier_pe = torch.cat([xy_scaled.sin().flatten(-2), xy_scaled.cos().flatten(-2)], dim=-1)
    return torch.cat([x, fourier_pe], dim=-1)


def compute_te_features(
    raw_xy: torch.Tensor,
    is_surface: torch.Tensor,
    saf_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_coords = raw_xy[..., 0]
    y_coords = raw_xy[..., 1]
    inf = 1e6

    fore_surf = is_surface & (saf_norm <= 0.005)
    fore_x_masked = x_coords * fore_surf.float() - inf * (~fore_surf).float()
    fore_te_idx = fore_x_masked.topk(1, dim=1)[1].squeeze(1)
    fore_te_x = x_coords.gather(1, fore_te_idx.unsqueeze(1)).squeeze(1)
    fore_te_y = y_coords.gather(1, fore_te_idx.unsqueeze(1)).squeeze(1)

    aft_surf = is_surface & (saf_norm > 0.005)
    is_tandem = aft_surf.any(dim=1).float()[:, None]
    aft_surf_safe = aft_surf | (~aft_surf.any(dim=1, keepdim=True))
    aft_x_masked = x_coords * aft_surf.float() - inf * (~aft_surf_safe).float()
    aft_te_idx = aft_x_masked.topk(1, dim=1)[1].squeeze(1)
    aft_te_x = x_coords.gather(1, aft_te_idx.unsqueeze(1)).squeeze(1) * is_tandem.squeeze(1)
    aft_te_y = y_coords.gather(1, aft_te_idx.unsqueeze(1)).squeeze(1) * is_tandem.squeeze(1)

    dx_fore = x_coords - fore_te_x[:, None]
    dy_fore = y_coords - fore_te_y[:, None]
    r_fore = (dx_fore.square() + dy_fore.square()).sqrt().clamp(min=1e-6)

    dx_aft = (x_coords - aft_te_x[:, None]) * is_tandem
    dy_aft = (y_coords - aft_te_y[:, None]) * is_tandem
    r_aft = (dx_aft.square() + dy_aft.square()).sqrt().clamp(min=1e-6) * is_tandem

    return torch.stack([dx_fore, dy_fore, r_fore, dx_aft, dy_aft, r_aft], dim=-1), fore_te_x, fore_te_y


def compute_wake_deficit_features(
    raw_xy: torch.Tensor,
    is_surface: torch.Tensor,
    saf_norm: torch.Tensor,
    gap_raw: torch.Tensor,
    *,
    fore_te_x: torch.Tensor | None = None,
    fore_te_y: torch.Tensor | None = None,
    include_angle: bool = False,
) -> torch.Tensor:
    x_coords = raw_xy[..., 0]
    y_coords = raw_xy[..., 1]

    if fore_te_x is None or fore_te_y is None:
        _, fore_te_x, fore_te_y = compute_te_features(raw_xy, is_surface, saf_norm)

    gap_safe = gap_raw.clamp(min=0.05)
    dx_norm = (x_coords - fore_te_x[:, None]) / gap_safe[:, None]
    dy_norm = (y_coords - fore_te_y[:, None]) / gap_safe[:, None]

    aft_surf = is_surface & (saf_norm > 0.005)
    is_tandem = aft_surf.any(dim=1).float()[:, None]
    dx_norm = dx_norm * is_tandem
    dy_norm = dy_norm * is_tandem

    channels = [dx_norm, dy_norm]
    if include_angle:
        wake_angle = torch.atan2(dy_norm, dx_norm + 1e-8) / torch.pi
        wake_angle = wake_angle * is_tandem
        channels.append(wake_angle)
    return torch.stack(channels, dim=-1)


def compute_cp_panel(
    raw_xy: torch.Tensor,
    aoa_rad: torch.Tensor,
    is_surface: torch.Tensor,
    saf_norm: torch.Tensor,
) -> torch.Tensor:
    x_coords = raw_xy[..., 0]
    y_coords = raw_xy[..., 1]

    fore_surf = is_surface & (saf_norm <= 0.005)
    aft_surf = is_surface & (saf_norm > 0.005)
    inf = 1e6

    fore_x = x_coords.clone()
    fore_x[~fore_surf] = inf
    fore_x_min = fore_x.min(dim=1, keepdim=True).values.clamp(max=inf - 1)
    fore_x[~fore_surf] = -inf
    fore_x_max = fore_x.max(dim=1, keepdim=True).values.clamp(min=-inf + 1)
    fore_chord = (fore_x_max - fore_x_min).clamp(min=1e-6)
    t_fore = ((x_coords - fore_x_min) / fore_chord).clamp(0.02, 0.98)

    aft_x = x_coords.clone()
    aft_x[~aft_surf] = inf
    aft_x_min = aft_x.min(dim=1, keepdim=True).values.clamp(max=inf - 1)
    aft_x[~aft_surf] = -inf
    aft_x_max = aft_x.max(dim=1, keepdim=True).values.clamp(min=-inf + 1)
    aft_chord = (aft_x_max - aft_x_min).clamp(min=1e-6)
    t_aft = ((x_coords - aft_x_min) / aft_chord).clamp(0.02, 0.98)
    t = torch.where(aft_surf, t_aft, t_fore)

    denom = torch.sqrt(t * (1.0 - t)).clamp(min=1e-4)
    fore_y_mean = (y_coords * fore_surf.float()).sum(dim=1, keepdim=True) / fore_surf.float().sum(dim=1, keepdim=True).clamp(min=1)
    aft_y_mean = (y_coords * aft_surf.float()).sum(dim=1, keepdim=True) / aft_surf.float().sum(dim=1, keepdim=True).clamp(min=1)
    y_ref = torch.where(aft_surf, aft_y_mean, fore_y_mean)
    side_sign = torch.sign(y_coords - y_ref)

    aoa = aoa_rad.squeeze(-1)
    cp_panel = -side_sign * 2.0 * torch.sin(aoa.abs().unsqueeze(1)) / denom
    cp_panel = cp_panel * is_surface.float()
    cp_panel = cp_panel.clamp(-4.0, 2.0)
    return cp_panel.unsqueeze(-1)


def compute_vortex_panel_velocity(
    raw_xy: torch.Tensor,
    aoa_rad: torch.Tensor,
    is_surface: torch.Tensor,
    saf_norm: torch.Tensor,
    *,
    n_panels: int = 64,
) -> torch.Tensor:
    batch_size, num_points, _ = raw_xy.shape
    two_pi = 2.0 * torch.pi

    fore_surf = is_surface & (saf_norm <= 0.005)
    aft_surf = is_surface & (saf_norm > 0.005)
    is_tandem = aft_surf.any(dim=1)
    gamma = aoa_rad.sin()

    out = torch.zeros(batch_size, num_points, 4, device=raw_xy.device, dtype=raw_xy.dtype)
    for batch_idx in range(batch_size):
        fore_idx = fore_surf[batch_idx].nonzero(as_tuple=False).view(-1)
        if fore_idx.numel() > 0:
            if fore_idx.numel() > n_panels:
                step = max(fore_idx.numel() // n_panels, 1)
                panel_idx = fore_idx[::step][:n_panels]
            else:
                panel_idx = fore_idx
            num_panel_points = panel_idx.numel()
            panel_xy = raw_xy[batch_idx, panel_idx]
            dx = raw_xy[batch_idx, :, 0].unsqueeze(1) - panel_xy[:, 0].unsqueeze(0)
            dy = raw_xy[batch_idx, :, 1].unsqueeze(1) - panel_xy[:, 1].unsqueeze(0)
            r2 = dx.square() + dy.square() + 1e-8
            g = gamma[batch_idx, 0].item() / num_panel_points
            out[batch_idx, :, 0] = (g / two_pi) * (dy / r2).sum(dim=1)
            out[batch_idx, :, 1] = -(g / two_pi) * (dx / r2).sum(dim=1)

        if is_tandem[batch_idx]:
            aft_idx = aft_surf[batch_idx].nonzero(as_tuple=False).view(-1)
            if aft_idx.numel() > 0:
                if aft_idx.numel() > n_panels:
                    step = max(aft_idx.numel() // n_panels, 1)
                    panel_idx = aft_idx[::step][:n_panels]
                else:
                    panel_idx = aft_idx
                num_panel_points = panel_idx.numel()
                panel_xy = raw_xy[batch_idx, panel_idx]
                dx = raw_xy[batch_idx, :, 0].unsqueeze(1) - panel_xy[:, 0].unsqueeze(0)
                dy = raw_xy[batch_idx, :, 1].unsqueeze(1) - panel_xy[:, 1].unsqueeze(0)
                r2 = dx.square() + dy.square() + 1e-8
                g = gamma[batch_idx, 0].item() / num_panel_points
                out[batch_idx, :, 2] = (g / two_pi) * (dy / r2).sum(dim=1)
                out[batch_idx, :, 3] = -(g / two_pi) * (dx / r2).sum(dim=1)
    return out


def _compute_tandem_cp_panel(x: torch.Tensor) -> torch.Tensor:
    raw_xy = x[:, :2].unsqueeze(0)
    is_surface = (x[:, TANDEM_IS_SURFACE_IDX] > 0.5).unsqueeze(0)
    aoa = x.new_full((1, 1), float(x[0, TANDEM_AOA0_IDX].item()))
    saf_norm = x[:, 2:4].norm(dim=1).unsqueeze(0)
    return compute_cp_panel(raw_xy, aoa, is_surface, saf_norm).squeeze(0)


def _compute_tandem_wake_deficit(x: torch.Tensor, include_angle: bool = False) -> torch.Tensor:
    raw_xy = x[:, :2].unsqueeze(0)
    is_surface = (x[:, TANDEM_IS_SURFACE_IDX] > 0.5).unsqueeze(0)
    saf_norm = x[:, 2:4].norm(dim=1).unsqueeze(0)
    gap_raw = x.new_full((1,), float(x[0, TANDEM_GAP_IDX].item()))
    return compute_wake_deficit_features(
        raw_xy,
        is_surface,
        saf_norm,
        gap_raw,
        include_angle=include_angle,
    ).squeeze(0)


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
    if sample.dataset_name == "tandemfoilset":
        return sample

    surface_x = sample.surface_x
    volume_x = sample.volume_x

    can_concat = volume_x is None or surface_x.shape[-1] == volume_x.shape[-1]
    if can_concat:
        full_x = torch.cat([surface_x, volume_x], dim=0) if volume_x is not None else surface_x
        if enable_fourier:
            full_x = append_fourier_features(full_x, sample.space_dim, fourier_freqs)
        appended_full: list[torch.Tensor] = []
        if enable_cp_panel and sample.dataset_name == "airfrans":
            appended_full.append(_compute_airfrans_cp_panel(full_x))
        if enable_wake_deficit and sample.dataset_name.startswith("tandemfoilset"):
            appended_full.append(_compute_tandem_wake_deficit(full_x, include_angle=enable_wake_angle))
        if appended_full:
            full_x = torch.cat([full_x, *appended_full], dim=1)
        num_surface = sample.surface_x.shape[0]
        surface_x = full_x[:num_surface]
        volume_x = full_x[num_surface:] if sample.volume_x is not None else None
    else:
        if enable_fourier:
            surface_x = append_fourier_features(surface_x, sample.space_dim, fourier_freqs)
            volume_x = append_fourier_features(volume_x, sample.space_dim, fourier_freqs)
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
