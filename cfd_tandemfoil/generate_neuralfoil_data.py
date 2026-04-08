"""Generate synthetic single-foil surface pressure samples using NeuralFoil.

Usage:
    cd cfd_tandemfoil
    python generate_neuralfoil_data.py --n_samples 5000 --out_dir data/neuralfoil_synthetic

Generates (x, y, is_surface) tuples where:
  x: 24-dim features (template mesh geometry, updated AoA/Re/NACA, stagger=99 as synthetic marker)
  y: pressure targets — surface nodes get NeuralFoil Cp*q, volume nodes get freestream velocity
  is_surface: bool mask from template

Volume supervision is disabled during training (detected via stagger sentinel=99.0).
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data.prepare import parse_naca
from data.prepare_multi import load_data

KINEMATIC_VISCOSITY = 1.35e-5  # m^2/s — derived from training data Re/Umag correspondence
SYNTHETIC_MARKER = 99.0        # sentinel value for x[:, 23] (stagger) to flag synthetic samples


def get_neuralfoil_cp(naca_str: str, aoa_deg: float, re: float, model_size: str = "large"):
    """Return (Cp_upper, Cp_lower) at 32 x/c positions using NeuralFoil."""
    import neuralfoil as nf
    import aerosandbox as asb

    af = asb.Airfoil(f"naca{naca_str}")
    result = nf.get_aero_from_airfoil(af, alpha=float(aoa_deg), Re=float(re), model_size=model_size)

    conf = float(result["analysis_confidence"])
    upper_ue = np.array([result[f"upper_bl_ue/vinf_{i}"] for i in range(32)]).flatten()
    lower_ue = np.array([result[f"lower_bl_ue/vinf_{i}"] for i in range(32)]).flatten()
    Cp_upper = 1.0 - upper_ue ** 2
    Cp_lower = 1.0 - lower_ue ** 2

    return Cp_upper, Cp_lower, conf


def interpolate_cp_to_nodes(pos, is_surface, Cp_upper, Cp_lower, x_nf):
    """Interpolate NeuralFoil Cp at 32 x/c points onto mesh surface node positions.

    Args:
        pos: (N, 2) node positions
        is_surface: (N,) bool
        Cp_upper: (32,) Cp at x_nf positions for upper surface
        Cp_lower: (32,) Cp at x_nf positions for lower surface
        x_nf: (32,) x/c positions from NeuralFoil (0.015625 to 0.984375)

    Returns:
        Cp_at_nodes: (n_surf,) Cp at surface node positions
    """
    surf_pos = pos[is_surface]  # (n_surf, 2)
    x_surf = surf_pos[:, 0].numpy()
    y_surf = surf_pos[:, 1].numpy()

    # Upper surface: y >= 0 (or more precisely, the top side of the airfoil)
    # For NACA airfoils at chord-length 1, x runs from 0 to 1, y > 0 is upper surface
    is_upper = y_surf >= 0.0

    # Clamp x to NeuralFoil's valid range
    x_clamped = np.clip(x_surf, x_nf[0], x_nf[-1])

    Cp_at_nodes = np.zeros(surf_pos.shape[0], dtype=np.float32)
    if is_upper.any():
        Cp_at_nodes[is_upper] = np.interp(x_clamped[is_upper], x_nf, Cp_upper)
    if (~is_upper).any():
        Cp_at_nodes[~is_upper] = np.interp(x_clamped[~is_upper], x_nf, Cp_lower)

    return Cp_at_nodes


def create_synthetic_sample(template_x, template_y, is_surface, aoa_deg, re, naca_str, Cp_upper, Cp_lower, x_nf):
    """Create synthetic (x, y, is_surface) from template mesh + NeuralFoil Cp.

    template_x: (N, 24) — template mesh features
    template_y: (N, 3)  — template mesh targets (not used for values, only structure)
    is_surface: (N,)    — surface mask
    """
    import neuralfoil as nf

    N = template_x.shape[0]
    n_surf = is_surface.sum().item()

    # Interpolate Cp to surface nodes
    pos = template_x[:, :2]  # raw node positions
    Cp_nodes = interpolate_cp_to_nodes(pos, is_surface, Cp_upper, Cp_lower, x_nf)

    # Convert Cp → raw pressure (density-normalized: p = Cp * 0.5 * V_inf^2)
    V_inf = re * KINEMATIC_VISCOSITY
    q_inf = 0.5 * V_inf ** 2
    p_surface = torch.from_numpy(Cp_nodes * q_inf)

    # Build synthetic y:
    # - surface nodes: no-slip (Ux=Uy=0), pressure from NeuralFoil
    # - volume nodes: freestream (Ux=V_inf, Uy=0, p=0)
    y_syn = torch.zeros(N, 3, dtype=torch.float32)
    vol_mask = ~is_surface
    y_syn[vol_mask, 0] = V_inf  # freestream x-velocity (needed for _umag_q)
    y_syn[is_surface, 2] = p_surface

    # Build synthetic x: copy template, update AoA/Re/NACA, set synthetic marker
    x_syn = template_x.clone()
    log_re = math.log(re)
    aoa_rad = aoa_deg * math.pi / 180.0
    naca_enc = torch.tensor(parse_naca(naca_str), dtype=torch.float32)

    x_syn[:, 13] = log_re    # log_Re feature
    x_syn[:, 14] = aoa_rad   # AoA0_rad
    x_syn[:, 15:18] = naca_enc.expand(N, 3)  # NACA0 encoding
    x_syn[:, 18] = 0.0       # AoA1_rad = 0 (single-foil)
    x_syn[:, 19:22] = 0.0    # NACA1 = (0, 0, 0)
    x_syn[:, 22] = 0.0       # gap = 0 (single-foil)
    x_syn[:, 23] = SYNTHETIC_MARKER  # stagger = sentinel marker

    return x_syn, y_syn, is_surface.clone()


def validate_samples(templates, x_nf, n_validate=5):
    """Compare NeuralFoil Cp against real training samples as a sanity check."""
    import neuralfoil as nf
    import aerosandbox as asb

    print("\n--- Validation: NeuralFoil vs real data ---")
    for i in range(min(n_validate, len(templates))):
        x_t, y_t, is_surf = templates[i]
        # Extract NACA and flow conditions from template
        naca_enc = x_t[0, 15:18].tolist()
        # Reverse parse_naca: naca_enc = (m/9, p/9, t/24)
        # Find closest matching NACA string
        m = round(naca_enc[0] * 9)
        p = round(naca_enc[1] * 9)
        t = round(naca_enc[2] * 24)
        naca_str = f"{m}{p}{t:02d}"
        log_re = x_t[0, 13].item()
        re = math.exp(log_re)
        aoa_rad = x_t[0, 14].item()
        aoa_deg = aoa_rad * 180 / math.pi

        try:
            Cp_up, Cp_lo, conf = get_neuralfoil_cp(naca_str, aoa_deg, re)
        except Exception as e:
            print(f"  Sample {i}: NACA{naca_str} AoA={aoa_deg:.1f} Re={re:.0f} → NeuralFoil failed: {e}")
            continue

        # Compute real Cp from template
        pos = x_t[:, :2]
        surf_pos = pos[is_surf]
        p_real = y_t[is_surf, 2].numpy()
        V_inf = re * KINEMATIC_VISCOSITY
        q_inf = 0.5 * V_inf ** 2
        Cp_real = p_real / q_inf if q_inf > 0 else p_real

        # NeuralFoil Cp interpolated
        Cp_nodes = interpolate_cp_to_nodes(pos, is_surf, Cp_up, Cp_lo, x_nf)

        mae = np.abs(Cp_nodes - Cp_real).mean()
        print(f"  Sample {i}: NACA{naca_str} AoA={aoa_deg:.1f}° Re={re:.0f} | "
              f"Cp MAE(NF vs CFD)={mae:.3f} | conf={conf:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--out_dir", type=str, default="data/neuralfoil_synthetic")
    parser.add_argument("--manifest", type=str, default="data/split_manifest.json")
    parser.add_argument("--stats_file", type=str, default="data/split_stats.json")
    parser.add_argument("--n_templates", type=int, default=20, help="Number of template meshes to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_size", type=str, default="large")
    parser.add_argument("--validate", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    import neuralfoil as nf
    x_nf = nf.bl_x_points  # 32 x/c positions

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data for template meshes...")
    train_ds, _, stats, _ = load_data(args.manifest, args.stats_file, debug=False)

    # Sample diverse template meshes from the training set (only single-foil: gap=0)
    template_indices = []
    for idx in range(len(train_ds)):
        x_t, y_t, is_surf = train_ds[idx]
        gap = x_t[0, 22].item()
        if abs(gap) < 0.01 and len(template_indices) < args.n_templates:
            template_indices.append(idx)
        if len(template_indices) >= args.n_templates:
            break

    templates = [train_ds[i] for i in template_indices]
    print(f"Using {len(templates)} template meshes (single-foil only)")

    if args.validate:
        validate_samples(templates, x_nf, n_validate=5)

    print(f"Generating {args.n_samples} synthetic samples...")
    n_failed = 0
    n_saved = 0

    for i in range(args.n_samples):
        # Sample random NACA geometry
        t_pct = random.randint(8, 20)   # thickness percent
        m_pct = random.randint(0, 6)    # max camber percent
        p_pct = random.randint(2, 6) if m_pct > 0 else 0  # camber position (0 if symmetric)
        naca_str = f"{m_pct}{p_pct}{t_pct:02d}"

        # Sample flow conditions
        aoa_deg = random.uniform(-15.0, 20.0)
        re = random.uniform(5e4, 2e6)

        # Run NeuralFoil
        try:
            Cp_up, Cp_lo, conf = get_neuralfoil_cp(naca_str, aoa_deg, re, model_size=args.model_size)
        except Exception as e:
            n_failed += 1
            continue

        # Skip low-confidence predictions (likely stalled/separated flow)
        if conf < 0.3:
            n_failed += 1
            continue

        # Pick random template mesh
        x_t, y_t, is_surf = random.choice(templates)

        # Create synthetic sample
        x_syn, y_syn, is_surf_syn = create_synthetic_sample(
            x_t, y_t, is_surf, aoa_deg, re, naca_str, Cp_up, Cp_lo, x_nf
        )

        # Save
        save_path = out_dir / f"sample_{n_saved:05d}.pt"
        torch.save({"x": x_syn, "y": y_syn, "is_surface": is_surf_syn}, save_path)
        n_saved += 1

        if (n_saved % 500) == 0:
            print(f"  {n_saved}/{args.n_samples} saved ({n_failed} failed/skipped)")

    print(f"\nDone. Saved {n_saved} samples to {out_dir} ({n_failed} failed/skipped).")


if __name__ == "__main__":
    main()
