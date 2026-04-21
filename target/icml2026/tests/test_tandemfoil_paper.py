from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import GroupedBatch  # noqa: E402
from tandemfoil_paper.data.split_paper_experiment4 import (  # noqa: E402
    split_tail_extrapolation_indices,
    split_uniform_indices,
)
from train import TargetTransform, evaluate_grouped  # noqa: E402


class _IdentityPaperModel(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, *, surface_x, surface_mask, volume_x, volume_mask):
        del surface_mask, volume_mask
        return {
            "surface_preds": surface_x[..., :3],
            "volume_preds": None if volume_x is None else volume_x[..., :3],
        }


def test_uniform_split_uses_80_10_10_counts():
    splits = split_uniform_indices(900, seed=42)
    assert len(splits["train"]) == 720
    assert len(splits["val"]) == 90
    assert len(splits["test"]) == 90
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])


def test_tail_extrapolation_split_uses_5_percent_tails():
    values = list(range(900))
    splits = split_tail_extrapolation_indices(values, seed=42)
    assert len(splits["train"]) == 720
    assert len(splits["val"]) == 90
    assert len(splits["test"]) == 90
    assert set(splits["test"][:45]) == set(range(45))
    assert set(splits["test"][45:]) == set(range(855, 900))


def test_paper_eval_reports_normalized_field_mse():
    surface_target = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]], dtype=torch.float32)
    volume_target = torch.tensor([[[2.0, 2.0, 2.0]]], dtype=torch.float32)
    surface_pred = torch.tensor([[[2.0, 2.0, 3.0], [0.0, 1.0, 2.0]]], dtype=torch.float32)
    volume_pred = torch.tensor([[[3.0, 1.0, 2.0]]], dtype=torch.float32)
    batch = GroupedBatch(
        case_ids=["paper-case"],
        dataset_name="tandemfoilset_paper",
        space_dim=2,
        surface_x=torch.cat([surface_pred, torch.zeros(1, 2, 1)], dim=-1),
        surface_y=surface_target,
        surface_mask=torch.ones(1, 2, dtype=torch.bool),
        volume_x=torch.cat([volume_pred, torch.zeros(1, 1, 1)], dim=-1),
        volume_y=volume_target,
        volume_mask=torch.ones(1, 1, dtype=torch.bool),
        metadata=[],
    )
    metrics = evaluate_grouped(
        _IdentityPaperModel(),
        None,
        [batch],
        TargetTransform(pressure_index=2, stats_mean=torch.zeros(3), stats_std=torch.ones(3)),
        torch.device("cpu"),
    )

    surface_sq = 3.0
    volume_sq = 2.0
    expected_surface = surface_sq / (2 * 3)
    expected_volume = volume_sq / (1 * 3)
    expected_field = (surface_sq + volume_sq) / (3 * 3)

    assert math.isclose(metrics["surface_mse"], expected_surface, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(metrics["volume_mse"], expected_volume, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(metrics["field_mse"], expected_field, rel_tol=1e-6, abs_tol=1e-6)
