from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import CaseSample, GroupedBatch  # noqa: E402
from core.datasets import DrivAerMLCaseDataset  # noqa: E402
from core import datasets as datasets_module  # noqa: E402
from train import TargetTransform, evaluate_grouped  # noqa: E402


class _FakeStore:
    def __init__(self, *args, **kwargs):
        surface = torch.arange(10, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
        volume = torch.arange(11, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
        self.calls: list[dict[str, object]] = []
        self._cases = {
            "case-a": CaseSample(
                case_id="case-a",
                dataset_name="drivaerml",
                space_dim=3,
                surface_x=surface,
                surface_y=surface.clone(),
                volume_x=volume,
                volume_y=volume.clone(),
                metadata={},
            )
        }

    def case_point_counts(self, case_id: str) -> dict[str, int]:
        case = self._cases[case_id]
        return {
            "n_surface": int(case.surface_x.shape[0]),
            "n_volume": int(case.volume_x.shape[0]),
        }

    def load_case(self, case_id: str, *, surface_rows=None, volume_rows=None) -> CaseSample:
        case = self._cases[case_id]
        self.calls.append(
            {
                "case_id": case_id,
                "surface_rows": None if surface_rows is None else surface_rows.tolist(),
                "volume_rows": None if volume_rows is None else volume_rows.tolist(),
            }
        )
        return CaseSample(
            case_id=case.case_id,
            dataset_name=case.dataset_name,
            space_dim=case.space_dim,
            surface_x=case.surface_x if surface_rows is None else case.surface_x.index_select(0, torch.as_tensor(surface_rows)),
            surface_y=case.surface_y if surface_rows is None else case.surface_y.index_select(0, torch.as_tensor(surface_rows)),
            volume_x=case.volume_x if volume_rows is None else case.volume_x.index_select(0, torch.as_tensor(volume_rows)),
            volume_y=case.volume_y if volume_rows is None else case.volume_y.index_select(0, torch.as_tensor(volume_rows)),
            metadata=dict(case.metadata),
        )


class _IdentitySurfaceModel(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, *, surface_x, surface_mask, volume_x, volume_mask):
        del surface_mask, volume_x, volume_mask
        return {"surface_preds": surface_x, "volume_preds": None}


def _patch_fake_store(monkeypatch):
    monkeypatch.setattr(datasets_module.prepare_drivaerml, "DrivAerMLCaseStore", _FakeStore)


def test_eval_chunk_covers_all_surface_and_volume_points_once(monkeypatch):
    _patch_fake_store(monkeypatch)
    dataset = DrivAerMLCaseDataset(
        ["case-a"],
        surface_only=False,
        max_surface_points=4,
        max_volume_points=3,
        sampling_mode="eval_chunk",
    )

    assert len(dataset) == 4

    surface_seen: list[int] = []
    volume_seen: list[int] = []
    for sample in dataset:
        assert sample.surface_x.shape[0] <= 4
        assert sample.volume_x is not None
        assert sample.volume_x.shape[0] <= 3
        assert sample.metadata["surface_view_count"] == 4
        assert sample.metadata["volume_view_count"] == 4
        surface_seen.extend(int(v) for v in sample.surface_x[:, 0].tolist())
        volume_seen.extend(int(v) for v in sample.volume_x[:, 0].tolist())

    assert sorted(surface_seen) == list(range(10))
    assert sorted(volume_seen) == list(range(11))
    assert dataset.store.calls[0]["surface_rows"] == [0, 4, 8]
    assert dataset.store.calls[0]["volume_rows"] == [0, 4, 8]


def test_train_random_repeats_case_enough_times(monkeypatch):
    _patch_fake_store(monkeypatch)
    torch.manual_seed(0)
    dataset = DrivAerMLCaseDataset(
        ["case-a"],
        surface_only=False,
        max_surface_points=4,
        max_volume_points=3,
        sampling_mode="train_random",
    )

    assert len(dataset) == 4

    total_surface_loaded = 0
    total_volume_loaded = 0
    for sample in dataset:
        assert sample.surface_x.shape[0] == 4
        assert sample.volume_x is not None
        assert sample.volume_x.shape[0] == 3
        assert sample.metadata["surface_sampling_mode"] == "train_random"
        assert sample.metadata["volume_sampling_mode"] == "train_random"
        total_surface_loaded += sample.surface_x.shape[0]
        total_volume_loaded += sample.volume_x.shape[0]

    assert total_surface_loaded >= 10
    assert total_volume_loaded >= 11
    assert all(call["surface_rows"] is not None for call in dataset.store.calls)
    assert all(call["volume_rows"] is not None for call in dataset.store.calls)


def test_chunked_eval_reaggregates_per_case_rel_l2():
    def batch(case_id: str, preds: list[float], targets: list[float]) -> GroupedBatch:
        pred_tensor = torch.tensor(preds, dtype=torch.float32).view(1, -1, 1)
        target_tensor = torch.tensor(targets, dtype=torch.float32).view(1, -1, 1)
        mask = torch.ones(1, len(preds), dtype=torch.bool)
        return GroupedBatch(
            case_ids=[case_id],
            dataset_name="drivaerml",
            space_dim=3,
            surface_x=pred_tensor,
            surface_y=target_tensor,
            surface_mask=mask,
            volume_x=None,
            volume_y=None,
            volume_mask=None,
            metadata=[],
        )

    loader = [
        batch("case-a", [1.0, 2.0], [1.0, 2.0]),
        batch("case-a", [3.0, 5.0], [3.0, 4.0]),
        batch("case-b", [4.0], [2.0]),
        batch("case-b", [2.0], [2.0]),
    ]
    metrics = evaluate_grouped(
        _IdentitySurfaceModel(),
        None,
        loader,
        TargetTransform(pressure_index=0, stats_mean=None, stats_std=None),
        torch.device("cpu"),
    )

    case_a = math.sqrt(1.0 / 30.0)
    case_b = math.sqrt(4.0 / 8.0)
    expected = (case_a + case_b) / 2.0
    assert metrics["surface_rel_l2"] == expected
    assert metrics["surface_rel_l2_pct"] == expected * 100.0
