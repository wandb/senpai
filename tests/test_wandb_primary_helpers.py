from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / ".claude/skills/wandb-primary/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from curve_plots import _load_history, plot_grad_norm_by_layer
from training_diagnostics import grad_histogram_features, lr_schedule_features
from wandb_helpers import fast_scan_history


class StaticRun:
    def __init__(self, rows: list[dict], run_id: str = "run-1", name: str = "run-name") -> None:
        self._rows = rows
        self.id = run_id
        self.name = name
        self.history_calls: list[dict] = []

    def scan_history(self, **kwargs):
        keys = kwargs.get("keys")
        for row in self._rows:
            if keys is None:
                yield dict(row)
            else:
                yield {key: row.get(key) for key in keys}

    def history(self, **kwargs):
        self.history_calls.append(kwargs)
        keys = kwargs["keys"]
        return pd.DataFrame([{key: row.get(key) for key in keys} for row in self._rows])


class PartialBetaRun:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def beta_scan_history(self, **_kwargs):
        yield from self._rows[:2]
        raise RuntimeError("beta scan failed mid-stream")

    def scan_history(self, **_kwargs):
        yield from self._rows


def test_fast_scan_history_fallback_skips_already_yielded_rows():
    rows = [
        {"_step": 0, "loss": 5.0},
        {"_step": 1, "loss": 4.0},
        {"_step": 2, "loss": 3.0},
        {"_step": 3, "loss": 2.0},
    ]

    observed = list(fast_scan_history(PartialBetaRun(rows), keys=["_step", "loss"]))

    assert observed == rows


def test_load_history_passes_confirmed_step_key_to_run_history():
    run = StaticRun(
        [
            {"global_step": 0, "loss": 5.0},
            {"global_step": 10, "loss": 4.0},
        ]
    )

    df = _load_history(run, keys=["global_step", "loss"], step_key="global_step", samples=25)

    assert list(df["global_step"]) == [0, 10]
    assert run.history_calls
    assert run.history_calls[0]["x_axis"] == "global_step"


def test_lr_schedule_features_detects_restart_below_initial_peak():
    feats = lr_schedule_features(
        values=[1.0, 0.8, 0.6, 0.7, 0.9, 0.5],
        steps=[0, 1, 2, 3, 4, 5],
    )

    assert feats["restart_steps"] == [3.0]


def test_grad_histogram_features_probes_past_first_20_rows():
    rows = [{"global_step": i} for i in range(100)]
    rows.append(
        {
            "global_step": 100,
            "gradients/layer1": {"bins": [-1.0, 0.0, 1.0], "values": [1.0, 3.0]},
            "gradients/layer2": {"bins": [-2.0, 0.0, 2.0], "values": [2.0, 2.0]},
        }
    )

    df = grad_histogram_features(StaticRun(rows), step_key="global_step", probe_rows=150)

    assert set(df["layer"]) == {"layer1", "layer2"}
    assert set(df["step"]) == {100}


def test_plot_grad_norm_by_layer_finds_sparse_scalar_keys(tmp_path):
    rows = [{"global_step": i} for i in range(100)]
    rows.append(
        {
            "global_step": 100,
            "parameters/layer1_grad_norm": 0.5,
            "parameters/layer2_grad_norm": 0.25,
        }
    )
    run = StaticRun(rows)

    png = plot_grad_norm_by_layer(
        run,
        step_key="global_step",
        probe_rows=150,
        out_dir=tmp_path,
    )

    assert png.exists()
