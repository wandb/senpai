import json
from pathlib import Path


def test_exported_run_supports_existing_analysis_without_credentials(
    tmp_path, monkeypatch
):
    helpers = (
        Path(__file__).resolve().parents[1]
        / "plugins/senpai/skills/wandb-primary/scripts"
    )
    monkeypatch.syspath_prepend(str(helpers))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    from run_snapshot import RunSnapshot
    from wandb_helpers import fast_scan_history, runs_to_dataframe

    metadata = tmp_path / "run.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "id": "run-17",
                "name": "sparse-validation",
                "state": "finished",
                "created_at": "2026-09-25T00:00:00Z",
                "config": {"lr": 0.01},
                "summary": {"loss": 1.0, "val_loss": 1.5},
            }
        )
        + "\n"
    )
    history = tmp_path / "history.jsonl"
    rows = [
        {"_step": 0, "epoch": 0, "loss": 4.0},
        {"_step": 1, "epoch": 1, "val_loss": 3.0},
        {"_step": 2, "epoch": 2, "loss": 1.0},
    ]
    history.write_text("".join(json.dumps(row) + "\n" for row in rows))

    run = RunSnapshot.from_exports(metadata, history)

    assert list(fast_scan_history(run)) == rows
    assert list(fast_scan_history(run, keys=["val_loss"], min_step=1, max_step=3)) == [
        {"_step": 1, "val_loss": 3.0},
        {"_step": 2},
    ]
    assert runs_to_dataframe([run], metric_keys=["loss", "val_loss"]) == [
        {
            "id": "run-17",
            "name": "sparse-validation",
            "state": "finished",
            "created_at": "2026-09-25T00:00:00Z",
            "config.lr": 0.01,
            "loss": 1.0,
            "val_loss": 1.5,
        }
    ]
