import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_module():
    path = Path(__file__).resolve().parents[1] / "plugins" / "senpai" / "scripts" / "in_flight.py"
    spec = importlib.util.spec_from_file_location("senpai_in_flight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_entry(root: Path, payload: dict) -> Path:
    path = root / ".senpai" / "in_flight" / "entry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_record_entry_writes_expected_json(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "current_pr_number", lambda root: 1842)
    monkeypatch.setattr(module, "current_branch", lambda root: "fern/test")
    monkeypatch.setattr(module, "ensure_recordable_launch_state", lambda root: "deadbeef")
    monkeypatch.setattr(module, "session_id", lambda: "session-123")
    monkeypatch.setenv("WANDB_ENTITY", "wandb-applied-ai-team")
    monkeypatch.setenv("WANDB_PROJECT", "senpai-v1")

    path = module.record_entry(tmp_path, "senpai-inflight-pr1842-test", 2)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["pr"] == 1842
    assert payload["branch"] == "fern/test"
    assert payload["git_commit"] == "deadbeef"
    assert payload["wandb_tag"] == "senpai-inflight-pr1842-test"
    assert payload["expected_runs"] == 2
    assert payload["launched_by_session"] == "session-123"


def test_harvest_waits_for_expected_runs(tmp_path, monkeypatch):
    module = load_module()
    entry = {
        "schema": 1,
        "pr": 1842,
        "branch": "fern/test",
        "git_commit": "deadbeef",
        "wandb_entity": "wandb-applied-ai-team",
        "wandb_project": "senpai-v1",
        "wandb_tag": "senpai-inflight-pr1842-test",
        "expected_runs": 2,
        "launched_at": "2026-04-17T12:00:00Z",
        "launched_by_session": "session-123",
    }
    entry_path = write_entry(tmp_path, entry)
    calls = []

    monkeypatch.setattr(
        module,
        "load_runs",
        lambda tracked: [
            SimpleNamespace(
                id="run-1",
                name="seed-42",
                state="finished",
                url="https://wandb.ai/run-1",
                summary={"_step": 50, "val/loss": 0.1234},
                config={"seed": 42},
            )
        ],
    )
    monkeypatch.setattr(module, "post_pr_comment", lambda root, pr, body: calls.append(("comment", pr, body)))
    monkeypatch.setattr(module, "mark_ready_for_review", lambda root, pr: calls.append(("ready", pr)))

    harvested = module.harvest(tmp_path)

    assert harvested == 0
    assert entry_path.exists()
    assert calls == []


def test_harvest_posts_marks_ready_and_clears(tmp_path, monkeypatch):
    module = load_module()
    entry = {
        "schema": 1,
        "pr": 1842,
        "branch": "fern/test",
        "git_commit": "deadbeef",
        "wandb_entity": "wandb-applied-ai-team",
        "wandb_project": "senpai-v1",
        "wandb_tag": "senpai-inflight-pr1842-test",
        "expected_runs": 2,
        "launched_at": "2026-04-17T12:00:00Z",
        "launched_by_session": "session-123",
    }
    entry_path = write_entry(tmp_path, entry)
    calls = []

    monkeypatch.setattr(
        module,
        "load_runs",
        lambda tracked: [
            SimpleNamespace(
                id="run-1",
                name="seed-42",
                state="finished",
                url="https://wandb.ai/run-1",
                summary={"_step": 50, "val/loss": 0.1234},
                config={"seed": 42},
            ),
            SimpleNamespace(
                id="run-2",
                name="seed-73",
                state="failed",
                url="https://wandb.ai/run-2",
                summary={"_step": 12, "val/loss": 0.4567},
                config={"seed": 73},
            ),
        ],
    )
    monkeypatch.setattr(module, "post_pr_comment", lambda root, pr, body: calls.append(("comment", pr, body)))
    monkeypatch.setattr(module, "mark_ready_for_review", lambda root, pr: calls.append(("ready", pr)))

    harvested = module.harvest(tmp_path)

    assert harvested == 1
    assert not entry_path.exists()
    assert calls[0][0] == "comment"
    assert calls[0][1] == 1842
    assert "HARNESS: Senpai in-flight harvester" in calls[0][2]
    assert "No LLM model authored this summary." in calls[0][2]
    assert calls[1] == ("ready", 1842)


def test_clear_entries_removes_matching_pr(tmp_path):
    module = load_module()
    write_entry(
        tmp_path,
        {
            "schema": 1,
            "pr": 1842,
            "branch": "fern/test",
            "git_commit": "deadbeef",
            "wandb_entity": "wandb-applied-ai-team",
            "wandb_project": "senpai-v1",
            "wandb_tag": "senpai-inflight-pr1842-test",
            "expected_runs": 1,
            "launched_at": "2026-04-17T12:00:00Z",
            "launched_by_session": "session-123",
        },
    )

    cleared = module.clear_entries(tmp_path, 1842)

    assert cleared == 1
    assert not list((tmp_path / ".senpai" / "in_flight").glob("*.json"))
