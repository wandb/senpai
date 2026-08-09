import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.senpai_tool_telemetry import build_report, parse_timestamp, render_table


ROOT = Path(__file__).resolve().parents[1]


def write_conversation(
    state_dir: Path,
    conversation_id: str,
    *,
    model: str,
    tools: list[dict],
    events: list[dict],
) -> Path:
    conversation = state_dir / conversation_id
    event_dir = conversation / "events"
    event_dir.mkdir(parents=True)
    (conversation / "base_state.json").write_text(
        json.dumps(
            {
                "id": conversation_id,
                "agent": {"llm": {"model": model}, "tools": tools},
            }
        ),
        encoding="utf-8",
    )
    for index, event in enumerate(events):
        (event_dir / f"event-{index:05d}-{index}.json").write_text(
            json.dumps(event),
            encoding="utf-8",
        )
    return event_dir


def action(call_id: str, tool: str, at: datetime, arguments: dict) -> dict:
    return {
        "kind": "ActionEvent",
        "timestamp": at.isoformat(),
        "tool_name": tool,
        "tool_call_id": call_id,
        "tool_call": {
            "id": call_id,
            "name": tool,
            "arguments": json.dumps(arguments),
        },
    }


def observation(call_id: str, tool: str, at: datetime, *, error=False) -> dict:
    return {
        "kind": "ObservationEvent",
        "timestamp": at.isoformat(),
        "tool_name": tool,
        "tool_call_id": call_id,
        "observation": {"is_error": error, "kind": "TestObservation"},
    }


def telemetry_fixture(tmp_path: Path, now: datetime) -> Path:
    state = tmp_path / "state"
    root_events = [
        action("launch", "run_training", now - timedelta(minutes=10), {"secret": "launch-secret"}),
        observation("launch", "run_training", now - timedelta(minutes=9, seconds=55)),
        action("status-1", "get_training_status", now - timedelta(minutes=8), {"training_id": "job-1"}),
        observation("status-1", "get_training_status", now - timedelta(minutes=7, seconds=59)),
        action("status-2", "get_training_status", now - timedelta(minutes=7, seconds=30), {"training_id": "job-1"}),
        observation("status-2", "get_training_status", now - timedelta(minutes=7, seconds=29), error=True),
        action("old", "terminal", now - timedelta(hours=13), {"command": "old-secret"}),
    ]
    write_conversation(
        state,
        "root-conversation",
        model="openai/gpt-test",
        tools=[{"name": "senpai_terminal", "params": {"role": "student"}}],
        events=root_events,
    )
    write_conversation(
        state / "children" / "task-1",
        "child-conversation",
        model="openai/gpt-fast",
        tools=[{"name": "terminal", "params": {}}],
        events=[
            action("child", "terminal", now - timedelta(minutes=6), {"command": "token=child-secret"}),
            observation("child", "terminal", now - timedelta(minutes=5, seconds=58)),
        ],
    )
    return state


def test_recursive_report_counts_outcomes_latency_polling_and_repetition(tmp_path: Path):
    now = datetime(2026, 8, 7, 12, tzinfo=UTC)
    state = telemetry_fixture(tmp_path, now)

    report = build_report(
        [state],
        now=now,
        hours=12,
        repeat_window_seconds=120,
    )

    assert report["conversations"] == 2
    assert report["tool_calls"] == 4
    assert report["arguments_redacted"] is True
    assert report["status_polling"] == {
        "checks": 2,
        "successful_job_launches": 1,
        "checks_per_successful_launch": 2.0,
        "rapid_repeats": 1,
    }
    assert report["repetition"]["rapid_repeats"] == 1

    rows = {
        (row["role"], row["model"], row["tool"]): row
        for row in report["by_source_scope_role_model_tool"]
    }
    assert rows[("student", "openai/gpt-test", "get_training_status")][
        "errors"
    ] == 1
    assert rows[("student", "openai/gpt-test", "get_training_status")][
        "latency"
    ]["p50_seconds"] == 1.0
    assert rows[("student", "openai/gpt-fast", "terminal")]["successes"] == 1
    assert {row["source"] for row in rows.values()} == {str(state.resolve())}
    assert rows[("student", "openai/gpt-test", "run_training")]["scope"] == "root"
    assert rows[("student", "openai/gpt-test", "run_training")]["depth"] == 0
    assert rows[("student", "openai/gpt-fast", "terminal")]["scope"] == "child"
    assert rows[("student", "openai/gpt-fast", "terminal")]["depth"] == 1

    encoded = json.dumps(report)
    assert "launch-secret" not in encoded
    assert "child-secret" not in encoded
    assert "old-secret" not in encoded
    assert "Status polling: 2 checks / 1 launches" in render_table(report)


def test_naive_timestamps_use_the_explicit_positive_offset_for_their_source(
    tmp_path: Path,
):
    now = datetime(2026, 8, 7, 12, tzinfo=UTC)
    state = tmp_path / "positive-offset-state"
    write_conversation(
        state,
        "root-conversation",
        model="openai/gpt-test",
        tools=[{"name": "senpai_terminal", "params": {"role": "student"}}],
        events=[
            action(
                "outside-window",
                "terminal",
                datetime(2026, 8, 7, 12, 30),
                {"command": "redacted-old"},
            ),
            action(
                "inside-window",
                "terminal",
                datetime(2026, 8, 7, 13, 30),
                {"command": "redacted-current"},
            ),
            observation(
                "inside-window",
                "terminal",
                datetime(2026, 8, 7, 13, 30, 2),
            ),
        ],
    )

    with pytest.raises(ValueError, match="naive event timestamp requires"):
        build_report(
            [state],
            now=now,
            hours=1,
            repeat_window_seconds=120,
        )

    report = build_report(
        [state],
        now=now,
        hours=1,
        repeat_window_seconds=120,
        default_naive_timezone=UTC,
        source_naive_timezones={state: timezone(timedelta(hours=2))},
    )

    assert report["tool_calls"] == 1
    assert report["by_source_scope_role_model_tool"][0]["latency"][
        "p50_seconds"
    ] == 2.0
    assert report["naive_timestamp_timezones"] == {
        "default": "UTC",
        "by_source": {str(state.resolve()): "UTC+02:00"},
    }


def test_aware_timestamp_keeps_its_embedded_offset():
    assert parse_timestamp(
        "2026-08-07T13:30:00+02:00",
        naive_timezone=timezone(-timedelta(hours=7)),
    ) == datetime(2026, 8, 7, 11, 30, tzinfo=UTC)


def test_cli_prints_table_and_writes_json(tmp_path: Path):
    now = datetime.now(UTC)
    state = telemetry_fixture(tmp_path, now)
    output = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "senpai_tool_telemetry.py"),
            str(state),
            "--hours",
            "12",
            "--json",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Tool telemetry: 4 calls, 2 conversations" in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["tool_calls"] == 4


def test_cli_applies_a_source_specific_timezone_to_naive_events(tmp_path: Path):
    now = datetime.now(UTC)
    local_now = now.astimezone(timezone(timedelta(hours=2))).replace(tzinfo=None)
    state = tmp_path / "source"
    write_conversation(
        state,
        "root-conversation",
        model="openai/gpt-test",
        tools=[{"name": "senpai_terminal", "params": {"role": "student"}}],
        events=[action("call", "terminal", local_now, {"secret": "never-print"})],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "senpai_tool_telemetry.py"),
            str(state),
            "--hours",
            "1",
            "--source-timezone",
            f"{state}=+02:00",
            "--json",
            "-",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["tool_calls"] == 1
    assert report["naive_timestamp_timezones"]["by_source"] == {
        str(state.resolve()): "UTC+02:00"
    }
    assert "never-print" not in completed.stdout
