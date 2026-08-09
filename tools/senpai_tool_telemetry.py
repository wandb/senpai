#!/usr/bin/env python3

"""Summarize tool use across a Senpai OpenHands state tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


EVENT_INDEX = re.compile(r"^event-(\d+)-")
STATUS_TOOLS = frozenset({"get_training_status", "get_job_status"})
LAUNCH_TOOLS = frozenset({"run_training", "run_job"})


@dataclass(frozen=True, slots=True)
class Conversation:
    source_root: Path
    event_dir: Path
    conversation_id: str
    model: str
    role: str | None
    depth: int

    @property
    def state_root(self) -> Path:
        return self.event_dir.parent.parent


@dataclass(frozen=True, slots=True)
class ToolCall:
    source_root: Path
    conversation_id: str
    role: str
    model: str
    depth: int
    tool: str
    occurred_at: datetime | None
    argument_fingerprint: str
    outcome: str
    latency_seconds: float | None


def parse_timezone(value: str) -> tzinfo:
    normalized = value.strip()
    if normalized.upper() in {"UTC", "Z"}:
        return UTC
    if re.fullmatch(r"[+-]\d{2}:\d{2}", normalized):
        parsed = datetime.fromisoformat(f"2000-01-01T00:00:00{normalized}")
        offset = parsed.utcoffset()
        if offset is None:
            raise ValueError(f"invalid UTC offset: {value}")
        return timezone(offset)
    try:
        return ZoneInfo(normalized)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            f"unknown timezone {value!r}; use an IANA name, UTC, or an offset like +02:00"
        ) from error


def parse_timestamp(
    value: object,
    *,
    naive_timezone: tzinfo | None,
) -> datetime | None:
    if not isinstance(value, str):
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        if naive_timezone is None:
            raise ValueError(
                "naive event timestamp requires --naive-timezone or a matching "
                "--source-timezone"
            )
        parsed = parsed.replace(tzinfo=naive_timezone)
    return parsed.astimezone(UTC)


def event_sort_key(path: Path) -> tuple[int, str]:
    match = EVENT_INDEX.match(path.name)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def discover_event_dirs(roots: Sequence[Path]) -> list[tuple[Path, Path]]:
    discovered: dict[Path, Path] = {}
    for source_root in roots:
        source_root = source_root.expanduser().resolve()
        if source_root.name == "events" and any(source_root.glob("event-*.json")):
            discovered.setdefault(source_root, source_root)
        if source_root.exists():
            for path in source_root.rglob("event-*.json"):
                if path.parent.name == "events":
                    discovered.setdefault(path.parent.resolve(), source_root)
    return sorted(
        ((source_root, event_dir) for event_dir, source_root in discovered.items()),
        key=lambda item: (str(item[0]), str(item[1])),
    )


def _base_state(event_dir: Path) -> dict[str, Any]:
    path = event_dir.parent / "base_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _explicit_role(base_state: dict[str, Any], event_dir: Path) -> str | None:
    tags = base_state.get("tags") or {}
    if tags.get("role") in {"advisor", "student"}:
        return str(tags["role"])

    tools = ((base_state.get("agent") or {}).get("tools") or [])
    for tool in tools:
        params = tool.get("params") or {}
        if params.get("role") in {"advisor", "student"}:
            return str(params["role"])
    if any(tool.get("name") == "senpai_training" for tool in tools):
        return "student"

    for part in reversed(event_dir.parts):
        lowered = part.lower()
        if lowered == "advisor" or lowered.startswith("advisor-"):
            return "advisor"
        if lowered == "student" or lowered.startswith("student-"):
            return "student"
    return None


def load_conversations(event_dirs: Sequence[tuple[Path, Path]]) -> list[Conversation]:
    partial: list[Conversation] = []
    root_roles: dict[tuple[Path, Path], str] = {}
    for source_root, event_dir in event_dirs:
        base_state = _base_state(event_dir)
        role = _explicit_role(base_state, event_dir)
        conversation = Conversation(
            source_root=source_root,
            event_dir=event_dir,
            conversation_id=str(base_state.get("id") or event_dir.parent.name),
            model=str(
                ((base_state.get("agent") or {}).get("llm") or {}).get("model")
                or "unknown"
            ),
            role=role,
            depth=sum(
                part == "children"
                for part in event_dir.relative_to(source_root).parts
            ),
        )
        partial.append(conversation)
        if role is not None and "children" not in event_dir.parts:
            root_roles[(source_root, conversation.state_root)] = role

    conversations = []
    for conversation in partial:
        role = conversation.role
        if role is None:
            role = next(
                (
                    inherited
                    for ancestor in conversation.event_dir.parents
                    if (
                        inherited := root_roles.get(
                            (conversation.source_root, ancestor)
                        )
                    )
                    is not None
                ),
                "unknown",
            )
        conversations.append(
            Conversation(
                source_root=conversation.source_root,
                event_dir=conversation.event_dir,
                conversation_id=conversation.conversation_id,
                model=conversation.model,
                role=role,
                depth=conversation.depth,
            )
        )
    return conversations


def _argument_fingerprint(event: dict[str, Any]) -> str:
    tool_call = event.get("tool_call") or {}
    arguments: object = tool_call.get("arguments")
    if arguments is None:
        arguments = {
            key: value
            for key, value in (event.get("action") or {}).items()
            if key != "kind"
        }
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    encoded = json.dumps(
        arguments,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _observation_failed(event: dict[str, Any]) -> bool:
    observation = event.get("observation") or {}
    return bool(
        event.get("is_error")
        or observation.get("is_error")
        or str(observation.get("kind", "")).lower().startswith("error")
    )


def load_tool_calls(
    conversations: Sequence[Conversation],
    *,
    since: datetime,
    default_naive_timezone: tzinfo | None,
    source_naive_timezones: Mapping[Path, tzinfo],
) -> tuple[list[ToolCall], list[str], int]:
    calls: list[ToolCall] = []
    parse_errors: list[str] = []
    event_count = 0
    for conversation in conversations:
        naive_timezone = source_naive_timezones.get(
            conversation.source_root,
            default_naive_timezone,
        )
        events: list[dict[str, Any]] = []
        for path in sorted(conversation.event_dir.glob("event-*.json"), key=event_sort_key):
            try:
                events.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                parse_errors.append(f"{path}: {type(error).__name__}: {error}")
        event_count += len(events)
        observations = {
            str(event.get("tool_call_id")): event
            for event in events
            if event.get("kind") == "ObservationEvent" and event.get("tool_call_id")
        }
        for event in events:
            if event.get("kind") != "ActionEvent":
                continue
            occurred_at = parse_timestamp(
                event.get("timestamp"),
                naive_timezone=naive_timezone,
            )
            if occurred_at is not None and occurred_at < since:
                continue
            tool = str(event.get("tool_name") or "unknown")
            observation = observations.get(str(event.get("tool_call_id")))
            latency = None
            outcome = "pending"
            if observation is not None:
                outcome = "error" if _observation_failed(observation) else "success"
                observed_at = parse_timestamp(
                    observation.get("timestamp"),
                    naive_timezone=naive_timezone,
                )
                if occurred_at is not None and observed_at is not None:
                    latency = max(0.0, (observed_at - occurred_at).total_seconds())
            calls.append(
                ToolCall(
                    source_root=conversation.source_root,
                    conversation_id=conversation.conversation_id,
                    role=conversation.role or "unknown",
                    model=conversation.model,
                    depth=conversation.depth,
                    tool=tool,
                    occurred_at=occurred_at,
                    argument_fingerprint=_argument_fingerprint(event),
                    outcome=outcome,
                    latency_seconds=latency,
                )
            )
    return calls, parse_errors, event_count


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean_seconds": round(sum(values) / len(values), 3) if values else None,
        "p50_seconds": _rounded(percentile(values, 0.50)),
        "p95_seconds": _rounded(percentile(values, 0.95)),
        "max_seconds": round(max(values), 3) if values else None,
    }


def _rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def build_report(
    roots: Sequence[Path],
    *,
    now: datetime,
    hours: float,
    repeat_window_seconds: float,
    default_naive_timezone: tzinfo | None = None,
    source_naive_timezones: Mapping[Path, tzinfo] | None = None,
) -> dict[str, Any]:
    now = now.astimezone(UTC)
    since = now - timedelta(hours=hours)
    event_dirs = discover_event_dirs(roots)
    conversations = load_conversations(event_dirs)
    resolved_source_timezones = {
        path.expanduser().resolve(): zone
        for path, zone in (source_naive_timezones or {}).items()
    }
    calls, parse_errors, event_count = load_tool_calls(
        conversations,
        since=since,
        default_naive_timezone=default_naive_timezone,
        source_naive_timezones=resolved_source_timezones,
    )

    grouped: dict[tuple[Path, int, str, str, str], list[ToolCall]] = defaultdict(
        list
    )
    for call in calls:
        grouped[
            (call.source_root, call.depth, call.role, call.model, call.tool)
        ].append(call)

    rows = []
    for (source_root, depth, role, model, tool), selected in sorted(grouped.items()):
        latencies = [
            call.latency_seconds
            for call in selected
            if call.latency_seconds is not None
        ]
        rows.append(
            {
                "source": str(source_root),
                "scope": "root" if depth == 0 else "child",
                "depth": depth,
                "role": role,
                "model": model,
                "tool": tool,
                "calls": len(selected),
                "successes": sum(call.outcome == "success" for call in selected),
                "errors": sum(call.outcome == "error" for call in selected),
                "pending": sum(call.outcome == "pending" for call in selected),
                "latency": latency_summary(latencies),
            }
        )

    repeats: dict[tuple[Path, int, str, str, str], int] = defaultdict(int)
    previous: dict[tuple[Path, str, str, str], datetime] = {}
    for call in sorted(
        (call for call in calls if call.occurred_at is not None),
        key=lambda call: call.occurred_at or datetime.min.replace(tzinfo=UTC),
    ):
        key = (
            call.source_root,
            call.conversation_id,
            call.tool,
            call.argument_fingerprint,
        )
        last = previous.get(key)
        if last is not None and (call.occurred_at - last).total_seconds() <= repeat_window_seconds:
            repeats[
                (call.source_root, call.depth, call.role, call.model, call.tool)
            ] += 1
        previous[key] = call.occurred_at

    status_calls = [call for call in calls if call.tool in STATUS_TOOLS]
    successful_launches = sum(
        call.tool in LAUNCH_TOOLS and call.outcome == "success" for call in calls
    )
    repeated_rows = [
        {
            "source": str(source_root),
            "scope": "root" if depth == 0 else "child",
            "depth": depth,
            "role": role,
            "model": model,
            "tool": tool,
            "rapid_repeats": count,
        }
        for (source_root, depth, role, model, tool), count in sorted(repeats.items())
    ]
    return {
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "window": {
            "hours": hours,
            "since": since.isoformat().replace("+00:00", "Z"),
        },
        "roots": [str(path.expanduser().resolve()) for path in roots],
        "naive_timestamp_timezones": {
            "default": str(default_naive_timezone)
            if default_naive_timezone is not None
            else None,
            "by_source": {
                str(path): str(zone)
                for path, zone in sorted(
                    resolved_source_timezones.items(),
                    key=lambda item: str(item[0]),
                )
            },
        },
        "arguments_redacted": True,
        "model_source": "current conversation base_state.json",
        "conversations": len(conversations),
        "events": event_count,
        "tool_calls": len(calls),
        "parse_errors": parse_errors,
        "status_polling": {
            "checks": len(status_calls),
            "successful_job_launches": successful_launches,
            "checks_per_successful_launch": (
                round(len(status_calls) / successful_launches, 3)
                if successful_launches
                else None
            ),
            "rapid_repeats": sum(
                count
                for (
                    _source_root,
                    _depth,
                    _role,
                    _model,
                    tool,
                ), count in repeats.items()
                if tool in STATUS_TOOLS
            ),
        },
        "repetition": {
            "window_seconds": repeat_window_seconds,
            "rapid_repeats": sum(repeats.values()),
            "by_source_scope_role_model_tool": repeated_rows,
        },
        "by_source_scope_role_model_tool": rows,
    }


def render_table(report: dict[str, Any]) -> str:
    rows = report["by_source_scope_role_model_tool"]
    lines = [
        (
            f"Tool telemetry: {report['tool_calls']} calls, "
            f"{report['conversations']} conversations, "
            f"{report['window']['hours']:g}h window"
        )
    ]
    if not rows:
        lines.append("No tool calls found.")
    else:
        headers = (
            "Source",
            "Scope",
            "Role",
            "Model",
            "Tool",
            "Calls",
            "OK",
            "Err",
            "Open",
            "P50s",
            "P95s",
        )
        body = []
        for row in rows:
            latency = row["latency"]
            body.append(
                (
                    row["source"],
                    f"{row['scope']}:{row['depth']}",
                    row["role"],
                    row["model"],
                    row["tool"],
                    str(row["calls"]),
                    str(row["successes"]),
                    str(row["errors"]),
                    str(row["pending"]),
                    _display_seconds(latency["p50_seconds"]),
                    _display_seconds(latency["p95_seconds"]),
                )
            )
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in body))
            for index in range(len(headers))
        ]
        lines.extend(
            (
                _format_row(headers, widths),
                _format_row(tuple("-" * width for width in widths), widths),
                *(_format_row(row, widths) for row in body),
            )
        )

    status = report["status_polling"]
    ratio = status["checks_per_successful_launch"]
    lines.append(
        "Status polling: "
        f"{status['checks']} checks / {status['successful_job_launches']} launches"
        + (f" = {ratio:g} checks/launch" if ratio is not None else "")
        + f"; {status['rapid_repeats']} rapid repeats"
    )
    lines.append(
        "Repeated identical calls: "
        f"{report['repetition']['rapid_repeats']} within "
        f"{report['repetition']['window_seconds']:g}s"
    )
    if report["parse_errors"]:
        lines.append(f"Parse errors: {len(report['parse_errors'])}")
    lines.append("Arguments: redacted (only in-memory fingerprints were compared).")
    return "\n".join(lines)


def _format_row(values: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True))


def _display_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def timezone_argument(value: str) -> tzinfo:
    try:
        return parse_timezone(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def source_timezone_argument(value: str) -> tuple[Path, tzinfo]:
    path, separator, timezone_name = value.rpartition("=")
    if not separator or not path or not timezone_name:
        raise argparse.ArgumentTypeError(
            "expected ROOT=TIMEZONE, for example /state/cedar=+02:00"
        )
    return Path(path), timezone_argument(timezone_name)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description=(
            "Recursively summarize OpenHands tool events without exposing tool arguments."
        )
    )
    cli.add_argument("roots", nargs="+", type=Path, help="State directories to scan.")
    cli.add_argument(
        "--hours",
        type=float,
        default=12,
        help="Include calls from this many recent hours (default: 12).",
    )
    cli.add_argument(
        "--repeat-window-seconds",
        type=float,
        default=120,
        help="Window for identical-call repetition signals (default: 120).",
    )
    cli.add_argument(
        "--naive-timezone",
        type=timezone_argument,
        metavar="TIMEZONE",
        help=(
            "Timezone for naive event timestamps in every source, as an IANA name, "
            "UTC, or a fixed offset such as +02:00. Naive timestamps fail clearly "
            "when no default or source-specific timezone is provided."
        ),
    )
    cli.add_argument(
        "--source-timezone",
        action="append",
        default=[],
        type=source_timezone_argument,
        metavar="ROOT=TIMEZONE",
        help=(
            "Timezone for one positional state root; repeat for roots captured in "
            "different timezones. Overrides --naive-timezone for that root."
        ),
    )
    cli.add_argument(
        "--json",
        dest="json_output",
        metavar="PATH",
        help="Also write the full JSON report; use - for JSON on stdout.",
    )
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.hours <= 0 or args.repeat_window_seconds <= 0:
        raise SystemExit("--hours and --repeat-window-seconds must be positive")
    roots = [path.expanduser().resolve() for path in args.roots]
    missing = [str(path) for path in roots if not path.exists()]
    if missing:
        raise SystemExit(f"state path does not exist: {', '.join(missing)}")
    source_timezones: dict[Path, tzinfo] = {}
    for path, zone in args.source_timezone:
        resolved = path.expanduser().resolve()
        if resolved not in roots:
            raise SystemExit(f"--source-timezone path is not a state root: {resolved}")
        if resolved in source_timezones:
            raise SystemExit(f"duplicate --source-timezone path: {resolved}")
        source_timezones[resolved] = zone

    try:
        report = build_report(
            roots,
            now=datetime.now(UTC),
            hours=args.hours,
            repeat_window_seconds=args.repeat_window_seconds,
            default_naive_timezone=args.naive_timezone,
            source_naive_timezones=source_timezones,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    table = render_table(report)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output == "-":
        print(table, file=sys.stderr)
        sys.stdout.write(encoded)
    else:
        print(table)
        if args.json_output:
            Path(args.json_output).expanduser().write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
