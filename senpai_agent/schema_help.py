"""Explain a tool call that fails validation: the field, its schema, an example.

OpenHands answers a malformed call with pydantic's error text and the names of
the parameters it received, which leaves the model to guess the shape it
should have sent. Every Senpai tool, and the OpenHands terminal and file
editor, is registered through ``explaining_validation_errors`` so a failing
call is answered with each failing field, the schema at that field, the
tool's required fields and one valid example payload.
"""

from __future__ import annotations

import json
from typing import Any

from openhands.sdk.tool import Action, ToolDefinition
from pydantic import ValidationError

# One valid payload per action type, keyed by the action class name; a test
# validates each one against its class so an example cannot rot.
EXAMPLES: dict[str, dict[str, Any]] = {
    "SubmitExperimentResultAction": {
        "branch": "student-one/wider-tile",
        "remote_branch_sha_before_push": "9f2d4c0b1e7a6d5c4b3a2f1e0d9c8b7a6f5e4d3c",
        "result": {
            "assignment": {
                "repo": "acme/widgets",
                "pr_number": 17,
                "assignment_id": "assignment-17",
                "revision_id": "revision-1",
                "expected_head_sha": "1c3e5a7b9d2f4e6a8c0b1d3f5a7c9e2b4d6f8a0c",
                "student": "student-one",
            },
            "status": "succeeded",
            "hypothesis": "A wider tile raises occupancy without spilling.",
            "summary": "Run ab12cd34 finished 1.2% faster than the baseline at commit 1c3e5a7.",
            "runs": [
                {
                    "run_id": "ab12cd34",
                    "url": "https://wandb.ai/acme/widgets/runs/ab12cd34",
                    "state": "finished",
                }
            ],
            "primary_metric": {
                "name": "latency_us",
                "direction": "minimize",
                "baseline": 118.2,
                "candidate": 116.8,
            },
            "commit_sha": "1c3e5a7b9d2f4e6a8c0b1d3f5a7c9e2b4d6f8a0c",
        },
    },
    "PostAssignmentCommentAction": {
        "assignment": {
            "pr_number": 17,
            "assignment_id": "assignment-17",
            "revision_id": "revision-1",
            "expected_pr_head_sha": "1c3e5a7b9d2f4e6a8c0b1d3f5a7c9e2b4d6f8a0c",
        },
        "comment_id": "progress-2",
        "comment": "# Progress\n\n## Baseline reproduced at 118.2 us\n\nRun ab12cd34 finished.",
    },
    "PostPeerCommentAction": {
        "assignment": {
            "pr_number": 17,
            "assignment_id": "assignment-17",
            "revision_id": "revision-1",
            "expected_pr_head_sha": "1c3e5a7b9d2f4e6a8c0b1d3f5a7c9e2b4d6f8a0c",
        },
        "target_pr_number": 21,
        "comment_id": "peer-1",
        "comment": "# Question\n\n## Does your twin pin the clock?",
    },
    "BroadcastMessageAction": {
        "assignment": {
            "pr_number": 17,
            "assignment_id": "assignment-17",
            "revision_id": "revision-1",
            "expected_pr_head_sha": "1c3e5a7b9d2f4e6a8c0b1d3f5a7c9e2b4d6f8a0c",
        },
        "broadcast_id": "finding-1",
        "message": "# Finding\n\n## Same-name extension modules are cached; rename twins.",
    },
    "PushExperimentCommitAction": {
        "assignment": {
            "pr_number": 17,
            "assignment_id": "assignment-17",
            "revision_id": "revision-1",
            "expected_pr_head_sha": "1c3e5a7b9d2f4e6a8c0b1d3f5a7c9e2b4d6f8a0c",
        },
        "local_commit_sha": "7e9a1c3b5d7f9a1c3e5b7d9f1a3c5e7b9d1f3a5c",
    },
    "FileEditorAction": {
        "command": "str_replace",
        "path": "/workspace/senpai/target/train.py",
        "old_str": "lr = 1e-3",
        "new_str": "lr = 3e-4",
    },
    "TerminalAction": {
        "command": "python -m pytest -q tests/test_model.py",
        "timeout": 600,
    },
    "RunTrainingAction": {
        "spec": {
            "argv": ["python", "train.py", "--config", "configs/base.yaml"],
            "cwd": "/workspace/senpai/target",
            "timeout_seconds": 3600,
        },
    },
    "GetTrainingStatusAction": {"training_id": "b81440b1-b803-471e-9fe0-6dcabd756b83"},
    "CancelTrainingAction": {"training_id": "b81440b1-b803-471e-9fe0-6dcabd756b83"},
    "AwaitAgentsAction": {
        "task_ids": ["task-1a2b3c4d"],
        "join": "all",
        "timeout_seconds": 300,
    },
}

_NODE_CHARS = 800


def describe_validation_error(error: ValidationError, action_type: type[Action]) -> str:
    """Return the failing fields, the schema at each, and a valid example."""

    schema = action_type.model_json_schema()
    errors = error.errors(include_url=False)
    lines = [f"{len(errors)} validation error(s) for {action_type.__name__}."]
    for item in errors:
        loc = tuple(item["loc"])
        path = _path(loc) or "<arguments>"
        detail = (
            "missing"
            if item["type"] == "missing"
            else f"got {type(item.get('input')).__name__}"
        )
        lines.append(f"- {path}: {item['msg']} ({detail}).")
        node = _schema_at(schema, loc)
        if node is not None:
            lines.append(f"  schema at {path}: {_dumps(_compact(node, schema))}")
    required = schema.get("required", ())
    # "kind" is the SDK's discriminator, never an argument the model sends.
    fields = [name for name in schema.get("properties", {}) if name != "kind"]
    lines.append(
        f"Required: {', '.join(required) or 'none'}; all fields: {', '.join(fields)}."
    )
    example = EXAMPLES.get(action_type.__name__)
    if example is not None:
        lines.append(f"Valid example: {_dumps(example)}")
    return "\n".join(lines)


class SchemaHelp:
    """Mixin for a ToolDefinition whose failing calls explain themselves."""

    __slots__ = ()

    def action_from_arguments(self, arguments: dict[str, Any]) -> Action:
        try:
            return super().action_from_arguments(arguments)  # type: ignore[misc]
        except ValidationError as error:
            action_type = self.action_type  # type: ignore[attr-defined]
            # A response-schema error is not about the action's own fields.
            if error.title != action_type.__name__:
                raise
            raise ValueError(describe_validation_error(error, action_type)) from error


# The SDK keys concrete ToolDefinition subclasses by class name and refuses a
# duplicate, so each generated class gets a name of its own, which is also its
# ``kind``. ToolDefinition derives a subclass's tool name from its class name
# unless the subclass sets one, so each keeps the original tool's name.
_HELPED: dict[type[ToolDefinition], type[ToolDefinition]] = {}
_EXPLAINING: dict[type[ToolDefinition], type[ToolDefinition]] = {}


def with_schema_help(tool: ToolDefinition) -> ToolDefinition:
    """Return the tool as an instance of its class with the SchemaHelp mixin.

    The copy keeps the executor, schema and name; its ``kind`` becomes
    ``<Class>SchemaHelp``. The mixin adds no fields or slots, so the layout
    is unchanged.
    """

    if isinstance(tool, SchemaHelp):
        return tool
    cls = type(tool)
    helped = _HELPED.get(cls)
    if helped is None:
        helped = _HELPED[cls] = type(
            f"{cls.__name__}SchemaHelp", (SchemaHelp, cls), {"__slots__": (), "name": cls.name}
        )
    copied = tool.model_copy()
    object.__setattr__(copied, "__class__", helped)
    return copied


def explaining_validation_errors(tool_class: type[ToolDefinition]) -> type[ToolDefinition]:
    """Return a registrable subclass whose created tools explain bad calls."""

    if issubclass(tool_class, SchemaHelp):
        return tool_class
    explaining = _EXPLAINING.get(tool_class)
    if explaining is None:

        def create(cls: type, *args: Any, **kwargs: Any) -> list[ToolDefinition]:
            return [with_schema_help(tool) for tool in tool_class.create(*args, **kwargs)]

        explaining = _EXPLAINING[tool_class] = type(
            f"{tool_class.__name__}Explaining",
            (SchemaHelp, tool_class),
            {"__slots__": (), "name": tool_class.name, "create": classmethod(create)},
        )
    return explaining


def _path(loc: tuple[Any, ...]) -> str:
    path = ""
    for part in loc:
        path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
    return path


def _deref(node: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in node:
        node = schema.get("$defs", {}).get(node["$ref"].rsplit("/", 1)[-1], {})
    if "anyOf" in node:
        options = [_deref(option, schema) for option in node["anyOf"]]
        typed = [option for option in options if option.get("type") != "null"]
        if len(typed) == 1:
            node = {**typed[0], **{k: v for k, v in node.items() if k != "anyOf"}}
    return node


def _schema_at(schema: dict[str, Any], loc: tuple[Any, ...]) -> dict[str, Any] | None:
    node: dict[str, Any] | None = schema
    for part in loc:
        if node is None:
            return None
        node = _deref(node, schema)
        if isinstance(part, int):
            node = node.get("items")
        else:
            node = (node.get("properties") or {}).get(part)
    return None if node is None else _deref(node, schema)


def _compact(node: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Keep a node readable: an object lists its fields by type, not in full."""

    if "items" in node:
        return {**node, "items": _compact(_deref(node["items"], schema), schema)}
    if "properties" not in node:
        return node
    compact = {key: value for key, value in node.items() if key not in {"properties", "$defs"}}
    # "kind" is the SDK's discriminator, never an argument the model sends.
    compact["properties"] = {
        name: _type_name(_deref(prop, schema), schema)
        for name, prop in node["properties"].items()
        if name != "kind"
    }
    return compact


def _type_name(node: dict[str, Any], schema: dict[str, Any]) -> str:
    if "enum" in node:
        return "enum[" + "|".join(str(value) for value in node["enum"][:6]) + "]"
    if "const" in node:
        return f"const={node['const']!r}"
    if "anyOf" in node:
        return "|".join(_type_name(_deref(option, schema), schema) for option in node["anyOf"])
    kind = node.get("type", "object")
    if kind == "array" and "items" in node:
        return f"array[{_type_name(_deref(node['items'], schema), schema)}]"
    return str(kind)


def _dumps(value: Any) -> str:
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return text if len(text) <= _NODE_CHARS else text[: _NODE_CHARS - 1] + "…"
