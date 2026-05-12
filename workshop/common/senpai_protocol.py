"""Teaching implementations of core SENPAI workflow protocols."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Assignment:
    number: int
    title: str
    student: str
    advisor_branch: str
    hypothesis: str
    metric_name: str
    baseline_value: float
    lower_is_better: bool = True

    @property
    def labels(self) -> list[str]:
        return [self.advisor_branch, f"student:{self.student}", "status:wip"]


def senpai_result_line(result: dict[str, Any]) -> str:
    return "SENPAI-RESULT: " + json.dumps(result, separators=(",", ":"), sort_keys=True)


def parse_senpai_result(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if "SENPAI-RESULT:" in line:
            return json.loads(line.split("SENPAI-RESULT:", 1)[1].strip())
    raise ValueError("No SENPAI-RESULT marker found.")


def terminal_result_errors(result: dict[str, Any]) -> list[str]:
    errors = []
    if not result.get("terminal"):
        errors.append("Latest SENPAI-RESULT is not terminal=true.")
    if result.get("pending_arms") or result.get("pending_runs"):
        errors.append("Latest SENPAI-RESULT still reports pending arms/runs.")
    if not result.get("wandb_run_ids"):
        errors.append("SENPAI-RESULT has no W&B run IDs.")
    primary = result.get("primary_metric") or {}
    if not primary.get("name") or not isinstance(primary.get("value"), int | float):
        errors.append("SENPAI-RESULT primary_metric is missing or non-numeric.")
    return errors


def merge_decision(metric_value: float, baseline: float, *, lower_is_better: bool = True) -> str:
    improved = metric_value < baseline if lower_is_better else metric_value > baseline
    return "merge" if improved else "do_not_merge"
