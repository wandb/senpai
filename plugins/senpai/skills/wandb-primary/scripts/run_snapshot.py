# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Use exported W&B run data with the existing offline analysis helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSnapshot:
    id: str
    name: str
    state: str
    created_at: str
    config: dict[str, Any]
    summary_metrics: dict[str, Any]
    history_path: Path

    @classmethod
    def from_exports(
        cls, run_path: str | Path, history_path: str | Path
    ) -> RunSnapshot:
        """Load one `run` export and its complete, unsampled `history` export."""
        record = json.loads(Path(run_path).read_text())
        return cls(
            id=record["id"],
            name=record["name"],
            state=record["state"],
            created_at=record["created_at"],
            config=record["config"],
            summary_metrics=record["summary"],
            history_path=Path(history_path),
        )

    def scan_history(
        self,
        keys: list[str] | None = None,
        min_step: int = 0,
        max_step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream local rows without sampling or dropping sparse observations."""
        selected = None if keys is None else {*keys, "_step", "_timestamp", "_runtime"}
        with self.history_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if min_step or max_step is not None:
                    step = row["_step"]
                    if step < min_step or (max_step is not None and step >= max_step):
                        continue
                yield (
                    row
                    if selected is None
                    else {key: value for key, value in row.items() if key in selected}
                )
