"""Private, deterministic storage for oversized GitHub retrieval results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from senpai_agent.github.pull_requests import PRManifestEntry

_MAX_AGE_SECONDS = 24 * 60 * 60


def store_pull_requests(
    *,
    repo: str,
    numbers: tuple[int, ...],
    date_range: tuple[str, str] | None,
    search: str | None,
    manifest: tuple[PRManifestEntry, ...],
    markdown: str,
    artifact_dir: str | Path | None,
    target_workspace: str | Path | None,
) -> Path:
    output_dir = _external_artifact_dir(artifact_dir, target_workspace)
    path = output_dir / _artifact_name(
        repo=repo,
        numbers=numbers,
        date_range=date_range,
        search=search,
        manifest=manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_expired_artifacts(output_dir)
    if not path.exists() or path.read_text(encoding="utf-8") != markdown:
        path.write_text(markdown, encoding="utf-8")
    path.chmod(0o600)
    return path


def _external_artifact_dir(
    artifact_dir: str | Path | None,
    target_workspace: str | Path | None,
) -> Path:
    target = Path(
        target_workspace
        or os.environ.get("SENPAI_OPENHANDS_WORKSPACE")
        or Path.cwd()
    ).expanduser().resolve()
    if artifact_dir is None:
        state_dir = os.environ.get("SENPAI_OPENHANDS_STATE_DIR")
        artifact_dir = (
            Path(state_dir) / "github"
            if state_dir
            else Path(tempfile.gettempdir()) / "senpai-github"
        )
    output = Path(artifact_dir).expanduser().resolve()
    if output == target or output.is_relative_to(target):
        raise ValueError("GitHub artifacts must be outside the target workspace")
    return output


def _remove_expired_artifacts(output_dir: Path) -> None:
    cutoff = time.time() - _MAX_AGE_SECONDS
    for path in output_dir.glob("pull-requests-*.md"):
        if path.stat().st_mtime < cutoff:
            path.unlink()


def _artifact_name(
    *,
    repo: str,
    numbers: tuple[int, ...],
    date_range: tuple[str, str] | None,
    search: str | None,
    manifest: tuple[PRManifestEntry, ...],
) -> str:
    identity = {
        "repo": repo,
        "numbers": numbers,
        "date_range": date_range,
        "search": search,
        "heads": [(entry.number, entry.head_sha) for entry in manifest],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"pull-requests-{digest}.md"
