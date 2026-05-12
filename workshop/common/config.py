"""Workshop configuration helpers."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSHOP_ROOT.parent
ARTIFACTS_DIR = WORKSHOP_ROOT / "artifacts"
ENV_PATH = WORKSHOP_ROOT / ".env"

REQUIRED_KEYS = [
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "TARGET_REPO_URL",
    "TARGET_REPO_BRANCH",
    "ADVISOR_BRANCH",
]


@dataclass(frozen=True)
class WorkshopConfig:
    anthropic_api_key: str
    github_token: str
    wandb_api_key: str
    wandb_entity: str
    wandb_project: str
    target_repo_url: str
    target_repo_branch: str
    advisor_branch: str
    exa_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-latest"

    @property
    def wandb_path(self) -> str:
        return f"{self.wandb_entity}/{self.wandb_project}"

    def redacted(self) -> dict[str, str]:
        values = asdict(self)
        for key in ("anthropic_api_key", "github_token", "wandb_api_key", "exa_api_key"):
            values[key] = redact(values.get(key, ""))
        return values


def redact(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]}"


def read_env(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in os.environ.items():
        if key in REQUIRED_KEYS or key in {"EXA_API_KEY", "ANTHROPIC_MODEL"}:
            values[key] = value
    return values


def load_config(require: bool = True) -> WorkshopConfig:
    values = read_env()
    if require:
        missing = [key for key in REQUIRED_KEYS if not values.get(key)]
        if missing:
            raise RuntimeError(
                "Missing workshop credentials: "
                + ", ".join(missing)
                + ". Run `uv run python workshop/setup_credentials.py`."
            )
    return WorkshopConfig(
        anthropic_api_key=values.get("ANTHROPIC_API_KEY", ""),
        github_token=values.get("GITHUB_TOKEN", ""),
        wandb_api_key=values.get("WANDB_API_KEY", ""),
        wandb_entity=values.get("WANDB_ENTITY", ""),
        wandb_project=values.get("WANDB_PROJECT", ""),
        target_repo_url=values.get("TARGET_REPO_URL", ""),
        target_repo_branch=values.get("TARGET_REPO_BRANCH", "main"),
        advisor_branch=values.get("ADVISOR_BRANCH", "workshop-r1"),
        exa_api_key=values.get("EXA_API_KEY", ""),
        anthropic_model=values.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
    )


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def write_artifact(name: str, payload: object) -> Path:
    ensure_artifacts_dir()
    path = ARTIFACTS_DIR / name
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def protect_env_file(path: Path = ENV_PATH) -> None:
    if path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
