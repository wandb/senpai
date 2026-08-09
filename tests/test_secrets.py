import os

import pytest

from senpai_agent.secrets import (
    SUPERVISOR_SECRET_DIR_ENV,
    consume_supervisor_secret_directory,
    scrub_github_credentials,
)


def test_scrub_github_credentials_removes_every_handoff():
    environment = {
        "GITHUB_TOKEN": "token",
        "GH_TOKEN": "token",
        "SENPAI_GITHUB_TOKEN_FILE": "/secret",
        "SENPAI_GITHUB_TOKEN_FD": "47",
        "WANDB_API_KEY": "keep",
    }

    scrub_github_credentials(environment)

    assert environment == {"WANDB_API_KEY": "keep"}


def test_supervisor_secret_handoff_is_consumed_without_mutating_input(tmp_path):
    secret_dir = tmp_path / "handoff"
    secret_dir.mkdir(mode=0o700)
    values = {
        "GITHUB_TOKEN": "github-sentinel",
        "WANDB_API_KEY": "wandb-sentinel",
        "OPENAI_API_KEY": "openai-sentinel",
    }
    for name, value in values.items():
        path = secret_dir / name
        path.write_text(value)
        path.chmod(0o600)
    environment = {
        "VISIBLE": "yes",
        SUPERVISOR_SECRET_DIR_ENV: str(secret_dir),
    }

    hydrated = consume_supervisor_secret_directory(environment)

    assert hydrated == {"VISIBLE": "yes", **values}
    assert environment[SUPERVISOR_SECRET_DIR_ENV] == str(secret_dir)
    assert not secret_dir.exists()
    assert all(value not in os.environ.values() for value in values.values())


def test_supervisor_secret_handoff_fails_closed_when_required():
    with pytest.raises(RuntimeError, match="SENPAI_SUPERVISOR_SECRET_DIR is required"):
        consume_supervisor_secret_directory({}, required=True)


@pytest.mark.parametrize("kind", ("public", "empty", "symlink", "unknown"))
def test_supervisor_secret_handoff_rejects_unsafe_entries(tmp_path, kind):
    secret_dir = tmp_path / "handoff"
    secret_dir.mkdir(mode=0o700)
    name = "UNEXPECTED" if kind == "unknown" else "WANDB_API_KEY"
    path = secret_dir / name
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_text("secret")
        path.symlink_to(target)
    else:
        path.write_text("" if kind == "empty" else "secret")
        path.chmod(0o644 if kind == "public" else 0o600)

    with pytest.raises(RuntimeError):
        consume_supervisor_secret_directory(
            {SUPERVISOR_SECRET_DIR_ENV: str(secret_dir)}
        )
