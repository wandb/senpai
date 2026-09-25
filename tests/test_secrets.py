import os

import pytest

import senpai_agent.secrets as secrets_module
from senpai_agent.secrets import (
    CUSTOM_SECRET_ENV_NAMES_ENV,
    MODEL_CREDENTIALS_FD_ENV,
    SHELL_STARTUP_ENV_NAMES,
    consume_model_credential_fd,
    configured_custom_secret_env_names,
    scrub_github_credentials,
    set_process_nondumpable,
    validate_custom_secret_env_names,
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


def test_linux_credential_holders_disable_process_dumping(monkeypatch):
    calls = []

    class LibC:
        @staticmethod
        def prctl(*arguments):
            calls.append(arguments)
            return 0

    monkeypatch.setattr(secrets_module.sys, "platform", "linux")
    monkeypatch.setattr(secrets_module.ctypes, "CDLL", lambda *_args, **_kwargs: LibC())

    set_process_nondumpable()

    assert calls == [(4, 0, 0, 0, 0)]


def test_model_credential_bundle_is_consumed_once_and_closes_its_fd():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'{"OPENAI_API_KEY":"private-key"}')
    os.close(write_fd)
    environment = {MODEL_CREDENTIALS_FD_ENV: str(read_fd)}

    assert consume_model_credential_fd(environment) == {
        "OPENAI_API_KEY": "private-key"
    }
    assert environment == {}
    with pytest.raises(OSError):
        os.fstat(read_fd)


@pytest.mark.parametrize(
    "payload",
    [b"not-json", b"{}", b'{"NOT-VALID":"secret"}', b'{"OPENAI_API_KEY":""}'],
)
def test_model_credential_bundle_rejects_malformed_payloads(payload: bytes):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload)
    os.close(write_fd)

    with pytest.raises(RuntimeError, match="credential bundle is invalid"):
        consume_model_credential_fd({MODEL_CREDENTIALS_FD_ENV: str(read_fd)})

    with pytest.raises(OSError):
        os.fstat(read_fd)


def test_configured_custom_secret_names_are_parsed_in_order():
    assert configured_custom_secret_env_names(
        {CUSTOM_SECRET_ENV_NAMES_ENV: " PRIVATE_AUTH,MODEL_REGISTRY_TOKEN "}
    ) == ("PRIVATE_AUTH", "MODEL_REGISTRY_TOKEN")


@pytest.mark.parametrize("environment", [{}, {CUSTOM_SECRET_ENV_NAMES_ENV: "  "}])
def test_configured_custom_secret_names_default_to_empty(environment):
    assert configured_custom_secret_env_names(environment) == ()


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ([""], "empty names"),
        (["PRIVATE-AUTH"], "invalid"),
        (["PRIVATE_AUTH", "PRIVATE_AUTH"], "duplicate"),
        (["OPENAI_API_KEY"], "reserved"),
        (["GH_PRIVATE_KEY"], "reserved"),
        (["GITHUB_APP_TOKEN"], "reserved"),
        (["SENPAI_INTERNAL_KEY"], "reserved"),
        (["WANDB_API_KEY_FERN"], "reserved"),
        (["RESEARCH_TAG"], "reserved"),
        (["PYTHONSAFEPATH"], "reserved"),
        (["PATH"], "reserved"),
    ],
)
def test_custom_secret_name_validator_rejects_unsafe_names(
    names: list[str],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        validate_custom_secret_env_names(names)


def test_custom_secret_name_validator_accepts_valid_names():
    validate_custom_secret_env_names(
        ["PRIVATE_AUTH", "MODEL_REGISTRY_TOKEN", "TIMEOUT_MINUTES", "MAX_EPOCHS"]
    )


@pytest.mark.parametrize("name", sorted(SHELL_STARTUP_ENV_NAMES))
def test_custom_secret_name_validator_rejects_shell_startup_variables(name):
    with pytest.raises(ValueError, match="reserved"):
        validate_custom_secret_env_names([name])


def test_configured_custom_secret_name_errors_identify_the_runtime_marker():
    with pytest.raises(RuntimeError, match=CUSTOM_SECRET_ENV_NAMES_ENV):
        configured_custom_secret_env_names(
            {CUSTOM_SECRET_ENV_NAMES_ENV: "PRIVATE_AUTH,,OTHER_AUTH"}
        )
