import ctypes
import errno
import os
import tempfile

import pytest

import senpai_agent.secrets as secrets_module

from senpai_agent.secrets import (
    CUSTOM_SECRET_ENV_NAMES_ENV,
    MAX_MODEL_CREDENTIAL_BUNDLE_BYTES,
    MODEL_CREDENTIALS_FD_ENV,
    consume_model_credential_fd,
    configured_custom_secret_env_names,
    scrub_github_credentials,
    set_process_nondumpable,
    validate_custom_secret_env_names,
)


@pytest.mark.parametrize("result", [0, -1])
def test_linux_credential_holders_disable_process_dumping(monkeypatch, result):
    calls = []

    class LibC:
        @staticmethod
        def prctl(*arguments):
            calls.append(arguments)
            ctypes.set_errno(errno.EPERM)
            return result

    monkeypatch.setattr(secrets_module.sys, "platform", "linux")
    monkeypatch.setattr(secrets_module.ctypes, "CDLL", lambda *_args, **_kwargs: LibC())

    if result == 0:
        set_process_nondumpable()
    else:
        with pytest.raises(OSError) as raised:
            set_process_nondumpable()
        assert raised.value.errno == errno.EPERM

    assert calls == [(4, 0, 0, 0, 0)]


def test_model_credential_bundle_is_consumed_once_and_closes_its_fd():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'{"OPENAI_API_KEY":"private-key"}')
    os.close(write_fd)
    environment = {MODEL_CREDENTIALS_FD_ENV: str(read_fd)}

    assert consume_model_credential_fd(environment) == {"OPENAI_API_KEY": "private-key"}
    assert environment == {}
    assert consume_model_credential_fd(environment) == {}
    with pytest.raises(OSError):
        os.fstat(read_fd)


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b"{}",
        b"[]",
        b'{"NOT-VALID":"secret"}',
        b'{"OPENAI_API_KEY":" "}',
        b'{"OPENAI_API_KEY":1}',
        b"x" * (MAX_MODEL_CREDENTIAL_BUNDLE_BYTES + 1),
    ],
)
def test_model_credential_bundle_rejects_invalid_payloads_and_closes_its_fd(payload):
    with tempfile.TemporaryFile() as stream:
        stream.write(payload)
        stream.seek(0)
        descriptor = os.dup(stream.fileno())
    environment = {MODEL_CREDENTIALS_FD_ENV: str(descriptor)}

    with pytest.raises(RuntimeError, match="credential bundle is"):
        consume_model_credential_fd(environment)

    assert environment == {}
    with pytest.raises(OSError):
        os.fstat(descriptor)


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
        (["RESEARCH_TAG"], "reserved"),
        (["PATH"], "reserved"),
        (["PYTHONSAFEPATH"], "reserved"),
        (["BASHOPTS"], "reserved"),
        (["BASH_XTRACEFD"], "reserved"),
        (["SHELLOPTS"], "reserved"),
        (["PROMPT_COMMAND"], "reserved"),
        (["PS0"], "reserved"),
        (["PS1"], "reserved"),
        (["PS2"], "reserved"),
        (["PS3"], "reserved"),
        (["PS4"], "reserved"),
        (["MAILCHECK"], "reserved"),
        (["MAILPATH"], "reserved"),
        (["ZDOTDIR"], "reserved"),
    ],
)
def test_custom_secret_name_validator_rejects_unsafe_names(
    names: list[str],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        validate_custom_secret_env_names(names)


@pytest.mark.parametrize(
    "name",
    [
        "NODES_PER_STUDENT",
        "GPUS_PER_STUDENT_NODE",
        "CPU_PER_STUDENT_GPU",
        "MEMORY_GI_PER_STUDENT_GPU",
        "PVC_CLAIM_NAME",
    ],
)
def test_custom_secret_name_validator_rejects_launcher_owned_topology_names(name):
    with pytest.raises(ValueError, match="reserved"):
        validate_custom_secret_env_names([name])


def test_custom_secret_name_validator_accepts_valid_names():
    validate_custom_secret_env_names(
        ["PRIVATE_AUTH", "MODEL_REGISTRY_TOKEN", "TIMEOUT_MINUTES", "MAX_EPOCHS"]
    )


def test_configured_custom_secret_name_errors_identify_the_runtime_marker():
    with pytest.raises(RuntimeError, match=CUSTOM_SECRET_ENV_NAMES_ENV):
        configured_custom_secret_env_names(
            {CUSTOM_SECRET_ENV_NAMES_ENV: "PRIVATE_AUTH,,OTHER_AUTH"}
        )
