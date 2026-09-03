import pytest

from senpai_agent.secrets import (
    CUSTOM_SECRET_ENV_NAMES_ENV,
    configured_custom_secret_env_names,
    scrub_github_credentials,
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
        (["CHATGPT_OAUTH_CREDENTIALS"], "reserved"),
        (["GH_PRIVATE_KEY"], "reserved"),
        (["GITHUB_APP_TOKEN"], "reserved"),
        (["SENPAI_INTERNAL_KEY"], "reserved"),
        (["RESEARCH_TAG"], "reserved"),
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


def test_configured_custom_secret_name_errors_identify_the_runtime_marker():
    with pytest.raises(RuntimeError, match=CUSTOM_SECRET_ENV_NAMES_ENV):
        configured_custom_secret_env_names(
            {CUSTOM_SECRET_ENV_NAMES_ENV: "PRIVATE_AUTH,,OTHER_AUTH"}
        )
