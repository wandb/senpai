"""ChatGPT-subscription authentication for `chatgpt/<model>` profiles."""

import base64
import dataclasses
import json
import stat
import time
from pathlib import Path

import openhands.sdk.llm.auth.openai as sdk_openai_auth
import pytest
from pydantic import SecretStr

import senpai_agent.weave_monitoring as monitoring
from launch_test_support import launch_helpers
from openhands_support import runtime_config, runtime_env
from senpai_agent.openhands_runner import (
    apply_reasoning_profile,
    chatgpt_credential_dir,
    chatgpt_subscription_llm,
    model_runtime_configuration,
    parse_runner_args,
    resolve_config,
    scrub_model_credentials,
)

ACCOUNT_ID = "acct-0123"
CODEX_BACKEND = "https://chatgpt.com/backend-api/codex"


def jwt(claims: dict) -> str:
    def segment(payload: dict) -> str:
        raw = json.dumps(payload).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{segment({'alg': 'RS256'})}.{segment(claims)}.signature"


def access_token(
    *,
    expires_in: float = 10 * 86400,
    account_id: str | None = ACCOUNT_ID,
    plan: str = "pro",
) -> str:
    now = int(time.time())
    auth_claims: dict[str, str] = {"chatgpt_plan_type": plan}
    if account_id is not None:
        auth_claims["chatgpt_account_id"] = account_id
    return jwt(
        {
            "iat": now,
            "exp": int(now + expires_in),
            "https://api.openai.com/auth": auth_claims,
        }
    )


def codex_login(codex_home: Path, **token_kwargs) -> dict:
    codex_home.mkdir(parents=True, exist_ok=True)
    auth = {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": jwt({"sub": "user"}),
            "access_token": access_token(**token_kwargs),
            "refresh_token": "refresh-secret",
            "account_id": ACCOUNT_ID,
        },
        "last_refresh": "2026-09-01T00:00:00Z",
    }
    (codex_home / "auth.json").write_text(json.dumps(auth))
    return auth


def credentials_json(*, refresh_token: str = "refresh-secret", **token_kwargs) -> str:
    token = access_token(**token_kwargs)
    return json.dumps(
        {
            "type": "oauth",
            "vendor": "openai",
            "access_token": token,
            "refresh_token": refresh_token,
            "expires_at": launch_helpers.jwt_claims(token)["exp"] * 1000,
        }
    )


@pytest.fixture
def offline_account_id(monkeypatch):
    """The SDK verifies real tokens against OpenAI's JWKS; tests stay offline."""

    def unverified_account_id(token: str) -> str | None:
        auth_claims = launch_helpers.jwt_claims(token).get("https://api.openai.com/auth")
        return (auth_claims or {}).get("chatgpt_account_id")

    def no_network(*_args, **_kwargs):
        raise AssertionError("tests must not fetch OpenAI's JWKS")

    monkeypatch.setattr(sdk_openai_auth, "_extract_chatgpt_account_id", unverified_account_id)
    monkeypatch.setattr(sdk_openai_auth._JWKSCache, "get_key_set", no_network)


def chatgpt_options(**overrides) -> dict:
    options = {
        "timeout": 30,
        "num_retries": 1,
        "reasoning_effort": "high",
        "usage_id": "senpai",
        **model_runtime_configuration(
            "chatgpt/gpt-5.5", "high", compaction_trigger_tokens=200_000
        ),
    }
    options.update(overrides)
    return options


# --- launcher -----------------------------------------------------------------


def test_launcher_converts_the_codex_login_into_openhands_credentials(tmp_path):
    codex_home = tmp_path / "codex"
    auth = codex_login(codex_home)

    resolved = json.loads(launch_helpers.resolve_chatgpt_oauth_credentials(codex_home))

    token = auth["tokens"]["access_token"]
    assert resolved == {
        "type": "oauth",
        "vendor": "openai",
        "access_token": token,
        "refresh_token": "refresh-secret",
        "expires_at": launch_helpers.jwt_claims(token)["exp"] * 1000,
    }


def test_launcher_requires_a_codex_login(tmp_path):
    with pytest.raises(SystemExit, match="codex login"):
        launch_helpers.resolve_chatgpt_oauth_credentials(tmp_path / "codex")


def test_launcher_rejects_an_api_key_only_codex_login(tmp_path):
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "sk-key", "tokens": None})
    )

    with pytest.raises(SystemExit, match="sign in with ChatGPT"):
        launch_helpers.resolve_chatgpt_oauth_credentials(codex_home)


def test_codex_home_follows_the_codex_cli_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "elsewhere"))
    assert launch_helpers.codex_home() == tmp_path / "elsewhere"

    monkeypatch.delenv("CODEX_HOME")
    assert launch_helpers.codex_home() == Path.home() / ".codex"


def test_chatgpt_preflight_reports_the_plan_without_network(monkeypatch, capsys):
    def no_network(*_args, **_kwargs):
        raise AssertionError("preflight must not contact OpenAI")

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", no_network)

    launch_helpers.preflight_check_chatgpt_oauth_credentials(credentials_json(plan="pro"))

    output = capsys.readouterr().out
    assert "plan=pro" in output
    assert "refresh-secret" not in output


@pytest.mark.parametrize(
    ("token_kwargs", "message"),
    [
        ({"expires_in": 60}, "expired"),
        ({"account_id": None}, "chatgpt_account_id"),
    ],
)
def test_chatgpt_preflight_rejects_unusable_tokens(token_kwargs, message):
    with pytest.raises(SystemExit, match=message):
        launch_helpers.preflight_check_chatgpt_oauth_credentials(
            credentials_json(**token_kwargs)
        )


def test_launch_secret_carries_the_chatgpt_credentials():
    secret = launch_helpers.render_launch_secret(
        "track",
        "github",
        "exa",
        "wandb",
        chatgpt_oauth_credentials='{"type": "oauth"}',
        custom_secrets={},
    )

    encoded = base64.b64encode(b'{"type": "oauth"}').decode()
    assert f"  chatgpt-oauth-credentials: {encoded}\n" in secret
    assert "openai-api-key" not in secret


# --- runner -------------------------------------------------------------------


def test_chatgpt_profiles_resolve_the_subscription_credential(tmp_path):
    env = runtime_env(tmp_path)
    env.pop("OPENAI_API_KEY")
    env.update(
        {
            "CHATGPT_OAUTH_CREDENTIALS": credentials_json(),
            "SENPAI_OPENHANDS_MODEL": "chatgpt/gpt-5.5",
            "SENPAI_OPENHANDS_SMART_MODEL": "chatgpt/gpt-5.5",
            "SENPAI_OPENHANDS_FAST_MODEL": "chatgpt/gpt-5.4-mini",
            "SENPAI_OPENHANDS_FRONTIER_MODEL": "chatgpt/gpt-5.6",
            "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT": "xhigh",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert {
        config.api_key_env,
        config.smart_api_key_env,
        config.fast_api_key_env,
        config.frontier_api_key_env,
    } == {"CHATGPT_OAUTH_CREDENTIALS"}
    assert config.api_key.get_secret_value() == env["CHATGPT_OAUTH_CREDENTIALS"]
    assert "CHATGPT_OAUTH_CREDENTIALS" not in config.conversation_secrets

    scrub_model_credentials(env, config)
    assert "CHATGPT_OAUTH_CREDENTIALS" not in env


def test_chatgpt_profiles_reject_max_effort(tmp_path):
    env = runtime_env(tmp_path)
    env.update(
        {
            "CHATGPT_OAUTH_CREDENTIALS": credentials_json(),
            "SENPAI_OPENHANDS_MODEL": "chatgpt/gpt-5.6",
            "SENPAI_OPENHANDS_REASONING_EFFORT": "max",
        }
    )

    with pytest.raises(ValueError, match="unsupported"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


def test_chatgpt_runtime_configuration_is_a_stateless_responses_chain():
    assert model_runtime_configuration(
        "chatgpt/gpt-5.5", "high", compaction_trigger_tokens=200_000
    ) == {
        "api_mode": "responses",
        "reasoning_summary": "auto",
        "reasoning_context": "all_turns",
        "responses_store": False,
        "responses_use_previous_response_id": False,
    }


def test_chatgpt_llm_uses_the_codex_backend_with_subscription_auth(
    tmp_path, offline_account_id
):
    credentials = credentials_json()

    llm = chatgpt_subscription_llm(
        "chatgpt/gpt-5.5",
        SecretStr(credentials),
        credential_dir=tmp_path / "chatgpt-auth",
        **chatgpt_options(),
    )

    assert llm.model == "openai/gpt-5.5"
    assert llm.is_subscription
    assert llm.api_key is None
    assert llm.base_url == CODEX_BACKEND
    assert llm.extra_headers["chatgpt-account-id"] == ACCOUNT_ID
    assert llm.extra_headers["originator"] == "senpai"
    assert llm.litellm_extra_body == {"store": False}
    assert llm.responses_store is False
    assert llm.responses_use_previous_response_id is False
    assert llm.reasoning_effort == "high"

    store_file = tmp_path / "chatgpt-auth" / "openai_oauth.json"
    assert json.loads(store_file.read_text()) == json.loads(credentials)
    assert stat.S_IMODE(store_file.stat().st_mode) == 0o600

    token = json.loads(credentials)["access_token"]
    assert llm._get_litellm_auth_values() == (
        token,
        {"chatgpt-account-id": ACCOUNT_ID},
    )


def test_chatgpt_llm_survives_the_reasoning_profile_copy(tmp_path, offline_account_id):
    llm = chatgpt_subscription_llm(
        "chatgpt/gpt-5.5",
        SecretStr(credentials_json()),
        credential_dir=tmp_path / "chatgpt-auth",
        **chatgpt_options(),
    )

    profiled = apply_reasoning_profile(llm)

    assert profiled.is_subscription
    assert profiled.reasoning_effort == "high"
    assert profiled.litellm_extra_body == {"store": False}
    assert profiled._get_litellm_auth_values() == llm._get_litellm_auth_values()


def test_chatgpt_llm_keeps_whichever_credential_expires_later(
    tmp_path, offline_account_id
):
    credential_dir = tmp_path / "chatgpt-auth"
    launched = credentials_json(expires_in=5 * 86400)
    refreshed = credentials_json(expires_in=9 * 86400, refresh_token="rotated")
    stale = credentials_json(expires_in=86400, refresh_token="stale")
    credential_dir.mkdir()
    (credential_dir / "openai_oauth.json").write_text(refreshed)

    llm = chatgpt_subscription_llm(
        "chatgpt/gpt-5.5",
        SecretStr(launched),
        credential_dir=credential_dir,
        **chatgpt_options(),
    )
    assert llm._get_litellm_auth_values()[0] == json.loads(refreshed)["access_token"]
    assert json.loads((credential_dir / "openai_oauth.json").read_text()) == json.loads(
        refreshed
    )

    (credential_dir / "openai_oauth.json").write_text(stale)
    llm = chatgpt_subscription_llm(
        "chatgpt/gpt-5.5",
        SecretStr(launched),
        credential_dir=credential_dir,
        **chatgpt_options(),
    )
    assert llm._get_litellm_auth_values()[0] == json.loads(launched)["access_token"]
    assert json.loads((credential_dir / "openai_oauth.json").read_text()) == json.loads(
        launched
    )


def test_chatgpt_llm_rejects_models_outside_the_codex_catalogue(
    tmp_path, offline_account_id
):
    with pytest.raises(ValueError, match="not supported for subscription"):
        chatgpt_subscription_llm(
            "chatgpt/gpt-4o",
            SecretStr(credentials_json()),
            credential_dir=tmp_path / "chatgpt-auth",
            **chatgpt_options(),
        )


def test_chatgpt_credential_store_is_shared_with_delegated_children(tmp_path):
    config = runtime_config(tmp_path)

    assert chatgpt_credential_dir(config) == config.state_dir / "chatgpt-auth"
    child = dataclasses.replace(
        config,
        state_dir=tmp_path / "child-state",
        delegation_root_state_dir=config.state_dir,
    )
    assert chatgpt_credential_dir(child) == config.state_dir / "chatgpt-auth"


def test_trace_redaction_hides_each_chatgpt_token():
    credentials = credentials_json(refresh_token="refresh-secret")
    token = json.loads(credentials)["access_token"]

    redact = monitoring.secret_redactor({"CHATGPT_OAUTH_CREDENTIALS": credentials})

    assert redact(f"{credentials} {token} refresh-secret") == " ".join(
        ["<secret-hidden>"] * 3
    )
