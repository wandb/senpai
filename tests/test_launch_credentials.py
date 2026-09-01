import base64
import io
import json
import subprocess
import urllib.error
import urllib.parse

import pytest

from launch_test_support import launch_helpers


class JSONResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


def capture_request(monkeypatch, payload):
    captured = {}

    def urlopen(request, timeout):
        captured.update(request=request, timeout=timeout)
        return JSONResponse(payload)

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)
    return captured


def test_custom_secrets_resolve_only_the_explicit_allowlist(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        'SHELL_SECRET="dotenv-value"\n'
        'DOTENV_SECRET="${MUST_STAY_LITERAL}"\n'
        "UNLISTED_SECRET=dotenv-unlisted\n"
    )
    monkeypatch.setenv("SHELL_SECRET", "shell-value")
    monkeypatch.setenv("UNLISTED_SECRET", "shell-unlisted")

    resolved = launch_helpers.resolve_custom_secrets(
        dotenv, ["SHELL_SECRET", "DOTENV_SECRET"]
    )

    assert resolved == {
        "SHELL_SECRET": "shell-value",
        "DOTENV_SECRET": "${MUST_STAY_LITERAL}",
    }


def test_custom_secrets_fall_back_from_blank_shell_values(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("HF_TOKEN=dotenv-token\n")
    monkeypatch.setenv("HF_TOKEN", " \t")

    assert launch_helpers.resolve_custom_secrets(dotenv, ["HF_TOKEN"]) == {
        "HF_TOKEN": "dotenv-token"
    }


def test_custom_secrets_report_every_missing_or_blank_name(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("BLANK_SECRET=\n")
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    monkeypatch.delenv("BLANK_SECRET", raising=False)

    with pytest.raises(SystemExit) as raised:
        launch_helpers.resolve_custom_secrets(
            dotenv, ["MISSING_SECRET", "BLANK_SECRET"]
        )

    message = str(raised.value)
    assert "MISSING_SECRET" in message
    assert "BLANK_SECRET" in message


def test_student_wandb_keys_are_derived_and_must_be_distinct(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "WANDB_API_KEY_FERN=fern-key\n"
        "WANDB_API_KEY_TEAM_FRIEREN=frieren-key\n"
    )
    for name in ("WANDB_API_KEY_FERN", "WANDB_API_KEY_TEAM_FRIEREN"):
        monkeypatch.delenv(name, raising=False)

    assert launch_helpers.resolve_student_wandb_api_keys(
        dotenv, ["fern", "team-frieren"]
    ) == {"fern": "fern-key", "team-frieren": "frieren-key"}

    dotenv.write_text(
        "WANDB_API_KEY_FERN=shared-key\n"
        "WANDB_API_KEY_FRIEREN=shared-key\n"
    )
    with pytest.raises(SystemExit, match="distinct W&B training API key"):
        launch_helpers.resolve_student_wandb_api_keys(
            dotenv, ["fern", "frieren"]
        )


def test_student_wandb_key_names_must_not_normalize_to_the_same_env_name(tmp_path):
    with pytest.raises(SystemExit) as raised:
        launch_helpers.resolve_student_wandb_api_keys(
            tmp_path / ".env", ["team-fern", "team_fern"]
        )

    message = str(raised.value)
    assert "team-fern" in message
    assert "team_fern" in message
    assert "WANDB_API_KEY_TEAM_FERN" in message


def test_github_cli_fallback_does_not_inherit_custom_secrets(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "custom-secret")
    monkeypatch.setenv("VISIBLE_SETTING", "safe")

    def run(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, "github-token\n", "")

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    token = launch_helpers.resolve_github_token(tmp_path / "missing.env", ["HF_TOKEN"])

    assert token == "github-token"
    assert "HF_TOKEN" not in captured["kwargs"]["env"]
    assert captured["kwargs"]["env"]["VISIBLE_SETTING"] == "safe"


def test_openai_preflight_authenticates_against_the_models_endpoint(monkeypatch):
    captured = capture_request(monkeypatch, {"data": []})

    launch_helpers.preflight_check_openai_api_key("openai-secret")

    request = captured["request"]
    assert request.full_url == "https://api.openai.com/v1/models"
    assert request.headers["Authorization"] == "Bearer openai-secret"


def test_exa_preflight_runs_one_instant_publication_search(monkeypatch):
    captured = capture_request(
        monkeypatch,
        {"results": [{"id": "publication"}]},
    )

    launch_helpers.preflight_check_exa_api_key("exa-secret")

    request = captured["request"]
    assert request.full_url == "https://api.exa.ai/search"
    assert request.headers["X-api-key"] == "exa-secret"
    assert json.loads(request.data) == {
        "query": "api credential preflight",
        "type": "instant",
        "category": "publication",
        "numResults": 1,
    }


def test_exa_preflight_rejects_a_success_response_without_search_results(monkeypatch):
    capture_request(monkeypatch, {"status": "ok"})

    with pytest.raises(SystemExit, match="invalid search response"):
        launch_helpers.preflight_check_exa_api_key("exa-secret")


def test_wandb_preflight_authenticates_with_the_minimal_viewer_query(monkeypatch):
    captured = capture_request(
        monkeypatch,
        {"data": {"viewer": {"id": "user"}}},
    )

    assert launch_helpers.preflight_check_wandb_api_key("wandb-secret") == "user"

    request = captured["request"]
    assert request.full_url == "https://api.wandb.ai/graphql"
    assert request.headers["Authorization"] == (
        "Basic " + base64.b64encode(b"api:wandb-secret").decode()
    )
    assert json.loads(request.data) == {
        "query": "query SenpaiPreflight { viewer { id } }"
    }


def test_wandb_preflight_redacts_credentials_from_graphql_errors(monkeypatch):
    basic_auth = base64.b64encode(b"api:wandb-secret").decode()
    capture_request(
        monkeypatch,
        {
            "errors": [
                {"message": f"wandb-secret ({basic_auth}) was rejected"}
            ]
        },
    )

    with pytest.raises(SystemExit) as raised:
        launch_helpers.preflight_check_wandb_api_key("wandb-secret")

    message = str(raised.value)
    assert "wandb-secret" not in message
    assert basic_auth not in message
    assert "<redacted>" in message


@pytest.mark.parametrize(
    "viewer",
    [{}, {"id": ""}, {"id": 123}, None],
)
def test_wandb_preflight_requires_a_viewer_identity(monkeypatch, viewer):
    capture_request(monkeypatch, {"data": {"viewer": viewer}})

    with pytest.raises(SystemExit, match="failed to resolve a viewer"):
        launch_helpers.preflight_check_wandb_api_key("wandb-secret")


def test_wandb_viewers_must_be_distinct_across_controller_and_students():
    launch_helpers.require_distinct_wandb_viewers(
        "controller-viewer",
        {"fern": "fern-viewer", "frieren": "frieren-viewer"},
        inference_viewers=("inference-viewer",),
        active_controller_viewers=("old-controller-viewer",),
        active_inference_viewers=("old-inference-viewer",),
        active_student_viewers={"fern": ("old-fern-viewer",)},
    )

    with pytest.raises(SystemExit, match="controller and W&B Inference"):
        launch_helpers.require_distinct_wandb_viewers(
            "shared-viewer",
            {},
            inference_viewers=("shared-viewer",),
        )

    with pytest.raises(SystemExit, match="W&B Inference.*student 'fern'"):
        launch_helpers.require_distinct_wandb_viewers(
            "controller-viewer",
            {"fern": "shared-viewer"},
            inference_viewers=("shared-viewer",),
        )

    with pytest.raises(SystemExit, match="controller.*student 'fern'.*same viewer"):
        launch_helpers.require_distinct_wandb_viewers(
            "shared-viewer",
            {"fern": "shared-viewer"},
        )

    with pytest.raises(SystemExit, match="student 'fern'.*student 'frieren'.*same viewer"):
        launch_helpers.require_distinct_wandb_viewers(
            "controller-viewer",
            {"fern": "shared-viewer", "frieren": "shared-viewer"},
        )

    with pytest.raises(SystemExit, match="W&B Inference.*student 'fern'"):
        launch_helpers.require_distinct_wandb_viewers(
            "controller-viewer",
            {"fern": "old-inference-viewer"},
            active_inference_viewers=("old-inference-viewer",),
        )

    with pytest.raises(SystemExit, match="controller.*student 'fern'"):
        launch_helpers.require_distinct_wandb_viewers(
            "new-controller-viewer",
            {"fern": "old-controller-viewer"},
            active_controller_viewers=("old-controller-viewer",),
        )

    launch_helpers.require_distinct_wandb_viewers(
        "controller-viewer",
        {"fern": "fern-viewer"},
        active_controller_viewers=("controller-viewer", "old-controller-viewer"),
        active_inference_viewers=("inference-viewer", "old-inference-viewer"),
        active_student_viewers={"fern": ("fern-viewer", "old-fern-viewer")},
    )

    with pytest.raises(SystemExit, match="older-track.*current-track.*same viewer"):
        launch_helpers.require_distinct_wandb_viewers(
            "controller-viewer",
            {"fern": "shared-viewer"},
            active_viewer_owners={
                "tag 'older-track' student 'frieren'": ("shared-viewer",)
            },
            owner_scope="tag 'current-track'",
        )


def test_wandb_inference_preflight_routes_to_the_requested_project(monkeypatch):
    captured = capture_request(monkeypatch, {"data": []})

    launch_helpers.preflight_check_wandb_inference(
        "wandb-secret",
        "research-team",
        "mlxfast",
    )

    request = captured["request"]
    assert request.full_url == "https://api.inference.wandb.ai/v1/models"
    assert request.headers["Authorization"] == "Bearer wandb-secret"
    assert request.headers["Openai-project"] == "research-team/mlxfast"


def test_repo_access_uses_an_impossible_ref_write_probe(monkeypatch):
    captured = {}

    def urlopen(request, timeout):
        captured["request"] = request
        raise urllib.error.HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            {},
            io.BytesIO(b'{"message":"Object does not exist"}'),
        )

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    launch_helpers.preflight_check_target_repo_access(
        "https://github.com/example/problem.git",
        "github-secret",
    )

    request = captured["request"]
    assert request.full_url == "https://api.github.com/repos/example/problem/git/refs"
    assert request.headers["Authorization"] == "Bearer github-secret"
    assert json.loads(request.data) == {
        "ref": "refs/heads/senpai-write-preflight",
        "sha": "0" * 40,
    }


def test_existing_advisor_branch_head_becomes_the_program_source_commit(
    monkeypatch,
):
    source_commit = "c" * 40
    monkeypatch.setattr(
        launch_helpers,
        "preflight_check_target_repo_branch",
        lambda *_args: "main",
    )
    monkeypatch.setattr(
        launch_helpers,
        "_github_api",
        lambda *_args, **_kwargs: {"commit": {"sha": source_commit}},
    )

    assert launch_helpers.ensure_advisor_branch(
        "https://github.com/example/problem.git",
        "github-secret",
        "main",
        "research",
    ) == source_commit


def test_repo_access_rejects_and_redacts_a_non_validation_error(monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"message":"github-secret cannot access this resource"}'),
        )

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    with pytest.raises(SystemExit) as raised:
        launch_helpers.preflight_check_target_repo_access(
            "https://github.com/example/problem.git",
            "github-secret",
        )

    message = str(raised.value)
    assert "HTTP 403" in message
    assert "Contents: Read and write" in message
    assert "github-secret" not in message
    assert "<redacted>" in message


def test_repo_access_fails_closed_if_the_impossible_write_is_accepted(monkeypatch):
    capture_request(monkeypatch, {})

    with pytest.raises(SystemExit, match="unexpectedly accepted"):
        launch_helpers.preflight_check_target_repo_access(
            "https://github.com/example/problem.git",
            "github-secret",
        )


def test_student_name_preflight_uses_open_assignment_workflow_labels(monkeypatch):
    captured = {}

    def urlopen(request, timeout):
        captured.update(request=request, timeout=timeout)
        return JSONResponse(
            [
                {
                    "number": 12,
                    "pull_request": {},
                    "labels": [
                        {"name": "student:fern"},
                        {"name": "status:merged"},
                    ],
                },
                {
                    "number": 13,
                    "labels": [
                        {"name": "student:fern"},
                        {"name": "status:wip"},
                    ],
                },
            ]
        )

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    launch_helpers.preflight_check_student_name_availability(
        "https://github.com/example/problem.git",
        "github-secret",
        ["fern"],
        "advisor",
    )

    request = captured["request"]
    assert request.get_method() == "GET"
    assert request.full_url == (
        "https://api.github.com/repos/example/problem/issues?"
        "state=open&labels=student%3Afern&per_page=100&page=1"
    )


def test_student_name_preflight_reports_active_prs_and_prefix_fix(monkeypatch):
    assignments = {
        "student:fern": [
            {
                "number": 17,
                "pull_request": {},
                "labels": [
                    {"name": "student:fern"},
                    {"name": "status:wip"},
                ],
            }
        ],
        "student:frieren": [
            {
                "number": 23,
                "pull_request": {},
                "labels": [
                    {"name": "student:frieren"},
                    {"name": "status:review"},
                ],
            }
        ],
    }
    bases = {17: "old-track", 23: "other-track"}

    def urlopen(request, timeout):
        url = urllib.parse.urlsplit(request.full_url)
        if url.path.endswith("/issues"):
            query = urllib.parse.parse_qs(url.query)
            return JSONResponse(assignments[query["labels"][0]])
        return JSONResponse({"base": {"ref": bases[int(url.path.rsplit("/", 1)[1])]}})

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    with pytest.raises(SystemExit) as raised:
        launch_helpers.preflight_check_student_name_availability(
            "https://github.com/example/problem.git",
            "github-secret",
            ["fern", "frieren"],
            "advisor",
        )

    message = str(raised.value)
    assert "student:fern: #17 (base old-track)" in message
    assert "student:frieren: #23 (base other-track)" in message
    assert "--student_prefix <prefix>" in message


def test_student_name_preflight_allows_resume_on_same_advisor_branch(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request.full_url)
        if request.full_url.endswith("/pulls/31"):
            return JSONResponse({"base": {"ref": "advisor"}})
        return JSONResponse(
            [
                {
                    "number": 31,
                    "pull_request": {},
                    "labels": [
                        {"name": "student:fern"},
                        {"name": "status:review"},
                    ],
                }
            ]
        )

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    launch_helpers.preflight_check_student_name_availability(
        "https://github.com/example/problem.git",
        "github-secret",
        ["fern"],
        "advisor",
    )

    assert requests[-1].endswith("/pulls/31")


def test_student_name_preflight_checks_every_issue_page(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request.full_url)
        url = urllib.parse.urlsplit(request.full_url)
        if url.path.endswith("/pulls/117"):
            return JSONResponse({"base": {"ref": "other-track"}})
        page = urllib.parse.parse_qs(url.query)["page"][0]
        if page == "1":
            return JSONResponse(
                [
                    {
                        "number": number,
                        "labels": [{"name": "student:fern"}],
                    }
                    for number in range(1, 101)
                ]
            )
        return JSONResponse(
            [
                {
                    "number": 117,
                    "pull_request": {},
                    "labels": [
                        {"name": "student:fern"},
                        {"name": "status:wip"},
                    ],
                }
            ]
        )

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    with pytest.raises(SystemExit, match=r"#117 \(base other-track\)"):
        launch_helpers.preflight_check_student_name_availability(
            "https://github.com/example/problem.git",
            "github-secret",
            ["fern"],
            "advisor",
        )

    assert any("page=1" in request for request in requests)
    assert any("page=2" in request for request in requests)
