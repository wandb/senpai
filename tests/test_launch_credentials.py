import base64
import io
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from launch_test_support import launch_helpers


def test_hivemind_token_prefers_shell_over_dotenv(tmp_path: Path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("HIVEMIND_TOKEN=sa_dotenv\n", encoding="utf-8")
    monkeypatch.setenv("HIVEMIND_TOKEN", "sa_shell")

    assert launch_helpers.resolve_hivemind_token(dotenv) == "sa_shell"


def test_hivemind_token_uses_dotenv_without_a_login_fallback(
    tmp_path: Path,
    monkeypatch,
):
    dotenv = tmp_path / ".env"
    dotenv.write_text("HIVEMIND_TOKEN=sa_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("HIVEMIND_TOKEN", raising=False)

    assert launch_helpers.resolve_hivemind_token(dotenv) == "sa_dotenv"


def test_hivemind_requires_static_token_form(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("HIVEMIND_TOKEN", "personal-token")

    with pytest.raises(SystemExit) as raised:
        launch_helpers.resolve_hivemind_token(tmp_path / ".env")

    message = str(raised.value)
    assert "sa_" in message
    assert "Personal Access Token" in message
    assert "personal-token" not in message


def test_hivemind_enabled_launch_fails_clearly_without_a_token(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.delenv("HIVEMIND_TOKEN", raising=False)

    with pytest.raises(SystemExit) as raised:
        launch_helpers.resolve_hivemind_token(tmp_path / ".env")

    message = str(raised.value)
    assert "no Hivemind token" in message
    assert "HIVEMIND_TOKEN" in message


class JSONResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


class EmptyResponse:
    status = 204

    def __init__(self, url=launch_helpers.HIVEMIND_HEARTBEAT_URL):
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self):
        return self.url

    def read(self):
        return b""


def test_hivemind_preflight_authenticates_write_access(monkeypatch, capsys):
    captured = {}

    def urlopen(request, timeout):
        captured.update(request=request, timeout=timeout)
        return EmptyResponse()

    monkeypatch.setattr(launch_helpers, "_open_without_redirects", urlopen)

    launch_helpers.preflight_check_hivemind_token("sa_hivemind-secret")

    request = captured["request"]
    assert request.full_url == launch_helpers.HIVEMIND_HEARTBEAT_URL
    assert request.method == "POST"
    assert request.data == b""
    assert request.headers["Authorization"] == "Bearer sa_hivemind-secret"
    assert captured["timeout"] == 10
    output = capsys.readouterr().out
    assert "write access" in output
    assert "sa_hivemind-secret" not in output


def test_hivemind_preflight_rejects_and_redacts_auth_errors(monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            io.BytesIO(
                b'{"error":{"code":"insufficient_scope",'
                b'"message":"sa_hivemind-secret lacks write"}}'
            ),
        )

    monkeypatch.setattr(launch_helpers, "_open_without_redirects", urlopen)

    with pytest.raises(SystemExit) as raised:
        launch_helpers.preflight_check_hivemind_token("sa_hivemind-secret")

    message = str(raised.value)
    assert "HTTP 403" in message
    assert "insufficient_scope" in message
    assert "sa_hivemind-secret" not in message
    assert "<redacted>" in message


def test_hivemind_preflight_rejects_unexpected_status(monkeypatch):
    response = EmptyResponse()
    response.status = 200
    monkeypatch.setattr(
        launch_helpers,
        "_open_without_redirects",
        lambda _request, timeout: response,
    )

    with pytest.raises(SystemExit, match="unexpected response"):
        launch_helpers.preflight_check_hivemind_token("sa_hivemind-secret")


def test_hivemind_preflight_never_forwards_authorization_on_redirect():
    request = urllib.request.Request(
        launch_helpers.HIVEMIND_HEARTBEAT_URL,
        headers={"Authorization": "Bearer sa_hivemind-secret"},
    )

    redirected = launch_helpers._NoRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {"Location": "https://attacker.example/collect"},
        "https://attacker.example/collect",
    )

    assert redirected is None


def test_hivemind_preflight_opener_installs_the_no_redirect_handler(monkeypatch):
    captured = {}

    class Opener:
        def open(self, request, *, timeout):
            captured.update(request=request, timeout=timeout)
            return EmptyResponse()

    def build_opener(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(launch_helpers.urllib.request, "build_opener", build_opener)
    request = urllib.request.Request(launch_helpers.HIVEMIND_HEARTBEAT_URL)

    launch_helpers._open_without_redirects(request, timeout=10)

    assert len(captured["handlers"]) == 1
    assert isinstance(captured["handlers"][0], launch_helpers._NoRedirectHandler)
    assert captured["request"] is request
    assert captured["timeout"] == 10


def capture_request(monkeypatch, payload):
    captured = {}

    def urlopen(request, timeout):
        captured.update(request=request, timeout=timeout)
        return JSONResponse(payload)

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)
    return captured


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

    launch_helpers.preflight_check_wandb_api_key("wandb-secret")

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
