import base64
import hashlib
import io
import json
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from senpai_agent.github import http
from senpai_agent.github.http import GitHubReadError
from senpai_agent.github.tools import (
    GitHubWorkflowToolSet,
    clear_github_credentials,
    configure_github_credentials,
)


BASE = "a" * 40
HEAD = "b" * 40
MERGE_BASE = "c" * 40
TEXT = b"def evaluate():\n    return 'complete source'\n"
BLOB = hashlib.sha1(b"blob " + str(len(TEXT)).encode() + b"\0" + TEXT).hexdigest()
BASE_TEXT = b"def evaluate():\n    return 'original source'\n"
BASE_BLOB = hashlib.sha1(b"blob " + str(len(BASE_TEXT)).encode() + b"\0" + BASE_TEXT).hexdigest()


class Response(io.BytesIO):
    def __init__(self, value):
        super().__init__(json.dumps(value).encode())
        self.headers = {}


@pytest.fixture
def source_tool(tmp_path, monkeypatch):
    configure_github_credentials("acme/private", SecretStr("private-credential"))
    calls = []
    details = {"number": 7, "base": {"sha": BASE}, "head": {"sha": HEAD},
               "changed_files": 1, "state": "closed", "merged": True}
    files = [{"filename": "src/evaluate.py", "sha": BLOB, "status": "modified",
              "patch": "@@ -1 +1 @@\n-old\n+new"}]
    responses = {
        "/repos/acme/private/pulls/7": details,
        f"/repos/acme/private/compare/{BASE}...{HEAD}?per_page=1": {
            "merge_base_commit": {"sha": MERGE_BASE}, "files": files,
        },
        f"/repos/acme/private/git/trees/{MERGE_BASE}": {
            "truncated": False, "tree": [{"path": "src", "type": "tree", "sha": "d" * 40}],
        },
        f"/repos/acme/private/git/trees/{'d' * 40}": {
            "truncated": False, "tree": [{"path": "evaluate.py", "type": "blob", "sha": BASE_BLOB}],
        },
        f"/repos/acme/private/git/blobs/{BASE_BLOB}": {
            "sha": BASE_BLOB, "size": len(BASE_TEXT), "encoding": "base64",
            "content": base64.b64encode(BASE_TEXT).decode(),
        },
        f"/repos/acme/private/git/blobs/{BLOB}": {
            "sha": BLOB, "size": len(TEXT), "encoding": "base64",
            "content": base64.b64encode(TEXT).decode(),
        },
    }

    def urlopen(request, timeout):
        assert request.headers["Authorization"] == "Bearer private-credential"
        assert request.full_url.startswith("https://api.github.com/")
        path = request.full_url.removeprefix("https://api.github.com")
        calls.append(path)
        value = responses[path]
        return Response(value() if callable(value) else value)

    monkeypatch.setattr(http.request, "urlopen", urlopen)
    try:
        tools = GitHubWorkflowToolSet.create(
            role="advisor", advisor_branch="advisor", student_names=("student",),
            workspace=tmp_path / "target", state_dir=tmp_path / "state",
        )
        matching = [tool for tool in tools if tool.name == "get_pr_source"]
        assert matching, "private PR source is unavailable through authenticated tools"
        yield matching[0], responses, calls, tmp_path
    finally:
        clear_github_credentials()


def read(tool, **overrides):
    return tool(tool.action_type.model_validate({
        "number": 7, "expected_base_sha": BASE, "expected_head_sha": HEAD,
        **overrides,
    }))


def test_private_merged_pr_source_is_pinned_and_available_to_reviewers(source_tool):
    tool, _, calls, tmp_path = source_tool
    manifest = read(tool)
    assert [file.path for file in manifest.files] == ["src/evaluate.py"]
    assert not any("/blobs/" in call for call in calls)
    result = read(tool, paths=("src/evaluate.py",))
    artifact = Path(result.path)
    text = artifact.read_text()
    assert TEXT.decode() in text
    assert BASE_TEXT.decode() in text
    assert "-    return 'original source'\n+    return 'complete source'" in text
    assert BASE in text and HEAD in text and BLOB in text
    assert result.base_sha == BASE and result.head_sha == HEAD
    assert result.comparison_base_sha == MERGE_BASE
    assert artifact.is_relative_to(tmp_path / "state")
    assert artifact.stat().st_mode & 0o777 == 0o600
    assert "private-credential" not in text + result.model_dump_json()
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("field", ["head", "base"])
def test_source_refuses_moved_pr_before_returning_an_artifact(source_tool, field):
    tool, responses, calls, tmp_path = source_tool
    details = responses["/repos/acme/private/pulls/7"]
    def moved():
        value = json.loads(json.dumps(details))
        if calls.count("/repos/acme/private/pulls/7") > 1:
            value[field]["sha"] = "c" * 40
        return value
    responses["/repos/acme/private/pulls/7"] = moved
    with pytest.raises(GitHubReadError, match="changed"):
        read(tool, paths=("src/evaluate.py",))
    assert not list((tmp_path / "state").glob("*.md"))


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "https://elsewhere/x", "src/../secret", "src\\secret"])
def test_source_rejects_unsafe_paths_before_http(source_tool, path):
    tool, _, calls, _ = source_tool
    with pytest.raises((ValueError, ValidationError)):
        read(tool, paths=(path,))
    assert not calls


def test_source_cannot_address_another_repo_or_unchanged_path(source_tool):
    tool, _, calls, _ = source_tool
    with pytest.raises(ValidationError):
        read(tool, repo="other/private")
    assert not calls
    with pytest.raises(GitHubReadError, match="changed path"):
        read(tool, paths=("secrets.txt",))
    assert not any("/blobs/" in call for call in calls)


@pytest.mark.parametrize("fault", ["file_limit", "incomplete", "blob_hash", "base_blob", "response_limit"])
def test_source_refuses_incomplete_or_unbounded_evidence(source_tool, fault):
    tool, responses, _, tmp_path = source_tool
    if fault == "file_limit":
        responses["/repos/acme/private/pulls/7"]["changed_files"] = 101
    elif fault == "incomplete":
        responses[f"/repos/acme/private/compare/{BASE}...{HEAD}?per_page=1"]["files"] = []
    elif fault == "blob_hash":
        responses[f"/repos/acme/private/git/blobs/{BLOB}"]["content"] = base64.b64encode(b"wrong source").decode()
    elif fault == "base_blob":
        responses[f"/repos/acme/private/git/blobs/{BASE_BLOB}"]["sha"] = BLOB
    else:
        responses[f"/repos/acme/private/git/blobs/{BLOB}"]["content"] = "x" * (2 * 1024 * 1024)
    with pytest.raises(GitHubReadError):
        read(tool, paths=("src/evaluate.py",))
    assert not list((tmp_path / "state").glob("*.md"))


@pytest.mark.parametrize("status", ["added", "removed", "renamed"])
def test_source_preserves_added_removed_and_renamed_file_sides(source_tool, status):
    tool, responses, calls, _ = source_tool
    file = responses[f"/repos/acme/private/compare/{BASE}...{HEAD}?per_page=1"]["files"][0]
    file["status"] = status
    if status == "renamed":
        file["previous_filename"] = "src/old.py"
        responses[f"/repos/acme/private/git/trees/{'d' * 40}"]["tree"][0]["path"] = "old.py"
    result = read(tool, paths=("src/evaluate.py",))
    text = Path(result.path).read_text()
    if status == "added":
        assert "--- /dev/null\n+++ b/src/evaluate.py" in text
        assert not any("/trees/" in call for call in calls)
    elif status == "removed":
        assert "--- a/src/evaluate.py\n+++ /dev/null" in text
        assert f"/repos/acme/private/git/blobs/{BLOB}" not in calls
    else:
        assert "--- a/src/old.py\n+++ b/src/evaluate.py" in text
        assert result.files[0].previous_path == "src/old.py"
