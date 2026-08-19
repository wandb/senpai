import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.tool import Tool, resolve_tool
from pydantic import SecretStr

from github_workflow_support import FakeGitHub, pull_request, workflow
from senpai_agent.github import tools as github_tools_module
from senpai_agent.github import PRManifestEntry, PRRetrievalResult
from senpai_agent.github.tools import (
    GetPRsAction,
    GetPRsTool,
    GitHubMutationObservation,
    GitHubToolRuntime,
    GitHubWorkflowToolSet,
    RespondToHumanIssueAction,
    RespondToHumanIssueTool,
    clear_github_credentials,
    configure_github_credentials,
)
from senpai_agent.github.workflow import MutationResult
from senpai_agent.tools import register_senpai_tools


ADVISOR_GITHUB_TOOLS = {
    "get_prs",
    "respond_to_human_issue",
    "create_assignment",
    "publish_advisor_branch",
    "repair_assignment_routing",
    "send_assignment_feedback",
    "request_assignment_revision",
    "accept_result_on_current_base",
    "merge_experiment",
    "close_experiment",
}
STUDENT_GITHUB_TOOLS = {
    "get_prs",
    "post_assignment_comment",
    "respond_to_human_issue",
    "submit_experiment_result",
}


def test_github_tools_package_preserves_public_contract_types():
    expected = {
        "GetPRsObservation",
        "PRManifestObservation",
        "GitHubMutationObservation",
        "GitHubCredentials",
        "PostAssignmentCommentAction",
        "PostAssignmentCommentTool",
    }

    assert expected <= set(github_tools_module.__all__)
    assert expected <= set(dir(github_tools_module))


@pytest.mark.parametrize(
    ("version", "expected_suffix"),
    [
        ("a" * 40, f"\n- Version: `{'a' * 40}`"),
        (None, ""),
    ],
)
def test_github_mutation_result_renders_as_markdown(version, expected_suffix):
    observation = GitHubMutationObservation(
        changed=False,
        resource_url="https://github.test/pull/17",
        state="post_assignment_comment",
        version=version,
    )

    assert observation.to_llm_content[0].text == (
        "## GitHub Update\n\n"
        "- State: `post_assignment_comment`\n"
        "- Changed: No\n"
        "- Resource: <https://github.test/pull/17>"
        f"{expected_suffix}"
    )


def test_github_mutation_resource_cannot_inject_markdown():
    observation = GitHubMutationObservation(
        changed=True,
        resource_url="https://good.test/)[Injected](https://evil.test)",
        state="updated",
    )

    assert observation.to_llm_content[0].text == (
        "## GitHub Update\n\n"
        "- State: `updated`\n"
        "- Changed: Yes\n"
        "- Resource: "
        "<https://good.test/%29%5BInjected%5D%28https://evil.test%29>"
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [("advisor", ADVISOR_GITHUB_TOOLS), ("student", STUDENT_GITHUB_TOOLS)],
)
def test_registered_github_toolset_exposes_only_role_owned_tools(
    tmp_path: Path,
    role: str,
    expected: set[str],
):
    configure_github_credentials(
        "acme/widgets",
        SecretStr("github-secret"),
        trusted_actor="senpai-bot",
    )
    register_senpai_tools()
    spec = Tool(
        name="senpai_github",
        params={
            "role": role,
            "state_dir": str(tmp_path / "state"),
            "advisor_branch": "advisor-branch" if role == "advisor" else None,
            "student_names": ["student-one"] if role == "advisor" else None,
            "student_name": "student-one" if role == "student" else None,
        },
    )
    workspace = tmp_path / "target"
    workspace.mkdir()
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=workspace))

    try:
        resolved = resolve_tool(spec, state)
    finally:
        clear_github_credentials()

    assert {tool.name for tool in resolved} == expected
    assert "senpai_github" not in expected
    assert "github_transition" not in expected
    assert "github-secret" not in json.dumps(spec.model_dump())
    assert all("github-secret" not in repr(tool.executor) for tool in resolved)


def test_toolset_rejects_a_runtime_role_mismatch(tmp_path: Path):
    with pytest.raises(ValueError, match="workflow role"):
        GitHubWorkflowToolSet.create(
            workflow=workflow(FakeGitHub(pull_request())),
            role="student",
            student_name="student-one",
            state_dir=tmp_path / "state",
            workspace=tmp_path,
        )


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_both_roles_can_respond_to_a_verified_human_message(
    role: str,
    tmp_path: Path,
):
    calls = []

    class Workflow:
        repo = "acme/widgets"

        def respond_to_issue(self, number, **kwargs):
            calls.append((number, kwargs))
            return MutationResult(
                changed=True,
                resource_url=f"https://github.test/issues/{number}",
                state="issue_response_upserted",
                version=str(kwargs["human_message_id"]),
            )

    runtime = GitHubToolRuntime(
        workflow=Workflow(),
        workspace=tmp_path,
        git_token=None,
        role=role,
        advisor_branch="advisor-branch" if role == "advisor" else None,
        student_names=frozenset({"student-one"}) if role == "advisor" else frozenset(),
        student_name="student-one" if role == "student" else None,
    )
    tool = RespondToHumanIssueTool.create(runtime)[0]
    observation = tool(
        RespondToHumanIssueAction(
            issue_number=23,
            human_message_id=987,
            response="bounded response",
        )
    )

    assert observation.state == "issue_response_upserted"
    assert calls == [
        (
            23,
            {
                "human_message_id": 987,
                "response": "bounded response",
                "audience_labels": (
                    {"team", "advisor-branch"}
                    if role == "advisor"
                    else {"team", "student:student-one"}
                ),
                "responder": "advisor" if role == "advisor" else "student-one",
            },
        )
    ]


def test_get_prs_is_scoped_to_the_configured_repo_and_credential(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    calls = []
    configure_github_credentials(
        "acme/widgets",
        SecretStr("github-secret"),
        trusted_actor="senpai-bot",
    )

    try:
        tool = GetPRsTool.create(
            state_dir=tmp_path / "state" / "github",
            workspace=workspace,
        )[0]

        def retrieve(repo: str, **kwargs) -> PRRetrievalResult:
            calls.append((repo, kwargs))
            return PRRetrievalResult(
                manifest=(
                    PRManifestEntry(
                        number=17,
                        title="Try spectral loss",
                        head_sha="abc123",
                        url="https://github.test/acme/widgets/pull/17",
                    ),
                ),
                markdown="# PR #17\n\nComplete context.\n",
                path=None,
            )

        tool.executor.get_prs = retrieve
        with pytest.raises(PermissionError, match="configured GitHub credentials"):
            tool(GetPRsAction(repo="other/widgets"))

        observation = tool(
            GetPRsAction(
                repo="acme/widgets",
                numbers=(17,),
                search="label:status:review",
            )
        )

        assert observation.manifest[0].head_sha == "abc123"
        assert observation.to_llm_content[0].text.startswith("# PR #17")
        assert calls[0][0] == "acme/widgets"
        assert calls[0][1]["numbers"] == (17,)
        assert calls[0][1]["search"] == "label:status:review"
        assert calls[0][1]["target_workspace"] == workspace.resolve()
        assert calls[0][1]["token"].get_secret_value() == "github-secret"
    finally:
        clear_github_credentials()


def test_get_prs_artifacts_must_live_outside_the_target_checkout(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside the target workspace"):
        GetPRsTool.create(
            get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
            state_dir=workspace / "state",
            workspace=workspace,
        )


def test_registered_github_tools_ignore_ambient_write_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    clear_github_credentials()
    monkeypatch.setenv("GH_REPO", "acme/widgets")
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-write-token")
    register_senpai_tools()
    workspace = tmp_path / "target"
    workspace.mkdir()
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=workspace))

    with pytest.raises(RuntimeError, match="configure GitHub credentials"):
        resolve_tool(
            Tool(
                name="senpai_github",
                params={
                    "role": "student",
                    "state_dir": str(tmp_path / "state"),
                    "student_name": "student-one",
                },
            ),
            state,
        )
