import io
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from senpai_agent.github import operator
from senpai_agent.github.tools import (
    AdoptAssignmentAction,
    CreateAssignmentAction,
    GitHubMutationObservation,
)
from senpai_agent.github.workflow import MutationResult


ENVIRONMENT = {
    "GH_REPO": "acme/widgets",
    "GITHUB_TOKEN": "operator-secret",
    "SENPAI_GITHUB_ACTOR": "senpai-bot",
}


def command(workspace: Path, operation: str, source: str) -> list[str]:
    return [
        "--workspace",
        str(workspace),
        "--advisor-branch",
        "advisor-branch",
        "--student-names",
        "student-one,student-two",
        operation,
        source,
    ]


def create_action() -> CreateAssignmentAction:
    return CreateAssignmentAction(
        assignment_id="assignment-18",
        revision_id="revision-1",
        student="student-one",
        expected_base_sha="b" * 40,
        head_branch="student-one/lower-lr",
        title="Try a lower learning rate",
        body="Run one bounded comparison.",
    )


def adopt_action() -> AdoptAssignmentAction:
    return AdoptAssignmentAction(
        pr_number=17,
        assignment_id="assignment-17",
        revision_id="revision-1",
        student="student-two",
        expected_base_sha="b" * 40,
        head_branch="student-two/existing-experiment",
        expected_pr_head_sha="c" * 40,
    )


def install_recorders(monkeypatch: pytest.MonkeyPatch):
    calls = []

    class Workflow:
        def __init__(self, repo, token, **kwargs):
            self.repo = repo
            calls.append(("workflow", repo, token, kwargs))

    def executor(name):
        class RecordingExecutor:
            def __init__(self, runtime):
                self.runtime = runtime

            def __call__(self, action):
                calls.append((name, self.runtime, action))
                return GitHubMutationObservation(
                    changed=True,
                    resource_url="https://github.test/acme/widgets/pull/17",
                    state=name,
                    version="c" * 40,
                )

        return RecordingExecutor

    monkeypatch.setattr(operator, "GitHubWorkflow", Workflow)
    monkeypatch.setattr(
        operator, "CreateAssignmentExecutor", executor("assignment_created")
    )
    monkeypatch.setattr(
        operator, "AdoptAssignmentExecutor", executor("assignment_adopted")
    )
    return calls


def test_create_assignment_reads_json_file_and_uses_the_typed_executor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls = install_recorders(monkeypatch)
    source = tmp_path / "action.json"
    source.write_text(create_action().model_dump_json(), encoding="utf-8")
    output = io.StringIO()

    assert operator.operator_main(
        command(tmp_path, "create-assignment", str(source)),
        environment=ENVIRONMENT,
        stdout=output,
    ) == 0

    _, repo, token, workflow_options = calls[0]
    name, runtime, action = calls[1]
    assert repo == "acme/widgets"
    assert token.get_secret_value() == "operator-secret"
    assert workflow_options == {"role": "advisor", "trusted_actor": "senpai-bot"}
    assert name == "assignment_created"
    assert runtime.workspace == tmp_path.resolve()
    assert runtime.advisor_branch == "advisor-branch"
    assert runtime.student_names == frozenset({"student-one", "student-two"})
    assert action == create_action()
    assert json.loads(output.getvalue()) == {
        "changed": True,
        "resource_url": "https://github.test/acme/widgets/pull/17",
        "state": "assignment_created",
        "version": "c" * 40,
    }
    assert "operator-secret" not in output.getvalue()


def test_adopt_assignment_reads_stdin_and_uses_the_typed_executor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls = install_recorders(monkeypatch)
    output = io.StringIO()

    assert operator.operator_main(
        command(tmp_path, "adopt-assignment", "-"),
        environment={"GH_REPO": "acme/widgets", "GH_TOKEN": "fallback-secret"},
        stdin=io.StringIO(adopt_action().model_dump_json()),
        stdout=output,
    ) == 0

    _, _, token, workflow_options = calls[0]
    name, runtime, action = calls[1]
    assert token.get_secret_value() == "fallback-secret"
    assert workflow_options == {"role": "advisor", "trusted_actor": None}
    assert name == "assignment_adopted"
    assert runtime.git_token is token
    assert action == adopt_action()
    assert "fallback-secret" not in output.getvalue()


def test_operator_adoption_runs_the_real_guarded_executor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls = []

    class Workflow:
        repo = "acme/widgets"

        def __init__(self, *_args, **_kwargs):
            pass

        @contextmanager
        def serialized_assignment_mutation(self):
            calls.append("lock_enter")
            yield
            calls.append("lock_exit")

        def adopt_assignment(self, number, *, assignment):
            calls.append(("adopt", number, assignment))
            return MutationResult(
                changed=True,
                resource_url=f"https://github.test/pull/{number}",
                state="assignment_adopted",
                version=assignment.head_sha,
            )

    monkeypatch.setattr(operator, "GitHubWorkflow", Workflow)
    monkeypatch.setattr(
        "senpai_agent.github.tools.advisor.git_assignment.require_remote_assignment_history",
        lambda *_args, **kwargs: calls.append(("history", kwargs)),
    )

    assert operator.operator_main(
        command(tmp_path, "adopt-assignment", "-"),
        environment=ENVIRONMENT,
        stdin=io.StringIO(adopt_action().model_dump_json()),
        stdout=io.StringIO(),
    ) == 0

    assert calls[0] == "lock_enter"
    assert calls[1][0] == "history"
    assert calls[2][0] == "adopt"
    assert calls[2][2].assignment_id == "assignment-17"
    assert calls[3][0] == "history"
    assert calls[4] == "lock_exit"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"GITHUB_TOKEN": "secret"}, "GH_REPO must use owner/name form"),
        ({"GH_REPO": "acme/widgets"}, "GITHUB_TOKEN or GH_TOKEN is required"),
    ],
)
def test_operator_requires_environment_only_github_authority(
    environment: dict[str, str],
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(SystemExit) as raised:
        operator.operator_main(
            command(tmp_path, "create-assignment", "-"),
            environment=environment,
            stdin=io.StringIO(create_action().model_dump_json()),
        )

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_operator_rejects_a_student_outside_its_allowlist_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    class Workflow:
        repo = "acme/widgets"

        def __init__(self, *_args, **_kwargs):
            pass

        def serialized_assignment_mutation(self):
            pytest.fail("mutation lock reached")

    monkeypatch.setattr(operator, "GitHubWorkflow", Workflow)
    action = create_action().model_copy(update={"student": "other-student"})

    with pytest.raises(SystemExit) as raised:
        operator.operator_main(
            command(tmp_path, "create-assignment", "-"),
            environment=ENVIRONMENT,
            stdin=io.StringIO(action.model_dump_json()),
        )

    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "outside this launch" in error
    assert "operator-secret" not in error


def test_operator_has_no_credential_argument():
    help_text = operator.build_parser().format_help()

    assert "--token" not in help_text
    with pytest.raises(SystemExit):
        operator.build_parser().parse_args(["--token", "secret"])
