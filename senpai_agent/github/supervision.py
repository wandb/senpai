"""Route bounded Supervisor requests through the existing GitHub control plane."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Annotated, Self
from urllib.parse import urlencode
from uuid import UUID

from openhands.sdk.tool import Action, ToolDefinition, ToolExecutor
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from senpai_agent.git_transport import run_git
from senpai_agent.github.pull_requests import get_prs
from senpai_agent.github.tools.contracts import (
    AssignmentVersion,
    GitHubMutationObservation,
)
from senpai_agent.github.tools.runtime import GitHubToolRuntime, tool_annotations
from senpai_agent.github.workflow import (
    GitHubWorkflow,
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import authoritative_marker_line
from senpai_agent.PROMPTS import SUPERVISOR_REPAIR_PROMPT, render_prompt

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation

    from senpai_agent.github.mailbox.core import GitHubMailbox
    from senpai_agent.openhands_runner import RunnerConfig

_PREFIX = "<!-- senpai-supervisor:"
_RESULT_PREFIX = "<!-- senpai-supervisor-result:"


class SupervisorRequest(Action):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request_id: str = Field(
        min_length=1,
        max_length=128,
        description="Stable ID for one repair attempt. Reuse it only to retry the same request.",
    )
    target: str = Field(
        min_length=1,
        description="'advisor' or the exact configured student name whose pod needs help.",
    )
    assignment: AssignmentVersion | None = Field(
        default=None,
        description="Current assignment and PR head. Required for a student repair or a request from a student.",
    )
    task: str = Field(
        min_length=1,
        max_length=12_000,
        description="The specific problem and desired repair, with relevant evidence from either role. Do not include secrets or full conversation histories.",
    )
    context_prs: list[Annotated[int, Field(gt=0)]] = Field(
        default_factory=list,
        max_length=4,
        description="Additional PRs whose full body and discussion help explain the problem, including related work by other students.",
    )

    @model_validator(mode="after")
    def require_student_assignment(self) -> Self:
        if self.target != "advisor" and self.assignment is None:
            raise ValueError("a student repair requires its exact assignment")
        return self


class SupervisorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    advisor_branch: str
    requester: str
    request: SupervisorRequest
    parent_conversation_id: UUID | None = None


def _identity(envelope: SupervisorEnvelope) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                envelope.repo,
                envelope.advisor_branch,
                envelope.requester,
                envelope.request.request_id,
            ]
        ).encode()
    ).hexdigest()


def render_request(envelope: SupervisorEnvelope) -> str:
    marker = f"{_PREFIX}{_identity(envelope)}:{envelope.model_dump_json()} -->"
    return f"{marker}\n\n## Supervisor request: {envelope.request.target}\n\n{envelope.request.task}"


def parse_request(body: str) -> SupervisorEnvelope | None:
    marker = authoritative_marker_line(body)
    if not marker.startswith(_PREFIX) or not marker.endswith(" -->"):
        return None
    try:
        identity, encoded = marker[len(_PREFIX) : -4].split(":", 1)
        envelope = SupervisorEnvelope.model_validate_json(encoded)
    except (ValueError, ValidationError):
        return None
    return envelope if identity == _identity(envelope) else None


def _assignment(
    workflow: GitHubWorkflow, request: SupervisorRequest, advisor_branch: str
):
    if request.assignment is None:
        return None
    version = request.assignment
    pull, assignment = workflow._routed_assignment_at_head(
        version.pr_number,
        assignment_id=version.assignment_id,
        revision_id=version.revision_id,
        expected_head_sha=version.expected_pr_head_sha,
        allowed_statuses=frozenset({"status:wip", "status:review"}),
    )
    if pull.base_ref != advisor_branch:
        raise WorkflowPreconditionError(
            "Supervisor request belongs to another advisor branch"
        )
    if "status:hold" in pull.labels:
        raise WorkflowPreconditionError(
            "Supervisor cannot override an assignment on hold"
        )
    if request.target != "advisor" and assignment.student != request.target:
        raise WorkflowPreconditionError(
            "Supervisor target does not own this assignment"
        )
    return assignment


class RequestSupervisorExecutor(
    ToolExecutor[SupervisorRequest, GitHubMutationObservation]
):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self, action: SupervisorRequest, conversation: LocalConversation | None = None
    ) -> GitHubMutationObservation:
        runtime = self.runtime
        if not runtime.advisor_branch:
            raise RuntimeError("Supervisor requests require an advisor branch")
        requester = (
            "advisor" if runtime.role == "advisor" else runtime.current_student()
        )
        if runtime.role == "student":
            if action.target not in {"advisor", requester} or action.assignment is None:
                raise PermissionError(
                    "students may request help for themselves or their advisor, with their current assignment"
                )
        elif action.target != "advisor":
            runtime.require_configured_student(action.target)
        assignment = _assignment(runtime.workflow, action, runtime.advisor_branch)
        if runtime.role == "student" and assignment.student != requester:
            raise PermissionError("a student request must reference its own assignment")
        envelope = SupervisorEnvelope(
            repo=runtime.workflow.repo,
            advisor_branch=runtime.advisor_branch,
            requester=requester,
            request=action,
            parent_conversation_id=str(conversation.id)
            if conversation is not None
            else None,
        )
        workflow = runtime.workflow
        prefix = f"{_PREFIX}{_identity(envelope)}:"
        body = render_request(envelope)
        with workflow.serialized_assignment_mutation():
            existing = workflow._created_human_issue(prefix)
            if existing is not None:
                if parse_request(existing.body or "") != envelope:
                    raise WorkflowPreconditionError(
                        "request_id already belongs to a different Supervisor request"
                    )
                return GitHubMutationObservation(
                    changed=False,
                    resource_url=existing.html_url,
                    state="supervisor_requested",
                    version=action.request_id,
                )
            workflow._mutate(
                "POST",
                f"/repos/{workflow.repo}/issues",
                json_body={
                    "title": f"Supervisor: {action.target} — {action.request_id}",
                    "body": body,
                    "labels": ["supervisor", runtime.advisor_branch],
                },
                expected_statuses={201},
            )
            created = workflow._created_human_issue(prefix)
            if (
                created is None
                or created.body != body
                or not {"supervisor", runtime.advisor_branch}.issubset(
                    label.name for label in created.labels
                )
            ):
                raise ReconciliationError(
                    "GitHub did not persist the Supervisor request"
                )
            return GitHubMutationObservation(
                changed=True,
                resource_url=created.html_url,
                state="supervisor_requested",
                version=action.request_id,
            )


class RequestSupervisorTool(
    ToolDefinition[SupervisorRequest, GitHubMutationObservation]
):
    @classmethod
    def create(cls, runtime: GitHubToolRuntime):
        return [
            cls(
                description="Ask a fresh Supervisor to diagnose and repair the advisor or a student. The target pod handles the request at its next safe boundary and returns feedback; other pods can continue working.",
                action_type=SupervisorRequest,
                observation_type=GitHubMutationObservation,
                annotations=tool_annotations("Request Supervisor"),
                executor=RequestSupervisorExecutor(runtime),
            )
        ]


def supervisor_events(mailbox: GitHubMailbox) -> tuple[ControllerEvent, ...]:
    from senpai_agent.supervisor_worker import SupervisorResult

    target = "advisor" if mailbox.role == "advisor" else mailbox.student_name
    query = urlencode(
        {
            "state": "all",
            "labels": f"supervisor,{mailbox.advisor_branch}",
            "per_page": 100,
        }
    )
    issues = mailbox._github.objects(f"/repos/{mailbox.repo}/issues?{query}")
    if not issues:
        return ()
    actor = mailbox._github.actor().casefold()
    events = []
    for issue in issues:
        body = str(issue.get("body") or "")
        envelope = parse_request(body)
        labels = {label["name"] for label in issue.get("labels", [])}
        if (
            "pull_request" in issue
            or envelope is None
            or str(issue.get("user", {}).get("login", "")).casefold() != actor
            or envelope.repo != mailbox.repo
            or envelope.advisor_branch != mailbox.advisor_branch
            or not {"supervisor", mailbox.advisor_branch}.issubset(labels)
        ):
            continue
        payload = {
            "number": int(issue["number"]),
            "url": str(issue["html_url"]),
            "request": envelope.request.model_dump(mode="json"),
        }
        identity = _identity(envelope)
        if issue["state"] == "open" and envelope.request.target == target:
            events.append(
                ControllerEvent(
                    "supervisor_requested", f"supervisor_requested:{identity}", payload
                )
            )
        elif issue["state"] == "closed" and envelope.requester == target:
            line = body.splitlines()[-1]
            if not line.startswith(_RESULT_PREFIX) or not line.endswith(" -->"):
                continue
            try:
                result = SupervisorResult.model_validate_json(
                    line[len(_RESULT_PREFIX) : -4]
                )
            except ValidationError:
                continue
            payload["result"] = result.model_dump(mode="json")
            if envelope.parent_conversation_id is not None:
                payload["parent_conversation_id"] = str(envelope.parent_conversation_id)
            events.append(
                ControllerEvent(
                    "supervisor_completed", f"supervisor_completed:{identity}", payload
                )
            )
    return tuple(events)


class SupervisorGateway:
    def __init__(self, config: RunnerConfig):
        if config.github_token is None or not config.advisor_branch:
            raise RuntimeError("Supervisor requires authenticated controller context")
        self.config = config
        self.workflow = GitHubWorkflow(
            config.github_repo,
            config.github_token,
            role=config.role,
            trusted_actor=config.github_trusted_actor,
        )

    def validate(self, request: SupervisorRequest) -> None:
        config = self.config
        target = "advisor" if config.role == "advisor" else config.student_name
        if request.target != target:
            raise PermissionError("Supervisor request targets a different pod")
        assignment = _assignment(self.workflow, request, config.advisor_branch)
        branch = (
            assignment.head_ref if config.role == "student" else config.advisor_branch
        )
        if run_git(config.workspace, "branch", "--show-current") != branch:
            raise WorkflowPreconditionError(
                "Supervisor repair requires the target's current checkout; no branch will be switched"
            )
        if config.role == "student":
            run_git(
                config.workspace,
                "merge-base",
                "--is-ancestor",
                request.assignment.expected_pr_head_sha,
                "HEAD",
            )

    def prepare(self, request: SupervisorRequest) -> str:
        self.validate(request)
        numbers = list(request.context_prs)
        if request.assignment is not None:
            numbers.append(request.assignment.pr_number)
        context = ""
        if numbers:
            retrieved = get_prs(
                self.config.github_repo, numbers=numbers, token=self.config.github_token
            )
            assert retrieved.markdown is not None  # At most five explicit PRs.
            context = retrieved.markdown
        self.validate(request)
        status = run_git(self.config.workspace, "status", "--short")
        head = run_git(self.config.workspace, "rev-parse", "HEAD")
        return render_prompt(
            SUPERVISOR_REPAIR_PROMPT,
            TASK=request.task,
            TARGET=request.target,
            WORKSPACE=str(self.config.workspace),
            ASSIGNMENT=request.assignment.model_dump_json()
            if request.assignment
            else "advisor",
            HEAD=head,
            STATUS=status or "(clean)",
            PR_CONTEXT=context,
        )

    def complete(
        self,
        issue_number: int,
        request: SupervisorRequest,
        result_summary: str,
        resolved: bool,
    ) -> None:
        from senpai_agent.supervisor_worker import SupervisorResult

        workflow = self.workflow
        path = f"/repos/{workflow.repo}/issues/{issue_number}"
        issue = workflow._request("GET", path, expected_statuses={200}).json_body
        envelope = parse_request(str(issue.get("body") or ""))
        if (
            envelope is None
            or envelope.request != request
            or envelope.repo != workflow.repo
            or envelope.advisor_branch != self.config.advisor_branch
            or issue["user"]["login"].casefold() != workflow._actor().casefold()
        ):
            raise WorkflowPreconditionError(
                "Supervisor request changed before completion"
            )
        result = SupervisorResult(resolved=resolved, repair_summary=result_summary)
        body = f"{render_request(envelope)}\n\n## Supervisor result\n\n{result_summary}\n\n{_RESULT_PREFIX}{result.model_dump_json()} -->"
        workflow._mutate(
            "PATCH",
            path,
            json_body={"body": body, "state": "closed"},
            expected_statuses={200},
        )
        saved = workflow._request("GET", path, expected_statuses={200}).json_body
        if saved.get("state") != "closed" or saved.get("body") != body:
            raise ReconciliationError("GitHub did not persist the Supervisor result")
