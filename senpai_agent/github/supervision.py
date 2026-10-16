"""Route bounded Supervisor requests through the existing GitHub control plane."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Annotated, Self
from urllib.parse import urlencode
from uuid import UUID

from openhands.sdk.tool import Action, ToolExecutor
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from senpai_agent.git_transport import run_git
from senpai_agent.github.pull_requests import get_prs
from senpai_agent.github.tools.contracts import (
    AssignmentVersion,
    GitHubMutationObservation,
)
from senpai_agent.github.tools.runtime import GitHubToolRuntime
from senpai_agent.github.workflow import (
    GitHubWorkflow,
    MutationResult,
    WorkflowPreconditionError,
)
from senpai_agent.models import _marker_payload, authoritative_marker_line
from senpai_agent.PROMPTS import SUPERVISOR_REPAIR_PROMPT, render_prompt

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation

    from senpai_agent.openhands_runner import RunnerConfig

_PREFIX = "<!-- senpai-supervisor:"


class SupervisorRequest(Action):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request_id: str = Field(
        min_length=1,
        max_length=128,
        description="Stable ID for one repair attempt. Reuse it to retry delivery; after a blocked or failed attempt, use a new request_id.",
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


class SupervisorResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    resolved: bool = Field(
        description="True only when the requested issue is resolved and verified.",
    )
    repair_summary: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12_000)
    ] = Field(
        description=(
            "Report the diagnosis, changes and verification. Give actionable feedback "
            "for any remaining issue, including evidence and the next required action."
        ),
    )

    @model_validator(mode="after")
    def fit_issue_body(self) -> Self:
        if len(_marker_payload(self)) + len(self.repair_summary) > 32_000:
            raise ValueError("Supervisor result is too large; shorten the summary")
        return self


class SupervisorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    advisor_branch: str
    requester: str
    request: SupervisorRequest
    parent_conversation_id: UUID
    result: SupervisorResult | None = None


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
    encoded = _marker_payload(envelope)
    marker = f"{_PREFIX}{_identity(envelope)}:{encoded} -->"
    body = f"{marker}\n\n## Supervisor request: {envelope.request.target}\n\n{envelope.request.task}"
    if envelope.result is not None:
        body += f"\n\n## Supervisor result\n\n{envelope.result.repair_summary}"
    # Reserve half the Issue body for a result, including JSON escape expansion.
    if len(body) > (65_536 if envelope.result else 32_000):
        raise ValueError("Supervisor request is too large; shorten the task")
    return body


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
        if conversation is None:
            raise RuntimeError("Supervisor requests require a conversation")
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
            parent_conversation_id=str(conversation.id),
        )

        def validate_existing(body: str) -> None:
            original = parse_request(body)
            if (
                original is None
                or original.model_copy(update={"result": None}) != envelope
            ):
                raise WorkflowPreconditionError(
                    "request_id already belongs to a different Supervisor request"
                )

        changed, issue = runtime.workflow._create_marked_issue(
            prefix=f"{_PREFIX}{_identity(envelope)}:",
            title=f"Supervisor: {action.target} — {action.request_id}",
            labels={"supervisor", runtime.advisor_branch},
            render_body=lambda: render_request(envelope),
            validate_existing=validate_existing,
        )
        return GitHubMutationObservation.from_result(
            MutationResult(
                changed,
                issue.html_url,
                "supervisor_requested",
                action.request_id,
            )
        )


class SupervisorGateway:
    def __init__(self, config: RunnerConfig):
        if config.github_token is None or not config.advisor_branch:
            raise RuntimeError("Supervisor requires authenticated controller context")
        self.config = config
        self.recipient = "advisor" if config.role == "advisor" else config.student_name
        self.workflow = GitHubWorkflow(
            config.github_repo,
            config.github_token,
            role=config.role,
            trusted_actor=config.github_trusted_actor,
        )

    def _trusted_envelope(
        self, issue: Mapping[str, object]
    ) -> SupervisorEnvelope | None:
        envelope = parse_request(str(issue.get("body") or ""))
        labels = {label["name"] for label in issue.get("labels", [])}
        if (
            "pull_request" in issue
            or envelope is None
            or str(issue.get("user", {}).get("login", "")).casefold()
            != self.workflow._actor().casefold()
            or envelope.repo != self.workflow.repo
            or envelope.advisor_branch != self.config.advisor_branch
            or not {"supervisor", self.config.advisor_branch}.issubset(labels)
        ):
            return None
        return envelope

    def pending(self) -> Iterator[tuple[Mapping[str, object], SupervisorEnvelope]]:
        query = urlencode(
            {
                "state": "open",
                "labels": f"supervisor,{self.config.advisor_branch}",
                "per_page": 100,
            }
        )
        for issue in self.workflow._objects(
            f"/repos/{self.workflow.repo}/issues?{query}"
        ):
            envelope = self._trusted_envelope(issue)
            if envelope is None or issue["state"] != "open":
                continue
            if (
                envelope.result is None
                and envelope.request.target == self.recipient
                or envelope.result is not None
                and envelope.requester == self.recipient
            ):
                yield issue, envelope

    def close_reply(self, number: int, original: SupervisorEnvelope) -> None:
        """Close only the exact trusted reply already stored by its requester."""
        workflow = self.workflow
        path = f"/repos/{workflow.repo}/issues/{number}"
        issue = workflow._request("GET", path, expected_statuses={200}).json_body
        if (
            self._trusted_envelope(issue) == original
            and original.requester == self.recipient
            and original.result is not None
            and issue["state"] != "closed"
        ):
            workflow._update_issue(number, {"state": "closed"})

    def validate(self, request: SupervisorRequest) -> None:
        config = self.config
        if request.target != self.recipient:
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

    def prepare(self, envelope: SupervisorEnvelope) -> str:
        request = envelope.request
        render_request(envelope)
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
        status = run_git(self.config.workspace, "status", "--short")
        head = run_git(self.config.workspace, "rev-parse", "HEAD")
        return render_prompt(
            SUPERVISOR_REPAIR_PROMPT,
            REQUESTER=envelope.requester,
            ROLE="advisor" if envelope.requester == "advisor" else "student",
            TASK=request.task,
            TARGET=request.target,
            WORKSPACE=str(self.config.workspace),
            ASSIGNMENT=request.assignment.model_dump_json()
            if request.assignment
            else "(none; advisor repair)",
            HEAD=head,
            STATUS=status or "(clean)",
            PR_CONTEXT=context,
        )

    def complete(
        self,
        issue_number: int,
        original: SupervisorEnvelope,
        result_summary: str,
        resolved: bool,
        *,
        delivered_to: UUID,
    ) -> None:
        workflow = self.workflow
        path = f"/repos/{workflow.repo}/issues/{issue_number}"
        issue = workflow._request("GET", path, expected_statuses={200}).json_body
        envelope = self._trusted_envelope(issue)
        if envelope is None or envelope.model_copy(
            update={"result": None}
        ) != original.model_copy(update={"result": None}):
            raise WorkflowPreconditionError(
                "Supervisor request changed before completion"
            )
        result = SupervisorResult(resolved=resolved, repair_summary=result_summary)
        if envelope.result is not None and envelope.result != result:
            raise WorkflowPreconditionError(
                "Supervisor request already has a different result"
            )
        body = render_request(envelope.model_copy(update={"result": result}))
        delivered_locally = (
            envelope.requester == original.request.target
            and envelope.parent_conversation_id == delivered_to
        )
        if issue.get("body") == body and (
            not delivered_locally or issue["state"] == "closed"
        ):
            return
        workflow._update_issue(
            issue_number,
            {
                "body": body,
                **({"state": "closed"} if delivered_locally else {}),
            },
        )
