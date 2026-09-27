"""Require a current assignment conversation before new training is launched."""

from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID

from senpai_agent.github.http import GitHubReader
from senpai_agent.github.tools.runtime import current_github_credentials
from senpai_agent.github.workflow.responses import (
    IssueSearchResponse,
    PullRequestResponse,
    validated_response,
)
from senpai_agent.github.workflow.validation import (
    require_active_assignment_routing,
    require_assignment_identity,
    require_open,
)
from senpai_agent.models import parse_assignment_markers
from senpai_agent.state import AssignmentConversationRegistry


class TrainingAssignmentGuard:
    """Check fresh GitHub routing against the controller's durable identity map."""

    def __init__(self, registry_path: Path, student_name: str):
        self.registry = AssignmentConversationRegistry(registry_path)
        self.student_name = student_name

    def require_current(self, conversation_id: UUID) -> None:
        credentials = current_github_credentials()
        if credentials is None or not self.student_name:
            raise RuntimeError(
                "training admission requires GitHub credentials and a student name"
            )
        github = GitHubReader(credentials.token)
        query = urlencode(
            {
                "state": "open",
                "labels": f"student:{self.student_name},status:wip",
                "per_page": 100,
            }
        )
        issues = (
            validated_response(IssueSearchResponse, item, "training assignment")
            for item in github.objects(f"/repos/{credentials.repo}/issues?{query}")
        )
        numbers = [issue.number for issue in issues if issue.pull_request is not None]
        if len(numbers) != 1:
            raise PermissionError(
                "training requires exactly one open WIP assignment for "
                f"{self.student_name!r}; "
                f"found {len(numbers)}"
            )
        snapshot = validated_response(
            PullRequestResponse,
            github.get(f"/repos/{credentials.repo}/pulls/{numbers[0]}"),
            "training assignment pull request",
        ).snapshot()
        require_open(snapshot)
        markers = parse_assignment_markers(snapshot.body)
        if len(markers) != 1:
            raise PermissionError("training requires exactly one assignment marker")
        assignment = require_assignment_identity(
            snapshot, repo=credentials.repo, assignment_id=markers[0].assignment_id
        )
        require_active_assignment_routing(snapshot, assignment)
        if assignment.student != self.student_name:
            raise PermissionError("training assignment does not belong to this student")
        self.registry.require_assignment(
            conversation_id, assignment.assignment_id, assignment.revision_id
        )
