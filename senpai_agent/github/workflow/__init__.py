"""Typed, role-safe GitHub workflow operations."""

from senpai_agent.github.workflow.errors import (
    GitHubAPIError,
    GitHubTransportError,
    GitHubWorkflowError,
    PullHeadMismatchError,
    ReconciliationError,
    StaleAssignmentRevisionError,
    StaleResearchBaseError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import (
    HttpResponse,
    HttpTransport,
    MutationResult,
    PullRequestSnapshot,
    SubmitResultPreflight,
)
from senpai_agent.github.workflow.workflow import GitHubWorkflow

__all__ = [
    "GitHubAPIError",
    "GitHubTransportError",
    "GitHubWorkflow",
    "GitHubWorkflowError",
    "HttpResponse",
    "HttpTransport",
    "MutationResult",
    "PullHeadMismatchError",
    "PullRequestSnapshot",
    "ReconciliationError",
    "StaleAssignmentRevisionError",
    "StaleResearchBaseError",
    "SubmitResultPreflight",
    "WorkflowPreconditionError",
]
