"""Errors raised by Senpai's GitHub workflow."""

from urllib.parse import urlsplit


class GitHubWorkflowError(RuntimeError):
    """Base error for GitHub workflow operations."""


class GitHubAPIError(GitHubWorkflowError):
    """GitHub returned an unexpected HTTP response."""

    def __init__(self, method: str, url: str, status_code: int):
        endpoint = urlsplit(url)
        path = endpoint.path
        if endpoint.query:
            path = f"{path}?{endpoint.query}"
        super().__init__(f"GitHub {method} {path} returned HTTP {status_code}")
        self.status_code = status_code


class GitHubTransportError(GitHubWorkflowError):
    """GitHub could not be reached."""

    def __init__(self, method: str, url: str):
        endpoint = urlsplit(url)
        super().__init__(
            f"GitHub {method} {endpoint.path} failed before an HTTP response"
        )


class WorkflowPreconditionError(GitHubWorkflowError):
    """Current GitHub state does not permit the requested transition."""


class PullHeadMismatchError(WorkflowPreconditionError):
    """GitHub's pull-request snapshot has not reached the expected head."""


class StaleAssignmentRevisionError(WorkflowPreconditionError):
    """The requested operation belongs to another assignment revision."""


class StaleResearchBaseError(WorkflowPreconditionError):
    """The result was not reviewed against the live research base."""


class ReconciliationError(GitHubWorkflowError):
    """GitHub did not reach the requested state."""
