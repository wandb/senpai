"""Public GitHub workflow client assembled from cohesive domain operations."""

from senpai_agent.github.workflow.assignments import AssignmentMixin
from senpai_agent.github.workflow.adoption import AdoptionMixin
from senpai_agent.github.workflow.comments import CommentsMixin
from senpai_agent.github.workflow.core import WorkflowCore
from senpai_agent.github.workflow.issues import HumanIssueMixin
from senpai_agent.github.workflow.lookup import LookupMixin
from senpai_agent.github.workflow.merge import MergeMixin
from senpai_agent.github.workflow.results import ResultMixin
from senpai_agent.github.workflow.review import ReviewMixin
from senpai_agent.github.workflow.revisions import RevisionMixin


class GitHubWorkflow(
    HumanIssueMixin,
    MergeMixin,
    ReviewMixin,
    ResultMixin,
    RevisionMixin,
    AdoptionMixin,
    AssignmentMixin,
    LookupMixin,
    CommentsMixin,
    WorkflowCore,
):
    """Desired-state GitHub client for Senpai research workflows."""

    __slots__ = ()
