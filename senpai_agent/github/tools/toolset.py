"""Assemble the GitHub tools allowed for one authenticated role."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from openhands.sdk.tool import ToolDefinition

from senpai_agent.github import PRRetrievalResult, get_prs
from senpai_agent.github.workflow import GitHubWorkflow

from .definitions import (
    AcceptResultOnCurrentBaseTool,
    AdoptAssignmentTool,
    CloseExperimentTool,
    CreateAssignmentTool,
    MergeExperimentTool,
    PublishAdvisorBranchTool,
    RepairAssignmentRoutingTool,
    RequestAssignmentRevisionTool,
    RespondToHumanIssueTool,
    SendAssignmentFeedbackTool,
    SubmitExperimentResultTool,
)
from .pull_requests import GetPRsAction, GetPRsObservation, GetPRsTool
from .runtime import (
    GitHubToolRuntime,
    configured_student_names,
    current_github_credentials,
)


class GitHubWorkflowToolSet(
    ToolDefinition[GetPRsAction, GetPRsObservation]
):
    """Resolve the reader and workflow tools allowed for one role."""

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        workflow: GitHubWorkflow | None = None,
        *,
        role: str | None = None,
        state_dir: str | Path | None = None,
        workspace: str | Path | None = None,
        advisor_branch: str | None = None,
        student_names: Sequence[str] | str | None = None,
        student_name: str | None = None,
        get_prs_fn: Callable[..., PRRetrievalResult] = get_prs,
    ) -> Sequence[ToolDefinition]:
        role = role or os.environ.get("SENPAI_ROLE")
        if role not in {"advisor", "student"}:
            raise ValueError("role must be advisor or student")
        if workspace is None:
            if conv_state is None:
                raise ValueError("senpai_github requires its OpenHands workspace")
            workspace = Path(conv_state.workspace.working_dir)

        credentials = current_github_credentials()
        git_token = None
        if workflow is None:
            if credentials is None:
                raise RuntimeError(
                    "configure GitHub credentials before initializing workflows"
                )
            workflow = GitHubWorkflow(
                credentials.repo,
                credentials.token,
                role=role,
                trusted_actor=credentials.trusted_actor,
            )
            git_token = credentials.token
        elif workflow.role != role:
            raise ValueError("workflow role must match the GitHub tool role")

        runtime = GitHubToolRuntime(
            workflow=workflow,
            workspace=Path(workspace),
            git_token=git_token,
            role=role,
            advisor_branch=advisor_branch or os.environ.get("ADVISOR_BRANCH"),
            student_names=configured_student_names(student_names),
            student_name=student_name or os.environ.get("STUDENT_NAME"),
        )
        if role == "advisor":
            runtime.assignment_base_branch()
            if not runtime.student_names:
                raise ValueError("advisor GitHub tools require configured student names")
        else:
            runtime.human_issue_audience()

        common = (
            *GetPRsTool.create(
                conv_state,
                get_prs_fn=get_prs_fn,
                state_dir=state_dir,
                workspace=workspace,
            ),
            *RespondToHumanIssueTool.create(runtime),
        )
        if role == "student":
            return (*common, *SubmitExperimentResultTool.create(runtime))
        return (
            *common,
            *CreateAssignmentTool.create(runtime),
            *AdoptAssignmentTool.create(runtime),
            *PublishAdvisorBranchTool.create(runtime),
            *RepairAssignmentRoutingTool.create(runtime),
            *SendAssignmentFeedbackTool.create(runtime),
            *RequestAssignmentRevisionTool.create(runtime),
            *AcceptResultOnCurrentBaseTool.create(runtime),
            *MergeExperimentTool.create(runtime),
            *CloseExperimentTool.create(runtime),
        )
