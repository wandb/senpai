# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Names of structured events that Senpai renders for models."""

from enum import StrEnum


class EventKind(StrEnum):
    ADVISOR_ACTION = "advisor_action"
    AGENT_ERROR = "agent_error"
    AGENT_RESULT = "agent_result"
    DUPLICATE_ASSIGNMENT = "duplicate_assignment"
    HUMAN_ISSUE = "human_issue"
    MALFORMED_ASSIGNMENT = "malformed_assignment"
    RESEARCH_BASE_CHANGED = "research_base_changed"
    REVIEW_READY = "review_ready"
    STUDENT_ASSIGNMENT = "student_assignment"
    STUDENT_ASSIGNMENT_COMMENT = "student_assignment_comment"
    STUDENT_AVAILABLE_FOR_ASSIGNMENT = "student_available_for_assignment"
    STUDENT_PR_FEEDBACK = "student_pr_feedback"
    TRAINING_MONITOR = "training_monitor"
    WORKSPACE_DIVERGED = "workspace_diverged"


EVENT_KINDS = frozenset(kind.value for kind in EventKind)
