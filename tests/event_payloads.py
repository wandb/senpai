"""Complete production-shaped payloads for event delivery tests."""

from __future__ import annotations

from senpai_agent.event_kinds import EventKind


def event_payload(
    kind: str | EventKind,
    /,
    **overrides: object,
) -> dict[str, object]:
    parent_conversation_id = "00000000-0000-0000-0000-000000000017"
    pull = {
        "number": 17,
        "url": "https://github.test/pull/17",
        "head_ref": "student/experiment",
        "head_sha": "abc",
    }
    payloads: dict[EventKind, dict[str, object]] = {
        EventKind.ADVISOR_ACTION: {**pull, "reasons": ["blocked"]},
        EventKind.AGENT_ERROR: {
            "task_id": "task-17",
            "parent_conversation_id": parent_conversation_id,
            "task": "Inspect the experiment.",
            "error": "Inspection failed.",
        },
        EventKind.AGENT_RESULT: {
            "task_id": "task-17",
            "parent_conversation_id": parent_conversation_id,
            "task": "Inspect the experiment.",
            "result": "Inspection complete.",
        },
        EventKind.DUPLICATE_ASSIGNMENT: {
            "student": "Fern",
            "pull_requests": [17, 18],
        },
        EventKind.HUMAN_ISSUE: {
            "number": 23,
            "title": "Experiment direction",
            "url": "https://github.test/issues/23",
            "human_message_id": 702,
            "author": "operator",
            "message": "Inspect the current experiment.",
            "created_at": "2026-08-19T15:49:43Z",
        },
        EventKind.MALFORMED_ASSIGNMENT: {
            **pull,
            "error": "Invalid assignment marker.",
        },
        EventKind.RESEARCH_BASE_CHANGED: {
            **pull,
            "assignment_id": "assignment-17",
            "revision_id": "initial",
            "student": "Fern",
            "base_ref": "research/main",
            "required_base_sha": "base-old",
            "current_base_sha": "base-new",
            "compare_url": "https://github.test/compare/base-old...base-new",
        },
        EventKind.REVIEW_READY: pull,
        EventKind.STUDENT_ASSIGNMENT: {
            **pull,
            "assignment_id": "assignment-17",
            "revision_id": "initial",
            "base_ref": "research/main",
            "base_sha": "base-sha",
            "blockers": [],
        },
        EventKind.STUDENT_ASSIGNMENT_COMMENT: {
            "number": 17,
            "pr_url": "https://github.test/pull/17",
            "comment_id": "comment-17",
            "assignment_id": "assignment-17",
            "revision_id": "initial",
            "student": "Fern",
            "message": "The run has started.",
            "content_digest": "digest-17",
        },
        EventKind.STUDENT_AVAILABLE_FOR_ASSIGNMENT: {"student": "Fern"},
        EventKind.STUDENT_PR_FEEDBACK: {
            "number": 17,
            "pr_url": "https://github.test/pull/17",
            "feedback_url": "https://github.test/pull/17#issuecomment-5344",
            "feedback_id": 5344,
            "feedback_type": "issue_comment",
            "assignment_id": "assignment-17",
            "revision_id": "initial",
            "base_ref": "research/main",
            "base_sha": "base-sha",
            "head_ref": "student/experiment",
            "head_sha": "abc",
            "author": "operator",
            "author_association": "OWNER",
            "message": "Review the current result.",
            "created_at": "2026-08-19T15:49:43Z",
        },
        EventKind.TRAINING_MONITOR: {
            "conversation_id": parent_conversation_id,
            "training_id": "training-17",
            "summary": "The metric crossed its gate.",
            "reason": "The registered monitor policy emitted this signal.",
            "signal": {
                "kind": "metric_gate",
                "dedupe_key": "training-17:gate:0",
                "training_id": "training-17",
                "metric": "val/loss",
                "value": 0.1,
                "state": "running",
                "detail": "val/loss crossed its gate at 0.1.",
                "hard_failure": False,
            },
        },
        EventKind.WORKSPACE_DIVERGED: {
            "head_ref": "student/experiment",
            "expected_remote_head": "abc",
            "preserved_local_head": "def",
            "base_ref": "research/main",
            "base_sha": "base-sha",
            "current_branch": "student/experiment",
            "worktree_fingerprint": "fingerprint-17",
            "instructions": "Reconcile the workspace.",
        },
    }
    if frozenset(payloads) != frozenset(EventKind):
        raise RuntimeError("event payload fixtures do not match EventKind")
    return {**payloads[EventKind(kind)], **overrides}
