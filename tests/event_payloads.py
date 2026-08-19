"""Complete production-shaped payloads for event delivery tests."""

from __future__ import annotations


def event_payload(kind: str, /, **overrides: object) -> dict[str, object]:
    parent_conversation_id = "00000000-0000-0000-0000-000000000017"
    pull = {
        "number": 17,
        "url": "https://github.test/pull/17",
        "head_ref": "student/experiment",
        "head_sha": "abc",
    }
    payloads: dict[str, dict[str, object]] = {
        "advisor_action": {**pull, "reasons": ["blocked"]},
        "agent_error": {
            "task_id": "task-17",
            "parent_conversation_id": parent_conversation_id,
            "task": "Inspect the experiment.",
            "error": "Inspection failed.",
        },
        "agent_result": {
            "task_id": "task-17",
            "parent_conversation_id": parent_conversation_id,
            "task": "Inspect the experiment.",
            "result": "Inspection complete.",
        },
        "duplicate_assignment": {
            "student": "Fern",
            "pull_requests": [17, 18],
        },
        "human_issue": {
            "number": 23,
            "title": "Experiment direction",
            "url": "https://github.test/issues/23",
            "human_message_id": 702,
            "author": "operator",
            "message": "Inspect the current experiment.",
            "created_at": "2026-08-19T15:49:43Z",
        },
        "malformed_assignment": {**pull, "error": "Invalid assignment marker."},
        "research_base_changed": {
            **pull,
            "assignment_id": "assignment-17",
            "revision_id": "initial",
            "student": "Fern",
            "base_ref": "research/main",
            "required_base_sha": "base-old",
            "current_base_sha": "base-new",
            "compare_url": "https://github.test/compare/base-old...base-new",
        },
        "review_ready": pull,
        "student_assignment": {
            **pull,
            "assignment_id": "assignment-17",
            "revision_id": "initial",
            "base_ref": "research/main",
            "base_sha": "base-sha",
            "blockers": [],
        },
        "student_assignment_comment": {
            "number": 17,
            "pr_url": "https://github.test/pull/17",
            "comment_id": "comment-17",
            "assignment_id": "assignment-17",
            "revision_id": "initial",
            "student": "Fern",
            "message": "The run has started.",
            "content_digest": "digest-17",
        },
        "student_available_for_assignment": {"student": "Fern"},
        "student_pr_feedback": {
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
        "training_monitor": {
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
        "workspace_diverged": {
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
    return {**payloads[kind], **overrides}
