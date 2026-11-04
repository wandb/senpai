import json
import os
import time
import uuid
from dataclasses import replace
from io import BytesIO, StringIO
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest
from git_workflow_support import commit_file, git, repository
from github_retrieval_support import inline_comment
from github_retrieval_support import review as review_submission
from github_workflow_support import (
    ASSIGNMENT_ID,
    REPO,
    FakeGitHub,
    assignment_record,
    comment,
    experiment_result,
    pull_request,
)
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.tool import resolve_tool
from openhands_support import runtime_config, runtime_env
from pydantic import SecretStr

import senpai_agent.github.code_review as code_review
import senpai_agent.github.merge_worker as merge_worker
import senpai_agent.openhands_runner as runner
from senpai_agent.github import http as github_http
from senpai_agent.github.mailbox import GitHubMailbox
from senpai_agent.github.pull_requests import PullRequest
from senpai_agent.github.tools import GitHubWorkflowToolSet
from senpai_agent.github.tools.contracts import AssignmentVersion, MergeExperimentAction
from senpai_agent.github.tools.runtime import (
    clear_github_credentials,
    configure_github_credentials,
)
from senpai_agent.github.workflow import HttpResponse
from senpai_agent.github.workflow.responses import PullRequestSnapshot
from senpai_agent.models import render_assignment_marker, render_result_comment
from senpai_agent.PROMPTS import CODE_QUALITY_REVIEW_PROMPT

APPROVED = code_review.CodeQualityVerdict(
    approved=True, review_summary="Focused change.", findings=[], notes=[],
)
FINDING_TEXT = "Line 1: remove the unused fallback; callers already require a value."
FINDING = code_review.ReviewFinding(path="model.py", finding=FINDING_TEXT)
REJECTED = code_review.CodeQualityVerdict(
    approved=False, review_summary="Unnecessary complexity.", findings=[FINDING],
    notes=[],
)
TOOLING_FINDING = code_review.ReviewFinding(
    path="tools/sass.py", finding="Exit with a non-zero status when the census fails.",
)
SAFETY_FINDING = code_review.ReviewFinding(
    path="tools/sass.py", finding="Remove the hard-coded API token.", safety=True,
)


def rejected_review_comment(index: int) -> dict:
    return comment(
        100 + index,
        f"<!-- senpai-merge-review:review-{index} -->\n\n**Supervisor**\n\n---\n\n"
        "**Senpai Supervisor — merge review**\n\n**Verdict: Rejected**\n\n"
        f"Round {index} findings.",
    )


def rejecting(*findings, summary="Unnecessary complexity."):
    verdict = code_review.CodeQualityVerdict(
        approved=False, review_summary=summary, findings=list(findings), notes=[],
    )

    def review(_prompt, _config, *, response_schema, on_structured_result):
        on_structured_result(verdict)
        return 0

    return review
NOTE = "README.md: The experiment notes could be shorter."
ADVISOR_HISTORY_SENTINEL = "Advisor-only history: the experiment improved the held-out metric."


@pytest.fixture
def merge_case(tmp_path, monkeypatch):
    workspace, remote, base_sha = repository(tmp_path)
    head_sha = commit_file(workspace, "model.py", "baseline = 2\n", "candidate")
    git(workspace, "push", "origin", "experiment-7")
    (workspace / "model.py").write_text("uncommitted = 'must not be reviewed'\n")
    assignment = assignment_record(base_sha=base_sha, head_sha=head_sha)
    fake = FakeGitHub(
        pull_request(
            labels={"status:review", "student:student-one"},
            head_sha=head_sha,
            body=render_assignment_marker(assignment),
        ),
        comments=[comment(1, render_result_comment(experiment_result(
            commit_sha=head_sha, expected_head_sha=head_sha,
        )))],
        branch_heads={"schmidhuber": base_sha},
    )
    monkeypatch.setattr("senpai_agent.github.workflow.core.UrllibTransport", lambda: fake)
    monkeypatch.setattr(code_review, "github_repository_url", lambda _repo: str(remote))
    reviews = []
    inline_comments = []
    discussions = {
        f"/repos/{REPO}/issues/7/comments": fake.comments,
        f"/repos/{REPO}/pulls/7/reviews": reviews,
        f"/repos/{REPO}/pulls/7/comments": inline_comments,
    }

    def urlopen(github_request, timeout):
        assert github_request.get_method() == "GET"
        assert github_request.headers["Authorization"] == "Bearer private-github-key"
        parsed = urlsplit(github_request.full_url)
        headers = {}
        if parsed.path in discussions:
            entries = discussions[parsed.path]
            page = int(parse_qs(parsed.query).get("page", ["1"])[0])
            payload = entries[page - 1:page]
            if page < len(entries):
                headers["Link"] = (
                    f'<https://api.github.com{parsed.path}?page={page + 1}>; rel="next"'
                )
        else:
            payload = fake.request(
                "GET", github_request.full_url, headers=github_request.headers,
            ).json_body
        response = BytesIO(json.dumps(payload).encode())
        response.headers = headers
        return response

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)
    config = runtime_config(
        tmp_path,
        workspace=workspace,
        child=True,
        agent_name="supervisor",
        model="anthropic/claude-sonnet-5",
        smart_model="openai/gpt-6-astra",
        smart_api_key_env="OPENAI_API_KEY",
        smart_api_key=SecretStr("smart-key"),
        smart_reasoning_effort="high",
        github_token=None,
        delegation_task_id="merge-task",
        delegation_root_state_dir=tmp_path / "advisor-state",
        delegation_deadline_epoch=time.time() + 600,
    )
    action = MergeExperimentAction(
        assignment=AssignmentVersion(
            pr_number=7,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-1",
            expected_pr_head_sha=head_sha,
        ),
        expected_current_base_sha=base_sha,
    )
    return SimpleNamespace(
        config=config, action=action, fake=fake,
        base_sha=base_sha, head_sha=head_sha,
        reviews=reviews, inline_comments=inline_comments,
    )


def execute(case):
    return merge_worker.merge_with_review(
        case.action,
        config=case.config,
        token=SecretStr("private-github-key"),
    )


@pytest.mark.parametrize("has_root", [True, False])
def test_merge_worker_requires_the_advisors_shared_lock(
    merge_case, monkeypatch, has_root,
):
    lock_paths = []
    original_workflow = merge_worker.GitHubWorkflow

    def workflow(*args, **kwargs):
        lock_paths.append(kwargs["mutation_lock_path"])
        return original_workflow(*args, **kwargs)

    monkeypatch.setattr(merge_worker, "GitHubWorkflow", workflow)
    monkeypatch.setattr("senpai_agent.github.tools.toolset.GitHubWorkflow", workflow)
    configure_github_credentials(REPO, SecretStr("private-github-key"))
    try:
        GitHubWorkflowToolSet.create(
            role="advisor", workspace=merge_case.config.workspace,
            state_dir=merge_case.config.delegation_root_state_dir / "github",
            advisor_branch="schmidhuber", student_names=["student-one"],
        )
    finally:
        clear_github_credentials()

    def review(_prompt, _config, *, response_schema, on_structured_result):
        on_structured_result(APPROVED)
        return 0

    monkeypatch.setattr(runner, "run_openhands", review)
    if has_root:
        assert execute(merge_case)["state"] == "experiment_merged"
        assert len(lock_paths) == 2
        assert lock_paths[0] == lock_paths[1]
    else:
        merge_case.config = replace(merge_case.config, delegation_root_state_dir=None)
        with pytest.raises(RuntimeError, match="root state directory"):
            execute(merge_case)
        assert merge_case.fake.mutations == []
        assert len(lock_paths) == 1


def test_approval_reviews_pinned_source_with_smart_profile_before_merging(
    merge_case, monkeypatch,
):
    case = merge_case
    reviewed_workspaces = []
    case.fake.pr["body"] += "\n\n" + "Detailed rationale.\n" * 100 + "Final scope exception."
    case.fake.comments.extend([
        comment(2, "Keep the required reproducibility document.\nIt records the seed."),
        comment(3, "The failed experiment helper must be removed."),
    ])
    case.reviews.extend([
        review_submission(4, "The initial implementation duplicated the data loader."),
        review_submission(5, "The revised implementation uses the existing loader."),
    ])
    case.inline_comments.extend([
        {**inline_comment(6, "This fallback belongs to the failed attempt."),
         "path": "model.py", "line": 1},
        {**inline_comment(7, "The current path already validates this value."),
         "path": "model.py", "line": 2, "in_reply_to_id": 6},
    ])

    def review(prompt, config, *, response_schema, on_structured_result):
        assert response_schema is code_review.CodeQualityVerdict
        assert case.fake.mutations == []
        assert git(config.workspace, "rev-parse", "--is-bare-repository") == "true"
        assert git(config.workspace, "show", f"{case.head_sha}:model.py") == "baseline = 2"
        assert git(config.workspace, "merge-base", case.base_sha, case.head_sha) == case.base_sha
        assert config.workspace != case.config.workspace
        assert config.model == "openai/gpt-6-astra"
        assert config.reasoning_effort == "high"
        assert config.api_key.get_secret_value() == "smart-key"
        assert config.github_token is None
        assert "GITHUB_TOKEN" not in config.conversation_secrets
        assert config.delegation_task_id is None
        assert config.instructions.program.content == "Test programme."
        assert case.base_sha in prompt and case.head_sha in prompt
        assert case.fake.pr["body"] in prompt
        for entry in (*case.fake.comments, *case.reviews, *case.inline_comments):
            assert entry["body"] in prompt
        assert "model.py:1" in prompt and "model.py:2" in prompt
        assert "model.py | 2" in prompt and "+baseline = 2" in prompt
        assert "Report everything else as non-blocking notes and approve" in prompt
        reviewed_workspaces.append(config.workspace)
        on_structured_result(APPROVED)
        return 0

    monkeypatch.setattr(runner, "run_openhands", review)

    result = execute(case)

    assert result["state"] == "experiment_merged"
    assert case.fake.pr["merged"] is True
    posted = case.fake.comments[-1]["body"]
    assert "Senpai Supervisor — merge review" in posted
    assert "**Verdict: Approved**" in posted
    assert APPROVED.review_summary in posted
    assert case.head_sha in posted and case.base_sha in posted
    assert case.fake.mutations == [(
        "POST", f"/repos/{REPO}/issues/7/comments", {"body": posted},
    ), (
        "PUT", f"/repos/{REPO}/pulls/7/merge",
        {"sha": case.head_sha, "merge_method": "squash"},
    )]
    assert len(reviewed_workspaces) == 1
    assert not reviewed_workspaces[0].exists()


@pytest.mark.parametrize(
    ("failure", "reason"),
    [("retrieval", "HTTP 503"), ("changed-head", "PR head changed")],
)
def test_missing_or_stale_pr_discussion_blocks_review_and_merge(
    merge_case, monkeypatch, failure, reason,
):
    original_urlopen = github_http.request.urlopen

    def urlopen(github_request, timeout):
        path = urlsplit(github_request.full_url).path
        if failure == "retrieval" and path.endswith("/reviews"):
            raise HTTPError(github_request.full_url, 503, "unavailable", {}, None)
        if failure == "changed-head" and path.endswith("/pulls/7"):
            merge_case.fake.pr["head_sha"] = "d" * 40
        return original_urlopen(github_request, timeout)

    def review(*_args, **_kwargs):
        pytest.fail("reviewer received incomplete or stale PR context")

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)
    monkeypatch.setattr(runner, "run_openhands", review)

    result = execute(merge_case)

    assert result["state"] in {"merge_blocked", "merge_failed"}
    assert reason in result["reason"]
    assert merge_case.fake.pr["merged"] is False
    assert not any(method == "PUT" for method, _path, _body in merge_case.fake.mutations)
    assert merge_case.fake.mutations == []
    assert "feedback_url" not in result


def test_large_pr_review_prompt_stays_within_its_context_budget():
    head_sha = "b" * 40
    pull = PullRequestSnapshot(
        number=7, node_id="PR_7", url=f"https://github.com/{REPO}/pull/7",
        title="Large experiment", body="", base_ref="schmidhuber",
        head_ref="experiment-7", head_sha=head_sha, labels=(), draft=False,
        state="open", merged=False, mergeable=True, merge_commit_sha=None,
    )
    discussion = tuple(
        {**comment(index, f"Discussion entry {index}. " + "x" * 5_000),
         "created_at": f"2026-10-29T{index // 60:02d}:{index % 60:02d}:00Z"}
        for index in range(1, 63)
    )
    record = PullRequest(
        details={
            "number": 7, "title": "Large experiment",
            "html_url": f"https://github.com/{REPO}/pull/7",
            "body": "Assignment details. " * 20_000,
        },
        issue_comments=discussion, reviews=(), inline_comments=(),
    )
    diff = (
        " kernel.py | 2886 ++++++++++\n 1 file changed, 2886 insertions(+)\n\n"
        + "".join(f"+line {index} of the kernel\n" for index in range(2886 * 3))
    )
    context_chars = 160_000

    prompt = code_review.review_prompt(
        pull, base_sha="a" * 40, merge_base="a" * 40, record=record, diff=diff,
        context_chars=context_chars, minutes=39, retry=False,
    )

    # The template, its three commit SHAs, and the context budget.
    assert len(prompt) <= len(CODE_QUALITY_REVIEW_PROMPT) + 3 * 40 + context_chars
    assert len(diff) + len(record.details["body"]) > 2 * context_chars
    assert " kernel.py | 2886" in prompt
    assert "Diff truncated to fit the review budget" in prompt
    assert "Truncated to fit the review budget" in prompt
    assert "Discussion entry 62." in prompt
    assert "Discussion entry 1." not in prompt
    assert "### Recent discussion (5 of 62 entries; 57 older omitted)" in prompt
    assert "within about 39 minutes" in prompt
    assert "ran out of time" not in prompt


@pytest.mark.parametrize("retry_succeeds", [True, False], ids=["verdict", "no-verdict"])
def test_unfinished_review_retries_once_with_a_compact_prompt(
    merge_case, monkeypatch, capsys, retry_succeeds,
):
    large_comment = "Earlier benchmark log. " + "y" * 8_000
    merge_case.fake.comments.append(comment(2, large_comment))
    attempts = []

    def review(prompt, config, *, response_schema, on_structured_result):
        attempts.append((prompt, config.delegation_deadline_epoch, config.conversation_id))
        if len(attempts) == 2 and retry_succeeds:
            on_structured_result(APPROVED)
            return 0
        return 1

    monkeypatch.setattr(runner, "run_openhands", review)
    started_at = time.time()

    result = execute(merge_case)

    (first_prompt, first_deadline, first_id), (retry_prompt, retry_deadline, retry_id) = (
        attempts
    )
    window = merge_case.config.delegation_deadline_epoch - 60 - started_at
    assert first_deadline - started_at == pytest.approx(window * 2 / 3, abs=5)
    assert first_deadline < retry_deadline <= started_at + window
    assert first_id == merge_case.config.conversation_id != retry_id
    assert large_comment in first_prompt and large_comment not in retry_prompt
    assert "ran out of time" not in first_prompt and "ran out of time" in retry_prompt
    assert "SENPAI_CODE_REVIEW_RETRY pr=7" in capsys.readouterr().err
    if retry_succeeds:
        assert result["state"] == "experiment_merged"
        assert merge_case.fake.pr["merged"] is True
    else:
        assert result["state"] == "merge_blocked"
        assert "did not complete with a verdict" in result["reason"]
        assert "compact retry" in result["reason"]
        assert merge_case.fake.mutations == []


def test_non_blocking_notes_merge_and_are_published_with_the_approval(
    merge_case, monkeypatch,
):
    noted = code_review.CodeQualityVerdict(
        approved=True, review_summary="Focused change.", findings=[], notes=[NOTE],
    )

    def review(_prompt, _config, *, response_schema, on_structured_result):
        on_structured_result(noted)
        return 0

    monkeypatch.setattr(runner, "run_openhands", review)

    result = execute(merge_case)

    assert result["state"] == "experiment_merged"
    assert merge_case.fake.pr["merged"] is True
    posted = merge_case.fake.comments[-1]["body"]
    assert "**Verdict: Approved**" in posted
    assert f"**Non-blocking notes**\n\n- {NOTE}" in posted
    with pytest.raises(ValueError, match="at least one blocking finding"):
        code_review.CodeQualityVerdict(
            approved=False, review_summary="Only nits.", findings=[], notes=[NOTE],
        )


def test_findings_outside_the_deliverable_become_notes_and_the_merge_proceeds(
    merge_case, monkeypatch,
):
    """About 40 of 46 rejections in one campaign were on lane tooling, README
    wording or generated copies rather than the submitted artefact."""

    merge_case.action = merge_case.action.model_copy(
        update={"deliverable_paths": ("model.py", "docs/*.md")},
    )
    prompts = []

    def review(prompt, _config, *, response_schema, on_structured_result):
        prompts.append(prompt)
        on_structured_result(code_review.CodeQualityVerdict(
            approved=False, review_summary="Tooling needs work.",
            findings=[TOOLING_FINDING], notes=[NOTE],
        ))
        return 0

    monkeypatch.setattr(runner, "run_openhands", review)

    result = execute(merge_case)

    assert result["state"] == "experiment_merged"
    assert merge_case.fake.pr["merged"] is True
    posted = merge_case.fake.comments[-1]["body"]
    assert "**Verdict: Approved**" in posted
    assert "**Blocking findings**" not in posted
    assert f"- Outside the deliverable: tools/sass.py: {TOOLING_FINDING.finding}" in posted
    assert f"- {NOTE}" in posted
    assert "outside the declared deliverable" in posted
    assert "## Deliverable\n\n- `model.py`\n- `docs/*.md`" in prompts[0]
    assert "Give every finding the repository path it concerns" in prompts[0]


def test_findings_on_the_deliverable_still_block(merge_case, monkeypatch):
    merge_case.action = merge_case.action.model_copy(
        update={"deliverable_paths": ("model.py",)},
    )
    monkeypatch.setattr(runner, "run_openhands", rejecting(FINDING, TOOLING_FINDING))

    result = execute(merge_case)

    assert result["state"] == "merge_blocked"
    assert merge_case.fake.pr["merged"] is False
    posted = merge_case.fake.comments[-1]["body"]
    assert "**Verdict: Rejected**" in posted
    assert f"- model.py: {FINDING_TEXT}" in posted
    assert f"- Outside the deliverable: tools/sass.py: {TOOLING_FINDING.finding}" in posted


@pytest.mark.parametrize(
    ("prior_rounds", "limit", "merged"),
    [(1, 2, False), (2, 2, True), (3, 2, True), (2, 3, False)],
    ids=["below-cap", "at-cap", "above-cap", "raised-cap"],
)
def test_blocking_review_rounds_are_capped_per_pr(
    merge_case, monkeypatch, prior_rounds, limit, merged,
):
    """One PR took 14 revisions and 17 hours after its science was accepted."""

    merge_case.fake.comments.extend(
        rejected_review_comment(index) for index in range(1, prior_rounds + 1)
    )
    merge_case.config = replace(merge_case.config, max_blocking_review_rounds=limit)
    monkeypatch.setattr(runner, "run_openhands", rejecting(FINDING))

    result = execute(merge_case)

    posted = merge_case.fake.comments[-1]["body"]
    assert merge_case.fake.pr["merged"] is merged
    if merged:
        assert result["state"] == "experiment_merged"
        assert "**Verdict: Approved**" in posted
        assert f"{prior_rounds} blocking review rounds, the configured limit of {limit}" in posted
        assert f"- After {prior_rounds} blocking rounds: model.py: {FINDING_TEXT}" in posted
    else:
        assert result["state"] == "merge_blocked"
        assert "**Verdict: Rejected**" in posted
        assert f"- model.py: {FINDING_TEXT}" in posted


def test_safety_findings_block_regardless_of_scope_and_rounds(merge_case, monkeypatch):
    merge_case.fake.comments.extend(rejected_review_comment(index) for index in (1, 2))
    merge_case.action = merge_case.action.model_copy(
        update={"deliverable_paths": ("model.py",)},
    )
    monkeypatch.setattr(runner, "run_openhands", rejecting(SAFETY_FINDING))

    result = execute(merge_case)

    assert result["state"] == "merge_blocked"
    assert merge_case.fake.pr["merged"] is False
    assert f"- tools/sass.py: {SAFETY_FINDING.finding}" in merge_case.fake.comments[-1]["body"]


def test_default_round_cap_is_two():
    assert runner.DEFAULT_MAX_BLOCKING_REVIEW_ROUNDS == 2
    assert MergeExperimentAction.model_fields["deliverable_paths"].default == ()


def test_worker_reconciles_a_merge_completed_during_review(merge_case, monkeypatch):
    def review(_prompt, _config, *, response_schema, on_structured_result):
        merge_case.fake.pr.update(merged=True, state="closed", merge_commit_sha="merge-sha")
        on_structured_result(APPROVED)
        return 0

    monkeypatch.setattr(runner, "run_openhands", review)

    result = execute(merge_case)

    assert result["state"] == "experiment_merged"
    assert result["version"] == "merge-sha"
    assert result["changed"] is False
    assert merge_case.fake.mutations == []


@pytest.mark.parametrize(
    ("verdict", "status", "remaining", "reason"),
    [
        (REJECTED, 0, 600, FINDING_TEXT),
        (None, 1, 600, "did not complete"),
        (RuntimeError("provider unavailable"), None, 600, "provider unavailable"),
        (None, None, -1, "insufficient time remains"),
    ],
    ids=("rejected", "unfinished", "runtime-failure", "expired"),
)
def test_only_review_findings_are_posted_without_waking_the_student(
    merge_case, monkeypatch, verdict, status, remaining, reason,
):
    merge_case.config = replace(
        merge_case.config, delegation_deadline_epoch=time.time() + remaining,
    )

    def review(_prompt, config, *, response_schema, on_structured_result):
        assert remaining > 0
        assert config.delegation_deadline_epoch <= (
            merge_case.config.delegation_deadline_epoch - 60
        )
        if isinstance(verdict, Exception):
            raise verdict
        if verdict is not None:
            on_structured_result(verdict)
        return status

    monkeypatch.setattr(runner, "run_openhands", review)

    result = execute(merge_case)

    assert result["state"] in {"merge_blocked", "merge_failed"}
    assert reason in result["reason"]
    assert merge_case.fake.pr["merged"] is False
    if verdict is REJECTED:
        body = merge_case.fake.comments[-1]["body"]
        assert reason in body
        assert body.split("\n\n", 1)[1].startswith("**Supervisor**\n\n---\n\n")
        assert "Senpai Supervisor — merge review" in body
        assert "**Verdict: Rejected**" in body
        assert result["feedback_url"] == merge_case.fake.comments[-1]["html_url"]
        assert [(method, path) for method, path, _body in merge_case.fake.mutations] == [
            ("POST", f"/repos/{REPO}/issues/7/comments"),
        ]
        mailbox = GitHubMailbox(
            repo=REPO, token=SecretStr("private-github-key"), role="student",
            advisor_branch="schmidhuber", student_name="student-one",
            trusted_actor="senpai-bot", human_issues_enabled=False,
        )
        pull = merge_case.fake.request(
            "GET", f"https://api.github.com/repos/{REPO}/pulls/7", headers={},
        ).json_body
        pull.update(
            user={"login": "senpai-bot"}, updated_at="2026-07-29T18:00:00Z",
            comments_url=f"https://api.github.com/repos/{REPO}/issues/7/comments",
        )
        pull["head"]["repo"] = {"full_name": REPO}
        for entry in merge_case.fake.comments:
            entry["created_at"] = "2026-07-29T18:01:00Z"
        monkeypatch.setattr(mailbox, "_pulls", lambda: [pull])
        monkeypatch.setattr(mailbox._github, "get", lambda _path: {"permission": "write"})
        monkeypatch.setattr(
            mailbox._github, "objects",
            lambda url: merge_case.fake.request("GET", url, headers={}).json_body,
        )
        assert mailbox.poll() == ()
    else:
        assert merge_case.fake.mutations == []
        assert "feedback_url" not in result


@pytest.mark.parametrize("verdict", [APPROVED, REJECTED], ids=["approved", "rejected"])
@pytest.mark.parametrize("feedback_failure", ["permission", "head-moved"])
def test_failed_feedback_preserves_review_reason_and_does_not_comment_on_new_head(
    merge_case, monkeypatch, feedback_failure, verdict,
):
    original_request = merge_case.fake.request

    def request(method, url, **kwargs):
        if method == "POST" and url.endswith("/issues/7/comments"):
            return HttpResponse(403, {})
        return original_request(method, url, **kwargs)

    if feedback_failure == "permission":
        monkeypatch.setattr(merge_case.fake, "request", request)

    def review(_prompt, _config, *, response_schema, on_structured_result):
        if feedback_failure == "head-moved":
            merge_case.fake.pr["head_sha"] = "d" * 40
        on_structured_result(verdict)
        return 0

    monkeypatch.setattr(runner, "run_openhands", review)

    result = execute(merge_case)

    assert result["state"] in {"merge_blocked", "merge_failed"}
    if verdict is REJECTED:
        assert FINDING_TEXT in result["reason"]
        assert "feedback_error" in result
    assert len(merge_case.fake.comments) == 1
    assert merge_case.fake.pr["merged"] is False


@pytest.mark.parametrize(
    ("response_kind", "review_summary"),
    [
        ("finish", "Focused change."),
        ("finish", "Reviewed. " * 15_001),
        ("plain-final", "Focused change."),
        ("malformed-finish", "Focused change."),
        ("inconsistent-finish", "Focused change."),
    ],
    ids=(
        "concise-verdict", "oversized-verdict", "plain-final", "malformed-finish",
        "inconsistent-finish",
    ),
)
def test_worker_requires_native_structured_review_in_one_private_conversation(
    merge_case, tmp_path, monkeypatch, capsys, response_kind, review_summary,
):
    program_content = "# Target programme\n\nResearch agents must run paired ablations.\n"
    environment = runtime_env(
        tmp_path, program_path="research/program.md", program_content=program_content,
    )
    credentials = {name: environment.pop(name) for name in (
        "GITHUB_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    )}
    credentials["GITHUB_TOKEN"] = "private-github-key"
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    read_fd, write_fd = os.pipe()
    with os.fdopen(write_fd, "w") as stream:
        json.dump(credentials, stream)
    monkeypatch.setenv("SENPAI_MODEL_CREDENTIALS_FD", str(read_fd))
    monkeypatch.setenv("SENPAI_DELEGATION_TASK_ID", "merge-task")
    private_environment = {
        "SENPAI_DELEGATION_TASK_ID": "merge-task",
        "SENPAI_DELEGATION_REGISTRY_PATH": str(tmp_path / "tasks.sqlite3"),
        "SENPAI_DELEGATION_EVENT_DB_PATH": str(tmp_path / "events.sqlite3"),
    }
    for name, value in private_environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SENPAI_DELEGATION_ROOT_STATE_DIR", str(tmp_path / "advisor-state"))
    monkeypatch.setenv("SENPAI_MERGE_REQUEST_JSON", merge_case.action.model_dump_json())
    inherited_stdin = StringIO(ADVISOR_HISTORY_SENTINEL)
    monkeypatch.setattr("sys.stdin", inherited_stdin)
    published = []
    inspected = []
    prompts = []
    verdict = {**APPROVED.model_dump(), "review_summary": review_summary}
    should_merge = response_kind == "finish"

    def record(task_id, **kwargs):
        assert merge_case.fake.pr["merged"] is should_merge
        assert all(kwargs["env"].get(name) == value for name, value in private_environment.items())
        published.append((task_id, kwargs))

    class ReviewConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.IDLE,
                view=SimpleNamespace(events=[]),
            )
            assert "GITHUB_TOKEN" not in kwargs["secrets"]
            assert "GITHUB_TOKEN" not in os.environ
            assert "SENPAI_MODEL_CREDENTIALS_FD" not in os.environ
            assert not (private_environment.keys() & os.environ.keys())
            assert "senpai_github" not in {tool.name for tool in kwargs["agent"].tools}
            agent = kwargs["agent"]
            inspected.append(agent.llm.model)
            suffix = agent.agent_context.system_message_suffix
            assert "research/program.md" in suffix
            assert program_content.strip() in suffix
            assert "independent Supervisor" in suffix
            assert "does not assign you the advisor or student role" in suffix
            specs = [tool for tool in agent.tools if tool.name == "FinishTool"]
            assert len(specs) == 1
            assert "FinishTool" not in agent.include_default_tools
            self.finish = resolve_tool(specs[0], self.state)[0]
            schema = self.finish.to_openai_tool()["function"]["parameters"]
            assert {"approved", "review_summary", "findings", "notes"} <= set(
                schema["required"]
            )
            fields = schema["properties"]
            assert fields["approved"]["type"] == "boolean"
            assert "complete" in fields["approved"]["description"]
            assert "actionable" in fields["review_summary"]["description"]
            assert "file or line" in fields["findings"]["description"]
            assert "Non-blocking" in fields["notes"]["description"]
            self.agent = SimpleNamespace(tools_map={self.finish.name: self.finish})

        def send_message(self, prompt):
            prompts.append(prompt)
            assert ADVISOR_HISTORY_SENTINEL not in prompt
            assert merge_case.fake.pr["body"] in prompt
            assert merge_case.fake.comments[0]["body"] in prompt
            assert "guides the research agents, not the reviewer" in prompt
            assert "Scientific validity is outside the scope of this review" in prompt

        async def arun(self):
            if response_kind == "plain-final":
                self.state.view.events.append(MessageEvent(
                    source="agent",
                    llm_message=Message(
                        role="assistant", content=[TextContent(text=json.dumps(verdict))],
                    ),
                ))
            else:
                arguments = {"message": "Review complete.", **verdict}
                if response_kind == "malformed-finish":
                    arguments["approved"] = "true"
                elif response_kind == "inconsistent-finish":
                    arguments["findings"] = [FINDING.model_dump()]
                action = self.finish.action_from_arguments(arguments)
                tool_call = MessageToolCall(
                    id="review-finish", name=self.finish.name,
                    arguments=json.dumps(arguments), origin="completion",
                )
                self.state.view.events.append(ActionEvent(
                    thought=[], action=action, tool_name=self.finish.name,
                    tool_call_id=tool_call.id, tool_call=tool_call,
                    llm_response_id="review-response",
                ))
            self.state.execution_status = ConversationExecutionStatus.FINISHED

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", ReviewConversation)

    def reject_extra_generation(*_args, **_kwargs):
        pytest.fail("structured review triggered an extra completion or text compaction")

    monkeypatch.setattr(runner, "compact_child_result", reject_extra_generation)
    monkeypatch.setattr(runner.LLM, "completion", reject_extra_generation)
    monkeypatch.setattr(runner.LLM, "responses", reject_extra_generation)
    monkeypatch.setattr(runner, "record_delegated_task_result", record)
    monkeypatch.setattr(merge_worker, "record_delegated_task_result", record)
    monkeypatch.setattr(merge_worker, "finish_weave_monitoring", lambda: None)

    status = merge_worker.main([
        "--child", "--agent", "supervisor", "--max-turns", "1",
        "--model", "anthropic/claude-opus-5-5", "--reasoning-effort", "xhigh",
        "--workspace", str(merge_case.config.workspace),
        "--state-dir", str(tmp_path / "worker-state"),
        "--conversation-id", str(uuid.uuid4()),
    ])

    assert status == 0
    assert inspected == ["anthropic/claude-opus-5-5"]
    assert len(prompts) == 1
    assert inherited_stdin.tell() == 0
    assert len(published) == 1
    assert published[0][0] == "merge-task"
    outcome = json.loads(published[0][1]["result"])
    if should_merge:
        assert outcome["state"] == "experiment_merged"
        posted = merge_case.fake.comments[-1]["body"]
        assert "**Verdict: Approved**" in posted
        assert len(posted) <= 65_536
        if len(review_summary) > 60_000:
            assert "[Review text truncated" in posted
        else:
            assert review_summary in posted
    else:
        assert outcome["state"] in {"merge_blocked", "merge_failed"}
        reason = {
            "plain-final": "required structured response",
            "malformed-finish": "valid boolean",
            "inconsistent-finish": "an approval must have no blocking findings",
        }[response_kind]
        assert reason in outcome["reason"]
        assert merge_case.fake.mutations == []
    output = capsys.readouterr().out
    final_record = json.loads(next(
        line.removeprefix("OPENHANDS_RESULT ")
        for line in reversed(output.splitlines())
        if line.startswith("OPENHANDS_RESULT ")
    ))
    assert json.loads(final_record["result"]) == outcome
    assert "private-github-key" not in output
