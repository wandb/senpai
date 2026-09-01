"""Stateful GitHub transport used by the workflow transition tests."""

from typing import Literal, cast
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import SecretStr

from senpai_agent.github.workflow import (
    GitHubTransportError,
    GitHubWorkflow,
    HttpResponse,
)
from senpai_agent.models import (
    AssignmentKey,
    AssignmentRecord,
    ExperimentResult,
    MetricComparison,
    ResultStatus,
    WandbRunRef,
    render_assignment_marker,
)

REPO = "acme/widgets"
API_URL = "https://api.github.test"
HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40
ASSIGNMENT_ID = "assignment-7"


def assignment_record(
    *,
    assignment_id: str = ASSIGNMENT_ID,
    revision_id: str = "revision-1",
    student: str = "student-one",
    base_ref: str = "schmidhuber",
    base_sha: str = BASE_SHA,
    head_ref: str = "student-one/lower-lr",
    head_sha: str = HEAD_SHA,
) -> AssignmentRecord:
    return AssignmentRecord(
        repo=REPO,
        assignment_id=assignment_id,
        revision_id=revision_id,
        student=student,
        base_ref=base_ref,
        base_sha=base_sha,
        head_ref=head_ref,
        head_sha=head_sha,
    )


def pull_request(
    *,
    labels: set[str] | None = None,
    draft: bool = False,
    state: str = "open",
    merged: bool = False,
    mergeable: bool | None = True,
    title: str = "Try lower learning rate",
    body: str | None = None,
    base_ref: str = "schmidhuber",
    head_ref: str = "student-one/lower-lr",
    head_sha: str = HEAD_SHA,
) -> dict[str, object]:
    if body is None:
        body = render_assignment_marker(
            assignment_record(base_ref=base_ref, head_ref=head_ref)
        )
    return {
        "number": 7,
        "node_id": "PR_node_7",
        "html_url": f"https://github.com/{REPO}/pull/7",
        "head_sha": head_sha,
        "labels": (
            set(labels) if labels is not None else {"student:one", "status:wip"}
        ),
        "draft": draft,
        "state": state,
        "merged": merged,
        "mergeable": mergeable,
        "merge_commit_sha": "merge-sha" if merged else None,
        "title": title,
        "body": body,
        "base_ref": base_ref,
        "head_ref": head_ref,
    }


def comment(
    comment_id: int,
    body: str,
    *,
    author: str = "senpai-bot",
    author_type: str | None = None,
    association: str = "MEMBER",
) -> dict[str, object]:
    if author_type is None:
        author_type = "Bot" if author.casefold() == "senpai-bot" else "User"
    return {
        "id": comment_id,
        "body": body,
        "user": {"login": author, "type": author_type},
        "author_association": association,
        "html_url": f"https://github.com/{REPO}/pull/7#issuecomment-{comment_id}",
    }


def human_issue(
    *,
    issue_id: int = 700,
    state: str = "open",
    labels: set[str] | None = None,
    author: str = "human-researcher",
    author_type: str = "User",
    association: str = "MEMBER",
    body: str | None = "Please investigate the new result.",
    pull_request_url: str | None = None,
) -> dict[str, object]:
    issue: dict[str, object] = {
        "id": issue_id,
        "number": 7,
        "html_url": f"https://github.com/{REPO}/issues/7",
        "state": state,
        "body": body,
        "labels": [
            {"name": label}
            for label in sorted(labels if labels is not None else {"human", "team"})
        ],
        "user": {"login": author, "type": author_type},
        "author_association": association,
    }
    if pull_request_url is not None:
        issue["pull_request"] = {"url": pull_request_url}
    return issue


def experiment_result(
    *,
    commit_sha: str = HEAD_SHA,
    repo: str = REPO,
    pr_number: int = 7,
    expected_head_sha: str = HEAD_SHA,
) -> ExperimentResult:
    return ExperimentResult(
        assignment=AssignmentKey(
            repo=repo,
            pr_number=pr_number,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-1",
            expected_head_sha=expected_head_sha,
            student="student-one",
        ),
        status=ResultStatus.SUCCEEDED,
        hypothesis="The candidate improves the primary metric.",
        summary="Terminal result with complete W&B evidence.",
        runs=(
            WandbRunRef(
                run_id="run-123",
                url="https://wandb.ai/acme/project/runs/run-123",
                state="finished",
            ),
        ),
        primary_metric=MetricComparison(
            name="val/loss",
            direction="minimize",
            baseline=0.42,
            candidate=0.38,
            delta=-0.04,
        ),
        commit_sha=commit_sha,
    )


class FakeGitHub:
    def __init__(
        self,
        pr: dict[str, object],
        *,
        comments: list[dict[str, object]] | None = None,
        issue: dict[str, object] | None = None,
        comment_page_size: int = 100,
        ignore_label_mutations: bool = False,
        ignore_draft_mutations: bool = False,
        branch_heads: dict[str, str] | None = None,
        actor_login: str = "senpai-bot",
        files: list[dict[str, object]] | None = None,
    ):
        self.pr = pr
        self.comments = list(comments or [])
        self.issue = issue
        self.comment_page_size = comment_page_size
        self.ignore_label_mutations = ignore_label_mutations
        self.ignore_draft_mutations = ignore_draft_mutations
        self.branch_heads = branch_heads or {str(pr["base_ref"]): BASE_SHA}
        self.actor_login = actor_login
        self.files = list(
            files
            if files is not None
            else [{"filename": "model.py", "status": "modified"}]
        )
        self.requests: list[tuple[str, str, object | None, dict[str, str]]] = []

    @property
    def mutations(self) -> list[tuple[str, str, object | None]]:
        return [
            (method, path, body)
            for method, path, body, _headers in self.requests
            if method != "GET"
        ]

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
    ) -> HttpResponse:
        parsed = urlsplit(url)
        path = parsed.path
        self.requests.append((method, path, json_body, dict(headers)))

        pull_path = f"/repos/{REPO}/pulls/7"
        pull_files_path = f"{pull_path}/files"
        pulls_path = f"/repos/{REPO}/pulls"
        issues_path = f"/repos/{REPO}/issues"
        issue_path = f"/repos/{REPO}/issues/7"
        comments_path = f"/repos/{REPO}/issues/7/comments"
        labels_path = f"/repos/{REPO}/issues/7/labels"

        if method == "GET" and path == "/user":
            return HttpResponse(200, {"login": self.actor_login})

        if method == "GET" and path == pull_path:
            return HttpResponse(200, self._pull_payload())

        if method == "GET" and path == pull_files_path:
            page = int(parse_qs(parsed.query).get("page", ["1"])[0])
            start = (page - 1) * 100
            return HttpResponse(200, self.files[start : start + 100])

        ref_prefix = f"/repos/{REPO}/git/ref/heads/"
        if method == "GET" and path.startswith(ref_prefix):
            branch = unquote(path.removeprefix(ref_prefix))
            return HttpResponse(
                200,
                {
                    "ref": f"refs/heads/{branch}",
                    "object": {"sha": self.branch_heads[branch]},
                },
            )

        if method == "GET" and path == issue_path and self.issue is not None:
            return HttpResponse(200, self.issue)

        if method == "GET" and path == pulls_path:
            requested_head = parse_qs(parsed.query).get("head", [":"])[0]
            requested_head = requested_head.split(":", 1)[-1]
            matches = (
                bool(self.pr.get("number"))
                and self.pr["head_ref"] == requested_head
            )
            return HttpResponse(200, [self._pull_payload()] if matches else [])

        if method == "GET" and path == issues_path:
            labels = set(parse_qs(parsed.query).get("labels", [""])[0].split(","))
            pr_labels = cast(set[str], self.pr["labels"])
            issues = []
            if labels.issubset(pr_labels):
                issues.append(
                    {
                        "number": self.pr["number"],
                        "pull_request": {"url": self.pr["html_url"]},
                        "labels": [{"name": label} for label in sorted(pr_labels)],
                    }
                )
            return HttpResponse(200, issues)

        if method == "POST" and path == pulls_path:
            payload = cast(dict[str, object], json_body)
            self.pr.update(
                number=7,
                title=payload["title"],
                body=payload["body"],
                base_ref=payload["base"],
                head_ref=payload["head"],
                draft=payload["draft"],
                state="open",
            )
            return HttpResponse(201, self._pull_payload())

        if method == "GET" and path == comments_path:
            page = int(parse_qs(parsed.query).get("page", ["1"])[0])
            start = (page - 1) * self.comment_page_size
            end = start + self.comment_page_size
            response_headers: tuple[tuple[str, str], ...] = ()
            if end < len(self.comments):
                next_url = f"{API_URL}{comments_path}?per_page=100&page={page + 1}"
                response_headers = (("Link", f'<{next_url}>; rel="next"'),)
            return HttpResponse(200, self.comments[start:end], response_headers)

        if method == "POST" and path == comments_path:
            body = cast(dict[str, str], json_body)["body"]
            created = comment(
                max((int(item["id"]) for item in self.comments), default=0) + 1,
                body,
            )
            self.comments.append(created)
            return HttpResponse(201, created)

        comment_prefix = f"/repos/{REPO}/issues/comments/"
        if method == "PATCH" and path.startswith(comment_prefix):
            comment_id = int(path.removeprefix(comment_prefix))
            body = cast(dict[str, str], json_body)["body"]
            existing = next(item for item in self.comments if item["id"] == comment_id)
            existing["body"] = body
            return HttpResponse(200, existing)

        if method == "POST" and path == labels_path:
            labels = cast(set[str], self.pr["labels"])
            if not self.ignore_label_mutations:
                labels.update(cast(dict[str, list[str]], json_body)["labels"])
            return HttpResponse(
                200,
                [{"name": label} for label in sorted(labels)],
            )

        label_prefix = f"{labels_path}/"
        if method == "DELETE" and path.startswith(label_prefix):
            labels = cast(set[str], self.pr["labels"])
            if not self.ignore_label_mutations:
                labels.discard(unquote(path.removeprefix(label_prefix)))
            return HttpResponse(
                200,
                [{"name": label} for label in sorted(labels)],
            )

        if method == "POST" and path == "/graphql":
            query = cast(str, cast(dict[str, object], json_body)["query"])
            if "convertPullRequestToDraft" in query:
                requested_draft = True
                field = "convertPullRequestToDraft"
            elif "markPullRequestReadyForReview" in query:
                requested_draft = False
                field = "markPullRequestReadyForReview"
            else:
                raise AssertionError(f"Unexpected GraphQL mutation: {query}")
            if not self.ignore_draft_mutations:
                self.pr["draft"] = requested_draft
            return HttpResponse(
                200,
                {
                    "data": {
                        field: {
                            "pullRequest": {
                                "id": self.pr["node_id"],
                                "isDraft": requested_draft,
                            }
                        }
                    }
                },
            )

        if method == "PATCH" and path == pull_path:
            update = cast(dict[str, object], json_body)
            for field in ("state", "title", "body"):
                if field in update:
                    self.pr[field] = update[field]
            return HttpResponse(200, self._pull_payload())

        if method == "PUT" and path == f"{pull_path}/merge":
            self.pr.update(state="closed", merged=True, merge_commit_sha="merge-sha")
            return HttpResponse(
                200,
                {
                    "merged": True,
                    "sha": "merge-sha",
                    "message": "Pull Request successfully merged",
                },
            )

        raise AssertionError(f"Unexpected request: {method} {url} {json_body!r}")

    def _pull_payload(self) -> dict[str, object]:
        return {
            "number": self.pr["number"],
            "node_id": self.pr["node_id"],
            "html_url": self.pr["html_url"],
            "base": {"ref": self.pr["base_ref"]},
            "head": {"sha": self.pr["head_sha"], "ref": self.pr["head_ref"]},
            "title": self.pr["title"],
            "body": self.pr["body"],
            "labels": [
                {"name": label}
                for label in sorted(cast(set[str], self.pr["labels"]))
            ],
            "draft": self.pr["draft"],
            "state": self.pr["state"],
            "merged": self.pr["merged"],
            "mergeable": self.pr["mergeable"],
            "merge_commit_sha": self.pr["merge_commit_sha"],
        }


class AmbiguousMutationGitHub(FakeGitHub):
    def __init__(self, *args, fail_method: str, fail_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_method = fail_method
        self.fail_path = fail_path
        self.failed = False

    def request(self, method, url, *, headers, json_body=None):
        response = super().request(method, url, headers=headers, json_body=json_body)
        if (
            not self.failed
            and method == self.fail_method
            and urlsplit(url).path == self.fail_path
        ):
            self.failed = True
            raise GitHubTransportError(method, url)
        return response


def workflow(
    fake: FakeGitHub,
    *,
    role: Literal["advisor", "student"] = "advisor",
) -> GitHubWorkflow:
    return GitHubWorkflow(
        REPO,
        SecretStr("github-secret"),
        role=role,
        transport=fake,
        api_url=API_URL,
        trusted_actor="senpai-bot",
    )
