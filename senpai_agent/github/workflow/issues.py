"""Open research questions and reply to trusted human issue messages."""

import json
import re
from collections.abc import Callable, Mapping
from hashlib import sha256
from urllib.parse import quote, urlencode

from senpai_agent.github.workflow.errors import (
    GitHubAPIError,
    GitHubTransportError,
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import (
    CreatedIssueResponse,
    GitHubAuthor,
    MutationResult,
    validated_response,
)
from senpai_agent.github.workflow.text import marker_body, role_prefixed_comment
from senpai_agent.github.workflow.validation import (
    positive_message_id,
    positive_number,
    require_trusted_human_message,
    validate_labels,
)
from senpai_agent.models import authoritative_marker_line


class HumanIssueMixin:
    __slots__ = ()

    def create_human_issue(
        self,
        *,
        issue_id: str,
        title: str,
        body: str,
        audience_label: str,
        creator: str,
    ) -> MutationResult:
        """Open one routed research question; exact retries reuse its issue."""

        title, body, issue_id = title.strip(), body.strip(), issue_id.strip()
        if not title or not body or not issue_id:
            raise ValueError("issue ID, title, and body must not be empty")
        validate_labels({audience_label})
        creator_key = self._issue_responder_key(creator)
        prefix = (
            f"<!-- senpai-human-issue:{quote(audience_label, safe='')}:"
            f"{creator_key}:{quote(issue_id, safe='')}:"
        )
        digest = sha256(json.dumps([title, body]).encode()).hexdigest()
        marker = f"{prefix}{digest} -->"

        def validate_existing(content: str) -> None:
            if authoritative_marker_line(content) != marker:
                raise WorkflowPreconditionError(
                    "issue_id already belongs to a different question; "
                    "reuse the original content or choose a new issue_id"
                )

        def render() -> str:
            mentions = self._token_owner_mention()
            content = f"{body}\n\n{mentions}" if mentions else body
            return role_prefixed_comment(marker_body(marker, content), self._role)

        changed, issue = self._create_marked_issue(
            prefix=prefix,
            title=title,
            labels={"human", audience_label},
            render_body=render,
            validate_existing=validate_existing,
        )
        return MutationResult(changed, issue.html_url, "human_issue_created", issue_id)

    def _create_marked_issue(
        self,
        *,
        prefix: str,
        title: str,
        labels: set[str],
        render_body: Callable[[], str],
        validate_existing: Callable[[str], None],
    ) -> tuple[bool, CreatedIssueResponse]:
        with self.serialized_assignment_mutation():
            existing = self._created_marked_issue(prefix)
            if existing is not None:
                validate_existing(existing.body or "")
                return False, existing
            body = render_body()
            self._mutate(
                "POST",
                f"/repos/{self._repo}/issues",
                json_body={"title": title, "body": body, "labels": sorted(labels)},
                expected_statuses={201},
            )
            created = self._created_marked_issue(prefix)
            if (
                created is None
                or created.title != title
                or created.body != body
                or not labels.issubset(label.name for label in created.labels)
            ):
                raise ReconciliationError("GitHub did not create the requested issue")
            return True, created

    def _update_issue(self, number: int, fields: Mapping[str, object]) -> None:
        path = f"/repos/{self._repo}/issues/{positive_number(number)}"
        self._mutate("PATCH", path, json_body=dict(fields), expected_statuses={200})
        saved = self._request("GET", path, expected_statuses={200}).json_body
        if any(saved.get(name) != value for name, value in fields.items()):
            raise ReconciliationError(
                f"GitHub did not persist requested changes to issue #{number}"
            )

    def _created_marked_issue(self, prefix: str) -> CreatedIssueResponse | None:
        actor = self._actor()
        query = urlencode({"state": "all", "creator": actor, "per_page": 100})
        matches = []
        for item in self._objects(f"/repos/{self._repo}/issues?{query}"):
            if item.get("pull_request") is not None:
                continue
            issue = validated_response(CreatedIssueResponse, item, "human issue")
            if (
                issue.user.login.casefold() == actor.casefold()
                and authoritative_marker_line(issue.body or "").startswith(prefix)
            ):
                matches.append(issue)
        if len(matches) > 1:
            raise ReconciliationError(
                "GitHub contains multiple issues for this issue_id"
            )
        return matches[0] if matches else None

    def _token_owner_mention(self) -> str:
        """Mention a user token's owner; optional discovery never blocks a write."""

        try:
            response = self._request("GET", "/user", expected_statuses={200})
            owner = validated_response(
                GitHubAuthor, response.json_body, "token owner"
            )
        except (GitHubAPIError, GitHubTransportError, ReconciliationError):
            return ""
        if (
            owner.type != "User"
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,39}", owner.login)
        ):
            return ""
        return f"@{owner.login}"

    def _issue_responder_key(self, responder: str) -> str:
        responder = responder.strip()
        if self._role == "advisor":
            if responder != "advisor":
                raise ValueError("advisor responder must be 'advisor'")
            return responder
        if not responder:
            raise ValueError("student responder must not be empty")
        return f"student:{quote(responder, safe='')}"

    def respond_to_issue(
        self,
        number: int,
        *,
        human_message_id: int,
        audience_labels: set[str],
        responder: str,
        response: str,
    ) -> MutationResult:
        """Reply once to one verified human-authored GitHub issue message."""

        with self.serialized_assignment_mutation():
            return self._respond_to_issue(
                number,
                human_message_id=human_message_id,
                audience_labels=audience_labels,
                responder=responder,
                response=response,
            )

    def _respond_to_issue(
        self,
        number: int,
        *,
        human_message_id: int,
        audience_labels: set[str],
        responder: str,
        response: str,
    ) -> MutationResult:
        number = positive_number(number)
        human_message_id = positive_message_id(human_message_id)
        body = response.strip()
        if not body:
            raise ValueError("response must not be empty")
        validate_labels(audience_labels)
        if not audience_labels:
            raise ValueError("audience_labels must not be empty")
        responder_key = self._issue_responder_key(responder)

        issue = self._human_issue(number, audience_labels=audience_labels)
        source = self._human_message(
            number,
            issue=issue,
            human_message_id=human_message_id,
        )
        require_trusted_human_message(
            author=source.author,
            author_type=source.author_type,
            association=source.author_association,
            body=source.body,
            actor=self._actor(),
        )

        marker = f"<!-- senpai-human-response:{responder_key}:{human_message_id} -->"
        senpai_opened = (
            issue.user.login.casefold() == self._actor().casefold()
            and authoritative_marker_line(issue.body or "").startswith(
                "<!-- senpai-human-issue:"
            )
        )
        if not senpai_opened:
            first_marker = self._first_issue_reply_marker(number)
            if first_marker == marker:
                mentions = self._token_owner_mention()
                if mentions:
                    body = f"{body}\n\n{mentions}"
        comment_body = marker_body(marker, body)
        changed, verified = self._upsert_marker_comment(
            number,
            marker=marker,
            body=comment_body,
        )
        # Elect the first reply only after GitHub has persisted it. Independent
        # pods may all observe no replies before posting their own comments.
        if (
            not senpai_opened
            and first_marker is None
            and self._first_issue_reply_marker(number) == marker
        ):
            mentions = self._token_owner_mention()
            if mentions:
                mentioned, verified = self._upsert_marker_comment(
                    number,
                    marker=marker,
                    body=marker_body(marker, f"{body}\n\n{mentions}"),
                )
                changed = changed or mentioned
        self._human_issue(number, audience_labels=audience_labels)
        return MutationResult(
            changed=changed,
            resource_url=verified.url,
            state="issue_response_upserted",
            version=str(human_message_id),
        )

    def _first_issue_reply_marker(self, number: int) -> str | None:
        actor = self._actor().casefold()
        first = min(
            (
                comment
                for comment in self._comments(number)
                if comment.author.casefold() == actor
                and authoritative_marker_line(comment.body).startswith(
                    "<!-- senpai-human-response:"
                )
            ),
            key=lambda comment: comment.id,
            default=None,
        )
        return authoritative_marker_line(first.body) if first is not None else None
