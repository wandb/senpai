"""Open research questions and reply to trusted human issue messages."""

import json
from hashlib import sha256
from urllib.parse import quote, urlencode

from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import (
    CollaboratorResponse,
    CreatedIssueResponse,
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

        with self._assignment_lifecycle_lock:
            existing = self._created_human_issue(prefix)
            if existing is not None:
                if authoritative_marker_line(existing.body or "") != marker:
                    raise WorkflowPreconditionError(
                        "issue_id already belongs to a different question; "
                        "reuse the original content or choose a new issue_id"
                    )
                return MutationResult(
                    False, existing.html_url, "human_issue_created", issue_id
                )

            mentions = self._maintainer_mentions()
            content = f"{body}\n\n{mentions}" if mentions else body
            rendered = role_prefixed_comment(marker_body(marker, content), self._role)
            labels = {"human", audience_label}
            self._mutate(
                "POST",
                f"/repos/{self._repo}/issues",
                json_body={"title": title, "body": rendered, "labels": sorted(labels)},
                expected_statuses={201},
            )
            created = self._created_human_issue(prefix)
            if (
                created is None
                or created.title != title
                or created.body != rendered
                or not labels.issubset(label.name for label in created.labels)
            ):
                raise ReconciliationError(
                    "GitHub did not create the requested human issue"
                )
            return MutationResult(
                True, created.html_url, "human_issue_created", issue_id
            )

    def _created_human_issue(self, prefix: str) -> CreatedIssueResponse | None:
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

    def _maintainer_mentions(self) -> str:
        collaborators = (
            validated_response(CollaboratorResponse, item, "repository collaborator")
            for item in self._objects(
                f"/repos/{self._repo}/collaborators?affiliation=all&per_page=100"
            )
        )
        # push includes write, maintain, admin, and custom roles with write access.
        handles = {
            collaborator.login.casefold()
            for collaborator in collaborators
            if collaborator.type == "User" and collaborator.permissions.push
        }
        return " ".join(f"@{handle}" for handle in sorted(handles))

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

        with self._assignment_lifecycle_lock:
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
                mentions = self._maintainer_mentions()
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
            mentions = self._maintainer_mentions()
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
