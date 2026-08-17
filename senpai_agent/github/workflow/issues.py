"""Reply exactly once to a trusted human issue message."""

from urllib.parse import quote

from senpai_agent.github.workflow.responses import MutationResult
from senpai_agent.github.workflow.text import marker_body
from senpai_agent.github.workflow.validation import (
    positive_message_id,
    positive_number,
    require_trusted_human_message,
    validate_labels,
)


class HumanIssueMixin:
    __slots__ = ()

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
        responder = responder.strip()
        if self._role == "advisor":
            if responder != "advisor":
                raise ValueError("advisor responder must be 'advisor'")
            responder_key = responder
        else:
            if not responder:
                raise ValueError("student responder must not be empty")
            responder_key = f"student:{quote(responder, safe='')}"

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
        comment_body = marker_body(marker, body)
        changed, verified = self._upsert_marker_comment(
            number,
            marker=marker,
            body=comment_body,
        )
        self._human_issue(number, audience_labels=audience_labels)
        return MutationResult(
            changed=changed,
            resource_url=verified.url,
            state="issue_response_upserted",
            version=str(human_message_id),
        )
