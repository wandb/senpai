"""Reply exactly once to a trusted human issue message."""

from senpai_agent.github.workflow.errors import WorkflowPreconditionError
from senpai_agent.github.workflow.responses import MutationResult
from senpai_agent.github.workflow.text import marker_body
from senpai_agent.github.workflow.validation import (
    positive_message_id,
    positive_number,
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
        response: str,
    ) -> MutationResult:
        """Reply once to one verified human-authored GitHub issue message."""

        number = positive_number(number)
        human_message_id = positive_message_id(human_message_id)
        body = response.strip()
        if not body:
            raise ValueError("response must not be empty")
        validate_labels(audience_labels)
        if not audience_labels:
            raise ValueError("audience_labels must not be empty")

        issue = self._human_issue(number, audience_labels=audience_labels)
        source_author = self._human_message_author(
            number,
            issue=issue,
            human_message_id=human_message_id,
        )
        if source_author.casefold() == self._actor().casefold():
            raise WorkflowPreconditionError(
                "human message must not be authored by the authenticated actor"
            )

        marker = f"<!-- senpai-human-response:{human_message_id} -->"
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
