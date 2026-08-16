# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import json

import pytest

from model_test_support import HEAD_SHA, assignment, result
from senpai_agent.models import (
    ResearchBaseAcceptanceRecord,
    ResultMarkerError,
    ResultStatus,
    WandbRunRef,
    authoritative_marker_line,
    experiment_result_digest,
    parse_assignment_markers,
    parse_research_base_acceptance_markers,
    parse_result_markers,
    render_assignment_marker,
    render_research_base_acceptance_marker,
    render_result_comment,
    render_result_marker,
)


@pytest.mark.parametrize("separator", ["\n", "\r", "\x85", "\u2028"])
def test_authoritative_marker_line_uses_parser_line_boundaries(separator: str):
    marker = render_result_marker(result())

    assert authoritative_marker_line(f"Visible prose{separator}{marker}") == (
        "Visible prose"
    )


def test_assignment_marker_round_trips_as_one_line():
    original = assignment()
    marker = render_assignment_marker(original)

    assert marker.startswith("<!-- senpai-assignment:v1 {")
    assert "\n" not in marker
    assert parse_assignment_markers(marker) == (original,)


@pytest.mark.parametrize(
    "line",
    [
        "<!-- senpai-assignment:v2 {} -->",
        "<!-- senpai-assignment:v1 not-json -->",
        "<!-- senpai-assignment:v1 {} --> trailing",
        "<!-- senpai-assignment:v1 {} -->",
    ],
)
def test_assignment_parser_rejects_malformed_or_invalid_markers(line: str):
    with pytest.raises(ValueError, match="assignment marker"):
        parse_assignment_markers(line)


def test_assignment_parser_ignores_marker_examples_in_prose():
    marker = render_assignment_marker(assignment())

    assert parse_assignment_markers(f"example: {marker}\n> {marker}") == ()


def test_research_base_acceptance_marker_round_trips_as_one_line():
    acceptance = ResearchBaseAcceptanceRecord(
        repo="acme/widgets",
        pr_number=17,
        assignment_id="assignment-17",
        revision_id="revision-2",
        result_head_sha="a" * 40,
        result_digest=experiment_result_digest(result()),
        evaluated_base_sha="b" * 40,
        base_ref="research",
        accepted_base_sha="c" * 40,
    )

    marker = render_research_base_acceptance_marker(acceptance)

    assert "\n" not in marker
    assert marker.startswith("<!-- senpai-research-base-acceptance:v1 {")
    assert parse_research_base_acceptance_markers(marker) == (acceptance,)


def test_result_digest_is_canonical_and_binds_every_result_field():
    original = result()

    assert experiment_result_digest(original) == experiment_result_digest(
        original.model_copy(deep=True)
    )
    assert experiment_result_digest(original) != experiment_result_digest(
        result(summary="A different terminal conclusion.")
    )


@pytest.mark.parametrize(
    "line",
    [
        "<!-- senpai-research-base-acceptance:v2 {} -->",
        "<!-- senpai-research-base-acceptance:v1 not-json -->",
        "<!-- senpai-research-base-acceptance:v1 {} --> trailing",
        "<!-- senpai-research-base-acceptance:v1 {} -->",
    ],
)
def test_research_base_acceptance_parser_rejects_invalid_markers(line: str):
    with pytest.raises(ValueError, match="research-base acceptance marker"):
        parse_research_base_acceptance_markers(line)


def test_result_marker_is_canonical_and_round_trips():
    original = result()
    marker = render_result_marker(original)
    payload = marker.removeprefix("<!-- senpai-result:v1 ").removesuffix(" -->")

    assert "\n" not in marker
    assert payload == json.dumps(
        json.loads(payload), sort_keys=True, separators=(",", ":")
    )
    assert parse_result_markers(marker) == (original,)


def test_result_marker_escapes_comment_terminators_without_changing_text():
    original = result(summary="The literal --> remains valid result text.")
    marker = render_result_marker(original)

    assert marker.count("-->") == 1
    assert r"The literal --\u003e remains valid result text." in marker
    assert parse_result_markers(marker) == (original,)


def test_visible_result_comment_contains_status_commit_summary_and_runs():
    runs = (
        WandbRunRef(
            run_id="run-123",
            url="https://wandb.ai/acme/widgets/runs/run-123",
            state="finished",
        ),
        WandbRunRef(
            run_id="run-456",
            url="https://wandb.ai/acme/widgets/runs/run-456",
            state="failed",
        ),
    )
    experiment_result = result(runs=runs)

    assert render_result_comment(experiment_result).splitlines() == [
        render_result_marker(experiment_result),
        "",
        "## Experiment result",
        "",
        "- **Status:** `succeeded`",
        "- **Student:** `student-one`",
        (
            f"- **Commit:** [`{HEAD_SHA[:12]}`]"
            f"(https://github.com/acme/widgets/commit/{HEAD_SHA})"
        ),
        "",
        "### Hypothesis",
        "",
        "Lowering the learning rate improves generalization.",
        "",
        "### Summary",
        "",
        "The candidate improved validation loss.",
        "",
        "### Primary metric",
        "",
        "- **Metric:** `validation/loss`",
        "- **Direction:** `minimize`",
        "- **Baseline:** `0.42`",
        "- **Candidate:** `0.38`",
        "- **Delta:** `-0.04`",
        "",
        "### W&B runs",
        "",
        "- **run-123** — `finished` — https://wandb.ai/acme/widgets/runs/run-123",
        "- **run-456** — `failed` — https://wandb.ai/acme/widgets/runs/run-456",
    ]
    assert parse_result_markers(render_result_comment(experiment_result)) == (
        experiment_result,
    )


def test_result_comment_preserves_markdown_summary_and_omits_empty_sections():
    summary = "\n".join(
        [
            "**Outcome:** The candidate is ready for review.",
            "",
            "**Validation:**",
            "- `pytest -q`",
            "- `git diff --check`",
        ]
    )
    experiment_result = result(summary=summary, runs=()).model_copy(
        update={"primary_metric": None}
    )

    body = render_result_comment(experiment_result)

    assert f"### Summary\n\n{summary}" in body
    assert "### Primary metric" not in body
    assert "### W&B runs" not in body
    assert parse_result_markers(body) == (experiment_result,)


def test_result_comment_quotes_protocol_marker_lines_from_visible_fields():
    hypothesis_marker = "<!-- senpai-assignment:v2 {} -->"
    result_marker = "  <!-- senpai-result:v2 {} -->"
    response_marker = "\t<!-- senpai-human-response:student:fern:700 -->"
    runs = (
        WandbRunRef(
            run_id="run-123",
            url=f"https://wandb.ai/acme/widgets/runs/run-123\n{response_marker}",
            state="finished",
        ),
    )
    experiment_result = result(
        summary=f"The run completed.\n{result_marker}\nEvidence follows.",
        runs=runs,
    ).model_copy(
        update={
            "hypothesis": (
                f"Test the bounded candidate.\n{hypothesis_marker}"
            )
        }
    )

    body = render_result_comment(experiment_result)

    assert f"> {hypothesis_marker}" in body.splitlines()
    assert f"> {result_marker}" in body.splitlines()
    assert f"> {response_marker}" in body.splitlines()
    assert parse_result_markers(body) == (experiment_result,)


def test_result_comment_quotes_protocol_markers_in_visible_fields():
    prior_marker = render_result_marker(
        result(
            assignment_id="assignment-prior",
            summary="Prior terminal evidence.",
        )
    )
    current = result(
        summary=f"Compare against the prior record.\n{prior_marker}",
        runs=(
            WandbRunRef(
                run_id="run-123",
                url=f"https://wandb.ai/acme/widgets/runs/run-123\n{prior_marker}",
                state="finished",
            ),
        ),
    )

    body = render_result_comment(current)

    assert body.splitlines().count(f"> {prior_marker}") == 2
    assert parse_result_markers(body) == (current,)


def test_result_parser_preserves_marker_order_and_duplicates():
    first = result()
    second = result(
        assignment_id="assignment-8",
        status=ResultStatus.INCONCLUSIVE,
        summary="The evidence was mixed.",
    )
    body = "\n".join(
        [
            "Visible introduction.",
            render_result_marker(first),
            render_result_marker(second),
            render_result_marker(first),
        ]
    )

    assert parse_result_markers(body) == (first, second, first)


def test_result_parser_ignores_marker_examples_in_prose_or_quotes():
    marker = render_result_marker(result())
    body = "\n".join(
        [
            f"Read this example: {marker}",
            f"> {marker}",
            f" {marker}",
            "`<!-- senpai-result:v1 {} -->`",
        ]
    )

    assert parse_result_markers(body) == ()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("<!-- senpai-result:v2 {} -->", "unknown senpai result marker version"),
        ("<!-- senpai-result:v1 not-json -->", "malformed senpai result marker"),
        ("<!-- senpai-result:v1 {} --> trailing", "malformed senpai result marker"),
        ("<!-- senpai-result:v1 {invalid} -->", "invalid JSON"),
        (
            '<!-- senpai-result:v1 {"schema_version":1} -->',
            "invalid senpai result payload",
        ),
    ],
)
def test_result_parser_reports_malformed_marker_categories(
    line: str,
    message: str,
):
    with pytest.raises(ResultMarkerError, match=message):
        parse_result_markers(line)


def test_result_parser_rejects_an_unknown_payload_schema_version():
    marker = render_result_marker(result()).replace(
        '"schema_version":1', '"schema_version":2'
    )

    with pytest.raises(ResultMarkerError, match="invalid senpai result payload"):
        parse_result_markers(marker)
