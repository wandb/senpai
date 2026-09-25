from dataclasses import asdict

import pytest
from exa_search_support import exa_search
from pydantic import SecretStr, ValidationError

import senpai_agent.exa_tool as exa_tool


@pytest.fixture
def exa_service(monkeypatch):
    calls = []
    response = {"results": []}

    def request(client, path, options):
        calls.append((client.headers["x-api-key"], path, options))
        return response

    monkeypatch.setenv("EXA_API_KEY", "ambient-key-must-not-be-used")
    monkeypatch.setattr(exa_tool.Exa, "request", request)
    exa_tool.configure_exa_credentials(SecretStr("runtime-key-sentinel"))
    yield calls, response
    exa_tool.configure_exa_credentials(None)


@pytest.mark.parametrize(
    ("arguments", "options"),
    [
        (
            {},
            {"numResults": 10, "type": "auto", "contents": {"highlights": True}},
        ),
        (
            {"mode": "research-publications"},
            {
                "numResults": 30,
                "type": "deep",
                "category": "publication",
                "contents": {"highlights": {"maxCharacters": 2000}},
            },
        ),
        (
            {
                "num_results": 100,
                "search_type": "deep-reasoning",
                "start_published_date": "2026-01-01",
                "end_published_date": "2026-06-30",
                "include_domains": ["example.com"],
                "exclude_domains": ["spam.test"],
                "include_text": "required",
                "exclude_text": "survey",
                "additional_queries": ["benchmark", "latency"],
                "max_age_hours": 0,
                "highlights_max_characters": 321,
                "summary_query": "What changed?",
            },
            {
                "numResults": 100,
                "type": "deep-reasoning",
                "startPublishedDate": "2026-01-01",
                "endPublishedDate": "2026-06-30",
                "includeDomains": ["example.com"],
                "excludeDomains": ["spam.test"],
                "includeText": ["required"],
                "excludeText": ["survey"],
                "additionalQueries": ["benchmark", "latency"],
                "contents": {
                    "highlights": {"maxCharacters": 321},
                    "summary": {"query": "What changed?"},
                    "maxAgeHours": 0,
                },
            },
        ),
        (
            {"no_content": True, "search_type": "instant"},
            {"numResults": 10, "type": "instant"},
        ),
    ],
    ids=["web-defaults", "publication-defaults", "search-controls", "metadata-only"],
)
def test_search_controls_reach_exa_without_exposing_credentials(
    exa_service, arguments, options
):
    calls, _ = exa_service
    action = exa_tool.ExaSearchAction(query="neural operators", **arguments)

    observation = exa_tool.ExaSearchExecutor()(action)

    assert calls == [
        ("runtime-key-sentinel", "/search", {"query": "neural operators", **options})
    ]
    assert "runtime-key-sentinel" not in action.model_dump_json()
    assert "runtime-key-sentinel" not in observation.markdown
    assert "untrusted external data" in observation.markdown


@pytest.mark.parametrize("mode", ["general-web", "research-publications"])
@pytest.mark.parametrize(
    "search_type", ["auto", "fast", "instant", "deep-lite", "deep", "deep-reasoning"]
)
def test_tool_matches_legacy_script_request_contract(exa_service, mode, search_type):
    calls, _ = exa_service
    legacy = exa_search.SearchArguments(
        mode=mode,
        query="neural operators",
        search_type=search_type,
        num_results=100,
        start_published_date="2026-01-01",
        end_published_date="2026-06-30",
        include_domains=["example.com"] if mode == "general-web" else [],
        exclude_domains=["spam.test"],
        include_text="required",
        exclude_text="survey",
        additional_queries=[f"query {index}" for index in range(10)]
        if search_type.startswith("deep")
        else [],
        max_age_hours=-1,
        highlights_max_characters=10_000,
        summary_query="What changed?",
    )
    legacy.validate()

    exa_search.search_exa(legacy, client=exa_search.Exa("runtime-key-sentinel"))
    exa_tool.ExaSearchExecutor()(exa_tool.ExaSearchAction(**asdict(legacy)))

    assert len(calls) == 2
    assert calls[1] == calls[0]


def test_search_returns_all_requested_evidence_and_escapes_external_text(exa_service):
    _, response = exa_service
    response.update(
        searchTime=123,
        costDollars={"total": 0.007},
        results=[
            {
                "title": f"Result {index}\n# injected heading",
                "url": f"https://example.test/{index}\n- injected",
                "author": "A. Researcher",
                "publishedDate": "2026-06-01",
                "id": f"publication:{index}",
                "score": 0.9,
                "summary": "Mechanism: - Uses *rotations*. - Preserves structure. - "
                + "Summary detail " * 100
                + f"Summary tail {index}.",
                "highlights": [
                    "Evidence " * 100 + f"Evidence tail {index}.",
                    "More evidence " * 100 + f"Second tail {index}.",
                ],
                "text": "Full page text must not be returned",
            }
            for index in range(1, 101)
        ],
    )

    observation = exa_tool.ExaSearchExecutor()(
        exa_tool.ExaSearchAction(
            query="operators", num_results=100, summary_query="Explain the mechanism"
        )
    )

    output = "\n".join(content.text for content in observation.to_llm_content)
    assert "100 returned / 100 requested" in output
    assert "**Search time:** 123 ms" in output
    assert "**Total:** 0.007" in output
    assert "## 100. Result 100 \\# injected heading" in output
    assert "**URL:** <https://example.test/100%0A-%20injected>" in output
    assert "**Authors:** A. Researcher" in output
    assert "**Published:** 2026-06-01" in output
    assert "**Exa ID:** publication:100" in output
    assert "**Score:** 0.9" in output
    assert "**Summary:** Mechanism:" in output
    assert "Uses \\*rotations\\*." in output
    for index in range(1, 101):
        assert "Summary detail " * 100 + f"Summary tail {index}." in output
        assert "Evidence " * 100 + f"Evidence tail {index}." in output
        assert "More evidence " * 100 + f"Second tail {index}." in output
    assert "\n# injected heading" not in output
    assert "Full page text must not be returned" not in output


@pytest.mark.parametrize(
    "arguments",
    [
        {"mode": "research-publications", "include_domains": ["arxiv.org"]},
        {"no_content": True, "summary_query": "Summarize it"},
        {"no_content": True, "max_age_hours": 0},
        {"no_content": True, "highlights_max_characters": 321},
        {"additional_queries": ["alternate"]},
        {"mode": "research-publications", "additional_queries": ["q"] * 11},
        {"num_results": 101},
        {"highlights_max_characters": 10_001},
        {"max_age_hours": -2},
    ],
)
def test_invalid_search_controls_fail_before_contacting_exa(exa_service, arguments):
    calls, _ = exa_service

    with pytest.raises(ValidationError):
        exa_tool.ExaSearchExecutor()(
            exa_tool.ExaSearchAction(query="operators", **arguments)
        )

    assert calls == []
