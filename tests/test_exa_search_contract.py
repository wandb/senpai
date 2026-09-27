from types import SimpleNamespace

import pytest

from exa_search_support import FakeClient, exa_search, result


def test_installed_sdk_serializes_publication_category(monkeypatch):
    captured = {}
    client = exa_search.Exa("test-key")

    def request(path, options, *, headers=None):
        captured.update(path=path, options=options)
        return {"results": []}

    monkeypatch.setattr(client, "request", request)

    client.search("neural operators", category="publication", contents=False)

    assert captured["path"] == "/search"
    assert captured["options"]["category"] == "publication"


def test_installed_sdk_serializes_general_scope_without_a_category(monkeypatch):
    captured = {}
    client = exa_search.Exa("test-key")

    def request(path, options, *, headers=None):
        captured.update(path=path, options=options)
        return {"results": []}

    monkeypatch.setattr(client, "request", request)

    client.search(
        "OpenHands SDK documentation",
        type="auto",
        num_results=10,
        include_domains=["docs.openhands.dev"],
        contents={"highlights": True, "max_age_hours": 0},
    )

    assert captured == {
        "path": "/search",
        "options": {
            "query": "OpenHands SDK documentation",
            "type": "auto",
            "numResults": 10,
            "includeDomains": ["docs.openhands.dev"],
            "contents": {"highlights": True, "maxAgeHours": 0},
        },
    }


@pytest.mark.parametrize(
    ("argv", "expected_options"),
    [
        (
            ["general-web", "current OpenHands SDK release"],
            {
                "num_results": 10,
                "type": "auto",
                "contents": {"highlights": True},
            },
        ),
        (
            ["research-publications", "neural operators for PDEs"],
            {
                "category": "publication",
                "num_results": 30,
                "type": "deep",
                "contents": {"highlights": {"max_characters": 2000}},
            },
        ),
    ],
    ids=["general-web", "research-publications"],
)
def test_search_modes_apply_their_bounded_defaults(argv, expected_options):
    client = FakeClient()
    args = exa_search.parse_args(argv)

    payload = exa_search.search_exa(args, client)

    assert client.calls == [(argv[1], expected_options)]
    assert payload["mode"] == argv[0]


def test_custom_search_constraints_are_forwarded_together():
    client = FakeClient()
    args = exa_search.parse_args(
        (
            "general-web operators --search-type deep "
            "--start-published-date 2026-01-01 --end-published-date 2026-06-30 "
            "--include-domains example.com --exclude-domains spam.test "
            "--include-text required --exclude-text survey "
            "--additional-queries benchmark latency "
            "--highlights-max-characters 321 --summary-query changes"
        ).split()
    )

    exa_search.search_exa(args, client)

    assert client.calls == [
        (
            "operators",
            {
                "num_results": 10,
                "type": "deep",
                "contents": {
                    "highlights": {"max_characters": 321},
                    "summary": {"query": "changes"},
                },
                "start_published_date": "2026-01-01",
                "end_published_date": "2026-06-30",
                "exclude_domains": ["spam.test"],
                "include_domains": ["example.com"],
                "include_text": ["required"],
                "exclude_text": ["survey"],
                "additional_queries": ["benchmark", "latency"],
            },
        )
    ]


def test_result_serialization_excludes_full_content_and_empty_fields():
    response = SimpleNamespace(
        results=[result()],
        search_time=None,
        cost_dollars=None,
    )
    client = FakeClient(response)
    args = exa_search.parse_args(
        ["research-publications", "neural operators for PDEs"]
    )

    payload = exa_search.search_exa(args, client)

    serialized = payload["results"][0]
    assert serialized["rank"] == 1
    assert serialized["highlights"] == [
        "We propose a Fourier neural operator."
    ]
    assert "text" not in serialized
    assert "summary" not in serialized
    assert "score" not in serialized
