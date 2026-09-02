from types import SimpleNamespace

from pydantic import SecretStr

import senpai_agent.exa_tool as exa_tool


def test_exa_search_uses_runtime_credentials_without_returning_them(monkeypatch):
    captured = {}

    class FakeExa:
        def __init__(self, api_key):
            captured["api_key"] = api_key

        def search(self, query, **options):
            captured.update(query=query, options=options)
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        title="External result",
                        url="https://example.test/result",
                        highlights=["Untrusted evidence"],
                    )
                ]
            )

    monkeypatch.setattr(exa_tool, "Exa", FakeExa)
    exa_tool.configure_exa_credentials(SecretStr("exa-secret-sentinel"))
    try:
        observation = exa_tool.ExaSearchExecutor()(
            exa_tool.ExaSearchAction(
                query="secure service boundary",
                include_domains=("example.test",),
            )
        )
    finally:
        exa_tool.configure_exa_credentials(None)

    assert captured["api_key"] == "exa-secret-sentinel"
    assert captured["options"]["include_domains"] == ["example.test"]
    assert "untrusted external data" in observation.markdown
    assert "https://example.test/result" in observation.markdown
    assert "exa-secret-sentinel" not in observation.markdown
