import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = (
    Path(__file__).parents[1]
    / "plugins"
    / "senpai"
    / "skills"
    / "exa-search"
    / "scripts"
    / "search_exa.py"
)
SPEC = importlib.util.spec_from_file_location("exa_search", SCRIPT)
assert SPEC and SPEC.loader
exa_search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exa_search)


class FakeClient:
    def __init__(self, response=None):
        self.response = response or empty_response()
        self.calls = []

    def search(self, query, **options):
        self.calls.append((query, options))
        return self.response


def empty_response():
    return SimpleNamespace(results=[], search_time=None, cost_dollars=None)


def result(**updates):
    values = {
        "title": "Fourier Neural Operator",
        "url": "https://example.com/fno",
        "id": "publication:fno",
        "published_date": "2020-10-23",
        "author": "Zongyi Li",
        "score": None,
        "highlights": ["We propose a Fourier neural operator."],
        "summary": None,
        "text": "full paper text must not be emitted",
    }
    values.update(updates)
    return SimpleNamespace(**values)
