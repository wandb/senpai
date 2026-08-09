"""GitHub pull-request retrieval and operational integrations."""

from senpai_agent.github.pull_requests import (
    PRManifestEntry,
    PRRetrievalResult,
    get_prs,
)

__all__ = ["PRManifestEntry", "PRRetrievalResult", "get_prs"]
