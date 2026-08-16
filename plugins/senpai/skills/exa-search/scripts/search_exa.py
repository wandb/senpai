#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Search Exa's web or publication index and emit compact Markdown."""

# ruff: noqa: RUF009 - simple_parsing stores CLI metadata in dataclass fields.

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, cast
from urllib.parse import quote

from dotenv import find_dotenv, load_dotenv
from exa_py import Exa
from simple_parsing import ArgumentParser, DashVariant
from simple_parsing.helpers import field

GENERAL_WEB_NUM_RESULTS = 10
PUBLICATION_NUM_RESULTS = 30
PUBLICATION_HIGHLIGHTS_MAX_CHARACTERS = 2000
SearchMode = Literal["general-web", "research-publications"]
SearchType = Literal[
    "auto",
    "fast",
    "instant",
    "deep-lite",
    "deep",
    "deep-reasoning",
]
DEEP_SEARCH_TYPES: set[SearchType] = {"deep-lite", "deep", "deep-reasoning"}
MARKDOWN_ESCAPES = str.maketrans(
    {character: f"\\{character}" for character in "\\`*_[]<>#"}
)


@dataclass
class SearchArguments:
    """Search Exa in one explicit mode."""

    mode: SearchMode = field(
        positional=True,
        help="general-web or research-publications",
    )

    query: str = field(
        positional=True,
        metavar="QUERY",
        help="natural-language search query",
    )
    num_results: int | None = field(
        default=None,
        nargs=None,
        alias="-n",
        metavar="N",
        help="results to return (default: 10 web, 30 publications)",
    )
    search_type: str | None = field(
        default=None,
        nargs=None,
        choices=[
            "auto",
            "fast",
            "instant",
            "deep-lite",
            "deep",
            "deep-reasoning",
        ],
        help="Exa search type (default: auto web, deep publications)",
    )
    start_published_date: str | None = field(
        default=None,
        nargs=None,
        metavar="DATE",
        help="ISO lower publication date",
    )
    end_published_date: str | None = field(
        default=None,
        nargs=None,
        metavar="DATE",
        help="ISO upper publication date",
    )
    exclude_domains: list[str] = field(
        default_factory=list,
        metavar="DOMAIN [DOMAIN ...]",
        help="exclude these domains",
    )
    include_domains: list[str] = field(
        default_factory=list,
        metavar="DOMAIN [DOMAIN ...]",
        help="restrict results to these domains",
    )
    max_age_hours: int | None = field(
        default=None,
        nargs=None,
        metavar="HOURS",
        help="content cache age; 0 always live-crawls, -1 uses cache only",
    )
    include_text: str | None = field(
        default=None,
        nargs=None,
        metavar="TEXT",
        help="required exact text constraint",
    )
    exclude_text: str | None = field(
        default=None,
        nargs=None,
        metavar="TEXT",
        help="excluded exact text constraint",
    )
    additional_queries: list[str] = field(
        default_factory=list,
        metavar="QUERY [QUERY ...]",
        help="additional queries for deep search",
    )
    highlights_max_characters: int | None = field(
        default=None,
        nargs=None,
        metavar="N",
        help="optional per-result highlight budget (1-10000)",
    )
    summary_query: str | None = field(
        default=None,
        nargs=None,
        metavar="QUESTION",
        help="request a per-result summary focused on this question",
    )
    no_content: bool = field(
        default=False,
        action="store_true",
        help="return metadata only",
    )

    @property
    def resolved_num_results(self) -> int:
        if self.num_results is not None:
            return self.num_results
        if self.mode == "general-web":
            return GENERAL_WEB_NUM_RESULTS
        return PUBLICATION_NUM_RESULTS

    @property
    def resolved_search_type(self) -> SearchType:
        if self.search_type is not None:
            return cast(SearchType, self.search_type)
        if self.mode == "general-web":
            return "auto"
        return "deep"

    def validate(self) -> None:
        if not 1 <= self.resolved_num_results <= 100:
            raise ValueError("--num-results must be between 1 and 100")
        if self.highlights_max_characters is not None and not (
            1 <= self.highlights_max_characters <= 10_000
        ):
            raise ValueError("--highlights-max-characters must be between 1 and 10000")
        if self.max_age_hours is not None and self.max_age_hours < -1:
            raise ValueError("--max-age-hours must be -1 or greater")
        if self.no_content and (
            self.summary_query
            or self.highlights_max_characters is not None
            or self.max_age_hours is not None
        ):
            raise ValueError("--no-content cannot be combined with content options")
        if self.mode == "research-publications" and self.include_domains:
            raise ValueError("--include-domains is only supported for general-web")
        if (
            self.additional_queries
            and self.resolved_search_type not in DEEP_SEARCH_TYPES
        ):
            raise ValueError("--additional-queries requires a deep search type")
        if len(self.additional_queries) > 10:
            raise ValueError("--additional-queries accepts at most 10 queries")


def parse_args(argv: Sequence[str] | None = None) -> SearchArguments:
    parser = ArgumentParser(
        add_option_string_dash_variants=DashVariant.DASH,
        description="Search Exa's general web or publication index.",
    )
    parser.add_arguments(SearchArguments, dest="options")
    args: SearchArguments = parser.parse_args(argv).options
    try:
        args.validate()
    except ValueError as error:
        parser.error(str(error))
    return args


def build_contents(args: SearchArguments) -> dict[str, Any] | bool:
    contents: dict[str, Any] = {}
    if not args.no_content:
        if args.mode == "general-web" and args.highlights_max_characters is None:
            contents["highlights"] = True
        else:
            contents["highlights"] = {
                "max_characters": (
                    args.highlights_max_characters
                    or PUBLICATION_HIGHLIGHTS_MAX_CHARACTERS
                ),
            }
    if args.summary_query:
        contents["summary"] = {"query": args.summary_query}
    if args.max_age_hours is not None:
        contents["max_age_hours"] = args.max_age_hours
    return contents or False


def without_empty(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {
            key: without_empty(item)
            for key, item in value.items()
            if item is not None and item != [] and item != {}
        }
    if isinstance(value, list):
        return [without_empty(item) for item in value]
    return value


def markdown_text(value: Any) -> str:
    return " ".join(str(value).split()).translate(MARKDOWN_ESCAPES)


def markdown_url(value: Any) -> str:
    return quote(
        str(value).strip(),
        safe="/:?#[]@!$&'()*+,;=%",
    )


def render_mapping(mapping: dict[str, Any], indent: int = 2) -> list[str]:
    lines = []
    for key, value in mapping.items():
        label = key.replace("_", " ").title()
        prefix = f"{' ' * indent}- **{label}:**"
        if isinstance(value, dict):
            lines.append(prefix)
            lines.extend(render_mapping(value, indent + 2))
        else:
            lines.append(f"{prefix} {markdown_text(value)}")
    return lines


def render_summary(value: Any) -> list[str]:
    parts = [line.strip() for line in str(value).splitlines() if line.strip()]
    if not parts:
        return []
    if len(parts) == 1 and " - " in parts[0]:
        parts = parts[0].split(" - ")
    parts = [part.removeprefix("- ").strip() for part in parts]
    if parts[0].rstrip(":").casefold() == "summary":
        parts = parts[1:]
    if not parts:
        return []
    lines = [f"- **Summary:** {markdown_text(parts[0])}"]
    lines.extend(f"  - {markdown_text(part)}" for part in parts[1:])
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    publications = payload["mode"] == "research-publications"
    lines = [
        "# Exa Publication Search" if publications else "# Exa Web Search",
        "",
        f"- **Query:** {markdown_text(payload['query'])}",
        f"- **Mode:** {markdown_text(payload['mode'])}",
    ]
    if "category" in payload:
        lines.append(f"- **Category:** {markdown_text(payload['category'])}")
    lines.extend(
        [
            f"- **Search type:** {markdown_text(payload['search_type'])}",
            (
                f"- **Results:** {payload['result_count']} returned / "
                f"{payload['requested_results']} requested"
            ),
        ]
    )
    if "search_time_ms" in payload:
        lines.append(f"- **Search time:** {payload['search_time_ms']} ms")
    if cost := payload.get("cost_dollars"):
        lines.append("- **Cost (USD):**")
        lines.extend(render_mapping(cost))

    results = payload.get("results", [])
    for result in results:
        fallback_title = "Untitled publication" if publications else "Untitled result"
        title = markdown_text(result.get("title") or fallback_title)
        lines.extend(["", f"## {result['rank']}. {title}", ""])
        if url := result.get("url"):
            lines.append(f"- **URL:** <{markdown_url(url)}>")
        for key, label in (
            ("author", "Authors"),
            ("published_date", "Published"),
            ("id", "Exa ID"),
            ("score", "Score"),
        ):
            if key in result:
                lines.append(f"- **{label}:** {markdown_text(result[key])}")
        if summary := result.get("summary"):
            lines.extend(render_summary(summary))
        if highlights := result.get("highlights"):
            lines.append("- **Highlights:**")
            lines.extend(f"  - {markdown_text(highlight)}" for highlight in highlights)

    if not results:
        empty = (
            "No publications were returned."
            if publications
            else "No web results were returned."
        )
        lines.extend(["", empty])
    return "\n".join(lines)


def create_exa_client() -> Exa:
    load_dotenv(dotenv_path=find_dotenv(usecwd=True), override=False)
    api_key = os.environ.get("EXA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set; add it to .env or the environment")
    return Exa(api_key)


def serialize_result(rank: int, result: Any) -> dict[str, Any]:
    return without_empty(
        {
            "rank": rank,
            "title": result.title,
            "url": result.url,
            "id": result.id,
            "published_date": result.published_date,
            "author": result.author,
            "score": result.score,
            "highlights": result.highlights,
            "summary": result.summary,
        }
    )


def search_exa(
    args: SearchArguments,
    client: Exa | None = None,
) -> dict[str, Any]:
    exa = client if client is not None else create_exa_client()
    options: dict[str, Any] = {
        "num_results": args.resolved_num_results,
        "type": args.resolved_search_type,
        "contents": build_contents(args),
    }
    if args.mode == "research-publications":
        options["category"] = "publication"
    optional = {
        "start_published_date": args.start_published_date,
        "end_published_date": args.end_published_date,
        "exclude_domains": args.exclude_domains,
        "include_domains": args.include_domains,
        "include_text": [args.include_text] if args.include_text else None,
        "exclude_text": [args.exclude_text] if args.exclude_text else None,
        "additional_queries": args.additional_queries,
    }
    options.update({key: value for key, value in optional.items() if value})

    response = exa.search(args.query, **options)
    return without_empty(
        {
            "query": args.query,
            "mode": args.mode,
            "category": (
                "publication" if args.mode == "research-publications" else None
            ),
            "search_type": args.resolved_search_type,
            "requested_results": args.resolved_num_results,
            "result_count": len(response.results),
            "search_time_ms": response.search_time,
            "cost_dollars": response.cost_dollars,
            "results": [
                serialize_result(rank, result)
                for rank, result in enumerate(response.results, start=1)
            ],
        }
    )


def main(argv: Sequence[str] | None = None) -> None:
    payload = search_exa(parse_args(argv))
    print(render_markdown(payload))


if __name__ == "__main__":
    main()
