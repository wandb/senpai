"""Present Markdown instructions without repository license boilerplate."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path


_WEB_SEARCH_GUIDANCE = (
    "- `search_general_web` for current public sources;\n",
    "- `search_research_publications` for scholarly literature and primary papers;\n",
    " The search agent and its search skills own source selection and search mechanics.",
)


def _is_spdx_line(line: str) -> bool:
    return line.strip().removeprefix("#").strip().startswith("SPDX-")


def strip_spdx_header(text: str) -> str:
    """Remove one leading SPDX header while preserving the document body."""

    lines = text.splitlines(keepends=True)
    if not lines:
        return text

    if lines[0].strip() == "<!--":
        try:
            end = next(
                index
                for index, line in enumerate(lines[1:], 1)
                if line.strip() == "-->"
            )
        except StopIteration:
            end = -1
        if end > 0 and all(
            not line.strip() or _is_spdx_line(line) for line in lines[1:end]
        ):
            del lines[: end + 1]
            if lines and not lines[0].strip():
                del lines[0]
            return "".join(lines)

    if lines[0].strip() == "---":
        try:
            end = next(
                index
                for index, line in enumerate(lines[1:], 1)
                if line.strip() == "---"
            )
        except StopIteration:
            end = -1
        if end > 0:
            frontmatter = [line for line in lines[1:end] if not _is_spdx_line(line)]
            while frontmatter and not frontmatter[0].strip():
                frontmatter.pop(0)
            return "".join([lines[0], *frontmatter, *lines[end:]])

    end = 0
    while end < len(lines) and _is_spdx_line(lines[end]):
        end += 1
    if end:
        if end < len(lines) and not lines[end].strip():
            end += 1
        return "".join(lines[end:])
    return text


def read_agent_markdown(path: Path) -> str:
    """Read one Markdown file exactly as it should be shown to an agent."""

    return strip_spdx_header(path.read_text(encoding="utf-8"))


def sanitize_markdown(paths: Iterable[Path]) -> None:
    """Strip headers from Markdown runtime copies, never source checkouts."""

    files = (
        child
        for path in paths
        for child in (path.rglob("*.md") if path.is_dir() else (path,))
    )
    for path in files:
        if path.suffix.lower() != ".md" or path.is_symlink():
            continue
        original = path.read_text(encoding="utf-8")
        cleaned = strip_spdx_header(original)
        if cleaned != original:
            path.write_text(cleaned, encoding="utf-8")


def remove_web_search_guidance(plugin: Path) -> None:
    """Remove unavailable search task advice from one runtime plugin copy."""

    guide = plugin / "skills" / "delegate-subagents" / "SKILL.md"
    content = guide.read_text(encoding="utf-8")
    for guidance in _WEB_SEARCH_GUIDANCE:
        if guidance not in content:
            raise RuntimeError(f"web-search guidance is missing from {guide}")
        content = content.replace(guidance, "", 1)
    guide.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Strip SPDX boilerplate from agent-facing Markdown copies."
    )
    parser.add_argument("--without-web-search", type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.without_web_search:
        remove_web_search_guidance(args.without_web_search)
    sanitize_markdown(args.paths)


if __name__ == "__main__":
    main()
