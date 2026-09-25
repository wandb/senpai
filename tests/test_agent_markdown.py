import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from openhands.sdk.plugin import Plugin

from senpai_agent.agent_markdown import sanitize_markdown, strip_spdx_header

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "plugins" / "senpai"

HTML_HEADER = """<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

"""
PLAIN_HEADER = """# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""


@pytest.mark.parametrize("header", [HTML_HEADER, PLAIN_HEADER])
def test_strip_spdx_header_removes_only_leading_boilerplate(header: str):
    body = "# Research contract\n\nKeep this SPDX-example literal.\n"

    assert strip_spdx_header(header + body) == body
    assert strip_spdx_header(body) == body


def test_strip_spdx_header_preserves_skill_frontmatter():
    source = """---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: review
description: Review one experiment.
---

# Review
"""

    assert strip_spdx_header(source) == """---
name: review
description: Review one experiment.
---

# Review
"""


def test_sanitize_markdown_changes_runtime_copies_only(tmp_path: Path):
    source = tmp_path / "source.md"
    runtime = tmp_path / "runtime" / "skill.md"
    runtime.parent.mkdir()
    source.write_text(HTML_HEADER + "# Source\n", encoding="utf-8")
    runtime.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    sanitize_markdown([runtime.parent])

    assert runtime.read_text(encoding="utf-8") == "# Source\n"
    assert source.read_text(encoding="utf-8").startswith("<!--\nSPDX-")


def test_human_issue_skill_keeps_the_single_intent_response_contract():
    content = (
        PLUGIN_DIR / "skills" / "check-human-issues" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "respond_to_human_issue" in content
    assert "without a role prefix" in content
    assert "github_transition" not in content
    assert "STUDENT $0" not in content


def test_plugin_sanitizer_builds_a_loadable_runtime_copy(tmp_path: Path):
    runtime_plugin = tmp_path / "plugin"
    shutil.copytree(PLUGIN_DIR, runtime_plugin)
    source_skill = PLUGIN_DIR / "skills" / "review-experiment" / "SKILL.md"
    original = source_skill.read_text(encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "senpai_agent.agent_markdown", str(runtime_plugin)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    plugin = Plugin.load(runtime_plugin)
    assert {skill.name for skill in plugin.skills} == {
        "alphaxiv-paper-lookup",
        "assign-experiment",
        "check-human-issues",
        "delegate-subagents",
        "exa-search",
        "maintain-research-state",
        "review-experiment",
        "senpai-status-check",
        "submit-experiment-results",
        "wandb-primary",
    }
    assert all(
        strip_spdx_header(text) == text
        for path in runtime_plugin.rglob("*.md")
        if (text := path.read_text(encoding="utf-8"))
    )
    assert source_skill.read_text(encoding="utf-8") == original


def test_delegate_subagents_skill_advertises_frontier_research_judgment():
    skill = (
        PLUGIN_DIR / "skills" / "delegate-subagents" / "SKILL.md"
    ).read_text(encoding="utf-8")
    frontmatter = " ".join(skill.split("---", 2)[1].split())

    assert "every task requires an" in frontmatter
    assert "explicit model tier" in frontmatter
    assert "delegation-capable subagents" in frontmatter
    assert "research ideation" in frontmatter
    assert "expensive experiment portfolios" in frontmatter
