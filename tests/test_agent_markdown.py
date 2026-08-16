import os
import subprocess
import sys
from pathlib import Path

import pytest
from openhands.sdk.plugin import Plugin
from openhands.sdk.subagent import discover_agents

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


def test_agent_context_installer_builds_loadable_sanitized_runtime_copies(
    tmp_path: Path,
):
    home = tmp_path / "home"
    runtime_root = tmp_path / "runtime"
    home.mkdir()
    runtime_root.mkdir()
    source_skill = PLUGIN_DIR / "skills" / "review-experiment" / "SKILL.md"
    operator_skill = ROOT / ".agents" / "skills" / "experiment-report" / "SKILL.md"
    source_agent = ROOT / ".agents" / "agents" / "bash-runner.md"
    originals = {
        source_skill: source_skill.read_text(encoding="utf-8"),
        operator_skill: operator_skill.read_text(encoding="utf-8"),
        source_agent: source_agent.read_text(encoding="utf-8"),
    }

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; install_senpai_agent_context "$2" "$3" "$4"',
            "bash",
            str(PLUGIN_DIR / "scripts" / "agent-context.sh"),
            str(ROOT),
            str(PLUGIN_DIR),
            str(runtime_root),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(home), "SENPAI_PYTHON": sys.executable},
    )
    runtime_plugin = Path(completed.stdout.strip())

    plugin = Plugin.load(runtime_plugin)
    agents = discover_agents(home, include_project=True, include_user=False)

    assert {skill.name for skill in plugin.skills} == {
        "alphaxiv-paper-lookup",
        "assign-experiment",
        "check-human-issues",
        "delegate-subagents",
        "exa-search",
        "review-experiment",
        "senpai-status-check",
        "submit-experiment-results",
        "wandb-primary",
    }
    assert "bash-runner" in {agent.name for agent in agents}
    assert not (home / ".agents/skills").exists()
    assert all(
        strip_spdx_header(text) == text
        for root in (runtime_plugin, home / ".agents")
        for path in root.rglob("*.md")
        if (text := path.read_text(encoding="utf-8"))
    )
    assert {
        path: path.read_text(encoding="utf-8") for path in originals
    } == originals
