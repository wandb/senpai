import json
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
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


@pytest.fixture
def readonly_runtime_plugin(tmp_path: Path):
    runtime_plugin = tmp_path / "plugin"
    shutil.copytree(PLUGIN_DIR, runtime_plugin)
    subprocess.run(
        [sys.executable, "-m", "senpai_agent.agent_markdown", str(runtime_plugin)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    modes = {path: path.stat().st_mode for path in runtime_plugin.rglob("*")}
    modes[runtime_plugin] = runtime_plugin.stat().st_mode
    try:
        for path, mode in modes.items():
            path.chmod(mode & ~0o222)
        yield runtime_plugin
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


def test_plugin_sanitizer_builds_a_loadable_runtime_copy(readonly_runtime_plugin: Path):
    runtime_plugin = readonly_runtime_plugin

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


def test_bundled_helpers_run_without_syncing_target_dependencies(
    readonly_runtime_plugin: Path, tmp_path: Path,
):
    target = tmp_path / "target-env"
    subprocess.run(
        [
            "uv", "venv", "--no-project", "--no-config", "--no-python-downloads",
            "--python", sys.executable, str(target),
        ],
        check=True,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
        capture_output=True,
        text=True,
    )
    target_site = Path(sysconfig.get_path("purelib", vars={"base": str(target)}))
    (target_site / "senpai-runtime.pth").write_text(sysconfig.get_path("purelib") + "\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text(
        '[project]\nname = "offline-analysis"\nversion = "0.0.0"\n'
        'requires-python = ">=3.13"\n'
        'dependencies = ["senpai-must-not-resolve==0"]\n'
    )
    (workspace / "analysis.py").write_text(textwrap.dedent("""\
        import json, os, sys
        from types import SimpleNamespace
        sys.path.insert(0, f"{os.environ['SENPAI_PLUGIN']}/skills/wandb-primary/scripts")
        from wandb_helpers import runs_to_dataframe
        from weave_helpers import get_token_usage
        from step_axis import list_candidate_step_keys

        rows = [{"_step": step, "loss": 20 - step} for step in range(20)]
        run = SimpleNamespace(
            id="offline", name="offline", state="finished", created_at=None,
            config={}, summary_metrics={"loss": 1},
            scan_history=lambda **kwargs: iter(rows),
        )
        usage = get_token_usage(SimpleNamespace(summary={
            "usage": {"model": {"input_tokens": 3, "output_tokens": 2}},
        }))
        print(json.dumps({
            "prefix": sys.prefix,
            "loss": runs_to_dataframe([run])[0]["loss"],
            "steps": list_candidate_step_keys(run),
            "tokens": usage["total_tokens"],
        }))
    """))
    environment = {
        "PATH": f"{target / 'bin'}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "SENPAI_PLUGIN": str(readonly_runtime_plugin),
        "UV_PROJECT_ENVIRONMENT": str(target),
        "UV_PYTHON": str(target / "bin/python"),
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }
    result = subprocess.run(
        ["uv", "run", "--offline", "--no-sync", "python", "analysis.py"],
        cwd=workspace, env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "prefix": str(target), "loss": 1, "steps": ["_step"],
        "tokens": 5,
    }
    assert not (workspace / "uv.lock").exists()
    exa = readonly_runtime_plugin / "skills/exa-search/scripts/search_exa.py"
    help_result = subprocess.run(
        ["uv", "run", "--offline", "--no-sync", "python", str(exa), "--help"],
        cwd=workspace, env=environment, capture_output=True, text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "general-web" in help_result.stdout
    assert "research-publications" in help_result.stdout


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
