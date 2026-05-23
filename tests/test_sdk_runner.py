# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "k8s" / "run_senpai_claude_sdk.py"

spec = importlib.util.spec_from_file_location("run_senpai_claude_sdk", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def test_parse_args_accepts_continue_flag():
    args = runner.parse_args(["--max-turns", "7", "--plugin-dir", "plugins/senpai", "-c"])

    assert args.max_turns == 7
    assert args.plugin_dir == Path("plugins/senpai")
    assert args.continue_conversation is True


def test_parse_args_rejects_old_cli_flags():
    with pytest.raises(SystemExit):
        runner.parse_args(["--max-turns", "7", "--plugin-dir", "plugins/senpai", "--model", "opus"])


def test_mcp_config_path_is_repo_root_relative():
    assert runner.mcp_config_path(ROOT / "plugins" / "senpai") == ROOT / ".mcp.json"


def test_build_options_adds_runner_root_for_nested_target_repos():
    args = argparse.Namespace(
        max_turns=3,
        plugin_dir=ROOT / "plugins" / "senpai",
        continue_conversation=False,
    )

    options = runner.build_options(args)

    assert options.add_dirs == [ROOT]
    assert options.model == "claude-opus-4-7"
    assert options.betas == []
    assert options.setting_sources == ["project"]


def test_validate_init_event_requires_senpai_surface():
    payload = {
        "plugins": [{"name": "senpai"}],
        "slash_commands": sorted(runner.REQUIRED_SLASH_COMMANDS),
    }

    runner.validate_init_event(payload)


def test_validate_init_event_fails_when_skill_missing():
    payload = {
        "plugins": [{"name": "senpai"}],
        "slash_commands": [],
    }

    with pytest.raises(RuntimeError, match="missing required skills"):
        runner.validate_init_event(payload)
