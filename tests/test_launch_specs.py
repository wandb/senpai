import base64
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from senpai.launch.local_backend import launch_local
from senpai.launch.specs import build_advisor_spec, build_student_spec


def args(**overrides):
    values = {
        "tag": "aws-r1",
        "repo_url": "https://github.com/wandb/senpai.git",
        "repo_branch": "codex/aws-compatible-senpai",
        "target_repo_url": "https://github.com/example/gemma-target.git",
        "target_repo_branch": "main",
        "problem_dir": "target/",
        "gpus_per_student": 0,
        "wandb_entity": "wandb",
        "wandb_project": "senpai-test",
        "human_issues": True,
        "advisor_branch": "gemma-advisor",
        "gh_history_scope": "branch",
        "pvc_mount_path": "/mnt/senpai",
        "extra_instructions": "external HF Jobs benchmark target",
        "timeout_minutes": 20.0,
        "max_epochs": 1,
        "poll_interval_s": 30,
        "poll_jitter_s": 5,
        "stale_wip_seconds": 600,
        "advisor_claude_watchdog_interval_s": 60,
        "advisor_claude_min_runtime_s": 600,
        "advisor_claude_stale_log_s": 1200,
        "student_claude_watchdog_interval_s": 30,
        "student_claude_watchdog_jitter_s": 5,
        "student_claude_min_runtime_s": 600,
        "student_claude_stale_log_s": 1200,
        "student_assignment_drift_grace_s": 120,
        "start_gate_path": "",
        "local_run_root": "~/.senpai/runs",
        "local_runner_source": "",
        "local_container_image": "",
        "local_student_gpu_ids": "",
        "local_skip_install": True,
        "local_disable_hivemind": True,
        "dry_run": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LaunchSpecTests(unittest.TestCase):
    def test_student_spec_contains_backend_independent_env(self):
        spec = build_student_spec(args(), "aws-r1", "fern", {"GITHUB_TOKEN": "secret"})

        self.assertEqual(spec.key, "student-fern")
        self.assertEqual(spec.env["GH_REPO"], "example/gemma-target")
        self.assertEqual(spec.env["GPUS_PER_STUDENT"], "0")
        self.assertEqual(spec.env["STUDENT_NAME"], "fern")
        decoded = base64.b64decode(spec.env["EXTRA_INSTRUCTIONS_B64"]).decode()
        self.assertIn("external HF Jobs benchmark target", decoded)
        self.assertEqual(spec.secrets["GITHUB_TOKEN"], "secret")

    def test_advisor_spec_lists_students(self):
        spec = build_advisor_spec(args(), "aws-r1", ["fern", "tanjiro"], {})

        self.assertEqual(spec.key, "advisor")
        self.assertEqual(spec.env["STUDENT_NAMES"], "fern,tanjiro")
        self.assertEqual(spec.env["SENPAI_STALE_WIP_SECONDS"], "600")

    def test_local_dry_run_does_not_create_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(local_run_root=tmp)
            spec = build_student_spec(run_args, "aws-r1", "fern", {"GITHUB_TOKEN": "secret"})
            output = io.StringIO()

            with redirect_stdout(output):
                launch_local(run_args, [spec])

            self.assertIn("--- Local student-fern ---", output.getvalue())
            self.assertIn("GITHUB_TOKEN='<redacted>'", output.getvalue())
            self.assertFalse((Path(tmp) / "aws-r1").exists())

    def test_local_container_dry_run_uses_docker_image_and_mounts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                local_run_root=tmp,
                local_container_image="vllm/vllm-openai",
                local_student_gpu_ids="fern:0",
            )
            spec = build_student_spec(run_args, "aws-r1", "fern", {"GITHUB_TOKEN": "secret"})
            output = io.StringIO()

            with redirect_stdout(output):
                launch_local(run_args, [spec])

            text = output.getvalue()
            self.assertIn("Container image: vllm/vllm-openai", text)
            self.assertIn("docker run", text)
            self.assertIn("--gpus device=0", text)
            self.assertIn("vllm/vllm-openai", text)
            self.assertIn("SENPAI_BACKEND='<redacted>'", text)
            self.assertFalse((Path(tmp) / "aws-r1").exists())


if __name__ == "__main__":
    unittest.main()
