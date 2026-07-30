import base64
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from senpai.launch.docker_backend import (
    _docker_command,
    _env_file_text,
    _role_values,
    launch_docker,
)
from senpai.launch.specs import build_advisor_spec, build_student_spec


def args(**overrides):
    values = {
        "tag": "aws-r1",
        "repo_url": "https://github.com/wandb/senpai.git",
        "repo_branch": "main",
        "target_repo_url": "https://github.com/example/problem.git",
        "target_repo_branch": "main",
        "problem_dir": "target/",
        "gpus_per_student": 0,
        "wandb_entity": "wandb",
        "wandb_project": "senpai-test",
        "human_issues": True,
        "advisor_branch": "research",
        "gh_history_scope": "branch",
        "pvc_mount_path": "/mnt/datasets",
        "extra_instructions": "AWS experiment",
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
        "docker_run_root": "~/.senpai/runs",
        "docker_runner_source": "",
        "docker_student_gpu_ids": "",
        "docker_data_dir": "",
        "docker_shm_size": "32g",
        "image": "ghcr.io/wandb/senpai:latest",
        "dry_run": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LaunchSpecTests(unittest.TestCase):
    def test_student_spec_contains_backend_independent_environment(self):
        spec = build_student_spec(args(), "aws-r1", "fern", {"GITHUB_TOKEN": "secret"})

        self.assertEqual(spec.key, "student-fern")
        self.assertEqual(spec.env["GH_REPO"], "example/problem")
        self.assertEqual(spec.env["STUDENT_NAME"], "fern")
        decoded = base64.b64decode(spec.env["EXTRA_INSTRUCTIONS_B64"]).decode()
        self.assertIn("AWS experiment", decoded)
        self.assertEqual(spec.secrets["GITHUB_TOKEN"], "secret")

    def test_advisor_spec_lists_students(self):
        spec = build_advisor_spec(args(), "aws-r1", ["fern", "tanjiro"], {})

        self.assertEqual(spec.key, "advisor")
        self.assertEqual(spec.env["STUDENT_NAMES"], "fern,tanjiro")
        self.assertEqual(spec.env["SENPAI_STALE_WIP_SECONDS"], "600")


class DockerBackendTests(unittest.TestCase):
    def test_dry_run_builds_restartable_gpu_containers_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            data = Path(tmp) / "data"
            source.mkdir()
            data.mkdir()
            run_root = Path(tmp) / "runs"
            run_args = args(
                docker_run_root=str(run_root),
                docker_runner_source=str(source),
                docker_data_dir=str(data),
                docker_student_gpu_ids="fern:0+2",
                gpus_per_student=2,
            )
            spec = build_student_spec(
                run_args, "aws-r1", "fern", {"GITHUB_TOKEN": "secret"}
            )
            output = io.StringIO()

            with redirect_stdout(output):
                launch_docker(run_args, [spec])

            text = output.getvalue()
            self.assertIn("docker run --detach --init --restart unless-stopped", text)
            self.assertIn("device=0,2", text)
            self.assertIn("--shm-size 32g", text)
            self.assertIn(f"{data.resolve()}:/mnt/datasets", text)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", text)
            self.assertFalse(run_root.exists())

    def test_launch_writes_private_environment_and_starts_detached_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                docker_runner_source=str(source),
                dry_run=False,
            )
            spec = build_student_spec(
                run_args, "aws-r1", "fern", {"GITHUB_TOKEN": "secret"}
            )
            completed = SimpleNamespace(
                returncode=0, stdout="abcdef123456\n", stderr=""
            )

            with (
                patch("senpai.launch.docker_backend._check_docker"),
                patch("senpai.launch.docker_backend._check_container_names_available"),
                patch("senpai.launch.docker_backend._prepare_runner_workdir"),
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    return_value=completed,
                ) as run,
            ):
                launch_docker(run_args, [spec])

            env_file = Path(tmp) / "runs" / "aws-r1" / "env" / "student-fern.env"
            self.assertEqual(env_file.stat().st_mode & 0o777, 0o600)
            self.assertIn("GITHUB_TOKEN=secret", env_file.read_text())
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["docker", "run", "--detach"])
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env_file.read_text())

    def test_docker_command_gives_no_gpu_or_shm_to_advisor(self):
        run_args = args()
        spec = build_advisor_spec(run_args, "aws-r1", ["fern"], {})
        command = _docker_command(
            run_args,
            Path("/tmp/run"),
            spec,
            Path("/tmp/workdir"),
            Path("/tmp/advisor.env"),
            [],
        )

        self.assertNotIn("--gpus", command)
        self.assertNotIn("--shm-size", command)
        self.assertEqual(command[-2:], ["bash", "k8s/entrypoint-advisor.sh"])

    def test_docker_role_values_rely_on_runtime_gpu_visibility(self):
        spec = build_student_spec(args(), "aws-r1", "fern", {})

        values = _role_values(spec)

        self.assertNotIn("CUDA_VISIBLE_DEVICES", values)
        self.assertEqual(values["SENPAI_BACKEND"], "docker")
        self.assertEqual(values["HOME"], "/senpai-run/home/student-fern")

    def test_gpu_students_require_explicit_assignments(self):
        run_args = args(gpus_per_student=1)
        spec = build_student_spec(run_args, "aws-r1", "fern", {})

        with self.assertRaisesRegex(ValueError, "docker_student_gpu_ids"):
            launch_docker(run_args, [spec])

    def test_gpu_assignments_must_match_count(self):
        run_args = args(gpus_per_student=2, docker_student_gpu_ids="fern:0")
        spec = build_student_spec(run_args, "aws-r1", "fern", {})

        with self.assertRaisesRegex(ValueError, "assignments differ"):
            launch_docker(run_args, [spec])

    def test_gpu_assignments_must_be_exclusive(self):
        run_args = args(
            gpus_per_student=1,
            docker_student_gpu_ids="fern:0,tanjiro:0",
        )
        specs = [
            build_student_spec(run_args, "aws-r1", "fern", {}),
            build_student_spec(run_args, "aws-r1", "tanjiro", {}),
        ]

        with self.assertRaisesRegex(ValueError, "exclusive"):
            launch_docker(run_args, specs)

    def test_duplicate_student_assignment_is_rejected(self):
        run_args = args(gpus_per_student=1, docker_student_gpu_ids="fern:0,fern:1")
        spec = build_student_spec(run_args, "aws-r1", "fern", {})

        with self.assertRaisesRegex(ValueError, "duplicate GPU assignment"):
            launch_docker(run_args, [spec])

    def test_env_file_rejects_multiline_values(self):
        with self.assertRaisesRegex(ValueError, "contains a newline"):
            _env_file_text({"TOKEN": "first\nsecond"})

    def test_empty_launch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no advisor or students"):
            launch_docker(args(), [])


if __name__ == "__main__":
    unittest.main()
