import base64
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from senpai.launch import remote
from senpai.launch.docker_backend import (
    _check_image,
    _check_runner_source,
    _docker_gpu_indices,
    _env_file_text,
    _path_beneath,
    _pull_image,
    _prepare_runner_workdir,
    _role_values,
    _wait_until_ready,
    launch_docker,
    logs_docker,
    plan_docker,
    preflight_docker,
    status_docker,
    terminate_docker,
)
from senpai.launch.specs import (
    CONTAINER_IMAGE_GROUP_ID,
    CONTAINER_USER_ID,
    build_advisor_spec,
    build_student_spec,
)


def args(**overrides):
    revision = "a" * 40
    values = {
        "tag": "aws-r1",
        "backend": "docker",
        "senpai_repo_url": "https://github.com/wandb/senpai.git",
        "senpai_repo_revision": revision,
        "advisor_image": f"ghcr.io/wandb/senpai-advisor:sha-{revision}",
        "student_image": f"ghcr.io/wandb/senpai-student:sha-{revision}",
        "target_repo_url": "https://github.com/example/problem.git",
        "target_repo_branch": "main",
        "program_path": "",
        "problem_dir": "target/",
        "gpus_per_student": 0,
        "cpu_per_gpu": 15,
        "memory_gi_per_gpu": 120,
        "wandb_entity": "wandb",
        "wandb_project": "senpai-test",
        "advisor_model": "anthropic/claude-opus-4-8",
        "advisor_reasoning_effort": "xhigh",
        "student_model": "anthropic/claude-opus-4-8",
        "student_reasoning_effort": "xhigh",
        "smart_model": "anthropic/claude-opus-4-8",
        "smart_reasoning_effort": "xhigh",
        "fast_model": "anthropic/claude-haiku-4-5",
        "fast_reasoning_effort": "low",
        "frontier_model": "openai/gpt-5.6-sol",
        "frontier_reasoning_effort": "max",
        "local_condenser_max_events": 0,
        "local_condenser_max_tokens": 0,
        "local_condenser_target_events": 0,
        "human_issues": True,
        "advisor_name": "advisor",
        "advisor_branch": "research",
        "gh_history_scope": "branch",
        "pvc_mount_path": "/mnt/datasets",
        "extra_instructions": "AWS experiment",
        "timeout_minutes": 20.0,
        "max_epochs": 1,
        "poll_interval_s": 30,
        "poll_jitter_s": 5,
        "stale_wip_seconds": 600,
        "start_gate_path": "",
        "docker_run_root": "~/.senpai/runs",
        "docker_student_gpu_ids": "",
        "data_dir": "",
        "docker_shm_size": "32g",
        "docker_ready_timeout_s": 120,
        "dry_run": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def student(run_args, name="fern"):
    return build_student_spec(run_args, run_args.tag, name, {})


def missing_image():
    return SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="Error response from daemon: No such image: test",
    )


class LaunchSpecTests(unittest.TestCase):
    def test_student_spec_contains_backend_independent_environment(self):
        spec = build_student_spec(args(), "aws-r1", "fern", {"GITHUB_TOKEN": "secret"})

        self.assertEqual(spec.key, "student-fern")
        self.assertEqual(spec.env["GH_REPO"], "example/problem")
        self.assertEqual(spec.env["STUDENT_NAME"], "fern")
        self.assertEqual(spec.env["SENPAI_REPO_REVISION"], "a" * 40)
        self.assertNotIn("REPO_BRANCH", spec.env)
        decoded = base64.b64decode(spec.env["EXTRA_INSTRUCTIONS_B64"]).decode()
        self.assertIn("AWS experiment", decoded)
        self.assertEqual(spec.secrets["GITHUB_TOKEN"], "secret")

    def test_advisor_spec_lists_students(self):
        spec = build_advisor_spec(
            args(advisor_name="aurora"),
            "aws-r1",
            ["fern", "tanjiro"],
            {},
        )

        self.assertEqual(spec.key, "advisor")
        self.assertEqual(spec.name, "aurora")
        self.assertEqual(spec.env["ADVISOR_NAME"], "aurora")
        self.assertEqual(spec.env["STUDENT_NAMES"], "fern,tanjiro")
        self.assertEqual(spec.env["SENPAI_STALE_WIP_SECONDS"], "600")


class DockerPlanTests(unittest.TestCase):
    def test_unsafe_identifiers_and_escaping_paths_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                tag="../outside",
                docker_run_root=str(Path(tmp) / "runs"),
            )

            with self.assertRaisesRegex(ValueError, "Docker tag"):
                plan_docker(run_args, [student(run_args)])
            with self.assertRaisesRegex(ValueError, "escapes run root"):
                _path_beneath(Path(tmp).resolve(), "..", "outside")

            self.assertFalse((Path(tmp) / "runs").exists())

    def test_unsafe_student_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(docker_run_root=str(Path(tmp) / "runs"))
            spec = build_student_spec(run_args, run_args.tag, "../../peer", {})

            with self.assertRaisesRegex(ValueError, "Docker student name"):
                plan_docker(run_args, [spec])

    def test_unsafe_advisor_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                advisor_name="../../peer",
                docker_run_root=str(Path(tmp) / "runs"),
            )
            spec = build_advisor_spec(run_args, run_args.tag, [], {})

            with self.assertRaisesRegex(ValueError, "Docker advisor name"):
                plan_docker(run_args, [spec])

    def test_existing_run_root_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs" / "aws-r1"
            run_root.mkdir(parents=True)
            run_args = args(docker_run_root=str(Path(tmp) / "runs"))

            with self.assertRaisesRegex(RuntimeError, "already exists"):
                plan_docker(run_args, [student(run_args)])

    def test_unwritable_run_root_fails_during_planning(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("senpai.launch.specs.os.access", return_value=False),
        ):
            run_args = args(docker_run_root=str(Path(tmp) / "runs"))
            with self.assertRaisesRegex(RuntimeError, "cannot be created"):
                plan_docker(run_args, [student(run_args)])

    def test_data_directory_cannot_contain_private_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "shared"
            data_dir.mkdir()
            run_args = args(
                docker_run_root=str(data_dir / "runs"),
                data_dir=str(data_dir),
            )

            with self.assertRaisesRegex(ValueError, "overlaps private run state"):
                plan_docker(run_args, [student(run_args)])

    def test_data_directory_cannot_mount_a_sibling_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_base = Path(tmp) / "runs"
            data_dir = run_base / "other-run"
            data_dir.mkdir(parents=True)
            run_args = args(
                docker_run_root=str(run_base),
                data_dir=str(data_dir),
            )

            with self.assertRaisesRegex(ValueError, "overlaps private run state"):
                plan_docker(run_args, [student(run_args)])

    def test_unrelated_data_directory_is_mounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "datasets"
            data_dir.mkdir()
            run_args = args(
                docker_run_root=str(root / "runs"),
                data_dir=str(data_dir),
            )

            plan = plan_docker(run_args, [student(run_args)])

            self.assertIn(
                f"{data_dir.resolve()}:{run_args.pvc_mount_path}",
                plan.roles[0].command,
            )

    def test_data_mount_cannot_override_private_container_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "datasets"
            data_dir.mkdir()
            run_args = args(
                docker_run_root=str(root / "runs"),
                data_dir=str(data_dir),
                pvc_mount_path="/var/lib/senpai/data",
            )

            with self.assertRaisesRegex(ValueError, "pvc_mount_path"):
                plan_docker(run_args, [student(run_args)])

    def test_plan_mounts_private_role_state_and_shared_launch_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                gpus_per_student=1,
                docker_student_gpu_ids="fern:2",
            )
            specs = [
                student(run_args),
                build_advisor_spec(run_args, run_args.tag, ["fern"], {}),
            ]

            plan = plan_docker(run_args, specs)
            student_plan, advisor_plan = plan.roles
            student_command = list(student_plan.command)
            advisor_command = list(advisor_plan.command)

            self.assertIn(
                f"{student_plan.state_root}:/var/lib/senpai",
                student_command,
            )
            self.assertIn(
                f"{plan.gate_root}:/senpai-launch:ro",
                student_command,
            )
            self.assertNotIn(f"{plan.run_root}:/var/lib/senpai", student_command)
            self.assertIn(
                f"{plan.gate_root}:/senpai-launch:ro",
                advisor_command,
            )
            self.assertIn(run_args.student_image, student_command)
            self.assertIn(run_args.advisor_image, advisor_command)
            self.assertEqual(student_command[student_command.index("--stop-timeout") + 1], "90")

    def test_role_environment_exposes_openhands_state_and_gate_contract(self):
        values = _role_values(student(args()))

        self.assertEqual(values["HOME"], "/var/lib/senpai/home")
        self.assertEqual(values["SENPAI_BACKEND"], "docker")
        self.assertEqual(values["SENPAI_UMASK"], "0002")
        self.assertEqual(
            values["SENPAI_LAUNCH_GATE_PATH"],
            "/senpai-launch/.launch",
        )
        self.assertNotIn("SENPAI_READY_FILE", values)
        self.assertNotIn("SENPAI_STATUS_DIR", values)

    def test_runner_checkout_is_detached_at_the_exact_revision(self):
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.docker_backend.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            workdir = Path(tmp) / "role" / "workdir"
            _prepare_runner_workdir(
                "https://github.com/example/runner.git",
                workdir,
                "a" * 40,
            )

        self.assertEqual(
            run.call_args_list[0].args[0],
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/example/runner.git",
                str(workdir),
            ],
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["git", "-C", str(workdir), "checkout", "--detach", "a" * 40],
        )

    def test_env_file_rejects_multiline_values(self):
        with self.assertRaisesRegex(ValueError, "contains a newline"):
            _env_file_text({"TOKEN": "first\nsecond"})

    def test_multi_gpu_selection_is_one_literal_docker_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                gpus_per_student=2,
                cpu_per_gpu=7,
                memory_gi_per_gpu=13,
                docker_student_gpu_ids="fern:0+2",
            )
            command = list(plan_docker(run_args, [student(run_args)]).roles[0].command)

        self.assertEqual(command[command.index("--gpus") + 1], '"device=0,2"')
        self.assertEqual(command[command.index("--cpus") + 1], "14")
        self.assertEqual(command[command.index("--memory") + 1], "26g")

    def test_zero_gpu_student_has_no_cpu_or_memory_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(docker_run_root=str(Path(tmp) / "runs"))
            command = plan_docker(run_args, [student(run_args)]).roles[0].command

        self.assertNotIn("--cpus", command)
        self.assertNotIn("--memory", command)


class DockerPreflightTests(unittest.TestCase):
    def test_advisor_only_preflight_checks_the_advisor_image_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(docker_run_root=str(Path(tmp) / "runs"))
            advisor = build_advisor_spec(run_args, run_args.tag, [], {})
            with (
                patch("senpai.launch.docker_backend._check_runner_source"),
                patch("senpai.launch.docker_backend._check_docker"),
                patch(
                    "senpai.launch.docker_backend._check_container_names_available"
                ),
                patch("senpai.launch.docker_backend._check_image") as check_image,
            ):
                preflight_docker(run_args, [advisor])

        check_image.assert_called_once_with(
            run_args.advisor_image,
            run_args.senpai_repo_revision,
        )

    def test_image_probe_requires_the_exact_source_revision(self):
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch(
            "senpai.launch.docker_backend.subprocess.run",
            return_value=completed,
        ) as run:
            _check_image("ghcr.io/wandb/senpai-advisor:test", "b" * 40)

        self.assertEqual(
            run.call_args_list[0].args[0],
            [
                "docker",
                "image",
                "inspect",
                "ghcr.io/wandb/senpai-advisor:test",
            ],
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--entrypoint",
                "/bin/bash",
                "ghcr.io/wandb/senpai-advisor:test",
                "-c",
                'test "$SENPAI_IMAGE_REVISION" = "$1"',
                "senpai-image-check",
                "b" * 40,
            ],
        )

    def test_cached_image_skips_the_registry_pull(self):
        cached = SimpleNamespace(returncode=0, stdout="[]", stderr="")

        with patch(
            "senpai.launch.docker_backend.subprocess.run",
            return_value=cached,
        ) as run:
            _pull_image("ghcr.io/wandb/senpai-student:test")

        self.assertEqual(
            run.call_args.args[0],
            ["docker", "image", "inspect", "ghcr.io/wandb/senpai-student:test"],
        )
        run.assert_called_once()

    def test_image_pull_retries_a_registry_connection_reset(self):
        reset = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                "docker: failed to copy: read tcp "
                "192.168.42.0:56036->185.199.109.154:443: "
                "read: connection reset by peer"
            ),
        )
        completed = SimpleNamespace(returncode=0, stdout="pulled", stderr="")
        output = io.StringIO()

        with (
            patch(
                "senpai.launch.docker_backend.subprocess.run",
                side_effect=[missing_image(), reset, completed],
            ) as run,
            redirect_stdout(output),
        ):
            _pull_image("ghcr.io/wandb/senpai-student:test")

        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                [
                    "docker",
                    "image",
                    "inspect",
                    "ghcr.io/wandb/senpai-student:test",
                ],
                ["docker", "pull", "ghcr.io/wandb/senpai-student:test"],
                ["docker", "pull", "ghcr.io/wandb/senpai-student:test"],
            ],
        )
        self.assertIn("retrying (2/3)", output.getvalue())

    def test_image_pull_connection_reset_retries_are_bounded(self):
        reset = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="read: connection reset by peer",
        )

        with (
            patch(
                "senpai.launch.docker_backend.subprocess.run",
                side_effect=[missing_image(), reset, reset, reset],
            ) as run,
            self.assertRaisesRegex(
                RuntimeError,
                "after 3 connection-reset attempts.*connection reset by peer",
            ),
        ):
            _pull_image("ghcr.io/wandb/senpai-student:test")

        self.assertEqual(run.call_count, 4)

    def test_image_pull_does_not_retry_a_deterministic_failure(self):
        rejected = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="manifest unknown",
        )

        with (
            patch(
                "senpai.launch.docker_backend.subprocess.run",
                side_effect=[missing_image(), rejected],
            ) as run,
            self.assertRaisesRegex(RuntimeError, "manifest unknown"),
        ):
            _pull_image("ghcr.io/wandb/senpai-student:missing")

        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                [
                    "docker",
                    "image",
                    "inspect",
                    "ghcr.io/wandb/senpai-student:missing",
                ],
                ["docker", "pull", "ghcr.io/wandb/senpai-student:missing"],
            ],
        )

    def test_image_inspect_failure_is_not_misclassified_as_a_cache_miss(self):
        unavailable = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon",
        )

        with (
            patch(
                "senpai.launch.docker_backend.subprocess.run",
                return_value=unavailable,
            ) as run,
            self.assertRaisesRegex(RuntimeError, "cannot inspect.*Cannot connect"),
        ):
            _pull_image("ghcr.io/wandb/senpai-student:test")

        run.assert_called_once()

    def test_image_probe_still_rejects_a_wrong_source_revision_after_pull(self):
        pulled = SimpleNamespace(returncode=0, stdout="pulled", stderr="")
        wrong_revision = SimpleNamespace(returncode=1, stdout="", stderr="")

        with (
            patch(
                "senpai.launch.docker_backend.subprocess.run",
                side_effect=[missing_image(), pulled, wrong_revision],
            ) as run,
            self.assertRaisesRegex(RuntimeError, "at source revision"),
        ):
            _check_image("ghcr.io/wandb/senpai-advisor:test", "b" * 40)

        self.assertEqual(run.call_count, 3)

    def test_preflight_checks_both_role_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(docker_run_root=str(Path(tmp) / "runs"))
            specs = [
                student(run_args),
                build_advisor_spec(run_args, run_args.tag, ["fern"], {}),
            ]
            with (
                patch("senpai.launch.docker_backend._check_runner_source"),
                patch("senpai.launch.docker_backend._check_docker"),
                patch(
                    "senpai.launch.docker_backend._check_container_names_available"
                ),
                patch("senpai.launch.docker_backend._check_image") as check_image,
            ):
                preflight_docker(run_args, specs)

        self.assertEqual(
            [call.args for call in check_image.call_args_list],
            [
                (run_args.advisor_image, run_args.senpai_repo_revision),
                (run_args.student_image, run_args.senpai_repo_revision),
            ],
        )

    def test_preflight_checks_data_access_from_every_active_role_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "datasets"
            data_dir.mkdir()
            run_args = args(
                docker_run_root=str(root / "runs"),
                data_dir=str(data_dir),
            )
            specs = [
                student(run_args),
                build_advisor_spec(run_args, run_args.tag, ["fern"], {}),
            ]
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("senpai.launch.docker_backend._check_runner_source"),
                patch("senpai.launch.docker_backend._check_docker"),
                patch(
                    "senpai.launch.docker_backend._check_container_names_available"
                ),
                patch("senpai.launch.docker_backend._check_image"),
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    return_value=completed,
                ) as run,
            ):
                preflight_docker(run_args, specs)

            probes = [
                call.args[0]
                for call in run.call_args_list
                if "senpai-data-check" in call.args[0]
            ]
            self.assertEqual(
                [command[command.index("--entrypoint") + 2] for command in probes],
                [run_args.advisor_image, run_args.student_image],
            )
            for command in probes:
                self.assertIn(
                    f"{data_dir.resolve()}:{run_args.pvc_mount_path}:rw",
                    command,
                )
                self.assertIn(
                    f"{CONTAINER_USER_ID}:{data_dir.stat().st_gid}",
                    command,
                )
                self.assertIn(str(CONTAINER_IMAGE_GROUP_ID), command)
                probe = command[command.index("-c") + 1]
                self.assertIn('test -r "$1"', probe)
                self.assertIn('test -x "$1"', probe)
                self.assertIn('test -w "$1"', probe)
                self.assertNotIn("chmod", command)
                self.assertNotIn("chown", command)

    def test_data_access_failure_is_actionable_and_does_not_mutate_user_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "datasets"
            data_dir.mkdir(mode=0o700)
            run_args = args(
                docker_run_root=str(root / "runs"),
                data_dir=str(data_dir),
            )
            advisor = build_advisor_spec(run_args, run_args.tag, [], {})
            failed = SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="permission denied",
            )
            original_mode = data_dir.stat().st_mode & 0o777

            with (
                patch("senpai.launch.docker_backend._check_runner_source"),
                patch("senpai.launch.docker_backend._check_docker"),
                patch(
                    "senpai.launch.docker_backend._check_container_names_available"
                ),
                patch("senpai.launch.docker_backend._check_image"),
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    return_value=failed,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "advisor.*read, traverse, and write.*host group GID",
                ),
            ):
                preflight_docker(run_args, [advisor])

            self.assertEqual(data_dir.stat().st_mode & 0o777, original_mode)

    def test_gpu_probe_executes_pytorch_cuda_in_the_actual_image(self):
        completed = SimpleNamespace(returncode=0, stdout="0\n1\n", stderr="")
        with patch(
            "senpai.launch.docker_backend.subprocess.run",
            return_value=completed,
        ) as run:
            self.assertEqual(
                _docker_gpu_indices("ghcr.io/wandb/senpai:test"),
                ["0", "1"],
            )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--entrypoint") + 1], "python3")
        self.assertEqual(command[command.index("--entrypoint") + 2], "ghcr.io/wandb/senpai:test")
        self.assertIn("torch.ones", command[-1])

    def test_runner_revision_accepts_an_older_reachable_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            subprocess.run(
                ["git", "init", str(source)],
                capture_output=True,
                text=True,
                check=True,
            )

            def git(*command):
                return subprocess.run(
                    ["git", "-C", str(source), *command],
                    capture_output=True,
                    text=True,
                    check=True,
                )

            git("config", "user.email", "senpai@example.com")
            git("config", "user.name", "Senpai Test")
            tracked = source / "tracked.txt"
            tracked.write_text("old\n")
            git("add", "tracked.txt")
            git("commit", "-m", "old")
            old_revision = git("rev-parse", "HEAD").stdout.strip()
            tracked.write_text("new\n")
            git("commit", "-am", "new")

            _check_runner_source(str(source), old_revision)

            self.assertEqual(git("status", "--porcelain").stdout, "")

    def test_preflight_discovers_and_assigns_visible_gpus_sequentially(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                gpus_per_student=1,
            )
            specs = [student(run_args, "fern"), student(run_args, "tanjiro")]

            with (
                patch("senpai.launch.docker_backend._check_runner_source"),
                patch("senpai.launch.docker_backend._check_docker"),
                patch(
                    "senpai.launch.docker_backend._docker_gpu_indices",
                    return_value=["2", "5", "7"],
                ) as gpu_indices,
                patch(
                    "senpai.launch.docker_backend._check_container_names_available"
                ),
                patch("senpai.launch.docker_backend._check_image"),
            ):
                plan = preflight_docker(run_args, specs)

            self.assertEqual([role.devices for role in plan.roles], [("2",), ("5",)])
            gpu_indices.assert_called_once_with(run_args.student_image)
            self.assertFalse(plan.run_root.exists())

    def test_explicit_gpu_map_overrides_automatic_placement_but_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                gpus_per_student=1,
                docker_student_gpu_ids="fern:5,tanjiro:2",
            )
            specs = [student(run_args, "fern"), student(run_args, "tanjiro")]

            with (
                patch("senpai.launch.docker_backend._check_runner_source"),
                patch("senpai.launch.docker_backend._check_docker"),
                patch(
                    "senpai.launch.docker_backend._docker_gpu_indices",
                    return_value=["2", "5"],
                ),
                patch(
                    "senpai.launch.docker_backend._check_container_names_available"
                ),
                patch("senpai.launch.docker_backend._check_image"),
            ):
                plan = preflight_docker(run_args, specs)

            self.assertEqual([role.devices for role in plan.roles], [("5",), ("2",)])

    def test_preflight_rejects_insufficient_visible_gpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                gpus_per_student=1,
            )
            specs = [student(run_args, "fern"), student(run_args, "tanjiro")]

            with (
                patch("senpai.launch.docker_backend._check_runner_source"),
                patch("senpai.launch.docker_backend._check_docker"),
                patch(
                    "senpai.launch.docker_backend._docker_gpu_indices",
                    return_value=["0"],
                ),
                patch("senpai.launch.docker_backend._check_image"),
            ):
                with self.assertRaisesRegex(RuntimeError, "only 1 are visible"):
                    preflight_docker(run_args, specs)

    def test_dry_run_is_docker_free_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                gpus_per_student=1,
            )
            output = io.StringIO()

            with (
                patch("senpai.launch.docker_backend.subprocess.run") as run,
                redirect_stdout(output),
            ):
                launch_docker(run_args, [student(run_args)])

            self.assertIn("device=0", output.getvalue())
            run.assert_not_called()
            self.assertFalse((Path(tmp) / "runs").exists())


class DockerLaunchTests(unittest.TestCase):
    @staticmethod
    def _create_workdir(_repo_url, workdir, _revision):
        workdir.mkdir(parents=True)

    def test_launch_waits_for_every_role_then_opens_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "datasets"
            data_dir.mkdir()
            run_args = args(
                docker_run_root=str(root / "custom runs"),
                data_dir=str(data_dir),
                dry_run=False,
            )
            specs = [
                student(run_args),
                build_advisor_spec(run_args, run_args.tag, ["fern"], {}),
            ]
            launched_commands = []

            def run(command, **_kwargs):
                if list(command[:2]) == ["docker", "container"]:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=json.dumps(
                            {
                                "State": {
                                    "Status": "running",
                                    "Restarting": False,
                                    "ExitCode": 0,
                                }
                            }
                        ),
                        stderr="",
                    )
                launched_commands.append(command)
                name = command[command.index("--name") + 1]
                ready_file = roles_by_name[name].ready_file
                ready_file.parent.mkdir(parents=True, exist_ok=True)
                ready_file.write_text(
                    json.dumps(
                        {
                            "pid": 7,
                            "phase": "start-gate",
                            "deadline": 1_000_000_000,
                        }
                    )
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout=f"{name}-id\n",
                    stderr="",
                )

            output = io.StringIO()
            sentinel_gid = os.getgid() + 1000
            with (
                patch(
                    "senpai.launch.docker_backend.os.getgid",
                    return_value=sentinel_gid,
                ),
                patch(
                    "senpai.launch.docker_backend._prepare_runner_workdir",
                    side_effect=self._create_workdir,
                ),
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    side_effect=run,
                ),
                redirect_stdout(output),
            ):
                plan = plan_docker(run_args, specs)
                roles_by_name = {role.container_name: role for role in plan.roles}
                launch_docker(run_args, specs, plan)

            self.assertTrue((plan.gate_root / ".launch").is_file())
            self.assertTrue(plan.gate_root.stat().st_mode & 0o010)
            self.assertEqual(
                (plan.gate_root / ".launch").stat().st_mode & 0o777,
                0o640,
            )
            self.assertEqual(
                plan.roles[0].ready_file.relative_to(plan.roles[0].state_root),
                Path("openhands_state/controller-lease.json"),
            )
            self.assertEqual(
                plan.roles[1].ready_file.relative_to(plan.roles[1].state_root),
                Path(
                    "aws-r1/advisor/openhands_state/controller-lease.json"
                ),
            )
            for role in plan.roles:
                self.assertTrue(role.ready_file.is_file())
                self.assertEqual(role.env_file.stat().st_mode & 0o777, 0o600)
            for role, command in zip(plan.roles, launched_commands, strict=True):
                group_ids = {
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "--group-add"
                }
                mounted_paths = (
                    role.workdir,
                    role.state_root,
                    plan.gate_root,
                    data_dir,
                )
                self.assertEqual(
                    group_ids,
                    {
                        *(str(path.stat().st_gid) for path in mounted_paths),
                        str(CONTAINER_IMAGE_GROUP_ID),
                    },
                )
                self.assertEqual(
                    command[command.index("--user") + 1],
                    f"{CONTAINER_USER_ID}:{data_dir.stat().st_gid}",
                )
                self.assertNotIn(str(sentinel_gid), group_ids)

            lifecycle = output.getvalue()
            resolved_run_root = str(Path(run_args.docker_run_root).resolve())
            self.assertEqual(lifecycle.count("python3 k8s/docker.py"), 3)
            self.assertEqual(lifecycle.count("--run-root"), 3)
            self.assertEqual(lifecycle.count(shlex.quote(resolved_run_root)), 3)

    def test_remote_launch_suppresses_host_lifecycle_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = Path(tmp) / "payload.json"
            payload.write_text(
                json.dumps(
                    {
                        "args": {"dry_run": False},
                        "roles": [
                            {
                                "role": "student",
                                "name": "fern",
                                "env": {},
                                "secrets": {},
                            }
                        ],
                    }
                )
            )

            with (
                patch("senpai.launch.remote.preflight_docker", return_value="plan"),
                patch("senpai.launch.remote.launch_docker") as launch,
            ):
                remote.launch_from_payload(payload)

            self.assertFalse(payload.exists())
            self.assertEqual(launch.call_args.args[2], "plan")
            self.assertEqual(launch.call_args.kwargs, {"show_lifecycle": False})

    def test_workdir_failure_removes_partial_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                dry_run=False,
            )
            plan = plan_docker(run_args, [student(run_args)])

            with (
                patch(
                    "senpai.launch.docker_backend._prepare_runner_workdir",
                    side_effect=RuntimeError("clone failed"),
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "clone failed"):
                    launch_docker(run_args, [student(run_args)], plan)

            self.assertFalse(plan.run_root.exists())

    def test_start_failure_removes_every_created_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                dry_run=False,
            )
            specs = [student(run_args, "fern"), student(run_args, "tanjiro")]
            plan = plan_docker(run_args, specs)
            docker_runs = 0

            def run(command, **_kwargs):
                nonlocal docker_runs
                if list(command[:2]) == ["docker", "run"]:
                    docker_runs += 1
                    if docker_runs == 2:
                        plan.roles[1].cid_file.write_text("failed-container-id")
                    return SimpleNamespace(
                        returncode=0 if docker_runs == 1 else 1,
                        stdout="container-id\n" if docker_runs == 1 else "",
                        stderr="start failed" if docker_runs == 2 else "",
                    )
                if command[:3] == ["docker", "rm", "--force"]:
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                return SimpleNamespace(returncode=1, stdout="", stderr="")

            with (
                patch(
                    "senpai.launch.docker_backend._prepare_runner_workdir",
                    side_effect=self._create_workdir,
                ),
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    side_effect=run,
                ) as subprocess_run,
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "start failed"):
                    launch_docker(run_args, specs, plan)

            removed = [
                invocation.args[0][-1]
                for invocation in subprocess_run.call_args_list
                if invocation.args[0][:3] == ["docker", "rm", "--force"]
            ]
            self.assertEqual(removed, ["failed-container-id", "container-id"])
            self.assertFalse(plan.run_root.exists())

    def test_failed_container_removal_preserves_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                dry_run=False,
            )
            plan = plan_docker(run_args, [student(run_args)])

            def run(command, **_kwargs):
                if list(command[:2]) == ["docker", "run"]:
                    plan.roles[0].cid_file.write_text("container-id")
                    return SimpleNamespace(
                        returncode=1,
                        stdout="",
                        stderr="start failed",
                    )
                if command[:3] == ["docker", "rm", "--force"]:
                    return SimpleNamespace(
                        returncode=1,
                        stdout="",
                        stderr="daemon unavailable",
                    )
                return SimpleNamespace(returncode=1, stdout="", stderr="")

            with (
                patch(
                    "senpai.launch.docker_backend._prepare_runner_workdir",
                    side_effect=self._create_workdir,
                ),
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    side_effect=run,
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rollback was incomplete",
                ):
                    launch_docker(run_args, [student(run_args)], plan)

            self.assertTrue(plan.run_root.is_dir())
            self.assertEqual(
                plan.roles[0].cid_file.read_text(),
                "container-id",
            )

    def test_failed_run_state_removal_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                dry_run=False,
            )
            plan = plan_docker(run_args, [student(run_args)])

            with (
                patch(
                    "senpai.launch.docker_backend._prepare_runner_workdir",
                    side_effect=RuntimeError("clone failed"),
                ),
                patch(
                    "senpai.launch.docker_backend.shutil.rmtree",
                    side_effect=OSError("permission denied"),
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "could not remove run state",
                ):
                    launch_docker(run_args, [student(run_args)], plan)

            self.assertTrue(plan.run_root.is_dir())

    def test_readiness_detects_exited_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(docker_run_root=str(Path(tmp) / "runs"))
            plan = plan_docker(run_args, [student(run_args)])
            plan.run_root.mkdir(parents=True)
            plan.roles[0].ready_file.parent.mkdir(parents=True)
            plan.roles[0].ready_file.touch()
            state = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "State": {
                            "Status": "exited",
                            "Restarting": False,
                            "ExitCode": 1,
                        }
                    }
                ),
                stderr="",
            )

            with patch(
                "senpai.launch.docker_backend.subprocess.run",
                return_value=state,
            ):
                with self.assertRaisesRegex(RuntimeError, "status=exited"):
                    _wait_until_ready(plan, 5)

    def test_readiness_wait_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(docker_run_root=str(Path(tmp) / "runs"))
            plan = plan_docker(run_args, [student(run_args)])
            plan.run_root.mkdir(parents=True)
            plan.roles[0].ready_file.parent.mkdir(parents=True)
            plan.roles[0].ready_file.write_text(
                json.dumps(
                    {
                        "pid": 7,
                        "phase": "startup",
                        "deadline": 1_000_000_000,
                    }
                )
            )
            running = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "State": {
                            "Status": "running",
                            "Restarting": False,
                            "ExitCode": 0,
                        }
                    }
                ),
                stderr="",
            )

            with (
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    return_value=running,
                ),
                patch(
                    "senpai.launch.docker_backend.time.monotonic",
                    side_effect=[0.0, 2.0],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Timed out after 1s"):
                    _wait_until_ready(plan, 1)


class DockerLifecycleTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(run_root: Path) -> tuple[Path, list[dict[str, str]]]:
        roles = [
            {
                "key": "student-fern",
                "container": "senpai-aws-r1-student-fern",
            },
            {
                "key": "advisor",
                "container": "senpai-aws-r1-advisor",
            },
        ]
        run_path = run_root / "aws-r1"
        run_path.mkdir(parents=True)
        (run_path / "manifest.json").write_text(
            json.dumps({"tag": "aws-r1", "roles": roles}),
            encoding="utf-8",
        )
        return run_path, roles

    def test_status_reports_each_container_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            _, roles = self._write_manifest(run_root)
            states = {
                roles[0]["container"]: {
                    "State": {
                        "Status": "running",
                        "Health": {"Status": "healthy"},
                    },
                    "RestartCount": 0,
                },
                roles[1]["container"]: {
                    "State": {
                        "Status": "running",
                        "Health": {"Status": "starting"},
                    },
                    "RestartCount": 2,
                },
            }

            def run(command, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(states[command[-1]]),
                    stderr="",
                )

            output = io.StringIO()
            with (
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    side_effect=run,
                ),
                redirect_stdout(output),
            ):
                status_docker("aws-r1", str(run_root))

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "student-fern: container=senpai-aws-r1-student-fern "
                "state=running health=healthy restarts=0",
                "advisor: container=senpai-aws-r1-advisor "
                "state=running health=starting restarts=2",
            ],
        )

    def test_logs_selects_role_and_passes_follow_and_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            self._write_manifest(run_root)
            with patch(
                "senpai.launch.docker_backend.subprocess.run"
            ) as run:
                logs_docker(
                    "aws-r1",
                    str(run_root),
                    role_key="advisor",
                    follow=True,
                    tail=80,
                )

        run.assert_called_once_with(
            [
                "docker",
                "logs",
                "--tail",
                "80",
                "--follow",
                "senpai-aws-r1-advisor",
            ],
            check=True,
        )

    def test_terminate_stops_containers_and_removes_private_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            run_path, roles = self._write_manifest(run_root)
            private_env = run_path / "roles" / "student-fern" / "role.env"
            private_env.parent.mkdir(parents=True)
            private_env.write_text("GITHUB_TOKEN=secret\n", encoding="utf-8")

            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            output = io.StringIO()
            with (
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    return_value=completed,
                ) as run,
                redirect_stdout(output),
            ):
                terminate_docker("aws-r1", str(run_root))

            commands = [call.args[0] for call in run.call_args_list]
            containers = [role["container"] for role in roles]
            self.assertIn(
                ["docker", "stop", "--time", "90", *containers],
                commands,
            )
            self.assertIn(["docker", "rm", *containers], commands)
            self.assertFalse(run_path.exists())
            self.assertIn("removed its private run state", output.getvalue())

    def test_terminate_preserves_state_when_container_presence_is_uncertain(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            run_path, _ = self._write_manifest(run_root)
            unavailable = SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Cannot connect to the Docker daemon",
            )

            with (
                patch(
                    "senpai.launch.docker_backend.subprocess.run",
                    return_value=unavailable,
                ),
                self.assertRaisesRegex(RuntimeError, "Run state was preserved"),
            ):
                terminate_docker("aws-r1", str(run_root))

            self.assertTrue(run_path.is_dir())


if __name__ == "__main__":
    unittest.main()
