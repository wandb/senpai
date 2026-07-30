import base64
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from senpai.launch.docker_backend import (
    _check_image,
    _check_runner_source,
    _docker_gpu_indices,
    _env_file_text,
    _path_beneath,
    _prepare_runner_workdir,
    _role_values,
    _wait_until_ready,
    launch_docker,
    plan_docker,
    preflight_docker,
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
        "docker_student_gpu_ids": "",
        "docker_data_dir": "",
        "docker_shm_size": "32g",
        "docker_ready_timeout_s": 120,
        "image": "ghcr.io/wandb/senpai:latest",
        "dry_run": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def student(run_args, name="fern"):
    return build_student_spec(run_args, run_args.tag, name, {})


class LaunchSpecTests(unittest.TestCase):
    def test_student_spec_contains_backend_independent_environment(self):
        spec = build_student_spec(args(), "aws-r1", "fern", {"GITHUB_TOKEN": "secret"})

        self.assertEqual(spec.key, "student-fern")
        self.assertEqual(spec.env["GH_REPO"], "example/problem")
        self.assertEqual(spec.env["STUDENT_NAME"], "fern")
        self.assertEqual(
            spec.env["SENPAI_STATUS_DIR"],
            "/mnt/datasets/.senpai-status/aws-r1",
        )
        decoded = base64.b64decode(spec.env["EXTRA_INSTRUCTIONS_B64"]).decode()
        self.assertIn("AWS experiment", decoded)
        self.assertEqual(spec.secrets["GITHUB_TOKEN"], "secret")

    def test_advisor_spec_lists_students(self):
        spec = build_advisor_spec(args(), "aws-r1", ["fern", "tanjiro"], {})

        self.assertEqual(spec.key, "advisor")
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
                docker_data_dir=str(data_dir),
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
                docker_data_dir=str(data_dir),
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
                docker_data_dir=str(data_dir),
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
                docker_data_dir=str(data_dir),
                pvc_mount_path="/senpai-run/data",
            )

            with self.assertRaisesRegex(ValueError, "pvc_mount_path"):
                plan_docker(run_args, [student(run_args)])

    def test_plan_mounts_private_role_state_and_narrow_shared_status(self):
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
                f"{student_plan.state_root}:/senpai-run",
                student_command,
            )
            self.assertIn(
                f"{plan.status_root}:/senpai-status:ro",
                student_command,
            )
            self.assertIn(
                f"{student_plan.status_dir}:/senpai-status/fern",
                student_command,
            )
            self.assertNotIn(f"{plan.run_root}:/senpai-run", student_command)
            self.assertIn(
                f"{plan.status_root}:/senpai-status:ro",
                advisor_command,
            )
            self.assertFalse(
                any(value.endswith(":/senpai-status/fern") for value in advisor_command)
            )

    def test_role_environment_exposes_readiness_status_and_gate_contract(self):
        values = _role_values(student(args()))

        self.assertEqual(values["HOME"], "/senpai-run/home")
        self.assertEqual(values["SENPAI_READY_FILE"], "/senpai-run/ready")
        self.assertEqual(values["SENPAI_STATUS_DIR"], "/senpai-status")
        self.assertEqual(
            values["SENPAI_LAUNCH_GATE_PATH"],
            "/senpai-status/.launch",
        )
        self.assertEqual(
            values["SENPAI_GIT_CREDENTIAL_FILE"],
            "/senpai-run/secrets/git-credentials",
        )

    def test_runner_checkout_uses_existing_repo_url_and_branch(self):
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
                "runner-branch",
            )

        self.assertEqual(
            run.call_args.args[0],
            [
                "git",
                "clone",
                "--branch",
                "runner-branch",
                "--single-branch",
                "https://github.com/example/runner.git",
                str(workdir),
            ],
        )

    def test_env_file_rejects_multiline_values(self):
        with self.assertRaisesRegex(ValueError, "contains a newline"):
            _env_file_text({"TOKEN": "first\nsecond"})

    def test_multi_gpu_selection_is_one_literal_docker_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                gpus_per_student=2,
                docker_student_gpu_ids="fern:0+2",
            )
            command = list(plan_docker(run_args, [student(run_args)]).roles[0].command)

        self.assertEqual(command[command.index("--gpus") + 1], "device=0,2")


class DockerPreflightTests(unittest.TestCase):
    def test_cpu_only_preflight_runs_the_actual_image(self):
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

        check_image.assert_called_once_with(run_args.image)

    def test_cpu_image_probe_runs_bash_in_the_actual_image(self):
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch(
            "senpai.launch.docker_backend.subprocess.run",
            return_value=completed,
        ) as run:
            _check_image("ghcr.io/wandb/senpai:test")

        self.assertEqual(
            run.call_args.args[0],
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "/bin/bash",
                "ghcr.io/wandb/senpai:test",
                "-c",
                "true",
            ],
        )

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

    def test_runner_branch_is_checked_without_cloning(self):
        result = SimpleNamespace(returncode=2, stdout="", stderr="not found")
        with patch(
            "senpai.launch.docker_backend.subprocess.run",
            return_value=result,
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "Runner branch"):
                _check_runner_source("https://github.com/example/runner.git", "missing")

        self.assertEqual(
            run.call_args.args[0],
            [
                "git",
                "ls-remote",
                "--exit-code",
                "--heads",
                "https://github.com/example/runner.git",
                "refs/heads/missing",
            ],
        )

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
            ):
                plan = preflight_docker(run_args, specs)

            self.assertEqual([role.devices for role in plan.roles], [("2",), ("5",)])
            gpu_indices.assert_called_once_with(run_args.image)
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
    def _create_workdir(_repo_url, workdir, _branch):
        workdir.mkdir(parents=True)

    def test_launch_waits_for_every_role_then_opens_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(
                docker_run_root=str(Path(tmp) / "runs"),
                dry_run=False,
            )
            specs = [
                student(run_args),
                build_advisor_spec(run_args, run_args.tag, ["fern"], {}),
            ]
            plan = plan_docker(run_args, specs)
            roles_by_name = {role.container_name: role for role in plan.roles}

            def run(command, **_kwargs):
                if list(command[:2]) == ["docker", "container"]:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=json.dumps(
                            {
                                "Status": "running",
                                "Restarting": False,
                                "ExitCode": 0,
                            }
                        ),
                        stderr="",
                    )
                name = command[command.index("--name") + 1]
                roles_by_name[name].ready_file.touch()
                return SimpleNamespace(
                    returncode=0,
                    stdout=f"{name}-id\n",
                    stderr="",
                )

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
                launch_docker(run_args, specs, plan)

            self.assertTrue((plan.status_root / ".launch").is_file())
            for role in plan.roles:
                self.assertTrue(role.ready_file.is_file())
                self.assertEqual(role.env_file.stat().st_mode & 0o777, 0o600)

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
            plan.roles[0].state_root.mkdir(parents=True)
            plan.roles[0].ready_file.touch()
            state = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"Status": "exited", "Restarting": False, "ExitCode": 1}
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
            plan.roles[0].state_root.mkdir(parents=True)
            running = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"Status": "running", "Restarting": False, "ExitCode": 0}
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


if __name__ == "__main__":
    unittest.main()
