import io
import json
import os
import plistlib
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from senpai.launch import native_backend
from senpai.launch.native_backend import (
    _prepare_runner_workdir,
    _bootstrap_role,
    _uninstall_role,
    launch_native,
    logs_native,
    plan_native,
    preflight_native,
    run_from_payload,
    run_role,
    status_native,
    terminate_native,
)
from senpai.launch.specs import RoleSpec


def args(root: Path, **overrides):
    values = {
        "tag": "mlxfast-r1",
        "repo_revision": "a" * 40,
        "native_run_root": str(root),
        "native_ready_timeout_s": 30,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def student(secret: str = "service-secret", *, name: str = "fern") -> RoleSpec:
    return RoleSpec(
        role="student",
        name=name,
        env={
            "PROBLEM_DIR": "target",
            "REPO_REVISION": "a" * 40,
            "RESEARCH_TAG": "mlxfast-r1",
        },
        secrets={
            "GITHUB_TOKEN": secret,
            "OPENAI_API_KEY": "openai-secret",
        },
    )


def create_runner_checkout(_source: Path, workdir: Path, _revision: str) -> None:
    (workdir / "k8s").mkdir(parents=True)
    (workdir / "k8s" / "entrypoint-student.sh").write_text("exit 0\n")
    (workdir / "k8s" / "native.py").write_text("# native runner\n")


class NativePlanTests(unittest.TestCase):
    def test_plan_gives_every_role_private_roots_and_a_launchd_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_native(args(Path(tmp) / "runs"), [student()])
            role = plan.roles[0]

        self.assertEqual(role.label, "com.wandb.senpai.mlxfast-r1.student-fern")
        self.assertEqual(plan.domain, "system")
        self.assertEqual(
            role.launchd_plist,
            Path("/Library/LaunchDaemons") / f"{role.label}.plist",
        )
        self.assertEqual(role.home.name, "home")
        self.assertEqual(role.workdir.name, "workspace")
        self.assertEqual(role.log_root.name, "logs")
        self.assertEqual(role.state_root.name, "state")
        self.assertTrue(role.lease.is_relative_to(role.state_root))

    def test_roles_get_distinct_tmux_socket_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_native(
                args(Path(tmp) / "runs"),
                [student(), student(name="tanjiro")],
            )

        socket_roots = {
            native_backend._role_environment(plan, role)["TMUX_TMPDIR"]
            for role in plan.roles
        }
        self.assertEqual(socket_roots, {str(role.tmp_root) for role in plan.roles})
        self.assertEqual(len(socket_roots), 2)

    def test_preflight_requires_apple_silicon_before_host_mutation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            native_backend.sys, "platform", "linux"
        ), self.assertRaisesRegex(RuntimeError, "Apple Silicon macOS"):
            preflight_native(args(Path(tmp) / "runs"), [student()])

    def test_runner_workspace_clones_the_current_checkout_at_exact_revision(self):
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp, patch(
            "senpai.launch.native_backend._run",
            return_value=completed,
        ) as run:
            source = Path(tmp) / "source"
            workdir = Path(tmp) / "role" / "workspace"
            source.mkdir()
            _prepare_runner_workdir(source, workdir, "b" * 40)

        self.assertEqual(
            run.call_args_list[0].args[0],
            [
                "git",
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                str(source),
                str(workdir),
            ],
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["git", "-C", str(workdir), "checkout", "--detach", "b" * 40],
        )

    def test_preflight_requires_noninteractive_sudo_for_the_system_domain(self):
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(native_backend.sys, "platform", "darwin"),
                patch.object(
                    native_backend.platform,
                    "machine",
                    return_value="arm64",
                ),
                patch.object(
                    native_backend.shutil,
                    "which",
                    return_value="/usr/bin/tool",
                ),
                patch.object(native_backend, "_check_source_revision"),
                patch.object(native_backend, "_job_state", return_value=None),
                patch.object(
                    native_backend,
                    "_sudo_run",
                    return_value=completed,
                ) as sudo,
            ):
                plan = preflight_native(args(Path(tmp) / "runs"), [student()])

        self.assertEqual(plan.domain, "system")
        self.assertEqual(
            [call.args[0] for call in sudo.call_args_list],
            [["true"], ["/bin/launchctl", "print", "system"]],
        )

    def test_privileged_commands_are_always_noninteractive(self):
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch.object(native_backend, "_run", return_value=completed) as run:
            result = native_backend._sudo_run(["/bin/launchctl", "print", "system"])

        self.assertIs(result, completed)
        run.assert_called_once_with(
            ["sudo", "-n", "/bin/launchctl", "print", "system"]
        )


class NativeLaunchTests(unittest.TestCase):
    def test_launch_writes_private_state_and_secret_free_persistent_plist(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(Path(tmp) / "runs")
            spec = student()
            plan = plan_native(run_args, [spec])
            with (
                patch(
                    "senpai.launch.native_backend._prepare_runner_workdir",
                    side_effect=create_runner_checkout,
                ),
                patch("senpai.launch.native_backend._bootstrap_role") as bootstrap,
                patch("senpai.launch.native_backend._wait_until_ready"),
            ):
                launch_native(run_args, [spec], plan, show_lifecycle=False)

            role = plan.roles[0]
            descriptor = json.loads(role.descriptor.read_text())
            plist = plistlib.loads(role.plist.read_bytes())

            self.assertEqual(role.descriptor.stat().st_mode & 0o777, 0o600)
            self.assertEqual(role.plist.stat().st_mode & 0o777, 0o600)
            self.assertEqual(role.descriptor.stat().st_uid, os.getuid())
            self.assertEqual(role.workdir.stat().st_uid, os.getuid())
            self.assertEqual(plan.run_root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(role.home.stat().st_mode & 0o777, 0o700)
            self.assertEqual(role.state_root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(role.log_root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(role.workdir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(plan.launch_gate.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                descriptor["environment"]["BROWSER_USE_DISABLE_EXTENSIONS"],
                "1",
            )
            self.assertEqual(descriptor["environment"]["HOME"], str(role.home))
            self.assertEqual(
                descriptor["environment"]["SENPAI_WORKDIR"], str(role.workdir)
            )
            self.assertEqual(
                descriptor["environment"]["SENPAI_LOGDIR"], str(role.state_root)
            )
            self.assertEqual(
                descriptor["environment"]["TMUX_TMPDIR"], str(role.tmp_root)
            )
            command_path = descriptor["environment"]["PATH"].split(os.pathsep)
            self.assertIn("/opt/homebrew/bin", command_path)
            self.assertIn("/opt/homebrew/opt/gettext/bin", command_path)
            self.assertIn("/usr/local/bin", command_path)
            self.assertTrue(plist["RunAtLoad"])
            self.assertTrue(plist["KeepAlive"])
            self.assertEqual(plist["UserName"], plan.user_name)
            self.assertEqual(plist["GroupName"], plan.group_name)
            self.assertEqual(plist["ProcessType"], "Interactive")
            self.assertEqual(
                plist["ProgramArguments"][:3],
                ["/usr/bin/caffeinate", "-is", native_backend.sys.executable],
            )
            self.assertNotIn("service-secret", json.dumps(plist))
            self.assertNotIn("openai-secret", json.dumps(plist))
            self.assertNotIn("service-secret", role.stdout_log.read_text())
            self.assertNotIn("service-secret", role.stderr_log.read_text())
            bootstrap.assert_called_once_with(plan, role)

    def test_launchdaemon_is_installed_root_owned_before_system_bootstrap(self):
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_native(args(Path(tmp) / "runs"), [student()])
            role = plan.roles[0]
            role.role_root.mkdir(parents=True)
            role.plist.write_bytes(b"plist")
            with patch.object(
                native_backend,
                "_sudo_run",
                return_value=completed,
            ) as sudo:
                _bootstrap_role(plan, role)

        self.assertEqual(
            [call.args[0] for call in sudo.call_args_list],
            [
                [
                    "/usr/bin/install",
                    "-o",
                    "root",
                    "-g",
                    "wheel",
                    "-m",
                    "0644",
                    str(role.plist),
                    str(role.launchd_plist),
                ],
                [
                    "/bin/launchctl",
                    "bootstrap",
                    "system",
                    str(role.launchd_plist),
                ],
            ],
        )

    def test_partial_launch_unloads_service_and_removes_installed_plist(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(Path(tmp) / "runs")
            spec = student()
            plan = plan_native(run_args, [spec])
            with (
                patch(
                    "senpai.launch.native_backend._prepare_runner_workdir",
                    side_effect=create_runner_checkout,
                ),
                patch(
                    "senpai.launch.native_backend._bootstrap_role",
                    side_effect=RuntimeError("bootstrap failed"),
                ),
                patch("senpai.launch.native_backend._uninstall_role") as uninstall,
                self.assertRaisesRegex(RuntimeError, "bootstrap failed"),
            ):
                launch_native(run_args, [spec], plan, show_lifecycle=False)

            uninstall.assert_called_once_with(plan, plan.roles[0])
            self.assertFalse(plan.run_root.exists())

    def test_distributed_launch_waits_for_leases_without_opening_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(Path(tmp) / "runs")
            spec = student()
            plan = plan_native(run_args, [spec])
            with (
                patch(
                    "senpai.launch.native_backend._prepare_runner_workdir",
                    side_effect=create_runner_checkout,
                ),
                patch("senpai.launch.native_backend._bootstrap_role"),
                patch("senpai.launch.native_backend._wait_until_ready") as wait,
            ):
                launch_native(
                    run_args,
                    [spec],
                    plan,
                    show_lifecycle=False,
                    open_gate=False,
                )

            wait.assert_called_once_with(plan, 30.0)
            self.assertFalse(plan.launch_gate.exists())

    def test_role_runner_uses_private_token_file_without_secret_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir = root / "workspace"
            state = root / "state"
            workdir.mkdir()
            state.mkdir()
            entrypoint = workdir / "entrypoint.sh"
            entrypoint.write_text("exit 0\n")
            descriptor = root / "role.json"
            descriptor.write_text(
                json.dumps(
                    {
                        "entrypoint": str(entrypoint),
                        "environment": {
                            "HOME": str(root / "home"),
                            "PATH": "/usr/bin:/bin",
                        },
                        "role": "student",
                        "secrets": {
                            "GITHUB_TOKEN": "github-secret",
                            "OPENAI_API_KEY": "openai-secret",
                        },
                        "token_file": str(state / "github-token"),
                        "workdir": str(workdir),
                    }
                )
            )
            descriptor.chmod(0o600)
            with (
                patch("senpai.launch.native_backend.os.chdir") as chdir,
                patch("senpai.launch.native_backend.os.execve") as execve,
            ):
                run_role(descriptor)

            executable, argv, environment = execve.call_args.args
            token_file = state / "github-token"
            chdir.assert_called_once_with(str(workdir))
            self.assertEqual(executable, "/bin/bash")
            self.assertEqual(argv, ["bash", str(entrypoint)])
            self.assertNotIn("github-secret", argv)
            self.assertNotIn("GITHUB_TOKEN", environment)
            self.assertEqual(environment["OPENAI_API_KEY"], "openai-secret")
            self.assertEqual(
                environment["SENPAI_GITHUB_TOKEN_FILE"],
                str(token_file),
            )
            self.assertEqual(token_file.read_text(), "github-secret")
            self.assertEqual(token_file.stat().st_mode & 0o777, 0o600)

    def test_launch_payload_is_private_one_use_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "launch.json"
            path.write_text(
                json.dumps(
                    {
                        "args": {
                            "tag": "mlxfast-r1",
                            "repo_revision": "a" * 40,
                            "native_run_root": str(Path(tmp) / "runs"),
                            "native_ready_timeout_s": 30,
                            "dry_run": False,
                        },
                        "roles": [
                            {
                                "role": "student",
                                "name": "fern",
                                "env": {},
                                "secrets": {"GITHUB_TOKEN": "secret"},
                            }
                        ],
                    }
                )
            )
            path.chmod(0o600)
            with (
                patch(
                    "senpai.launch.native_backend.preflight_native",
                    return_value="plan",
                ),
                patch("senpai.launch.native_backend.launch_native") as launch,
            ):
                run_from_payload("launch", path)

            self.assertFalse(path.exists())
            self.assertEqual(launch.call_args.args[2], "plan")
            self.assertEqual(launch.call_args.args[1][0].key, "student-fern")
            self.assertEqual(
                launch.call_args.kwargs,
                {"show_lifecycle": False, "open_gate": False},
            )


class NativeLifecycleTests(unittest.TestCase):
    @staticmethod
    def write_manifest(root: Path) -> tuple[Path, dict]:
        run_path = root / "mlxfast-r1"
        logs = run_path / "logs"
        logs.mkdir(parents=True)
        stdout = logs / "stdout.log"
        stderr = logs / "stderr.log"
        stdout.write_text("hello\n")
        stderr.write_text("warning\n")
        manifest = {
            "domain": "system",
            "tag": "mlxfast-r1",
            "roles": [
                {
                    "key": "student-fern",
                    "label": "com.wandb.senpai.mlxfast-r1.student-fern",
                    "lease": str(run_path / "lease.json"),
                    "plist": (
                        "/Library/LaunchDaemons/"
                        "com.wandb.senpai.mlxfast-r1.student-fern.plist"
                    ),
                    "stdout": str(stdout),
                    "stderr": str(stderr),
                }
            ],
        }
        (run_path / "manifest.json").write_text(json.dumps(manifest))
        return run_path, manifest

    def test_status_combines_launchd_and_controller_lease_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            _, manifest = self.write_manifest(root)
            output = io.StringIO()
            with (
                patch(
                    "senpai.launch.native_backend._job_state",
                    return_value=("running", 42),
                ),
                patch(
                    "senpai.launch.native_backend.lease_is_healthy",
                    return_value=True,
                ),
                redirect_stdout(output),
            ):
                status_native("mlxfast-r1", str(root))

        self.assertEqual(
            output.getvalue().strip(),
            f"student-fern: service={manifest['roles'][0]['label']} "
            "state=running pid=42 health=healthy",
        )

    def test_logs_are_bounded_and_terminate_unloads_before_removing_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run_path, manifest = self.write_manifest(root)
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch(
                "senpai.launch.native_backend.subprocess.run",
                return_value=completed,
            ) as run:
                logs_native("mlxfast-r1", str(root), role_key="student-fern", tail=80)
            self.assertEqual(run.call_args.args[0][:3], ["tail", "-n", "80"])

            with (
                patch("senpai.launch.native_backend._uninstall_recorded_role") as uninstall,
            ):
                terminate_native("mlxfast-r1", str(root))

            uninstall.assert_called_once_with("system", manifest["roles"][0])
            self.assertFalse(run_path.exists())

    def test_uninstall_boots_out_then_removes_the_root_launchdaemon(self):
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_native(args(Path(tmp) / "runs"), [student()])
            role = plan.roles[0]
            with (
                patch.object(
                    native_backend,
                    "_job_state",
                    return_value=("running", 42),
                ),
                patch.object(
                    native_backend,
                    "_sudo_run",
                    return_value=completed,
                ) as sudo,
            ):
                _uninstall_role(plan, role)

        self.assertEqual(
            [call.args[0] for call in sudo.call_args_list],
            [
                [
                    "/bin/launchctl",
                    "bootout",
                    f"system/{role.label}",
                ],
                ["/bin/rm", "-f", str(role.launchd_plist)],
            ],
        )

    def test_uninstall_still_removes_plist_when_bootout_fails(self):
        failed = SimpleNamespace(returncode=1, stdout="", stderr="bootout failed")
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_native(args(Path(tmp) / "runs"), [student()])
            role = plan.roles[0]
            with (
                patch.object(
                    native_backend,
                    "_job_state",
                    return_value=("running", 42),
                ),
                patch.object(
                    native_backend,
                    "_sudo_run",
                    side_effect=(failed, completed),
                ) as sudo,
                self.assertRaisesRegex(RuntimeError, "bootout failed"),
            ):
                _uninstall_role(plan, role)

        self.assertEqual(
            sudo.call_args_list[1].args[0],
            ["/bin/rm", "-f", str(role.launchd_plist)],
        )


class NativeEntrypointTests(unittest.TestCase):
    def test_entrypoints_accept_native_paths_and_student_has_apple_accelerator_path(self):
        root = Path(__file__).resolve().parents[1]
        advisor = (root / "k8s" / "entrypoint-advisor.sh").read_text()
        student_text = (root / "k8s" / "entrypoint-student.sh").read_text()
        for text in (advisor, student_text):
            self.assertIn("SENPAI_WORKDIR", text)
            self.assertIn("SENPAI_LOGDIR", text)
            self.assertIn("SENPAI_SKIP_EDITABLE_INSTALL", text)
            self.assertIn("SENPAI_TMPDIR", text)
        self.assertIn('command -v nvidia-smi', student_text)
        self.assertIn('"$(uname -s)" = "Darwin"', student_text)
        self.assertIn("Accelerator:  Apple Silicon", student_text)
        self.assertIn('git config user.name "senpai-$ADVISOR_NAME"', advisor)
        self.assertIn('git config user.email "senpai-$ADVISOR_NAME@senpai"', advisor)

    def test_payload_cli_names_match_the_aws_fleet_protocol(self):
        text = (Path(__file__).resolve().parents[1] / "k8s" / "native.py").read_text()
        self.assertIn('("preflight-payload", "launch-payload", "run-role")', text)


if __name__ == "__main__":
    unittest.main()
