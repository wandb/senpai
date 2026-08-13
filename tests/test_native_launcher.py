import io
import json
import os
import plistlib
import stat
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


class NativeTmuxTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tmux_tmp = tempfile.TemporaryDirectory(
            prefix="senpai-tmux-test-",
            dir="/tmp",
        )
        self.tmux_root_patch = patch.object(
            native_backend,
            "DEFAULT_NATIVE_TMUX_ROOT",
            str(Path(self.tmux_tmp.name) / "t"),
        )
        self.tmux_root_patch.start()
        self.addCleanup(self.tmux_root_patch.stop)
        self.addCleanup(self.tmux_tmp.cleanup)


class NativePlanTests(NativeTmuxTestCase):
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
        self.assertEqual(socket_roots, {str(role.tmux_root) for role in plan.roles})
        self.assertEqual(len(socket_roots), 2)
        self.assertTrue(
            socket_roots.isdisjoint(str(role.tmp_root) for role in plan.roles)
        )

    def test_tmux_socket_root_stays_below_the_macos_path_limit(self):
        with patch.object(
            native_backend,
            "DEFAULT_NATIVE_TMUX_ROOT",
            "/Users/ec2-user/.senpai/t",
        ):
            for campaign in ("maple", "cedar"):
                run_root = Path(
                    f"/Users/ec2-user/.senpai/native/mlxfast-{campaign}-20260804"
                )
                for student_name in ("frieren", "tanjiro"):
                    role_key = f"student-{campaign}-{student_name}"
                    legacy_socket = (
                        run_root / "roles" / role_key / "tmp" / "tmux-501" / "openhands"
                    )
                    socket = (
                        native_backend._tmux_root(run_root, role_key)
                        / f"tmux-{os.getuid()}"
                        / "openhands"
                    )
                    with self.subTest(role=role_key):
                        self.assertGreaterEqual(
                            len(os.fsencode(legacy_socket)),
                            native_backend.DARWIN_UNIX_SOCKET_PATH_MAX,
                        )
                        self.assertLess(
                            len(os.fsencode(socket)),
                            native_backend.DARWIN_UNIX_SOCKET_PATH_MAX,
                        )

    def test_tmux_socket_root_rejects_an_overlong_base(self):
        with (
            patch.object(
                native_backend,
                "DEFAULT_NATIVE_TMUX_ROOT",
                "/private/tmp/" + ("x" * 100),
            ),
            self.assertRaisesRegex(ValueError, "too long for the macOS tmux socket"),
        ):
            native_backend._tmux_root(Path("/private/tmp/run"), "advisor")

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


class NativeLaunchTests(NativeTmuxTestCase):
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
            manifest = json.loads((plan.run_root / "manifest.json").read_text())
            plist = plistlib.loads(role.plist.read_bytes())

            self.assertEqual(role.descriptor.stat().st_mode & 0o777, 0o600)
            self.assertEqual(role.plist.stat().st_mode & 0o777, 0o600)
            self.assertEqual(role.descriptor.stat().st_uid, os.getuid())
            self.assertEqual(role.workdir.stat().st_uid, os.getuid())
            self.assertEqual(plan.run_root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(role.home.stat().st_mode & 0o777, 0o700)
            self.assertEqual(role.state_root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(role.log_root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(role.tmux_root.stat().st_mode & 0o777, 0o700)
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
                descriptor["environment"]["TMUX_TMPDIR"], str(role.tmux_root)
            )
            self.assertEqual(manifest["roles"][0]["tmux_root"], str(role.tmux_root))
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
            self.assertFalse(plan.roles[0].tmux_root.exists())

    def test_launch_preserves_a_preexisting_tmux_socket_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_args = args(Path(tmp) / "runs")
            spec = student()
            plan = plan_native(run_args, [spec])
            role = plan.roles[0]
            role.tmux_root.mkdir(parents=True)
            marker = role.tmux_root / "stale-server"
            marker.write_text("preserve")
            with (
                patch(
                    "senpai.launch.native_backend._prepare_runner_workdir",
                    side_effect=create_runner_checkout,
                ),
                self.assertRaisesRegex(RuntimeError, "tmux socket root already exists"),
            ):
                launch_native(run_args, [spec], plan, show_lifecycle=False)

            self.assertEqual(marker.read_text(), "preserve")
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


class NativeInventoryTests(unittest.TestCase):
    def test_installed_role_keys_discovers_only_valid_campaign_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for role in ("advisor", "student-fern"):
                (root / f"com.wandb.senpai.mlxfast-r1.{role}.plist").touch()
            metadata = SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_uid=0)

            with (
                patch.object(native_backend, "LAUNCH_DAEMON_ROOT", root),
                patch.object(Path, "lstat", return_value=metadata),
            ):
                roles = native_backend._installed_role_keys("mlxfast-r1")

        self.assertEqual(roles, {"advisor", "student-fern"})

    def test_installed_role_keys_rejects_untrusted_inventory_entries(self):
        cases = {
            "symbolic link": SimpleNamespace(
                st_mode=stat.S_IFLNK | 0o777,
                st_uid=0,
            ),
            "non-root owner": SimpleNamespace(
                st_mode=stat.S_IFREG | 0o644,
                st_uid=501,
            ),
            "group-writable file": SimpleNamespace(
                st_mode=stat.S_IFREG | 0o664,
                st_uid=0,
            ),
        }
        for name, metadata in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "com.wandb.senpai.mlxfast-r1.student-fern.plist").touch()
                with (
                    patch.object(native_backend, "LAUNCH_DAEMON_ROOT", root),
                    patch.object(Path, "lstat", return_value=metadata),
                    self.assertRaisesRegex(RuntimeError, "not root-controlled"),
                ):
                    native_backend._installed_role_keys("mlxfast-r1")


class NativeLifecycleTests(NativeTmuxTestCase):
    def setUp(self):
        super().setUp()
        self.installed_roles_patch = patch.object(
            native_backend,
            "_installed_role_keys",
            return_value={"student-fern"},
        )
        self.installed_roles_patch.start()
        self.addCleanup(self.installed_roles_patch.stop)

    def write_manifest(
        self,
        root: Path,
        *,
        include_tmux: bool = False,
    ) -> tuple[Path, dict]:
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
        if include_tmux:
            tmux_root = native_backend._tmux_root(run_path, "student-fern")
            tmux_root.mkdir(parents=True)
            manifest["roles"][0]["tmux_root"] = str(tmux_root)
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

    def test_terminate_removes_the_recorded_tmux_root_after_bootout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run_path, manifest = self.write_manifest(root, include_tmux=True)
            tmux_root = Path(manifest["roles"][0]["tmux_root"])

            def uninstall(_domain, _role):
                self.assertTrue(tmux_root.exists())

            with patch(
                "senpai.launch.native_backend._uninstall_recorded_role",
                side_effect=uninstall,
            ):
                terminate_native("mlxfast-r1", str(root))

            self.assertFalse(tmux_root.exists())
            self.assertFalse(run_path.exists())

    def test_terminate_rejects_an_unexpected_tmux_root_and_preserves_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run_path, manifest = self.write_manifest(root)
            unexpected = Path(tmp) / "do-not-delete"
            unexpected.mkdir()
            manifest["roles"][0]["tmux_root"] = str(unexpected)
            (run_path / "manifest.json").write_text(json.dumps(manifest))

            with (
                patch("senpai.launch.native_backend._uninstall_recorded_role"),
                self.assertRaisesRegex(RuntimeError, "unexpected tmux root"),
            ):
                terminate_native("mlxfast-r1", str(root))

            self.assertTrue(unexpected.exists())
            self.assertTrue(run_path.exists())

    def test_terminate_does_not_delete_another_runs_valid_tmux_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run_path, manifest = self.write_manifest(root)
            sibling = native_backend._tmux_root(
                root / "another-run",
                "student-fern",
            )
            sibling.mkdir(parents=True)
            manifest["roles"][0]["tmux_root"] = str(sibling)
            (run_path / "manifest.json").write_text(json.dumps(manifest))

            with (
                patch("senpai.launch.native_backend._uninstall_recorded_role"),
                self.assertRaisesRegex(RuntimeError, "unexpected tmux root"),
            ):
                terminate_native("mlxfast-r1", str(root))

            self.assertTrue(sibling.exists())
            self.assertTrue(run_path.exists())

    def test_terminate_rejects_an_unrelated_launchdaemon_before_sudo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run_path, manifest = self.write_manifest(root)
            manifest["roles"].append(
                {
                    "key": "student-tanjiro",
                    "label": "com.vendor.security-agent",
                    "plist": (
                        "/Library/LaunchDaemons/com.vendor.security-agent.plist"
                    ),
                }
            )
            (run_path / "manifest.json").write_text(json.dumps(manifest))
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch(
                    "senpai.launch.native_backend._sudo_run",
                    return_value=completed,
                ) as sudo,
                self.assertRaisesRegex(RuntimeError, "unexpected native role"),
            ):
                terminate_native("mlxfast-r1", str(root))

            sudo.assert_not_called()
            self.assertTrue(run_path.exists())

    def test_terminate_rejects_a_manifest_that_omits_an_installed_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run_path, _manifest = self.write_manifest(root)

            with (
                patch.object(
                    native_backend,
                    "_installed_role_keys",
                    return_value={"student-fern", "student-tanjiro"},
                    create=True,
                ),
                patch.object(
                    native_backend,
                    "_uninstall_recorded_role",
                ) as uninstall,
                self.assertRaisesRegex(RuntimeError, "omits installed native role"),
            ):
                terminate_native("mlxfast-r1", str(root))

            uninstall.assert_not_called()
            self.assertTrue(run_path.exists())

    def test_terminate_retry_boots_out_a_loaded_job_after_its_plist_was_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run_path, manifest = self.write_manifest(root)
            failed = SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="temporary bootout failure",
            )
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch.object(
                    native_backend,
                    "_installed_role_keys",
                    side_effect=(
                        {"student-fern"},
                        set(),
                    ),
                ),
                patch.object(
                    native_backend,
                    "_job_state",
                    return_value=("running", 42),
                ),
                patch.object(
                    native_backend,
                    "_sudo_run",
                    side_effect=(failed, completed, completed, completed),
                ) as sudo,
            ):
                with self.assertRaisesRegex(RuntimeError, "temporary bootout failure"):
                    terminate_native("mlxfast-r1", str(root))
                self.assertTrue(run_path.exists())

                terminate_native("mlxfast-r1", str(root))

            bootouts = [
                call.args[0]
                for call in sudo.call_args_list
                if call.args[0][1] == "bootout"
            ]
            self.assertEqual(len(bootouts), 2)
            self.assertEqual(
                bootouts[0],
                [
                    "/bin/launchctl",
                    "bootout",
                    f"system/{manifest['roles'][0]['label']}",
                ],
            )
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
