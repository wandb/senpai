import io
import json
import plistlib
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from k8s import aws_mac as aws_mac_cli
from senpai.launch import aws_mac_backend
from senpai.launch.aws_mac_backend import (
    AwsMacHost,
    AwsMacPlan,
    _cleanup,
    _csv,
    _launchd_canary,
    _metal_toolchain_archive,
    _native_payload,
    _resolve_ami,
    _resolve_hosts,
    _remote_setup_script,
    _run_instance,
    _subnet_map,
    _user_data,
    _validate_network,
    _wait_instance,
    distribute_roles,
    launch_aws_mac,
    logs_aws_mac,
    preflight_aws_mac,
    status_aws_mac,
)
from senpai.launch.aws_backend import AwsCommandError, AwsContext
from senpai.launch.specs import RoleSpec


REVISION = "a" * 40


def student(name: str) -> RoleSpec:
    return RoleSpec(
        role="student",
        name=name,
        env={"SENPAI_REPO_REVISION": REVISION},
        secrets={"GITHUB_TOKEN": f"github-{name}"},
    )


def advisor() -> RoleSpec:
    return RoleSpec(
        role="advisor",
        name="advisor",
        env={"SENPAI_REPO_REVISION": REVISION},
        secrets={"GITHUB_TOKEN": "github-advisor"},
    )


def args(state_root: Path, **overrides) -> SimpleNamespace:
    values = {
        "tag": "mlxfast-r1",
        "aws_state_root": str(state_root),
        "aws_instance_type": "mac-m4pro.metal",
        "aws_ttl_hours": 24.0,
        "aws_ready_timeout_s": 1_200,
        "aws_data_timeout_s": 7_200,
        "senpai_repo_revision": REVISION,
        "senpai_repo_url": "https://github.com/wandb/senpai.git",
        "docker_ready_timeout_s": 91,
        "native_ready_timeout_s": 37,
        "dry_run": False,
        "aws_mac_host_ids": "h-a1,h-b2",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def mac_host(
    host_id: str,
    student_name: str,
    *,
    availability_zone: str = "us-east-1a",
    subnet_id: str = "subnet-a1",
) -> AwsMacHost:
    return AwsMacHost(
        host_id=host_id,
        availability_zone=availability_zone,
        subnet_id=subnet_id,
        student=student_name,
    )


def host_description(
    host_id: str,
    *,
    state: str = "available",
    instance_type: str = "mac-m4pro.metal",
    instances: list[dict] | None = None,
    capacity: int = 1,
) -> dict:
    return {
        "HostId": host_id,
        "State": state,
        "AvailabilityZone": "us-east-1a",
        "HostProperties": {"InstanceType": instance_type},
        "Instances": [] if instances is None else instances,
        "AvailableCapacity": {
            "AvailableInstanceCapacity": [
                {
                    "InstanceType": "mac-m4pro.metal",
                    "AvailableCapacity": capacity,
                }
            ]
        },
    }


class AwsMacInputTests(unittest.TestCase):
    def test_csv_and_subnet_map_parse_operator_inputs(self):
        self.assertEqual(_csv(" h-a1, h-b2 ,,"), ("h-a1", "h-b2"))
        self.assertEqual(
            _subnet_map("us-east-1a=subnet-a1, us-east-1b=subnet-b2"),
            {"us-east-1a": "subnet-a1", "us-east-1b": "subnet-b2"},
        )
        with self.assertRaisesRegex(ValueError, "AZ=subnet-id"):
            _subnet_map("subnet-a1")
        with self.assertRaisesRegex(ValueError, "repeats"):
            _subnet_map("us-east-1a=subnet-a1,us-east-1a=subnet-b2")

    def test_resolve_hosts_preserves_operator_order_and_binds_students(self):
        descriptions = [host_description("h-b2"), host_description("h-a1")]
        with patch.object(
            aws_mac_backend,
            "_aws_json",
            return_value={"Hosts": descriptions},
        ):
            hosts = _resolve_hosts(
                AwsContext("us-east-1", "sandbox"),
                ("h-a1", "h-b2"),
                {"us-east-1a": "subnet-a1"},
                [student("fern"), student("tanjiro")],
                "mac-m4pro.metal",
            )

        self.assertEqual(
            [(host.host_id, host.student) for host in hosts],
            [("h-a1", "fern"), ("h-b2", "tanjiro")],
        )

    def test_resolve_hosts_rejects_unavailable_occupied_wrong_or_full_hosts(self):
        cases = [
            (host_description("h-a1", state="pending"), "not available"),
            (
                host_description("h-a1", instances=[{"InstanceId": "i-live"}]),
                "already has an instance",
            ),
            (
                host_description("h-a1", instance_type="mac2.metal"),
                "expected mac-m4pro.metal",
            ),
            (host_description("h-a1", capacity=0), "no free capacity"),
        ]
        for description, message in cases:
            with self.subTest(message=message), patch.object(
                aws_mac_backend,
                "_aws_json",
                return_value={"Hosts": [description]},
            ), self.assertRaisesRegex(RuntimeError, message):
                _resolve_hosts(
                    AwsContext("us-east-1"),
                    ("h-a1",),
                    {"us-east-1a": "subnet-a1"},
                    [student("fern")],
                    "mac-m4pro.metal",
                )

    def test_students_are_one_per_host_and_advisor_is_on_host_zero(self):
        hosts = (
            mac_host("h-a1", "fern"),
            mac_host("h-b2", "tanjiro"),
        )
        groups = distribute_roles(
            [student("fern"), student("tanjiro"), advisor()],
            hosts,
        )

        self.assertEqual(
            [(host.host_id, [spec.key for spec in specs]) for host, specs in groups],
            [
                ("h-a1", ["student-fern", "advisor"]),
                ("h-b2", ["student-tanjiro"]),
            ],
        )
        with self.assertRaisesRegex(ValueError, "one host per student"):
            distribute_roles([student("fern")], hosts)

    def test_ttl_zero_disables_shutdown_but_still_publishes_ssh_host_keys(self):
        no_ttl = _user_data(0)
        self.assertNotIn("/sbin/shutdown", no_ttl)
        self.assertIn("SENPAI_SSH_HOST_KEY", no_ttl)
        self.assertIn("/etc/ssh/ssh_host_*_key.pub", no_ttl)

        with_ttl = _user_data(2.5)
        self.assertIn("/sbin/shutdown -h +150", with_ttl)
        self.assertIn("SENPAI_SSH_HOST_KEY", with_ttl)

    def test_negative_ttl_is_rejected(self):
        run_args = args(
            Path("/tmp/state"),
            aws_ttl_hours=-1,
            gpus_per_student=1,
            data_dir="",
            start_gate_path="",
            aws_volume_gib=250,
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            preflight_aws_mac(run_args, [student("fern")])

    def test_metal_toolchain_archive_is_required_and_has_one_exported_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "is required"):
                _metal_toolchain_archive(SimpleNamespace())

            invalid = root / "invalid.zip"
            invalid.write_bytes(b"not a zip")
            with self.assertRaisesRegex(RuntimeError, "not a zip file"):
                _metal_toolchain_archive(
                    SimpleNamespace(
                        aws_mac_metal_toolchain_archive=str(invalid),
                    )
                )

            wrong_shape = root / "wrong-shape.zip"
            with ZipFile(wrong_shape, "w") as archive:
                archive.writestr("README.txt", b"no exported bundle")
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                _metal_toolchain_archive(
                    SimpleNamespace(
                        aws_mac_metal_toolchain_archive=str(wrong_shape),
                    )
                )

            artifact = root / "MetalToolchain.zip"
            with ZipFile(artifact, "w") as archive:
                archive.writestr(
                    "MetalToolchain-17F109.exportedBundle/Restore/component.dmg",
                    b"metal",
                )

            resolved = _metal_toolchain_archive(
                SimpleNamespace(
                    aws_mac_metal_toolchain_archive=str(artifact),
                )
            )
            self.assertEqual(resolved, artifact.resolve())


class AwsMacInfrastructureValidationTests(unittest.TestCase):
    def test_campaign_ssh_group_is_created_in_the_selected_vpc_with_run_tags(self):
        host = mac_host("h-a1", "fern")
        plan = AwsMacPlan(
            context=AwsContext("us-east-1", "sandbox"),
            account_id="770934259321",
            ami_id="ami-mac",
            root_device="/dev/sda1",
            volume_gib=250,
            security_group_id="sg-a1",
            ssh_cidr="203.0.113.7/32",
            hosts=(host,),
            vpc_id="vpc-a1",
        )
        with patch.object(
            aws_mac_backend,
            "_aws_json",
            return_value={"GroupId": "sg-b1"},
        ) as aws_json:
            group_id = aws_mac_backend._create_ssh_security_group(
                plan,
                name="senpai-mlxfast-r1-ssh-abcd1234",
                tag="mlxfast-r1",
            )

        command = aws_json.call_args.args
        self.assertEqual(group_id, "sg-b1")
        self.assertEqual(command[1:3], ("ec2", "create-security-group"))
        self.assertEqual(command[command.index("--vpc-id") + 1], "vpc-a1")
        tags = json.loads(command[command.index("--tag-specifications") + 1])
        self.assertIn(
            {"Key": "senpai:run", "Value": "mlxfast-r1"},
            tags[0]["Tags"],
        )
        self.assertIn(
            {"Key": "SenpaiPurpose", "Value": "ssh"},
            tags[0]["Tags"],
        )

    def test_ssh_requires_a_pre_authorized_host_key(self):
        command = aws_mac_backend._ssh_base(
            Path("/tmp/state"),
            {"instance_id": "i-test", "public_ip": "198.51.100.1"},
        )

        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertNotIn("StrictHostKeyChecking=accept-new", command)

    def test_ssh_disconnects_stdin_when_no_payload_is_sent(self):
        node = {"instance_id": "i-test", "public_ip": "198.51.100.1"}
        completed = subprocess.CompletedProcess([], 0, stdout=b"", stderr=b"")

        with patch.object(
            aws_mac_backend.subprocess,
            "run",
            return_value=completed,
        ) as run:
            aws_mac_backend._ssh(Path("/tmp/state"), node, "true")

        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_launchd_canary_reconnects_before_removing_the_system_service(self):
        node = {"instance_id": "i-canary", "public_ip": "198.51.100.1"}
        calls = []

        def ssh(_run_dir, _node, command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess([], 0, stdout=b"", stderr=b"")

        with patch.object(aws_mac_backend, "_ssh", side_effect=ssh):
            _launchd_canary(Path("/tmp/state"), node)

        self.assertEqual(len(calls), 3)
        plist = plistlib.loads(calls[0][1]["input_bytes"])
        self.assertEqual(plist["ProcessType"], "Interactive")
        self.assertEqual(
            plist["ProgramArguments"][:2],
            ["/usr/bin/caffeinate", "-is"],
        )
        self.assertIn("launchctl bootstrap system", calls[0][0])
        self.assertIn("launchctl print system/", calls[1][0])
        self.assertIn("state = running", calls[1][0])
        self.assertIn("launchctl bootout system/", calls[2][0])

    def test_remote_setup_smoke_tests_native_tools_for_private_homes(self):
        script = _remote_setup_script(
            SimpleNamespace(
                senpai_repo_url="https://github.com/wandb/senpai.git",
                senpai_repo_revision=REVISION,
                advisor_model="wandb/zai-org/GLM-5.2",
            )
        ).decode()

        self.assertIn("/usr/local/bin/chromium", script)
        self.assertIn(
            "printf '%s\\n' '#!/bin/sh' 'exec \"'\"$chromium_path\"'\" \"$@\"'",
            script,
        )
        self.assertIn("/usr/local/bin/chromium --version", script)
        self.assertNotIn(
            'ln -sf "$chromium_path" /usr/local/bin/chromium',
            script,
        )
        remove_wrapper = script.index("sudo rm -f /usr/local/bin/chromium")
        write_wrapper = script.index("sudo tee /usr/local/bin/chromium")
        self.assertLess(remove_wrapper, write_wrapper)
        self.assertIn("senpai-browser-smoke-test.py", script)
        self.assertIn('HOME="$role_home"', script)
        self.assertIn("brew install uv gh gettext cmake jq", script)
        self.assertIn("command -v jq", script)
        self.assertIn("jq -n -e 'true'", script)
        self.assertIn("brew install uv gh gettext cmake jq bun", script)
        self.assertIn("brew install uv gh gettext cmake jq bun tmux", script)
        self.assertIn('HOME="$role_home" tmux -V', script)
        self.assertIn(
            'tmux -L "$tmux_socket" -f /dev/null new-session -d -s smoke',
            script,
        )
        self.assertIn('tmux -L "$tmux_socket" has-session -t smoke', script)
        self.assertIn('tmux -L "$tmux_socket" kill-server', script)
        self.assertIn("trap cleanup_role_runtime EXIT", script)
        tmux_smoke = script.index('HOME="$role_home" tmux -V')
        browser_smoke = script.index("senpai-browser-smoke-test.py")
        self.assertLess(tmux_smoke, browser_smoke)
        self.assertIn("trap - EXIT", script)
        self.assertIn("/usr/local/libexec/mlxfast.js", script)
        self.assertIn("/usr/local/bin/mlxfast", script)
        self.assertIn("MLXFAST_API_URL='https://api.mlx.fast'", script)
        self.assertIn("MLXFAST_BENCHMARK_REF='eigenlabs/mlxfast-challenge'", script)
        self.assertIn("mlxfast version", script)
        self.assertIn("HF_HOME=/Users/ec2-user/.senpai/huggingface", script)
        self.assertIn(
            'AutoTokenizer.from_pretrained("zai-org/GLM-5.2")',
            script,
        )
        self.assertIn("llm.has_chat_template_tokenizer()", script)
        self.assertIn("HF_HUB_OFFLINE=1", script)
        self.assertIn("tokenizer.apply_chat_template", script)
        self.assertIn('tokens.get("input_ids")', script)
        self.assertIn("token_count > 0", script)
        self.assertIn('"name": "echo"', script)

    def test_remote_setup_skips_glm_tokenizer_for_other_model_profiles(self):
        script = _remote_setup_script(
            SimpleNamespace(
                senpai_repo_url="https://github.com/wandb/senpai.git",
                senpai_repo_revision=REVISION,
                advisor_model="openai/gpt-5.6-sol",
                student_model="anthropic/claude-opus-5",
            )
        ).decode()

        self.assertNotIn("AutoTokenizer.from_pretrained", script)
        self.assertNotIn("llm.has_chat_template_tokenizer()", script)

    def test_remote_setup_imports_the_supplied_metal_toolchain(self):
        script = _remote_setup_script(
            SimpleNamespace(
                senpai_repo_url="https://github.com/wandb/senpai.git",
                senpai_repo_revision=REVISION,
            )
        ).decode()

        time_sync = "sudo /usr/bin/sntp -sS -t 10 169.254.169.123"
        metal_import = "xcodebuild -importComponent metalToolchain -importPath"
        self.assertIn(time_sync, script)
        self.assertIn("ditto -x -k /tmp/senpai-MetalToolchain.zip", script)
        self.assertIn(metal_import, script)
        self.assertNotIn("-downloadComponent", script)
        self.assertLess(script.index(time_sync), script.index(metal_import))
        self.assertLess(
            script.index(metal_import),
            script.index("xcrun -sdk macosx metal --version"),
        )

    def test_each_prepared_node_receives_each_local_artifact_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Xcode.zip"
            archive.write_bytes(b"xcode")
            metal = root / "MetalToolchain.zip"
            with ZipFile(metal, "w") as contents:
                contents.writestr(
                    "MetalToolchain-17F109.exportedBundle/Restore/component.dmg",
                    b"metal",
                )
            bundle = root / "mlxfast.js"
            bundle.write_bytes(b"#!/usr/bin/env bun\n")
            run_args = args(
                root / "state",
                aws_mac_metal_toolchain_archive=str(metal),
                aws_mac_mlxfast_bundle=str(bundle),
            )
            node = {
                "student": "fern",
                "instance_id": "i-fern",
                "public_ip": "198.51.100.1",
            }

            with (
                patch.object(aws_mac_backend, "_wait_ssh"),
                patch.object(aws_mac_backend, "_ssh") as ssh,
            ):
                aws_mac_backend._prepare_node(run_args, root, node, archive)

        commands = [call.args[2] for call in ssh.call_args_list]
        self.assertEqual(
            commands.count("umask 077; cat > /tmp/senpai-MetalToolchain.zip"),
            1,
        )
        self.assertEqual(
            commands.count("umask 077; cat > /tmp/senpai-mlxfast.js"),
            1,
        )

    def test_prepared_node_executes_setup_from_a_file_then_reconnects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Xcode.zip"
            archive.write_bytes(b"xcode")
            metal = root / "MetalToolchain.zip"
            with ZipFile(metal, "w") as contents:
                contents.writestr(
                    "MetalToolchain-17F109.exportedBundle/Restore/component.dmg",
                    b"metal",
                )
            bundle = root / "mlxfast.js"
            bundle.write_bytes(b"#!/usr/bin/env bun\n")
            run_args = args(
                root / "state",
                aws_mac_metal_toolchain_archive=str(metal),
                aws_mac_mlxfast_bundle=str(bundle),
            )
            node = {
                "student": "fern",
                "instance_id": "i-fern",
                "public_ip": "198.51.100.1",
            }

            with (
                patch.object(aws_mac_backend, "_wait_ssh"),
                patch.object(aws_mac_backend, "_ssh") as ssh,
            ):
                aws_mac_backend._prepare_node(run_args, root, node, archive)

        calls = ssh.call_args_list
        commands = [call.args[2] for call in calls]
        upload = commands.index(
            "set -eu; umask 077; cat > /tmp/senpai-setup.sh; "
            "chmod 0700 /tmp/senpai-setup.sh"
        )
        execute = next(
            index
            for index, command in enumerate(commands)
            if "/bin/bash /tmp/senpai-setup.sh </dev/null" in command
        )
        reconnect = next(
            index
            for index, command in enumerate(commands)
            if command.startswith(
                "/Users/ec2-user/.senpai/venv/bin/python -c"
            )
        )

        self.assertEqual(
            calls[upload].kwargs["input_bytes"],
            _remote_setup_script(run_args),
        )
        self.assertIn("trap 'rm -f /tmp/senpai-setup.sh' EXIT", commands[execute])
        self.assertIn("import openhands.sdk, weave_openhands", commands[reconnect])
        self.assertLess(upload, execute)
        self.assertLess(execute, reconnect)
        self.assertNotIn("/bin/bash -s", commands)

    def test_instance_launch_is_pinned_to_the_mapped_dedicated_host(self):
        host = mac_host("h-a1", "fern")
        plan = AwsMacPlan(
            context=AwsContext("us-east-1", "sandbox"),
            account_id="770934259321",
            ami_id="ami-mac",
            root_device="/dev/sda1",
            volume_gib=250,
            security_group_id="sg-a1",
            ssh_cidr="203.0.113.7/32",
            hosts=(host,),
        )
        with patch.object(
            aws_mac_backend,
            "_aws_json",
            return_value={"Instances": [{"InstanceId": "i-fern"}]},
        ) as aws_json:
            node = _run_instance(
                args(Path("/tmp/state")),
                plan,
                host,
                "senpai-key",
                "token-fern",
                ssh_security_group_id="sg-b2",
            )

        command = aws_json.call_args.args
        self.assertEqual(command[1:3], ("ec2", "run-instances"))
        placement = json.loads(command[command.index("--placement") + 1])
        self.assertEqual(placement, {"HostId": "h-a1", "Tenancy": "host"})
        network = json.loads(command[command.index("--network-interfaces") + 1])
        self.assertEqual(network[0]["Groups"], ["sg-a1", "sg-b2"])
        self.assertEqual(command[command.index("--count") + 1], "1")
        self.assertEqual(
            command[command.index("--client-token") + 1],
            "token-fern",
        )
        self.assertEqual(node["instance_id"], "i-fern")
        self.assertEqual(node["host_id"], "h-a1")
        self.assertEqual(node["client_token"], "token-fern")
        user_data = command[command.index("--user-data") + 1]
        self.assertIn("/sbin/shutdown -h +1440", user_data)
        self.assertIn("SENPAI_SSH_HOST_KEY", user_data)

    def test_zero_ttl_publishes_host_keys_without_scheduled_shutdown(self):
        host = mac_host("h-a1", "fern")
        plan = AwsMacPlan(
            context=AwsContext("us-east-1", "sandbox"),
            account_id="770934259321",
            ami_id="ami-mac",
            root_device="/dev/sda1",
            volume_gib=250,
            security_group_id="sg-a1",
            ssh_cidr="203.0.113.7/32",
            hosts=(host,),
        )
        with patch.object(
            aws_mac_backend,
            "_aws_json",
            return_value={"Instances": [{"InstanceId": "i-fern"}]},
        ) as aws_json:
            _run_instance(
                args(Path("/tmp/state"), aws_ttl_hours=0),
                plan,
                host,
                "senpai-key",
                "token-fern",
                ssh_security_group_id="sg-b2",
            )

        command = aws_json.call_args.args
        user_data = command[command.index("--user-data") + 1]
        self.assertNotIn("/sbin/shutdown", user_data)
        self.assertIn("SENPAI_SSH_HOST_KEY", user_data)
        self.assertIn("/etc/ssh/ssh_host_*_key.pub", user_data)
        self.assertEqual(
            command[command.index("--instance-initiated-shutdown-behavior") + 1],
            "stop",
        )

    def test_recorded_instance_authenticates_host_after_persisting_public_ip(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "state"
            run_dir.mkdir()
            host = mac_host("h-a1", "fern")
            plan = AwsMacPlan(
                context=AwsContext("us-east-1", "sandbox"),
                account_id="770934259321",
                ami_id="ami-mac",
                root_device="/dev/sda1",
                volume_gib=250,
                security_group_id="sg-a1",
                ssh_cidr="203.0.113.7/32",
                hosts=(host,),
                vpc_id="vpc-a1",
            )
            node = {
                **asdict(host),
                "client_token": "token-fern",
                "instance_id": "i-fern",
                "public_ip": "",
            }
            state = {"tag": "mlxfast-r1", "nodes": [node]}

            def authorize(context, recorded_run_dir, recorded_node, *, timeout_s):
                persisted = json.loads((run_dir / "state.json").read_text())
                self.assertEqual(persisted["nodes"][0]["public_ip"], "198.51.100.1")
                self.assertEqual(context, plan.context)
                self.assertEqual(recorded_run_dir, run_dir)
                self.assertIs(recorded_node, node)
                self.assertEqual(timeout_s, 1_200)

            with (
                patch.object(
                    aws_mac_backend,
                    "_wait_instance",
                    return_value={"PublicIpAddress": "198.51.100.1"},
                ),
                patch.object(
                    aws_mac_backend,
                    "_authorize_ssh_host",
                    side_effect=authorize,
                    create=True,
                ) as authorize_ssh_host,
            ):
                aws_mac_backend._wait_recorded_instance(
                    args(run_dir.parent),
                    plan,
                    run_dir,
                    state,
                    node,
                )

            authorize_ssh_host.assert_called_once_with(
                plan.context,
                run_dir,
                node,
                timeout_s=1_200,
            )

    def test_instance_readiness_polling_uses_health_status_without_stock_waiters(self):
        instance = {
            "InstanceId": "i-fern",
            "State": {"Name": "running"},
            "PublicIpAddress": "198.51.100.1",
        }
        status = {
            "InstanceStatuses": [
                {
                    "InstanceStatus": {"Status": "ok"},
                    "SystemStatus": {"Status": "ok"},
                }
            ]
        }
        with (
            patch.object(aws_mac_backend, "_instance", return_value=instance),
            patch.object(aws_mac_backend, "_aws_json", return_value=status) as aws_json,
            patch.object(aws_mac_backend, "_aws_raw") as aws_raw,
        ):
            ready = _wait_instance(AwsContext("us-east-1"), "i-fern", 900)

        self.assertEqual(ready, instance)
        self.assertEqual(aws_json.call_args.args[1:3], ("ec2", "describe-instance-status"))
        aws_raw.assert_not_called()

    def test_instance_readiness_polling_obeys_the_operator_timeout(self):
        pending = {"InstanceId": "i-fern", "State": {"Name": "pending"}}
        with (
            patch.object(aws_mac_backend, "_instance", return_value=pending),
            patch.object(
                aws_mac_backend,
                "_aws_json",
                return_value={"InstanceStatuses": []},
            ),
            patch.object(aws_mac_backend.time, "monotonic", side_effect=[10, 12]),
            patch.object(aws_mac_backend.time, "sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "Timed out after 1s"),
        ):
            _wait_instance(AwsContext("us-east-1"), "i-fern", 1)

        sleep.assert_not_called()

    def test_ami_resolution_requires_an_available_arm64_mac_image(self):
        image = {
            "State": "available",
            "Architecture": "arm64_mac",
            "RootDeviceName": "/dev/sda1",
            "BlockDeviceMappings": [
                {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 200}}
            ],
        }

        def aws_json(_context, service, operation, *arguments):
            if (service, operation) == ("ssm", "get-parameter"):
                return {"Parameter": {"Value": "ami-mac"}}
            self.assertEqual((service, operation), ("ec2", "describe-images"))
            self.assertEqual(arguments, ("--image-ids", "ami-mac"))
            return {"Images": [image]}

        with patch.object(aws_mac_backend, "_aws_json", side_effect=aws_json):
            resolved = _resolve_ami(AwsContext("us-east-1"), "")
        self.assertEqual(resolved, ("ami-mac", "/dev/sda1", 200))

        image["Architecture"] = "x86_64"
        with patch.object(
            aws_mac_backend,
            "_aws_json",
            return_value={"Images": [image]},
        ), self.assertRaisesRegex(RuntimeError, "arm64_mac"):
            _resolve_ami(AwsContext("us-east-1"), "ami-intel")

    def test_network_validation_requires_public_az_matched_subnets_and_sg_vpc(self):
        hosts = (
            mac_host("h-a1", "fern", subnet_id="subnet-a1"),
            mac_host("h-b2", "tanjiro", subnet_id="subnet-b2"),
        )
        subnets = [
            {
                "SubnetId": subnet_id,
                "State": "available",
                "AvailabilityZone": "us-east-1a",
                "MapPublicIpOnLaunch": True,
                "VpcId": "vpc-1",
            }
            for subnet_id in ("subnet-a1", "subnet-b2")
        ]

        def aws_json(_context, _service, operation, *arguments):
            if operation == "describe-subnets":
                self.assertEqual(
                    arguments,
                    ("--subnet-ids", "subnet-a1", "subnet-b2"),
                )
                return {"Subnets": subnets}
            self.assertEqual(operation, "describe-security-groups")
            return {"SecurityGroups": [{"GroupId": "sg-a1", "VpcId": "vpc-1"}]}

        with patch.object(aws_mac_backend, "_aws_json", side_effect=aws_json):
            _validate_network(AwsContext("us-east-1"), hosts, "sg-a1")

        subnets[1]["AvailabilityZone"] = "us-east-1b"
        with patch.object(
            aws_mac_backend,
            "_aws_json",
            side_effect=aws_json,
        ), self.assertRaisesRegex(RuntimeError, "not us-east-1a"):
            _validate_network(AwsContext("us-east-1"), hosts, "sg-a1")

    def test_native_payload_uses_native_timeout_and_contains_no_aws_credentials(self):
        run_args = args(Path("/tmp/state"))
        payload = json.loads(_native_payload(run_args, (student("fern"),)))

        self.assertEqual(
            payload["args"]["native_ready_timeout_s"],
            run_args.native_ready_timeout_s,
        )
        self.assertNotIn("HF_HOME", payload["roles"][0]["env"])
        self.assertNotIn("HF_HUB_OFFLINE", payload["roles"][0]["env"])
        serialized = json.dumps(payload)
        self.assertNotIn("AWS_ACCESS_KEY_ID", serialized)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", serialized)

    def test_glm_native_payload_does_not_disable_huggingface_network(self):
        run_args = args(
            Path("/tmp/state"),
            student_model="wandb/zai-org/GLM-5.2",
        )

        payload = json.loads(_native_payload(run_args, (student("fern"),)))

        self.assertEqual(
            payload["roles"][0]["env"]["HF_HOME"],
            "/Users/ec2-user/.senpai/huggingface",
        )
        self.assertNotIn("HF_HUB_OFFLINE", payload["roles"][0]["env"])


class AwsMacLaunchTests(unittest.TestCase):
    def test_key_pair_intent_is_recorded_before_the_aws_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = mac_host("h-a1", "fern")
            plan = AwsMacPlan(
                context=AwsContext("us-east-1", "sandbox"),
                account_id="770934259321",
                ami_id="ami-mac",
                root_device="/dev/sda1",
                volume_gib=250,
                security_group_id="sg-a1",
                ssh_cidr="203.0.113.7/32",
                hosts=(host,),
            )
            operations: list[tuple[str, ...]] = []

            def aws_raw(_context, _service, operation, *arguments):
                operations.append((operation, *arguments))
                return ""

            with (
                patch.object(
                    aws_mac_backend,
                    "_aws_json",
                    side_effect=AwsCommandError("connection lost after create response"),
                ),
                patch.object(aws_mac_backend, "_key_name", return_value="senpai-key"),
                patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw),
                self.assertRaisesRegex(AwsCommandError, "connection lost"),
            ):
                launch_aws_mac(args(root / "state"), [student("fern")], plan)

        self.assertIn(
            ("delete-key-pair", "--key-name", "senpai-key"),
            operations,
        )

    def test_key_pair_is_recorded_before_a_local_private_key_write_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = mac_host("h-a1", "fern")
            plan = AwsMacPlan(
                context=AwsContext("us-east-1", "sandbox"),
                account_id="770934259321",
                ami_id="ami-mac",
                root_device="/dev/sda1",
                volume_gib=250,
                security_group_id="sg-a1",
                ssh_cidr="203.0.113.7/32",
                hosts=(host,),
            )
            operations: list[tuple[str, ...]] = []

            def aws_raw(_context, _service, operation, *arguments):
                operations.append((operation, *arguments))
                return ""

            with (
                patch.object(
                    aws_mac_backend,
                    "_aws_json",
                    return_value={"KeyMaterial": "private-key"},
                ),
                patch.object(aws_mac_backend, "_key_name", return_value="senpai-key"),
                patch.object(
                    aws_mac_backend,
                    "_write_private_key",
                    side_effect=OSError("disk write failed"),
                ),
                patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw),
                self.assertRaisesRegex(OSError, "disk write failed"),
            ):
                launch_aws_mac(args(root / "state"), [student("fern")], plan)

        self.assertIn(
            ("delete-key-pair", "--key-name", "senpai-key"),
            operations,
        )

    def test_uncertain_ssh_authorization_is_revoked_during_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = mac_host("h-a1", "fern")
            plan = AwsMacPlan(
                context=AwsContext("us-east-1", "sandbox"),
                account_id="770934259321",
                ami_id="ami-mac",
                root_device="/dev/sda1",
                volume_gib=250,
                security_group_id="sg-a1",
                ssh_cidr="203.0.113.7/32",
                hosts=(host,),
            )

            def write_key(path, material):
                path.write_text(material)
                path.chmod(0o600)

            with (
                patch.object(
                    aws_mac_backend,
                    "_aws_json",
                    return_value={"KeyMaterial": "private-key"},
                ),
                patch.object(aws_mac_backend, "_key_name", return_value="senpai-key"),
                patch.object(aws_mac_backend, "_write_private_key", side_effect=write_key),
                patch.object(
                    aws_mac_backend,
                    "_create_ssh_security_group",
                    return_value="sg-b1",
                ),
                patch.object(
                    aws_mac_backend,
                    "_authorize_ssh",
                    side_effect=AwsCommandError("connection lost after authorize response"),
                ),
                patch.object(aws_mac_backend, "_aws_raw", return_value=""),
                patch.object(aws_mac_backend, "_revoke_ssh") as revoke_ssh,
                self.assertRaisesRegex(AwsCommandError, "connection lost"),
            ):
                launch_aws_mac(args(root / "state"), [student("fern")], plan)

        revoke_ssh.assert_called_once()

    def test_launch_orders_remote_preflight_github_native_launch_and_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Xcode.zip"
            archive.write_bytes(b"xcode")
            run_args = args(root / "state")
            hosts = (
                mac_host("h-a1", "fern"),
                mac_host("h-b2", "tanjiro"),
            )
            plan = AwsMacPlan(
                context=AwsContext("us-east-1", "sandbox"),
                account_id="770934259321",
                ami_id="ami-mac",
                root_device="/dev/sda1",
                volume_gib=250,
                security_group_id="sg-a1",
                ssh_cidr="203.0.113.7/32",
                hosts=hosts,
                vpc_id="vpc-a1",
            )
            events: list[str] = []

            def run_instance(
                _args,
                _plan,
                host,
                _key_name,
                client_token,
                *,
                ssh_security_group_id,
            ):
                self.assertEqual(ssh_security_group_id, "sg-b1")
                saved = json.loads(
                    (root / "state" / "mlxfast-r1" / "state.json").read_text()
                )
                recorded = next(
                    node for node in saved["nodes"] if node["student"] == host.student
                )
                self.assertEqual(recorded["client_token"], client_token)
                self.assertEqual(recorded["instance_id"], "")
                events.append(f"instance:{host.student}")
                return {
                    **asdict(host),
                    "client_token": client_token,
                    "instance_id": f"i-{host.student}",
                    "public_ip": "",
                }

            def wait_instance(_context, instance_id, timeout_s):
                self.assertEqual(timeout_s, run_args.aws_ready_timeout_s)
                student_name = instance_id.removeprefix("i-")
                events.append(f"wait:{student_name}")
                return {
                    "PublicIpAddress": (
                        f"198.51.100.{1 if student_name == 'fern' else 2}"
                    )
                }

            def authorize_host(context, _run_dir, node, *, timeout_s):
                self.assertEqual(context, plan.context)
                self.assertEqual(timeout_s, run_args.aws_ready_timeout_s)
                events.append(f"authorize:{node['student']}")

            def sequential(label, actions):
                events.append(label)
                for action in actions.values():
                    action()

            def native_action(action, _args, _run_dir, node, specs):
                keys = ",".join(spec.key for spec in specs)
                events.append(f"{action}:{node['student']}:{keys}")

            def open_gate(_args, _run_dir, node):
                events.append(f"gate:{node['student']}")

            def write_key(path, material):
                path.write_text(material)
                path.chmod(0o600)

            with (
                patch.object(
                    aws_mac_backend,
                    "_aws_json",
                    return_value={"KeyMaterial": "private-key"},
                ),
                patch.object(aws_mac_backend, "_key_name", return_value="senpai-key"),
                patch.object(aws_mac_backend, "_write_private_key", side_effect=write_key),
                patch.object(
                    aws_mac_backend,
                    "_create_ssh_security_group",
                    return_value="sg-b1",
                ),
                patch.object(
                    aws_mac_backend,
                    "_authorize_ssh",
                    side_effect=AwsCommandError(
                        "An error occurred (InvalidPermission.Duplicate)"
                    ),
                ),
                patch.object(aws_mac_backend, "_run_instance", side_effect=run_instance),
                patch.object(
                    aws_mac_backend,
                    "_wait_instance",
                    side_effect=wait_instance,
                ),
                patch.object(
                    aws_mac_backend,
                    "_authorize_ssh_host",
                    side_effect=authorize_host,
                ),
                patch.object(
                    aws_mac_backend,
                    "_xcode_archive",
                    return_value=(archive, False),
                ),
                patch.object(
                    aws_mac_backend,
                    "_prepare_node",
                    side_effect=lambda _args, _run_dir, node, _archive: events.append(
                        f"prepare:{node['student']}"
                    ),
                ),
                patch.object(aws_mac_backend, "_run_parallel", side_effect=sequential),
                patch.object(aws_mac_backend, "_native_action", side_effect=native_action),
                patch.object(
                    aws_mac_backend,
                    "_launchd_canary",
                    side_effect=lambda _run_dir, node: events.append(
                        f"canary:{node['student']}"
                    ),
                ),
                patch.object(aws_mac_backend, "_open_gate", side_effect=open_gate),
            ):
                launch_aws_mac(
                    run_args,
                    [student("fern"), student("tanjiro"), advisor()],
                    plan,
                    before_start=lambda: events.append("github"),
                )

            preflights = [
                index
                for index, event in enumerate(events)
                if event.startswith("preflight:")
            ]
            launches = [
                index
                for index, event in enumerate(events)
                if event.startswith("launch:")
            ]
            gates = [
                index
                for index, event in enumerate(events)
                if event.startswith("gate:")
            ]
            github = events.index("github")
            self.assertLess(
                events.index("canary:fern"),
                events.index("instance:tanjiro"),
            )
            for student_name in ("fern", "tanjiro"):
                self.assertLess(
                    events.index(f"wait:{student_name}"),
                    events.index(f"authorize:{student_name}"),
                )
                self.assertLess(
                    events.index(f"authorize:{student_name}"),
                    events.index(f"prepare:{student_name}"),
                )
            self.assertEqual(len(preflights), 2)
            self.assertEqual(len(launches), 2)
            self.assertEqual(len(gates), 2)
            self.assertLess(max(preflights), github)
            self.assertLess(github, min(launches))
            self.assertLess(max(launches), min(gates))
            self.assertIn("preflight:fern:student-fern,advisor", events)
            self.assertIn("preflight:tanjiro:student-tanjiro", events)
            self.assertIn("launch:fern:student-fern,advisor", events)
            self.assertIn("launch:tanjiro:student-tanjiro", events)
            state = json.loads(
                (root / "state" / "mlxfast-r1" / "state.json").read_text()
            )
            self.assertEqual(state["backend"], "aws-mac")
            self.assertEqual(state["state_version"], 2)
            self.assertEqual(state["phase"], "running")
            self.assertTrue(state["ssh_authorized"])
            self.assertEqual(state["ssh_security_group_id"], "sg-b1")
            self.assertEqual(len({node["client_token"] for node in state["nodes"]}), 2)

    def test_rollback_cleanup_terminates_instances_but_never_releases_hosts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "state" / "mlxfast-r1"
            run_dir.mkdir(parents=True)
            state = {
                "tag": "mlxfast-r1",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "senpai-key",
                "ssh_authorized": False,
                "ssh_cidr": "203.0.113.7/32",
                "security_group_id": "sg-a1",
                "nodes": [
                    {
                        "instance_id": "i-fern",
                        "host_id": "h-a1",
                        "public_ip": "",
                    },
                    {
                        "instance_id": "i-tanjiro",
                        "host_id": "h-b2",
                        "public_ip": "",
                    },
                ],
            }
            operations: list[tuple[str, ...]] = []

            def aws_raw(_context, _service, operation, *arguments):
                operations.append((operation, *arguments))
                return ""

            with (
                patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw),
                patch.object(aws_mac_backend, "_revoke_ssh"),
            ):
                errors = _cleanup(run_dir, state, AwsContext("us-east-1", "sandbox"))

        self.assertEqual(errors, [])
        terminate = [
            item for item in operations if item[0] == "terminate-instances"
        ]
        self.assertEqual(
            terminate,
            [
                ("terminate-instances", "--instance-ids", "i-fern"),
                ("terminate-instances", "--instance-ids", "i-tanjiro"),
            ],
        )
        self.assertFalse(any(item[0] == "release-hosts" for item in operations))
        self.assertFalse(any("h-a1" in item or "h-b2" in item for item in operations))

    def test_cleanup_handles_a_missing_and_live_instance_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "state" / "mlxfast-r1"
            run_dir.mkdir(parents=True)
            state = {
                "tag": "mlxfast-r1",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "ssh_authorized": False,
                "nodes": [
                    {
                        "instance_id": "i-missing",
                        "host_id": "h-a1",
                        "public_ip": "",
                    },
                    {
                        "instance_id": "i-live",
                        "host_id": "h-b2",
                        "public_ip": "",
                    },
                ],
            }
            terminated: list[tuple[str, ...]] = []

            def aws_raw(_context, _service, operation, *arguments):
                if operation != "terminate-instances":
                    return ""
                terminated.append(arguments)
                instance_id = arguments[-1]
                if instance_id == "i-missing":
                    raise AwsCommandError("InvalidInstanceID.NotFound")
                return ""

            with patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw):
                errors = _cleanup(run_dir, state, AwsContext("us-east-1", "sandbox"))

            self.assertEqual(errors, [])
            self.assertEqual(
                terminated,
                [
                    ("--instance-ids", "i-missing"),
                    ("--instance-ids", "i-live"),
                ],
            )
            self.assertFalse(run_dir.exists())

    def test_cleanup_persists_only_the_instance_that_failed_to_terminate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "state" / "mlxfast-r1"
            run_dir.mkdir(parents=True)
            state = {
                "tag": "mlxfast-r1",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "ssh_authorized": False,
                "nodes": [
                    {
                        "instance_id": "i-terminated",
                        "host_id": "h-a1",
                        "public_ip": "",
                    },
                    {
                        "instance_id": "i-denied",
                        "host_id": "h-b2",
                        "public_ip": "",
                    },
                ],
            }

            def aws_raw(_context, _service, operation, *arguments):
                if operation == "terminate-instances" and arguments[-1] == "i-denied":
                    raise AwsCommandError("UnauthorizedOperation")
                return ""

            with patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw):
                errors = _cleanup(run_dir, state, AwsContext("us-east-1", "sandbox"))

            persisted = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(len(errors), 1)
            self.assertIn("terminate instance i-denied", errors[0])
            self.assertEqual(persisted["nodes"][0]["instance_id"], "")
            self.assertEqual(persisted["nodes"][1]["instance_id"], "i-denied")
            self.assertEqual(persisted["phase"], "cleanup-failed")

    def test_same_cidr_campaigns_delete_only_their_owned_ssh_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "campaign-a"
            run_b = root / "campaign-b"
            run_a.mkdir()
            run_b.mkdir()
            common = {
                "backend": "aws-mac",
                "state_version": 2,
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "security_group_id": "sg-a1",
                "ssh_authorize_started": True,
                "ssh_authorized": True,
                "ssh_cidr": "203.0.113.7/32",
                "ssh_security_group_create_started": True,
                "ssh_security_group_owned": True,
                "vpc_id": "vpc-a1",
                "nodes": [],
            }
            state_a = {
                **common,
                "tag": "campaign-a",
                "ssh_security_group_id": "sg-b1",
                "ssh_security_group_name": "senpai-campaign-a-ssh",
            }
            state_b = {
                **common,
                "tag": "campaign-b",
                "ssh_security_group_id": "sg-b2",
                "ssh_security_group_name": "senpai-campaign-b-ssh",
            }
            operations: list[tuple[str, ...]] = []

            def aws_raw(_context, _service, operation, *arguments):
                operations.append((operation, *arguments))
                return ""

            with patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw):
                errors = _cleanup(
                    run_a,
                    state_a,
                    AwsContext("us-east-1", "sandbox"),
                )

            self.assertEqual(errors, [])
            self.assertFalse(run_a.exists())
            self.assertTrue(run_b.exists())
            serialized = json.dumps(operations)
            self.assertIn("sg-b1", serialized)
            self.assertNotIn("sg-a1", serialized)
            self.assertNotIn("sg-b2", serialized)
            self.assertIn("delete-security-group", serialized)

    def test_cleanup_recovers_an_ambiguous_owned_ssh_group_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "campaign-a"
            run_dir.mkdir()
            state = {
                "backend": "aws-mac",
                "state_version": 2,
                "tag": "campaign-a",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "security_group_id": "sg-a1",
                "ssh_authorize_started": False,
                "ssh_authorized": None,
                "ssh_cidr": "203.0.113.7/32",
                "ssh_security_group_create_started": True,
                "ssh_security_group_id": "",
                "ssh_security_group_name": "senpai-campaign-a-ssh",
                "ssh_security_group_owned": None,
                "vpc_id": "vpc-a1",
                "nodes": [],
            }
            operations: list[tuple[str, ...]] = []

            def aws_raw(_context, _service, operation, *arguments):
                operations.append((operation, *arguments))
                return ""

            with (
                patch.object(
                    aws_mac_backend,
                    "_aws_json",
                    return_value={"SecurityGroups": [{"GroupId": "sg-b1"}]},
                ) as aws_json,
                patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw),
            ):
                errors = _cleanup(
                    run_dir,
                    state,
                    AwsContext("us-east-1", "sandbox"),
                )

            self.assertEqual(errors, [])
            self.assertFalse(run_dir.exists())
            self.assertIn("describe-security-groups", aws_json.call_args.args)
            self.assertIn(
                ("delete-security-group", "--group-id", "sg-b1"),
                operations,
            )

    def test_cleanup_retries_an_owned_ssh_group_delete_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "campaign-a"
            run_dir.mkdir()
            state = {
                "backend": "aws-mac",
                "state_version": 2,
                "tag": "campaign-a",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "security_group_id": "sg-a1",
                "ssh_authorize_started": False,
                "ssh_authorized": False,
                "ssh_cidr": "203.0.113.7/32",
                "ssh_security_group_create_started": True,
                "ssh_security_group_id": "sg-b1",
                "ssh_security_group_name": "senpai-campaign-a-ssh",
                "ssh_security_group_owned": True,
                "vpc_id": "vpc-a1",
                "nodes": [],
            }
            attempts = 0

            def aws_raw(_context, _service, operation, *_arguments):
                nonlocal attempts
                if operation == "delete-security-group":
                    attempts += 1
                    if attempts == 1:
                        raise AwsCommandError("DependencyViolation")
                return ""

            with patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw):
                first_errors = _cleanup(
                    run_dir,
                    state,
                    AwsContext("us-east-1", "sandbox"),
                )
                persisted = json.loads((run_dir / "state.json").read_text())
                retry_state = json.loads(json.dumps(persisted))
                second_errors = _cleanup(
                    run_dir,
                    retry_state,
                    AwsContext("us-east-1", "sandbox"),
                )

            self.assertEqual(len(first_errors), 1)
            self.assertIn("delete SSH security group", first_errors[0])
            self.assertEqual(persisted["ssh_security_group_id"], "sg-b1")
            self.assertEqual(second_errors, [])
            self.assertEqual(attempts, 2)
            self.assertFalse(run_dir.exists())

    def test_cleanup_preserves_access_until_every_instance_is_terminated(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "campaign-a"
            run_dir.mkdir()
            state = {
                "backend": "aws-mac",
                "state_version": 2,
                "tag": "campaign-a",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "senpai-key",
                "key_create_started": True,
                "key_owned": True,
                "security_group_id": "sg-a1",
                "ssh_authorize_started": True,
                "ssh_authorized": True,
                "ssh_cidr": "203.0.113.7/32",
                "ssh_security_group_create_started": True,
                "ssh_security_group_id": "sg-b1",
                "ssh_security_group_name": "senpai-campaign-a-ssh",
                "ssh_security_group_owned": True,
                "vpc_id": "vpc-a1",
                "nodes": [
                    {
                        "instance_id": "i-live",
                        "host_id": "h-a1",
                        "public_ip": "",
                    }
                ],
            }
            operations: list[tuple[str, ...]] = []
            terminate_attempts = 0

            def aws_raw(_context, _service, operation, *arguments):
                nonlocal terminate_attempts
                operations.append((operation, *arguments))
                if operation == "terminate-instances":
                    terminate_attempts += 1
                    if terminate_attempts == 1:
                        raise AwsCommandError("UnauthorizedOperation")
                return ""

            with patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw):
                first_errors = _cleanup(
                    run_dir,
                    state,
                    AwsContext("us-east-1", "sandbox"),
                )
                persisted = json.loads((run_dir / "state.json").read_text())
                first_operations = list(operations)
                retry_state = json.loads(json.dumps(persisted))
                second_errors = _cleanup(
                    run_dir,
                    retry_state,
                    AwsContext("us-east-1", "sandbox"),
                )

            self.assertEqual(len(first_errors), 1)
            self.assertEqual(persisted["key_name"], "senpai-key")
            self.assertEqual(persisted["ssh_security_group_id"], "sg-b1")
            self.assertFalse(
                any(
                    operation
                    in {
                        "delete-key-pair",
                        "revoke-security-group-ingress",
                        "delete-security-group",
                    }
                    for operation, *_arguments in first_operations
                )
            )
            self.assertEqual(second_errors, [])
            self.assertFalse(run_dir.exists())
            self.assertIn(
                ("delete-key-pair", "--key-name", "senpai-key"),
                operations,
            )
            self.assertIn(
                ("delete-security-group", "--group-id", "sg-b1"),
                operations,
            )

    def test_legacy_cleanup_never_revokes_a_shared_ssh_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "campaign-a"
            run_dir.mkdir()
            state = {
                "backend": "aws-mac",
                "state_version": 1,
                "tag": "campaign-a",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "security_group_id": "sg-a1",
                "ssh_authorize_started": True,
                "ssh_authorized": True,
                "ssh_cidr": "203.0.113.7/32",
                "nodes": [],
            }

            with (
                patch.object(
                    aws_mac_backend,
                    "_aws_json",
                    return_value={
                        "SecurityGroups": [
                            {
                                "GroupId": "sg-a1",
                                "IpPermissions": [
                                    {
                                        "IpProtocol": "tcp",
                                        "FromPort": 22,
                                        "ToPort": 22,
                                        "IpRanges": [
                                            {"CidrIp": "203.0.113.7/32"}
                                        ],
                                    }
                                ],
                            }
                        ]
                    },
                ),
                patch.object(aws_mac_backend, "_aws_raw") as aws_raw,
            ):
                errors = _cleanup(
                    run_dir,
                    state,
                    AwsContext("us-east-1", "sandbox"),
                )

            self.assertEqual(len(errors), 1)
            self.assertIn("legacy shared SSH rule", errors[0])
            self.assertIn("rerun terminate", errors[0])
            self.assertTrue(run_dir.exists())
            aws_raw.assert_not_called()

    def test_legacy_cleanup_converges_after_the_shared_rule_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "campaign-a"
            run_dir.mkdir()
            state = {
                "backend": "aws-mac",
                "state_version": 1,
                "tag": "campaign-a",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "security_group_id": "sg-a1",
                "ssh_authorize_started": True,
                "ssh_authorized": True,
                "ssh_cidr": "203.0.113.7/32",
                "nodes": [],
            }

            with (
                patch.object(
                    aws_mac_backend,
                    "_aws_json",
                    return_value={
                        "SecurityGroups": [
                            {"GroupId": "sg-a1", "IpPermissions": []}
                        ]
                    },
                ),
                patch.object(aws_mac_backend, "_aws_raw") as aws_raw,
            ):
                errors = _cleanup(
                    run_dir,
                    state,
                    AwsContext("us-east-1", "sandbox"),
                )

            self.assertEqual(errors, [])
            self.assertFalse(run_dir.exists())
            aws_raw.assert_not_called()

    def test_cleanup_skips_native_termination_when_no_manifest_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "state" / "mlxfast-r1"
            run_dir.mkdir(parents=True)
            (run_dir / "id_ed25519").write_text("private")
            state = {
                "backend": "aws-mac",
                "state_version": 2,
                "tag": "mlxfast-r1",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "ssh_authorized": False,
                "nodes": [
                    {
                        "instance_id": "i-fern",
                        "host_id": "h-a1",
                        "public_ip": "198.51.100.1",
                    }
                ],
            }

            with (
                patch.object(aws_mac_backend, "_aws_raw", return_value=""),
                patch.object(aws_mac_backend, "_ssh") as ssh,
            ):
                errors = _cleanup(run_dir, state, AwsContext("us-east-1", "sandbox"))

        self.assertEqual(errors, [])
        self.assertIn("if test -f", ssh.call_args.args[2])

    def test_successful_instance_termination_supersedes_native_stop_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "state" / "mlxfast-r1"
            run_dir.mkdir(parents=True)
            (run_dir / "id_ed25519").write_text("private")
            state = {
                "backend": "aws-mac",
                "state_version": 2,
                "tag": "mlxfast-r1",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "ssh_authorized": False,
                "nodes": [
                    {
                        "instance_id": "i-fern",
                        "host_id": "h-a1",
                        "public_ip": "198.51.100.1",
                    }
                ],
            }

            with (
                patch.object(aws_mac_backend, "_aws_raw", return_value=""),
                patch.object(
                    aws_mac_backend,
                    "_ssh",
                    side_effect=RuntimeError("host is already unavailable"),
                ),
            ):
                errors = _cleanup(run_dir, state, AwsContext("us-east-1", "sandbox"))
                run_exists = run_dir.exists()

        self.assertEqual(errors, [])
        self.assertFalse(run_exists)

    def test_failed_instance_termination_preserves_cleanup_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "state" / "mlxfast-r1"
            run_dir.mkdir(parents=True)
            (run_dir / "id_ed25519").write_text("private")
            state = {
                "tag": "mlxfast-r1",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "ssh_authorized": False,
                "nodes": [
                    {
                        "instance_id": "i-fern",
                        "host_id": "h-a1",
                        "public_ip": "198.51.100.1",
                    }
                ],
            }

            with (
                patch.object(
                    aws_mac_backend,
                    "_aws_raw",
                    side_effect=AwsCommandError("UnauthorizedOperation"),
                ),
                patch.object(
                    aws_mac_backend,
                    "_ssh",
                    side_effect=RuntimeError("host is unavailable"),
                ),
            ):
                errors = _cleanup(run_dir, state, AwsContext("us-east-1", "sandbox"))
                run_exists = run_dir.exists()

        self.assertEqual(len(errors), 2)
        self.assertTrue(any(error.startswith("native stop i-fern") for error in errors))
        self.assertTrue(any(error.startswith("terminate instance") for error in errors))
        self.assertTrue(run_exists)

    def test_cleanup_recovers_an_instance_from_its_persisted_client_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "state" / "mlxfast-r1"
            run_dir.mkdir(parents=True)
            state = {
                "tag": "mlxfast-r1",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "",
                "ssh_authorized": False,
                "nodes": [
                    {
                        "student": "fern",
                        "client_token": "token-fern",
                        "instance_id": "",
                        "host_id": "h-a1",
                        "public_ip": "",
                    }
                ],
            }
            operations: list[tuple[str, ...]] = []

            def aws_raw(_context, _service, operation, *arguments):
                operations.append((operation, *arguments))
                return ""

            with (
                patch.object(
                    aws_mac_backend,
                    "_aws_json",
                    return_value={
                        "Reservations": [
                            {
                                "Instances": [
                                    {
                                        "InstanceId": "i-recovered",
                                        "Placement": {"HostId": "h-a1"},
                                        "State": {"Name": "running"},
                                    }
                                ]
                            }
                        ]
                    },
                ) as aws_json,
                patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw),
            ):
                errors = _cleanup(run_dir, state, AwsContext("us-east-1", "sandbox"))

        self.assertEqual(errors, [])
        self.assertIn(
            ("terminate-instances", "--instance-ids", "i-recovered"),
            operations,
        )
        self.assertIn("Name=client-token,Values=token-fern", aws_json.call_args.args)
        self.assertIn("Name=tag:senpai:run,Values=mlxfast-r1", aws_json.call_args.args)

    def test_cleanup_treats_missing_aws_resources_as_already_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "state" / "mlxfast-r1"
            run_dir.mkdir(parents=True)
            state = {
                "backend": "aws-mac",
                "state_version": 2,
                "tag": "mlxfast-r1",
                "region": "us-east-1",
                "profile": "sandbox",
                "key_name": "senpai-key",
                "key_create_started": True,
                "key_owned": True,
                "ssh_authorize_started": True,
                "ssh_authorized": True,
                "ssh_cidr": "203.0.113.7/32",
                "security_group_id": "sg-a1",
                "ssh_security_group_create_started": True,
                "ssh_security_group_id": "sg-b1",
                "ssh_security_group_name": "senpai-mlxfast-r1-ssh",
                "ssh_security_group_owned": True,
                "nodes": [
                    {
                        "instance_id": "i-fern",
                        "host_id": "h-a1",
                        "public_ip": "",
                    }
                ],
            }

            def aws_raw(_context, _service, operation, *_arguments):
                code = {
                    "terminate-instances": "InvalidInstanceID.NotFound",
                    "delete-key-pair": "InvalidKeyPair.NotFound",
                    "delete-security-group": "InvalidGroup.NotFound",
                }[operation]
                raise AwsCommandError(code)

            with (
                patch.object(aws_mac_backend, "_aws_raw", side_effect=aws_raw),
                patch.object(
                    aws_mac_backend,
                    "_revoke_ssh",
                    side_effect=AwsCommandError("InvalidPermission.NotFound"),
                ),
            ):
                errors = _cleanup(run_dir, state, AwsContext("us-east-1", "sandbox"))

        self.assertEqual(errors, [])
        self.assertFalse(run_dir.exists())


class AwsMacLifecycleTests(unittest.TestCase):
    @staticmethod
    def write_state(root: Path) -> None:
        run_dir = root / "mlxfast-r1"
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text(
            json.dumps(
                {
                    "account_id": "770934259321",
                    "backend": "aws-mac",
                    "phase": "running",
                    "profile": "sandbox",
                    "region": "us-east-1",
                    "instance_type": "mac-m4pro.metal",
                    "security_group_id": "sg-a1",
                    "ssh_authorize_started": False,
                    "ssh_authorized": False,
                    "state_version": 1,
                    "tag": "mlxfast-r1",
                    "nodes": [
                        {
                            "student": "fern",
                            "instance_id": "i-fern",
                            "host_id": "h-a1",
                            "public_ip": "198.51.100.1",
                        },
                        {
                            "student": "tanjiro",
                            "instance_id": "i-tanjiro",
                            "host_id": "h-b2",
                            "public_ip": "198.51.100.2",
                        },
                    ],
                }
            )
        )

    def assert_lifecycle_state_rejected(
        self,
        state: object,
        message: str,
    ) -> None:
        actions = {
            "status": lambda state_root: status_aws_mac(
                "mlxfast-r1", str(state_root)
            ),
            "logs": lambda state_root: logs_aws_mac(
                "mlxfast-r1", str(state_root)
            ),
            "terminate": lambda state_root: aws_mac_backend.terminate_aws_mac(
                "mlxfast-r1", str(state_root)
            ),
        }
        for action_name, action in actions.items():
            with self.subTest(action=action_name):
                with tempfile.TemporaryDirectory() as tmp:
                    state_root = Path(tmp) / "aws"
                    run_dir = state_root / "mlxfast-r1"
                    run_dir.mkdir(parents=True)
                    state_path = run_dir / "state.json"
                    state_path.write_text(json.dumps(state))
                    key_path = run_dir / "id_ed25519"
                    key_path.write_text("private-key")

                    with (
                        patch.object(aws_mac_backend, "_check_account") as account,
                        patch.object(aws_mac_backend, "_aws_json") as aws_json,
                        patch.object(aws_mac_backend, "_aws_raw") as aws_raw,
                        patch.object(aws_mac_backend, "_ssh") as ssh,
                        self.assertRaisesRegex(RuntimeError, message),
                    ):
                        action(state_root)

                    account.assert_not_called()
                    aws_json.assert_not_called()
                    aws_raw.assert_not_called()
                    ssh.assert_not_called()
                    self.assertEqual(json.loads(state_path.read_text()), state)
                    self.assertEqual(key_path.read_text(), "private-key")

    def test_lifecycle_rejects_non_object_state_before_aws_or_local_mutation(self):
        self.assert_lifecycle_state_rejected(
            ["not", "a", "state object"],
            "JSON object",
        )

    def test_lifecycle_rejects_standard_aws_state_before_aws_or_local_mutation(self):
        self.assert_lifecycle_state_rejected(
            {
                "account_id": "770934259321",
                "availability_zone": "us-east-1a",
                "instance_id": "i-standard-aws",
                "instance_type": "g6.12xlarge",
                "key_name": "senpai-standard-key",
                "key_owned": True,
                "phase": "running",
                "profile": "sandbox",
                "region": "us-east-1",
                "roles": ["student-fern"],
                "security_group_id": "sg-a1",
                "subnet_id": "subnet-a1",
                "tag": "mlxfast-r1",
                "vpc_id": "vpc-a1",
            },
            "not compatible with AWS Mac lifecycle",
        )

    def test_lifecycle_rejects_explicit_non_mac_backend(self):
        self.assert_lifecycle_state_rejected(
            {
                "account_id": "770934259321",
                "backend": "aws",
                "instance_type": "mac-m4pro.metal",
                "nodes": [],
                "profile": "sandbox",
                "region": "us-east-1",
                "security_group_id": "sg-a1",
                "ssh_authorize_started": False,
                "ssh_authorized": False,
                "state_version": 1,
                "tag": "mlxfast-r1",
            },
            "backend",
        )

    def test_lifecycle_rejects_unsupported_mac_state_version(self):
        self.assert_lifecycle_state_rejected(
            {
                "account_id": "770934259321",
                "backend": "aws-mac",
                "instance_type": "mac-m4pro.metal",
                "nodes": [],
                "profile": "sandbox",
                "region": "us-east-1",
                "security_group_id": "sg-a1",
                "ssh_authorize_started": False,
                "ssh_authorized": False,
                "state_version": 3,
                "tag": "mlxfast-r1",
            },
            "state version",
        )

    def test_terminate_accepts_unambiguous_legacy_mac_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / "aws"
            run_dir = state_root / "mlxfast-r1"
            run_dir.mkdir(parents=True)
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "account_id": "770934259321",
                        "instance_type": "mac-m4pro.metal",
                        "key_name": "",
                        "nodes": [],
                        "phase": "creating",
                        "profile": "sandbox",
                        "region": "us-east-1",
                        "security_group_id": "sg-a1",
                        "ssh_authorize_started": False,
                        "ssh_authorized": False,
                        "tag": "mlxfast-r1",
                    }
                )
            )

            with (
                patch.object(aws_mac_backend, "_check_account"),
                redirect_stdout(io.StringIO()),
            ):
                aws_mac_backend.terminate_aws_mac("mlxfast-r1", str(state_root))

            self.assertFalse(run_dir.exists())

    def test_status_and_logs_map_roles_to_their_recorded_hosts(self):
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout=b"native service healthy\n",
            stderr=b"",
        )
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / "aws"
            self.write_state(state_root)
            output = io.StringIO()
            with (
                patch.object(aws_mac_backend, "_check_account"),
                patch.object(
                    aws_mac_backend,
                    "_instance",
                    return_value={"State": {"Name": "running"}},
                ),
                patch.object(aws_mac_backend, "_ssh", return_value=completed) as ssh,
                redirect_stdout(output),
            ):
                status_aws_mac("mlxfast-r1", str(state_root))

            rendered = output.getvalue()
            self.assertIn("student-fern: instance=i-fern host=h-a1", rendered)
            self.assertIn("student-tanjiro: instance=i-tanjiro host=h-b2", rendered)
            self.assertEqual(
                [call.args[1]["student"] for call in ssh.call_args_list],
                ["fern", "tanjiro"],
            )

            with (
                patch.object(aws_mac_backend, "_check_account"),
                patch.object(aws_mac_backend, "_ssh", return_value=completed) as ssh,
                redirect_stdout(io.StringIO()),
            ):
                logs_aws_mac(
                    "mlxfast-r1",
                    str(state_root),
                    role_key="advisor",
                    tail=41,
                )
                advisor_node = ssh.call_args.args[1]
                advisor_command = ssh.call_args.args[2]
                logs_aws_mac(
                    "mlxfast-r1",
                    str(state_root),
                    role_key="student-tanjiro",
                    tail=42,
                )
                student_node = ssh.call_args.args[1]
                student_command = ssh.call_args.args[2]

        self.assertEqual(advisor_node["student"], "fern")
        self.assertIn("--role advisor --tail 41", advisor_command)
        self.assertEqual(student_node["student"], "tanjiro")
        self.assertIn("--role student-tanjiro --tail 42", student_command)

    def test_cli_parses_log_role_tail_profile_and_state_root(self):
        with (
            patch.object(
                sys,
                "argv",
                [
                    "aws_mac.py",
                    "logs",
                    "mlxfast-r1",
                    "--state-root",
                    "/tmp/aws-state",
                    "--profile",
                    "sandbox",
                    "--role",
                    "student-fern",
                    "--tail",
                    "73",
                ],
            ),
            patch.object(aws_mac_cli, "logs_aws_mac") as logs,
        ):
            aws_mac_cli.main()

        logs.assert_called_once_with(
            "mlxfast-r1",
            "/tmp/aws-state",
            profile="sandbox",
            role_key="student-fern",
            tail=73,
        )


if __name__ == "__main__":
    unittest.main()
