import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from senpai.launch import aws_backend, remote
from senpai.launch.aws_backend import (
    AwsCommandError,
    AwsContext,
    _automatic_instance_type,
    _aws_command,
    _aws_raw,
    _check_account,
    _cleanup,
    _provision,
    _remote_payload,
    _resolve_region,
    _select_subnet,
    _source_archive,
    _ssh,
    _ssh_cidr,
    _user_data,
    _wait_for_gpu,
    launch_aws,
    preflight_aws,
)
from senpai.launch.specs import RoleSpec


def args(**overrides):
    values = {
        "tag": "aws-r1",
        "aws_region": "us-east-1",
        "aws_profile": "research",
        "aws_instance_type": "",
        "aws_ami_id": "",
        "aws_subnet_id": "",
        "aws_volume_gib": 100,
        "aws_state_root": "~/.senpai/aws",
        "aws_ssh_cidr": "",
        "aws_ready_timeout_s": 900,
        "aws_ttl_hours": 24.0,
        "docker_run_root": "~/.senpai/runs",
        "docker_student_gpu_ids": "",
        "docker_data_dir": "",
        "docker_shm_size": "32g",
        "docker_ready_timeout_s": 600,
        "gpus_per_student": 1,
        "pvc_mount_path": "/mnt/data",
        "repo_url": "https://github.com/wandb/senpai.git",
        "repo_branch": "main",
        "image": "ghcr.io/wandb/senpai:latest",
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def student(secret="service-secret"):
    return RoleSpec(
        role="student",
        name="fern",
        env={"REPO_URL": "runner", "REPO_BRANCH": "main"},
        secrets={"WANDB_API_KEY": secret},
    )


class AwsPlanningTests(unittest.TestCase):
    def test_aws_command_uses_profile_and_region_without_a_shell(self):
        self.assertEqual(
            _aws_command(
                AwsContext("us-east-1", "research"),
                "sts",
                "get-caller-identity",
            ),
            [
                "aws",
                "--profile",
                "research",
                "--region",
                "us-east-1",
                "sts",
                "get-caller-identity",
            ],
        )

    def test_automatic_instance_type_tracks_requested_gpu_count(self):
        self.assertEqual(_automatic_instance_type(1), "g4dn.xlarge")
        self.assertEqual(_automatic_instance_type(2), "g4dn.12xlarge")
        self.assertEqual(_automatic_instance_type(4), "g4dn.12xlarge")
        self.assertEqual(_automatic_instance_type(8), "g5.48xlarge")
        with self.assertRaisesRegex(ValueError, "at most 8 GPUs"):
            _automatic_instance_type(9)

    def test_ssh_access_is_restricted_to_one_ipv4_address(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"203.0.113.9\n"
        with patch(
            "senpai.launch.aws_backend.urllib.request.urlopen",
            return_value=response,
        ):
            self.assertEqual(_ssh_cidr(""), "203.0.113.9/32")
        with self.assertRaisesRegex(ValueError, "IPv4 address"):
            _ssh_cidr("0.0.0.0/0")

    def test_user_data_configures_gpu_runtime_and_cost_backstop_once(self):
        text = _user_data(args(aws_ttl_hours=2.5))

        self.assertIn("shutdown -h +150", text)
        self.assertIn("nvidia-ctk runtime configure", text)
        self.assertNotIn("docker run", text)
        self.assertNotIn("AWS_", text)

    def test_local_contract_errors_happen_before_aws_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                (
                    args(
                        aws_state_root=str(root / "map"),
                        docker_student_gpu_ids="fern:0",
                    ),
                    [student()],
                    "dedicated GPUs automatically",
                ),
                (
                    args(
                        aws_state_root=str(root / "data"),
                        docker_data_dir="/datasets",
                    ),
                    [student()],
                    "no local dataset directory",
                ),
                (
                    args(aws_state_root=str(root / "name")),
                    [
                        RoleSpec(
                            role="student",
                            name="../fern",
                            env={},
                            secrets={},
                        )
                    ],
                    "student name",
                ),
                (
                    args(
                        aws_state_root=str(root / "capacity"),
                        gpus_per_student=9,
                    ),
                    [student()],
                    "at most 8 GPUs",
                ),
            ]
            with patch(
                "senpai.launch.aws_backend._resolve_region",
                side_effect=AssertionError("AWS should not be reached"),
            ):
                for launch_args, roles, message in cases:
                    with self.subTest(message=message):
                        with self.assertRaisesRegex(
                            (ValueError, RuntimeError),
                            message,
                        ):
                            preflight_aws(launch_args, roles)

    def test_invalid_state_root_fails_before_aws_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / "not-a-directory"
            state_root.write_text("file")
            with (
                patch(
                    "senpai.launch.aws_backend._resolve_region",
                    side_effect=AssertionError("AWS should not be reached"),
                ),
                self.assertRaisesRegex(ValueError, "not a directory"),
            ):
                preflight_aws(
                    args(aws_state_root=str(state_root)),
                    [student()],
                )

    def test_state_root_cannot_enter_the_source_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(aws_backend, "ROOT", root),
                patch(
                    "senpai.launch.aws_backend._resolve_region",
                    side_effect=AssertionError("AWS should not be reached"),
                ),
                self.assertRaisesRegex(ValueError, "outside the Senpai source"),
            ):
                preflight_aws(
                    args(aws_state_root=str(root / ".state")),
                    [student()],
                )

    def test_automatic_subnet_selection_refuses_multiple_vpcs(self):
        default_vpc = ""

        def aws_json(_context, service, operation, *arguments):
            self.assertEqual(service, "ec2")
            if operation == "describe-subnets":
                return {
                    "Subnets": [
                        {
                            "AvailabilityZone": availability_zone,
                            "AvailableIpAddressCount": 10,
                            "DefaultForAz": vpc == default_vpc,
                            "State": "available",
                            "SubnetId": subnet,
                            "Tags": [{"Key": "Name", "Value": name}],
                            "VpcId": vpc,
                        }
                        for vpc, subnet, availability_zone, name in (
                            ("vpc-a", "subnet-a", "us-east-1a", "alpha-public"),
                            ("vpc-b", "subnet-b", "us-east-1b", "beta-public"),
                        )
                    ]
                }
            if operation == "describe-route-tables":
                return {
                    "RouteTables": [
                        {
                            "VpcId": vpc,
                            "Associations": [{"SubnetId": subnet}],
                            "Routes": [
                                {
                                    "DestinationCidrBlock": "0.0.0.0/0",
                                    "GatewayId": f"igw-{suffix}",
                                }
                            ],
                        }
                        for vpc, subnet, suffix in (
                            ("vpc-a", "subnet-a", "a"),
                            ("vpc-b", "subnet-b", "b"),
                        )
                    ]
                }
            if operation == "describe-instance-type-offerings":
                return {
                    "InstanceTypeOfferings": [
                        {"Location": "us-east-1a"},
                        {"Location": "us-east-1b"},
                    ]
                }
            self.fail(f"unexpected AWS operation: {operation} {arguments}")

        with patch("senpai.launch.aws_backend._aws_json", side_effect=aws_json):
            with self.assertRaisesRegex(
                RuntimeError,
                r"--aws_subnet_id.*subnet-a \(alpha-public\)",
            ):
                _select_subnet(AwsContext("us-east-1"), "g4dn.xlarge", "")

            default_vpc = "vpc-b"
            self.assertEqual(
                _select_subnet(AwsContext("us-east-1"), "g4dn.xlarge", "")[
                    "SubnetId"
                ],
                "subnet-b",
            )

    def test_aws_errors_name_the_operation_without_misleading_auth_advice(self):
        failed = subprocess.CompletedProcess(
            ["aws"],
            1,
            stdout="",
            stderr="UnauthorizedOperation",
        )
        with (
            patch("senpai.launch.aws_backend.subprocess.run", return_value=failed),
            self.assertRaisesRegex(AwsCommandError, "aws ec2 run-instances") as raised,
        ):
            _aws_raw(AwsContext("us-east-1"), "ec2", "run-instances")
        self.assertNotIn("Authenticate", str(raised.exception))

        with (
            patch(
                "senpai.launch.aws_backend._aws_json",
                side_effect=AwsCommandError("`aws sts get-caller-identity` failed"),
            ),
            self.assertRaisesRegex(AwsCommandError, "Authenticate"),
        ):
            _check_account(AwsContext("us-east-1", "research"))

    def test_missing_aws_cli_has_install_guidance(self):
        with (
            patch(
                "senpai.launch.aws_backend.subprocess.run",
                side_effect=FileNotFoundError,
            ),
            self.assertRaisesRegex(AwsCommandError, "AWS CLI is not installed"),
        ):
            _resolve_region("", "research")

    def test_dry_run_does_not_call_aws_or_expose_role_secrets(self):
        output = io.StringIO()
        with (
            redirect_stdout(output),
            patch(
                "senpai.launch.aws_backend.preflight_aws",
                side_effect=AssertionError("AWS should not be called"),
            ),
        ):
            launch_aws(args(dry_run=True), [student()])

        self.assertIn("g4dn.xlarge", output.getvalue())
        self.assertIn("credentials redacted", output.getvalue())
        self.assertNotIn("service-secret", output.getvalue())


class AwsTransportTests(unittest.TestCase):
    def test_source_snapshot_requires_untracked_files_to_be_staged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text(".env\n")
            (root / "tracked.py").write_text("tracked\n")
            (root / "deleted.py").write_text("deleted\n")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "add",
                    ".gitignore",
                    "tracked.py",
                    "deleted.py",
                ],
                check=True,
            )
            (root / "deleted.py").unlink()
            (root / "new.py").write_text("new\n")
            (root / ".env").write_text("SECRET=never\n")
            archive = root / "source.tar.gz"

            with patch.object(aws_backend, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "untracked files"):
                    _source_archive(archive)
                subprocess.run(
                    ["git", "-C", str(root), "add", "new.py"],
                    check=True,
                )
                _source_archive(archive)

            with tarfile.open(archive) as source:
                names = set(source.getnames())
            self.assertIn("tracked.py", names)
            self.assertIn("new.py", names)
            self.assertNotIn("deleted.py", names)
            self.assertNotIn(".env", names)

    def test_remote_payload_reuses_role_specs_without_aws_credentials(self):
        launch_args = args()
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "temporary-access-id",
                "AWS_SECRET_ACCESS_KEY": "temporary-secret",
            },
        ):
            payload = json.loads(_remote_payload(launch_args, [student()]))
            encoded = json.dumps(payload)

        self.assertEqual(payload["args"]["backend"], "docker")
        self.assertEqual(payload["args"]["repo_url"], "/home/ubuntu/senpai-source")
        self.assertEqual(payload["roles"][0]["secrets"]["WANDB_API_KEY"], "service-secret")
        self.assertNotIn("temporary-access-id", encoded)
        self.assertNotIn("temporary-secret", encoded)

    def test_remote_unlinks_payload_before_docker_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            path.write_text(
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

            def preflight(*_):
                self.assertFalse(path.exists())
                return "plan"

            with (
                patch("senpai.launch.remote.preflight_docker", side_effect=preflight),
                patch("senpai.launch.remote.launch_docker") as launch,
            ):
                remote.launch_from_payload(path)

            self.assertEqual(launch.call_args.args[2], "plan")

    def test_ssh_failure_reports_redacted_stdout_and_stderr(self):
        failed = subprocess.CompletedProcess(
            ["ssh"],
            1,
            stdout=b"cloud-init status: error service-secret",
            stderr=b"module setup failed",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend.subprocess.run",
                return_value=failed,
            ),
            self.assertRaisesRegex(RuntimeError, "cloud-init status: error")
            as raised,
        ):
            _ssh(
                Path(tmp),
                {"public_ip": "203.0.113.10"},
                "sudo cloud-init status --wait",
                redactions=("service-secret",),
            )

        message = str(raised.exception)
        self.assertIn("stdout:", message)
        self.assertIn("stderr:", message)
        self.assertNotIn("service-secret", message)

    def test_ssh_timeout_is_reported_as_a_launcher_error(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["ssh"], 12),
            ),
            self.assertRaisesRegex(RuntimeError, "timed out after 12s"),
        ):
            _ssh(
                Path(tmp),
                {"public_ip": "203.0.113.10"},
                "slow-command",
                timeout=12,
            )

    def test_gpu_readiness_retries_until_nvidia_driver_is_ready(self):
        unavailable = subprocess.CompletedProcess(
            ["ssh"],
            1,
            stdout=b"",
            stderr=b"driver initialization in progress",
        )
        ready = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=b"0\n",
            stderr=b"",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._ssh",
                side_effect=[unavailable, ready],
            ) as ssh,
            patch("senpai.launch.aws_backend.time.sleep"),
        ):
            _wait_for_gpu(
                Path(tmp),
                {"instance_id": "i-123"},
                30,
            )

        self.assertEqual(ssh.call_count, 2)


class AwsCleanupTests(unittest.TestCase):
    def test_launch_prints_copyable_nondefault_lifecycle_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            launch_args = args(
                aws_state_root=str(Path(tmp) / "state"),
                aws_profile="research",
            )
            plan = SimpleNamespace(
                account_id="123456789012",
                ami_id="ami-123",
                availability_zone="us-east-1a",
                instance_type="g4dn.xlarge",
                context=AwsContext("us-east-1", "research"),
                ssh_cidr="203.0.113.9/32",
                subnet_id="subnet-123",
                vpc_id="vpc-123",
            )

            def provision(_args, _plan, _run_dir, state):
                state.update(
                    {
                        "instance_id": "i-123",
                        "public_ip": "203.0.113.10",
                    }
                )

            output = io.StringIO()
            with (
                redirect_stdout(output),
                patch(
                    "senpai.launch.aws_backend._provision",
                    side_effect=provision,
                ),
                patch("senpai.launch.aws_backend._prepare_host"),
                patch("senpai.launch.aws_backend._start_roles"),
            ):
                launch_aws(launch_args, [student()], plan)

            text = output.getvalue()
            self.assertIn(f"--state-root {launch_args.aws_state_root}", text)
            self.assertIn("--profile research", text)
            self.assertIn("status aws-r1", text)
            self.assertIn("terminate aws-r1", text)

    def test_provision_saves_key_and_client_token_before_risky_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = {}
            plan = SimpleNamespace(
                context=AwsContext("us-east-1", "research"),
                vpc_id="vpc-123",
                ssh_cidr="203.0.113.9/32",
                root_device="/dev/sda1",
                volume_gib=100,
                ami_id="ami-123",
                instance_type="g4dn.xlarge",
                subnet_id="subnet-123",
            )
            run_arguments = []

            def aws_json(_context, _service, operation, *arguments):
                if operation == "create-security-group":
                    return {"GroupId": "sg-123"}
                if operation == "authorize-security-group-ingress":
                    return {}
                if operation == "run-instances":
                    saved = json.loads((run_dir / "state.json").read_text())
                    self.assertTrue(saved["instance_launch_started"])
                    self.assertTrue(saved["client_token"])
                    run_arguments.extend(arguments)
                    return {"Instances": [{"InstanceId": "i-123"}]}
                if operation == "describe-instances":
                    return {
                        "Reservations": [
                            {"Instances": [{"PublicIpAddress": "203.0.113.10"}]}
                        ]
                    }
                self.fail(f"unexpected operation: {operation}")

            def write_key(_path, _material):
                saved = json.loads((run_dir / "state.json").read_text())
                self.assertTrue(saved["key_name"].startswith("senpai-aws-r1-"))

            with (
                patch(
                    "senpai.launch.aws_backend._aws_raw",
                    side_effect=["private-key", ""],
                ),
                patch(
                    "senpai.launch.aws_backend._aws_json",
                    side_effect=aws_json,
                ),
                patch(
                    "senpai.launch.aws_backend._write_private_key",
                    side_effect=write_key,
                ),
            ):
                _provision(args(), plan, run_dir, state)

            token_index = run_arguments.index("--client-token") + 1
            self.assertEqual(run_arguments[token_index], state["client_token"])

    def test_cleanup_recovers_interrupted_instance_by_client_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            state = {
                "region": "us-east-1",
                "profile": "research",
                "client_token": "launch-token",
                "instance_launch_started": True,
                "security_group_id": "sg-123",
                "key_name": "senpai-test",
            }
            recovered = {
                "Reservations": [
                    {"Instances": [{"InstanceId": "i-recovered"}]}
                ]
            }

            with (
                patch(
                    "senpai.launch.aws_backend._aws_json",
                    return_value=recovered,
                ) as aws_json,
                patch(
                    "senpai.launch.aws_backend._aws_raw",
                    return_value="",
                ) as aws,
            ):
                errors = _cleanup(run_dir, state)

            self.assertEqual(errors, [])
            self.assertFalse(run_dir.exists())
            self.assertIn(
                "Name=client-token,Values=launch-token",
                aws_json.call_args.args,
            )
            operations = [call.args[1:4] for call in aws.call_args_list]
            self.assertEqual(
                operations,
                [
                    ("ec2", "terminate-instances", "--instance-ids"),
                    ("ec2", "wait", "instance-terminated"),
                    ("ec2", "delete-security-group", "--group-id"),
                    ("ec2", "delete-key-pair", "--key-name"),
                ],
            )

    def test_cleanup_terminates_instance_before_network_and_key_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            state = {
                "region": "us-east-1",
                "profile": "research",
                "instance_id": "i-123",
                "security_group_id": "sg-123",
                "key_name": "senpai-test",
            }

            override = AwsContext("us-east-1", "renewed-profile")
            with patch("senpai.launch.aws_backend._aws_raw", return_value="") as aws:
                errors = _cleanup(run_dir, state, override)

            self.assertEqual(errors, [])
            self.assertFalse(run_dir.exists())
            self.assertTrue(
                all(call.args[0] == override for call in aws.call_args_list)
            )
            operations = [call.args[1:4] for call in aws.call_args_list]
            self.assertEqual(
                operations,
                [
                    ("ec2", "terminate-instances", "--instance-ids"),
                    ("ec2", "wait", "instance-terminated"),
                    ("ec2", "delete-security-group", "--group-id"),
                    ("ec2", "delete-key-pair", "--key-name"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
