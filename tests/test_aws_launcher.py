import io
import json
import os
import subprocess
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
    AwsInstanceShape,
    AwsLaunchPlan,
    AwsRequirements,
    AwsRootLayout,
    AwsSubnet,
    _automatic_instance_type,
    _aws_command,
    _aws_raw,
    _check_account,
    _check_source_revision,
    _cleanup,
    _ensure_remote_capacity,
    _grow_root_volume,
    _prepare_host,
    _prepare_remote_storage,
    _provision,
    _remote_payload,
    _resolve_region,
    _select_subnet,
    _select_subnets,
    _source_bundle,
    _ssh,
    _ssh_cidr,
    _stop_remote_roles,
    _stream_directory,
    _user_data,
    _wait_for_gpu,
    launch_aws,
    preflight_aws,
    status_aws,
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
        "aws_runtime_reserve_gib": 80,
        "aws_state_root": "~/.senpai/aws",
        "aws_ssh_cidr": "",
        "aws_ready_timeout_s": 900,
        "aws_data_timeout_s": 7200,
        "aws_ttl_hours": 24.0,
        "docker_run_root": "~/.senpai/runs",
        "docker_student_gpu_ids": "",
        "data_dir": "",
        "docker_shm_size": "32g",
        "docker_ready_timeout_s": 600,
        "gpus_per_student": 1,
        "cpu_per_gpu": 15,
        "memory_gi_per_gpu": 120,
        "start_gate_path": "",
        "pvc_mount_path": "/mnt/data",
        "repo_url": "https://github.com/wandb/senpai.git",
        "repo_revision": "a" * 40,
        "advisor_image": "ghcr.io/wandb/senpai-advisor:sha-" + "a" * 40,
        "student_image": "ghcr.io/wandb/senpai-student:sha-" + "a" * 40,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def student(secret="service-secret"):
    return RoleSpec(
        role="student",
        name="fern",
        env={"REPO_URL": "runner", "REPO_REVISION": "a" * 40},
        secrets={"WANDB_API_KEY": secret},
    )


def aws_plan(**overrides):
    values = {
        "context": AwsContext("us-east-1", "research"),
        "account_id": "123456789012",
        "instance_type": "g5.8xlarge",
        "ami_id": "ami-123",
        "root_device": "/dev/sda1",
        "volume_gib": 250,
        "subnets": (
            AwsSubnet("subnet-a", "vpc-123", "us-east-1a"),
        ),
        "ssh_cidr": "203.0.113.9/32",
    }
    values.update(overrides)
    return AwsLaunchPlan(**values)


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
        self.assertEqual(
            _automatic_instance_type(AwsRequirements(1, 15, 120)),
            "g6e.8xlarge",
        )
        self.assertEqual(
            _automatic_instance_type(AwsRequirements(2, 30, 240)),
            "g5.24xlarge",
        )
        self.assertEqual(
            _automatic_instance_type(AwsRequirements(4, 60, 480)),
            "g5.48xlarge",
        )
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            _automatic_instance_type(AwsRequirements(9, 135, 1080))

    def test_host_resources_are_reserved_beyond_student_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            requirements = aws_backend._validate_aws_inputs(
                args(aws_state_root=str(Path(tmp) / "state")),
                [student()],
            )

        self.assertEqual(requirements.gpus, 1)
        self.assertEqual(aws_backend.AWS_HOST_VCPU_HEADROOM, 4)
        self.assertEqual(requirements.vcpus, 19)
        self.assertEqual(
            requirements.memory_gib,
            120 + aws_backend.AWS_HOST_MEMORY_HEADROOM_GIB,
        )

    def test_preflight_rejects_non_nvidia_and_undersized_instance_types(self):
        requirements = AwsRequirements(gpus=1, vcpus=19, memory_gib=120)
        cases = (
            (
                AwsInstanceShape("custom", 0, 32, 128),
                "does not have an NVIDIA GPU",
            ),
            (
                AwsInstanceShape("custom", 1, 18, 128),
                "18 vCPUs.*needs 1 GPUs, 19 vCPUs",
            ),
            (
                AwsInstanceShape("custom", 1, 32, 119),
                "119 GiB RAM.*needs 1 GPUs, 19 vCPUs, and 120 GiB RAM",
            ),
            (
                AwsInstanceShape("custom", 1, 32, 128, ("arm64",)),
                "must support x86_64",
            ),
        )
        for shape, message in cases:
            with (
                self.subTest(shape=shape),
                patch(
                    "senpai.launch.aws_backend._validate_aws_inputs",
                    return_value=requirements,
                ),
                patch("senpai.launch.aws_backend.shutil.which", return_value="ssh"),
                patch("senpai.launch.aws_backend._check_source_revision"),
                patch(
                    "senpai.launch.aws_backend._resolve_region",
                    return_value="us-east-1",
                ),
                patch(
                    "senpai.launch.aws_backend._check_account",
                    return_value={"Account": "123456789012"},
                ),
                patch(
                    "senpai.launch.aws_backend._instance_type_details",
                    return_value=shape,
                ),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                preflight_aws(
                    args(aws_instance_type="custom", aws_ssh_cidr="203.0.113.9/32"),
                    [student()],
                )

    def test_dataset_size_does_not_change_the_initial_volume_floor(self):
        requirements = AwsRequirements(
            gpus=1,
            vcpus=15,
            memory_gib=120,
            data_files=3502,
            data_bytes=41 * aws_backend.GIB + 1,
        )
        output = io.StringIO()
        with (
            redirect_stdout(output),
            patch(
                "senpai.launch.aws_backend._validate_aws_inputs",
                return_value=requirements,
            ),
            patch("senpai.launch.aws_backend.shutil.which", return_value="ssh"),
            patch("senpai.launch.aws_backend._check_source_revision"),
            patch(
                "senpai.launch.aws_backend._resolve_region",
                return_value="us-east-1",
            ),
            patch(
                "senpai.launch.aws_backend._check_account",
                return_value={"Account": "123456789012"},
            ),
            patch(
                "senpai.launch.aws_backend._instance_type_details",
                return_value=AwsInstanceShape("g5.8xlarge", 1, 32, 128),
            ),
            patch(
                "senpai.launch.aws_backend._resolve_ami",
                return_value=("ami-123", "/dev/sda1", 80),
            ),
            patch(
                "senpai.launch.aws_backend._select_subnets",
                return_value=(AwsSubnet("subnet-a", "vpc-123", "us-east-1a"),),
            ),
            patch(
                "senpai.launch.aws_backend._ssh_cidr",
                return_value="203.0.113.9/32",
            ),
        ):
            plan = preflight_aws(args(aws_volume_gib=100), [student()])

        self.assertEqual(plan.volume_gib, 100)
        self.assertEqual(plan.data_files, 3502)
        self.assertEqual(plan.data_bytes, requirements.data_bytes)
        self.assertIn("volume=100 GiB", output.getvalue())

    def test_payload_beyond_gp3_limit_fails_before_aws_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            with (
                patch(
                    "senpai.launch.aws_backend._data_summary",
                    return_value=(
                        1,
                        (aws_backend.GP3_MAX_GIB - 79) * aws_backend.GIB,
                    ),
                ),
                patch(
                    "senpai.launch.aws_backend._resolve_region",
                    side_effect=AssertionError("AWS should not be reached"),
                ),
                self.assertRaisesRegex(ValueError, "beyond the gp3 limit"),
            ):
                aws_backend._validate_aws_inputs(
                    args(
                        aws_state_root=str(root / "state"),
                        data_dir=str(data_dir),
                        aws_runtime_reserve_gib=80,
                    ),
                    [student()],
                )

    def test_runtime_reserve_must_be_non_negative(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "aws_runtime_reserve_gib"
        ):
            aws_backend._validate_aws_inputs(
                args(
                    aws_state_root=str(Path(tmp) / "state"),
                    aws_runtime_reserve_gib=-1,
                ),
                [student()],
            )

    def test_gpu_aws_still_requires_a_positive_ttl(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "greater than 0"
        ):
            aws_backend._validate_aws_inputs(
                args(
                    aws_state_root=str(Path(tmp) / "state"),
                    aws_ttl_hours=0,
                ),
                [student()],
            )

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
                        data_dir="/datasets",
                    ),
                    [student()],
                    "data directory does not exist",
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
                    "cannot fit",
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

    def test_automatic_subnet_selection_prefers_the_best_public_vpc(self):
        default_vpc = ""

        def aws_json(_context, service, operation, *arguments):
            self.assertEqual(service, "ec2")
            if operation == "describe-subnets":
                return {
                    "Subnets": [
                        {
                            "AvailabilityZone": availability_zone,
                            "AvailableIpAddressCount": 20 if vpc == "vpc-a" else 10,
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
            self.assertEqual(
                _select_subnet(AwsContext("us-east-1"), "g4dn.xlarge", "")[
                    "SubnetId"
                ],
                "subnet-a",
            )

            default_vpc = "vpc-b"
            self.assertEqual(
                _select_subnet(AwsContext("us-east-1"), "g4dn.xlarge", "")[
                    "SubnetId"
                ],
                "subnet-b",
            )

    def test_subnets_in_one_vpc_are_ordered_for_capacity_fallback(self):
        def aws_json(_context, _service, operation, *arguments):
            if operation == "describe-subnets":
                return {
                    "Subnets": [
                        {
                            "AvailabilityZone": zone,
                            "AvailableIpAddressCount": addresses,
                            "DefaultForAz": default,
                            "State": "available",
                            "SubnetId": subnet,
                            "VpcId": "vpc-a",
                        }
                        for subnet, zone, addresses, default in (
                            ("subnet-other", "us-east-1a", 90, False),
                            ("subnet-default", "us-east-1c", 5, True),
                            ("subnet-capacity", "us-east-1b", 100, False),
                        )
                    ]
                }
            if operation == "describe-route-tables":
                return {
                    "RouteTables": [
                        {
                            "VpcId": "vpc-a",
                            "Associations": [{"Main": True}],
                            "Routes": [
                                {
                                    "DestinationCidrBlock": "0.0.0.0/0",
                                    "GatewayId": "igw-a",
                                }
                            ],
                        }
                    ]
                }
            if operation == "describe-instance-type-offerings":
                return {
                    "InstanceTypeOfferings": [
                        {"Location": zone}
                        for zone in ("us-east-1a", "us-east-1b", "us-east-1c")
                    ]
                }
            self.fail(f"unexpected AWS operation: {operation} {arguments}")

        with patch("senpai.launch.aws_backend._aws_json", side_effect=aws_json):
            selected = _select_subnets(
                AwsContext("us-east-1"),
                "g5.8xlarge",
                "",
            )

        self.assertEqual(
            [subnet.subnet_id for subnet in selected],
            ["subnet-default", "subnet-capacity", "subnet-other"],
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

        self.assertIn("g6e.8xlarge", output.getvalue())
        self.assertIn("credentials redacted", output.getvalue())
        self.assertNotIn("service-secret", output.getvalue())


class AwsTransportTests(unittest.TestCase):
    def test_source_bundle_preserves_the_exact_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "tracked.py").write_text("tracked\n")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "add",
                    "tracked.py",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Senpai",
                    "-c",
                    "user.email=senpai@example.com",
                    "commit",
                    "-qm",
                    "source",
                ],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            bundle = root / "source.bundle"

            with patch.object(aws_backend, "ROOT", root):
                _check_source_revision(revision)
                _source_bundle(bundle)

            heads = subprocess.run(
                ["git", "bundle", "list-heads", str(bundle)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn(f"{revision} HEAD", heads)

    def test_source_revision_rejects_dirty_or_mismatched_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "tracked.py").write_text("tracked\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Senpai",
                    "-c",
                    "user.email=senpai@example.com",
                    "commit",
                    "-qm",
                    "source",
                ],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            with patch.object(aws_backend, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "images use"):
                    _check_source_revision("b" * 40)
                (root / "tracked.py").write_text("dirty\n")
                with self.assertRaisesRegex(RuntimeError, "must be clean"):
                    _check_source_revision(revision)

    def test_remote_payload_reuses_role_specs_without_aws_credentials(self):
        launch_args = args(data_dir="/local/data")
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
        self.assertEqual(payload["args"]["repo_revision"], "a" * 40)
        self.assertEqual(payload["args"]["data_dir"], "/home/ubuntu/senpai-data")
        self.assertEqual(payload["roles"][0]["secrets"]["WANDB_API_KEY"], "service-secret")
        self.assertNotIn("temporary-access-id", encoded)
        self.assertNotIn("temporary-secret", encoded)

    def test_image_preflight_payload_has_no_data_or_role_secrets(self):
        role = student()
        payload = json.loads(
            _remote_payload(
                args(data_dir="/local/data"),
                [role],
                include_data=False,
                include_secrets=False,
            )
        )

        self.assertEqual(payload["args"]["data_dir"], "")
        self.assertEqual(payload["roles"][0]["secrets"], {})
        self.assertEqual(payload["roles"][0]["env"]["REPO_URL"], aws_backend.REMOTE_SOURCE)
        self.assertEqual(role.secrets, {"WANDB_API_KEY": "service-secret"})
        self.assertEqual(role.env["REPO_URL"], "runner")

    def test_image_preflight_invokes_remote_without_data_or_secrets(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._invoke_remote",
                return_value="",
            ) as invoke,
        ):
            aws_backend._preflight_roles(
                args(data_dir="/local/data"),
                [student()],
                Path(tmp),
                {"public_ip": "203.0.113.10"},
                include_data=False,
            )

        self.assertEqual(invoke.call_args.args[0], "preflight")
        self.assertFalse(invoke.call_args.kwargs["include_data"])
        self.assertFalse(invoke.call_args.kwargs["include_secrets"])

    def test_data_stream_is_verified_by_file_count_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data"
            source.mkdir()
            (source / "one.pt").write_bytes(b"123")
            (source / "two.json").write_bytes(b"45678")

            def receive(_run_dir, _state, command, *, input_file, **_kwargs):
                self.assertTrue(input_file.read())
                self.assertIn("find /home/ubuntu/senpai-data", command)
                self.assertTrue(_kwargs["compression"])
                return subprocess.CompletedProcess(
                    ["ssh"],
                    0,
                    stdout=b"2 8\n",
                    stderr=b"",
                )

            real_popen = subprocess.Popen
            with (
                patch("senpai.launch.aws_backend._ssh", side_effect=receive),
                patch(
                    "senpai.launch.aws_backend.subprocess.Popen",
                    wraps=real_popen,
                ) as popen,
            ):
                _stream_directory(
                    root,
                    {"public_ip": "203.0.113.10"},
                    source,
                    "/home/ubuntu/senpai-data",
                    30,
                )

            self.assertIn("--no-xattrs", popen.call_args.args[0])
            self.assertEqual(
                popen.call_args.kwargs["env"]["COPYFILE_DISABLE"],
                "1",
            )

    def test_ssh_compression_uses_openssh_transport_compression(self):
        completed = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=b"",
            stderr=b"",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            _ssh(
                Path(tmp),
                {"public_ip": "203.0.113.10"},
                "cat >/dev/null",
                input_bytes=b"payload",
                compression=True,
            )

        self.assertEqual(run.call_args.args[0][0:2], ["ssh", "-C"])

    def test_capacity_does_not_resize_when_runtime_reserve_already_fits(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._remote_free_bytes",
                return_value=80 * aws_backend.GIB,
            ),
            patch("senpai.launch.aws_backend._validate_shared_root_storage"),
            patch("senpai.launch.aws_backend._grow_root_volume") as grow,
        ):
            _ensure_remote_capacity(
                args(),
                AwsContext("us-east-1", "research"),
                Path(tmp),
                {"instance_id": "i-123"},
                0,
                0,
            )

        grow.assert_not_called()

    def test_capacity_rounds_growth_up_and_persists_target_before_modify(self):
        required = 200 * aws_backend.GIB
        available = required - 100 * aws_backend.GIB - 1
        layout = AwsRootLayout(
            "/dev/nvme0n1p1", "ext4", "/dev/nvme0n1", 1, "gpt"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = {"instance_id": "i-123", "phase": "sizing-storage"}

            def modify(*arguments):
                saved = json.loads((run_dir / "state.json").read_text())
                self.assertTrue(saved["volume_resize_started"])
                self.assertEqual(saved["volume_resize_target_gib"], 212)
                self.assertNotIn("volume_resize_requested", saved)
                self.assertEqual(
                    arguments[1:],
                    (
                        "ec2",
                        "modify-volume",
                        "--volume-id",
                        "vol-root",
                        "--size",
                        "212",
                    ),
                )
                return ""

            with (
                patch(
                    "senpai.launch.aws_backend._remote_free_bytes",
                    side_effect=[available, required],
                ) as free,
                patch("senpai.launch.aws_backend._validate_shared_root_storage"),
                patch(
                    "senpai.launch.aws_backend._root_volume",
                    return_value=("vol-root", 100),
                ),
                patch(
                    "senpai.launch.aws_backend._root_layout",
                    return_value=layout,
                ),
                patch("senpai.launch.aws_backend._aws_raw", side_effect=modify),
                patch("senpai.launch.aws_backend._wait_for_volume_resize") as wait,
                patch("senpai.launch.aws_backend._wait_for_guest_disk_size") as guest,
                patch("senpai.launch.aws_backend._grow_root_filesystem") as filesystem,
            ):
                _ensure_remote_capacity(
                    args(),
                    AwsContext("us-east-1", "research"),
                    run_dir,
                    state,
                    0,
                    120 * aws_backend.GIB,
                )

            self.assertEqual(free.call_count, 2)
            wait.assert_called_once_with(
                AwsContext("us-east-1", "research"),
                "vol-root",
                212,
                900,
            )
            guest.assert_called_once_with(
                args(), run_dir, state, "/dev/nvme0n1", 212
            )
            filesystem.assert_called_once_with(args(), run_dir, state, layout)
            self.assertEqual(state["volume_gib"], 212)
            self.assertTrue(state["volume_resize_completed"])
            self.assertEqual(state["phase"], "sizing-storage")

    def test_growth_rejects_gp3_limit_before_aws_mutation(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._root_volume",
                return_value=("vol-root", aws_backend.GP3_MAX_GIB),
            ),
            patch("senpai.launch.aws_backend._root_layout") as layout,
            patch("senpai.launch.aws_backend._aws_raw") as aws,
            self.assertRaisesRegex(RuntimeError, "beyond the gp3 limit"),
        ):
            _grow_root_volume(
                args(),
                AwsContext("us-east-1", "research"),
                Path(tmp),
                {"instance_id": "i-123"},
                1,
            )

        layout.assert_not_called()
        aws.assert_not_called()

    def test_capacity_recheck_rejects_short_growth(self):
        required = 80 * aws_backend.GIB
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._remote_free_bytes",
                side_effect=[required - 1, required - 1],
            ),
            patch("senpai.launch.aws_backend._validate_shared_root_storage"),
            patch("senpai.launch.aws_backend._grow_root_volume") as grow,
            self.assertRaisesRegex(RuntimeError, "after growing the root filesystem"),
        ):
            _ensure_remote_capacity(
                args(),
                AwsContext("us-east-1", "research"),
                Path(tmp),
                {"instance_id": "i-123"},
                0,
                0,
            )

        grow.assert_called_once()

    def test_capacity_accounts_for_remote_block_allocation_per_file(self):
        reserve = 80 * aws_backend.GIB
        allocated = aws_backend.REMOTE_FILESYSTEM_BLOCK_BYTES
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._remote_free_bytes",
                side_effect=[reserve, reserve + allocated],
            ),
            patch("senpai.launch.aws_backend._validate_shared_root_storage"),
            patch(
                "senpai.launch.aws_backend._remote_free_inodes",
                return_value=10_000,
            ),
            patch("senpai.launch.aws_backend._grow_root_volume") as grow,
        ):
            _ensure_remote_capacity(
                args(),
                AwsContext("us-east-1", "research"),
                Path(tmp),
                {"instance_id": "i-123"},
                1,
                0,
            )

        grow.assert_called_once_with(
            args(),
            AwsContext("us-east-1", "research"),
            Path(tmp),
            {"instance_id": "i-123"},
            allocated,
        )

    def test_capacity_rejects_inode_exhaustion_before_upload(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._remote_free_bytes",
                return_value=80 * aws_backend.GIB + 4096,
            ),
            patch("senpai.launch.aws_backend._validate_shared_root_storage"),
            patch(
                "senpai.launch.aws_backend._remote_free_inodes",
                return_value=1024,
            ),
            patch("senpai.launch.aws_backend._grow_root_volume") as grow,
            self.assertRaisesRegex(RuntimeError, "free inodes.*Increase"),
        ):
            _ensure_remote_capacity(
                args(),
                AwsContext("us-east-1", "research"),
                Path(tmp),
                {"instance_id": "i-123"},
                1,
                0,
            )

        grow.assert_not_called()

    def test_free_inode_query_uses_compatible_gnu_df_flags(self):
        completed = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=b"123456\n",
            stderr=b"",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("senpai.launch.aws_backend._ssh", return_value=completed) as ssh,
        ):
            available = aws_backend._remote_free_inodes(
                args(),
                Path(tmp),
                {"instance_id": "i-123"},
            )

        self.assertEqual(available, 123456)
        self.assertEqual(
            ssh.call_args.args[2],
            "df --output=iavail /home/ubuntu | tail -1",
        )

    def test_root_volume_is_resolved_from_the_live_instance_mapping(self):
        context = AwsContext("us-east-1", "research")
        instance = {
            "RootDeviceName": "/dev/sda1",
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeId": "vol-root"},
                }
            ],
        }
        with (
            patch(
                "senpai.launch.aws_backend._instance_details",
                return_value=instance,
            ),
            patch(
                "senpai.launch.aws_backend._aws_json",
                return_value={"Volumes": [{"VolumeId": "vol-root", "Size": 250}]},
            ) as aws,
        ):
            resolved = aws_backend._root_volume(
                context,
                {"instance_id": "i-123"},
            )

        self.assertEqual(resolved, ("vol-root", 250))
        aws.assert_called_once_with(
            context,
            "ec2",
            "describe-volumes",
            "--volume-ids",
            "vol-root",
        )

    def test_storage_mounts_are_validated_even_when_no_resize_is_needed(self):
        output = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=b"/dev/root\t/dev/root\t/dev/root\n",
            stderr=b"",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("senpai.launch.aws_backend._ssh", return_value=output),
            patch(
                "senpai.launch.aws_backend._remote_free_bytes",
                return_value=80 * aws_backend.GIB,
            ),
            patch("senpai.launch.aws_backend._grow_root_volume") as grow,
        ):
            _ensure_remote_capacity(
                args(),
                AwsContext("us-east-1", "research"),
                Path(tmp),
                {"instance_id": "i-123"},
                0,
                0,
            )

        grow.assert_not_called()

    def test_storage_mounts_reject_a_separate_docker_device(self):
        output = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=b"/dev/root\t/dev/root\t/dev/docker\n",
            stderr=b"",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("senpai.launch.aws_backend._ssh", return_value=output),
            patch("senpai.launch.aws_backend._remote_free_bytes") as free,
            self.assertRaisesRegex(RuntimeError, "share one root device"),
        ):
            _ensure_remote_capacity(
                args(),
                AwsContext("us-east-1", "research"),
                Path(tmp),
                {"instance_id": "i-123"},
                0,
                0,
            )

        free.assert_not_called()

    def test_root_layout_accepts_ext4_partition_and_xfs_disk(self):
        cases = (
            (
                b"/dev/nvme0n1p1\t/dev/nvme0n1p1\t/dev/nvme0n1p1\t"
                b"ext4\tpart\tnvme0n1\t1\tgpt\t1\n",
                AwsRootLayout(
                    "/dev/nvme0n1p1", "ext4", "/dev/nvme0n1", 1, "gpt"
                ),
                "growpart",
            ),
            (
                b"/dev/nvme0n1\t/dev/nvme0n1\t/dev/nvme0n1\t"
                b"xfs\tdisk\t-\t-\t-\t-\n",
                AwsRootLayout(
                    "/dev/nvme0n1", "xfs", "/dev/nvme0n1", None, None
                ),
                "xfs_growfs",
            ),
        )
        for output, expected, tool in cases:
            with (
                self.subTest(filesystem=expected.filesystem),
                tempfile.TemporaryDirectory() as tmp,
            ):
                completed = subprocess.CompletedProcess(
                    ["ssh"], 0, stdout=output, stderr=b""
                )
                checked = subprocess.CompletedProcess(
                    ["ssh"], 0, stdout=b"", stderr=b""
                )
                with patch(
                    "senpai.launch.aws_backend._ssh",
                    side_effect=[completed, checked],
                ) as ssh:
                    layout = aws_backend._root_layout(
                        args(), Path(tmp), {"instance_id": "i-123"}
                    )

                self.assertEqual(layout, expected)
                self.assertIn(tool, ssh.call_args_list[1].args[2])
                discovery = ssh.call_args_list[0].args[2]
                self.assertIn('start=$(cat "$directory/start")', discovery)
                self.assertIn('size=$(cat "$directory/size")', discovery)

    def test_root_layout_rejects_separate_home_or_docker_filesystem(self):
        output = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=(
                b"/dev/nvme0n1p1\t/dev/nvme0n1p1\t/dev/nvme1n1\t"
                b"ext4\tpart\tnvme0n1\t1\tgpt\t1\n"
            ),
            stderr=b"",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("senpai.launch.aws_backend._ssh", return_value=output),
            self.assertRaisesRegex(RuntimeError, "share one root device"),
        ):
            aws_backend._root_layout(
                args(), Path(tmp), {"instance_id": "i-123"}
            )

    def test_root_layout_rejects_non_final_partition(self):
        output = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=(
                b"/dev/nvme0n1p1\t/dev/nvme0n1p1\t/dev/nvme0n1p1\t"
                b"ext4\tpart\tnvme0n1\t1\tgpt\t2\n"
            ),
            stderr=b"",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("senpai.launch.aws_backend._ssh", return_value=output),
            self.assertRaisesRegex(RuntimeError, "non-final root partitions"),
        ):
            aws_backend._root_layout(
                args(), Path(tmp), {"instance_id": "i-123"}
            )

    def test_dos_partition_cannot_grow_beyond_two_tib(self):
        layout = AwsRootLayout(
            "/dev/nvme0n1p1", "ext4", "/dev/nvme0n1", 1, "dos"
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._root_volume",
                return_value=("vol-root", aws_backend.DOS_PARTITION_MAX_GIB - 1),
            ),
            patch(
                "senpai.launch.aws_backend._root_layout",
                return_value=layout,
            ),
            patch("senpai.launch.aws_backend._aws_raw") as aws,
            self.assertRaisesRegex(RuntimeError, "DOS/MBR"),
        ):
            _grow_root_volume(
                args(),
                AwsContext("us-east-1", "research"),
                Path(tmp),
                {"instance_id": "i-123"},
                aws_backend.GIB,
            )

        aws.assert_not_called()

    def test_guest_disk_size_is_polled_before_filesystem_growth(self):
        sizes = [100 * aws_backend.GIB, 103 * aws_backend.GIB]

        def completed(size):
            return subprocess.CompletedProcess(
                ["ssh"], 0, stdout=f"{size}\n".encode(), stderr=b""
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._ssh",
                side_effect=[completed(size) for size in sizes],
            ) as ssh,
            patch("senpai.launch.aws_backend.time.sleep") as sleep,
        ):
            aws_backend._wait_for_guest_disk_size(
                args(),
                Path(tmp),
                {"instance_id": "i-123"},
                "/dev/nvme0n1",
                103,
            )

        self.assertEqual(ssh.call_count, 2)
        self.assertIn("lsblk -bdno SIZE /dev/nvme0n1", ssh.call_args.args[2])
        sleep.assert_called_once()

    def test_filesystem_growth_uses_the_validated_layout(self):
        cases = (
            (
                AwsRootLayout(
                    "/dev/nvme0n1p1", "ext4", "/dev/nvme0n1", 1, "gpt"
                ),
                "sudo growpart /dev/nvme0n1 1; sudo resize2fs /dev/nvme0n1p1",
            ),
            (
                AwsRootLayout(
                    "/dev/nvme0n1", "xfs", "/dev/nvme0n1", None, None
                ),
                "sudo xfs_growfs /",
            ),
        )
        for layout, expected in cases:
            with (
                self.subTest(filesystem=layout.filesystem),
                tempfile.TemporaryDirectory() as tmp,
                patch("senpai.launch.aws_backend._ssh") as ssh,
            ):
                aws_backend._grow_root_filesystem(
                    args(),
                    Path(tmp),
                    {"instance_id": "i-123"},
                    layout,
                )

            self.assertIn(expected, ssh.call_args.args[2])

    def test_unsupported_root_layout_fails_before_modify_volume(self):
        output = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=(
                b"/dev/mapper/root\t/dev/mapper/root\t/dev/mapper/root\t"
                b"ext4\tlvm\t-\t-\t-\t-\n"
            ),
            stderr=b"",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "senpai.launch.aws_backend._root_volume",
                return_value=("vol-root", 100),
            ),
            patch("senpai.launch.aws_backend._ssh", return_value=output),
            patch("senpai.launch.aws_backend._aws_raw") as aws,
            self.assertRaisesRegex(RuntimeError, "LVM, device-mapper"),
        ):
            _grow_root_volume(
                args(),
                AwsContext("us-east-1", "research"),
                Path(tmp),
                {"instance_id": "i-123"},
                aws_backend.GIB,
            )

        aws.assert_not_called()

    def test_no_data_still_reserves_runtime_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = {"instance_id": "i-123", "phase": "sizing-storage"}
            with (
                patch("senpai.launch.aws_backend._ensure_remote_capacity") as ensure,
                patch("senpai.launch.aws_backend._stream_directory") as stream,
            ):
                _prepare_remote_storage(
                    args(data_dir=""),
                    AwsContext("us-east-1", "research"),
                    run_dir,
                    state,
                )

            ensure.assert_called_once_with(
                args(data_dir=""),
                AwsContext("us-east-1", "research"),
                run_dir,
                state,
                0,
                0,
            )
            stream.assert_not_called()
            self.assertEqual(state["runtime_reserve_gib"], 80)

    def test_storage_recomputes_data_and_rechecks_reserve_after_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            data_dir = run_dir / "data"
            data_dir.mkdir()
            (data_dir / "sample.bin").write_bytes(b"sample")
            (data_dir / "empty-directory").mkdir()
            state = {"instance_id": "i-123", "phase": "sizing-storage"}
            launch_args = args(data_dir=str(data_dir))
            with (
                patch("senpai.launch.aws_backend._ensure_remote_capacity") as ensure,
                patch("senpai.launch.aws_backend._stream_directory") as stream,
                patch(
                    "senpai.launch.aws_backend._remote_free_bytes",
                    return_value=79 * aws_backend.GIB,
                ),
                self.assertRaisesRegex(RuntimeError, "after dataset upload"),
            ):
                _prepare_remote_storage(
                    launch_args,
                    AwsContext("us-east-1", "research"),
                    run_dir,
                    state,
                )

            ensure.assert_called_once_with(
                launch_args,
                AwsContext("us-east-1", "research"),
                run_dir,
                state,
                3,
                6,
            )
            self.assertEqual(state["data_directories"], 1)
            self.assertEqual(
                state["data_allocation_bytes"],
                6 + 3 * aws_backend.REMOTE_FILESYSTEM_BLOCK_BYTES,
            )
            stream.assert_called_once_with(
                run_dir,
                state,
                data_dir.resolve(),
                aws_backend.REMOTE_DATA,
                7200,
                (1, 6),
            )
            saved = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(saved["data_files"], 1)
            self.assertEqual(saved["data_bytes"], 6)
            self.assertEqual(
                saved["data_allocation_bytes"],
                6 + 3 * aws_backend.REMOTE_FILESYSTEM_BLOCK_BYTES,
            )

    def test_source_bootstrap_rejects_an_existing_destination_and_is_atomic(self):
        completed = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=b"",
            stderr=b"",
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            def make_bundle(path):
                path.write_bytes(b"bundle")

            with (
                patch("senpai.launch.aws_backend._wait_for_ssh"),
                patch("senpai.launch.aws_backend._wait_for_gpu"),
                patch(
                    "senpai.launch.aws_backend._source_bundle",
                    side_effect=make_bundle,
                ),
                patch(
                    "senpai.launch.aws_backend._ssh",
                    return_value=completed,
                ) as ssh,
            ):
                _prepare_host(
                    args(),
                    run_dir,
                    {"instance_id": "i-123", "public_ip": "203.0.113.10"},
                )

            bootstrap = ssh.call_args_list[1]
            command = bootstrap.args[2]
            self.assertTrue(command.startswith("set -eu; umask 077;"))
            self.assertIn("bundle=$(mktemp /home/ubuntu/senpai.", command)
            self.assertIn("source_tmp=$(mktemp -d /home/ubuntu/senpai-source.", command)
            self.assertIn("trap \'rm -rf", command)
            self.assertIn(f"test ! -e {aws_backend.REMOTE_SOURCE}", command)
            self.assertIn('git clone -q "$bundle" "$source_tmp"', command)
            self.assertIn(f'mv "$source_tmp" {aws_backend.REMOTE_SOURCE}', command)
            self.assertIsNotNone(bootstrap.kwargs["input_file"])
            self.assertFalse((run_dir / "source.bundle").exists())

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
            self.assertEqual(launch.call_args.kwargs, {"show_lifecycle": False})

    def test_remote_preflight_does_not_start_roles(self):
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
            with (
                patch(
                    "senpai.launch.remote.preflight_docker",
                    return_value="plan",
                ) as preflight,
                patch("senpai.launch.remote.launch_docker") as launch,
            ):
                remote.run_from_payload("preflight", path)

            self.assertFalse(path.exists())
            preflight.assert_called_once()
            launch.assert_not_called()

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


class AwsLaunchFlowTests(unittest.TestCase):
    def test_images_and_data_are_validated_before_github_and_role_start(self):
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            launch_args = args(
                aws_state_root=str(root / "state"),
                data_dir=str(data),
            )
            plan = aws_plan(data_files=1, data_bytes=aws_backend.GIB)

            def provision(_args, _plan, _run_dir, state):
                events.append("provision")
                state.update(
                    {
                        "instance_id": "i-123",
                        "public_ip": "203.0.113.10",
                    }
                )

            def prepare(*_):
                events.append("prepare-host")

            def preflight(_args, _roles, _run_dir, _state, *, include_data):
                events.append(f"preflight-data={include_data}")

            def upload(*_):
                events.append("upload-data")

            def github():
                state = json.loads(
                    (root / "state" / "aws-r1" / "state.json").read_text()
                )
                self.assertEqual(state["phase"], "preparing-github")
                self.assertNotIn("roles_starting", state)
                events.append("github")

            def start(_args, _roles, _run_dir, state):
                self.assertTrue(state["roles_starting"])
                events.append("start-roles")

            with (
                redirect_stdout(io.StringIO()),
                patch(
                    "senpai.launch.aws_backend._provision",
                    side_effect=provision,
                ),
                patch(
                    "senpai.launch.aws_backend._prepare_host",
                    side_effect=prepare,
                ),
                patch(
                    "senpai.launch.aws_backend._preflight_roles",
                    side_effect=preflight,
                ),
                patch(
                    "senpai.launch.aws_backend._prepare_remote_storage",
                    side_effect=upload,
                ),
                patch(
                    "senpai.launch.aws_backend._start_roles",
                    side_effect=start,
                ),
            ):
                launch_aws(
                    launch_args,
                    [student()],
                    plan,
                    before_start=github,
                )

        self.assertEqual(
            events,
            [
                "provision",
                "prepare-host",
                "preflight-data=False",
                "upload-data",
                "preflight-data=True",
                "github",
                "start-roles",
            ],
        )

    def test_image_preflight_failure_never_uploads_or_mutates_github(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launch_args = args(
                aws_state_root=str(root / "state"),
                data_dir=str(root / "data"),
            )
            callback = unittest.mock.Mock()
            output = io.StringIO()

            def provision(_args, _plan, _run_dir, state):
                state.update(
                    {
                        "instance_id": "i-123",
                        "public_ip": "203.0.113.10",
                    }
                )

            def cleanup(*_):
                self.assertIn("AWS launch failed:\nimage is invalid", output.getvalue())
                return []

            with (
                redirect_stdout(output),
                patch(
                    "senpai.launch.aws_backend._provision",
                    side_effect=provision,
                ),
                patch("senpai.launch.aws_backend._prepare_host"),
                patch(
                    "senpai.launch.aws_backend._preflight_roles",
                    side_effect=RuntimeError("image is invalid"),
                ),
                patch("senpai.launch.aws_backend._prepare_remote_storage") as upload,
                patch("senpai.launch.aws_backend._start_roles") as start,
                patch(
                    "senpai.launch.aws_backend._cleanup",
                    side_effect=cleanup,
                ) as cleanup,
                self.assertRaisesRegex(RuntimeError, "image is invalid"),
            ):
                launch_aws(
                    launch_args,
                    [student()],
                    aws_plan(data_files=1, data_bytes=aws_backend.GIB),
                    before_start=callback,
                )

            upload.assert_not_called()
            callback.assert_not_called()
            start.assert_not_called()
            cleanup.assert_called_once()

    def test_image_pull_disk_exhaustion_explains_bootstrap_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launch_args = args(aws_state_root=str(root / "state"))

            def provision(_args, _plan, _run_dir, state):
                state.update(
                    {
                        "instance_id": "i-123",
                        "public_ip": "203.0.113.10",
                    }
                )

            with (
                redirect_stdout(io.StringIO()),
                patch(
                    "senpai.launch.aws_backend._provision",
                    side_effect=provision,
                ),
                patch("senpai.launch.aws_backend._prepare_host"),
                patch(
                    "senpai.launch.aws_backend._preflight_roles",
                    side_effect=RuntimeError("no space left on device"),
                ),
                patch("senpai.launch.aws_backend._prepare_remote_storage") as storage,
                patch("senpai.launch.aws_backend._cleanup", return_value=[]),
                self.assertRaisesRegex(
                    RuntimeError,
                    r"Increase --aws_volume_gib.*compressed/unpacked image-pull peak",
                ),
            ):
                launch_aws(launch_args, [student()], aws_plan(volume_gib=100))

            storage.assert_not_called()


class AwsLifecycleTests(unittest.TestCase):
    def test_status_delegates_to_remote_docker_status_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / "state"
            run_dir = state_root / "aws-r1"
            run_dir.mkdir(parents=True)
            (run_dir / "id_ed25519").write_text("private key")
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "account_id": "123456789012",
                        "instance_id": "i-123",
                        "instance_type": "g5.8xlarge",
                        "phase": "running",
                        "profile": "research",
                        "public_ip": "203.0.113.10",
                        "region": "us-east-1",
                        "roles_starting": True,
                        "tag": "aws-r1",
                    }
                )
            )
            running = {"State": {"Name": "running"}}
            completed = subprocess.CompletedProcess(
                ["ssh"],
                0,
                stdout=b"student: healthy\n",
                stderr=b"",
            )
            output = io.StringIO()
            with (
                redirect_stdout(output),
                patch("senpai.launch.aws_backend._check_account"),
                patch(
                    "senpai.launch.aws_backend._instance_details",
                    return_value=running,
                ),
                patch(
                    "senpai.launch.aws_backend._ssh",
                    return_value=completed,
                ) as ssh,
            ):
                status_aws("aws-r1", str(state_root))

            command = ssh.call_args.args[2]
            self.assertIn(f"{aws_backend.REMOTE_SOURCE}/k8s/docker.py", command)
            self.assertIn("status aws-r1", command)
            self.assertIn(f"--run-root {aws_backend.REMOTE_RUN_ROOT}", command)
            self.assertIn("student: healthy", output.getvalue())

            with (
                patch("senpai.launch.aws_backend._check_account"),
                patch(
                    "senpai.launch.aws_backend._instance_details",
                    return_value=running,
                ),
                patch(
                    "senpai.launch.aws_backend._ssh",
                    side_effect=RuntimeError("remote Docker status failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "Docker status failed"),
            ):
                status_aws("aws-r1", str(state_root))

    def test_status_reports_pre_role_launch_phases_without_remote_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / "state"
            run_dir = state_root / "aws-r1"
            run_dir.mkdir(parents=True)
            (run_dir / "id_ed25519").write_text("private key")
            for launch_phase in ("booting", "validating-images", "uploading-data"):
                with self.subTest(launch_phase=launch_phase):
                    (run_dir / "state.json").write_text(
                        json.dumps(
                            {
                                "account_id": "123456789012",
                                "instance_id": "i-123",
                                "instance_type": "g5.8xlarge",
                                "phase": launch_phase,
                                "profile": "research",
                                "public_ip": "203.0.113.10",
                                "region": "us-east-1",
                                "tag": "aws-r1",
                            }
                        )
                    )
                    output = io.StringIO()
                    with (
                        redirect_stdout(output),
                        patch("senpai.launch.aws_backend._check_account"),
                        patch(
                            "senpai.launch.aws_backend._instance_details",
                            return_value={"State": {"Name": "running"}},
                        ),
                        patch("senpai.launch.aws_backend._ssh") as ssh,
                    ):
                        status_aws("aws-r1", str(state_root))

                    self.assertIn(f"launcher={launch_phase}", output.getvalue())
                    self.assertIn("state=running", output.getvalue())
                    ssh.assert_not_called()


class AwsCleanupTests(unittest.TestCase):
    def test_booting_host_still_gets_a_graceful_container_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "id_ed25519").write_text("private key")
            state = {
                "tag": "aws-r1",
                "phase": "booting",
                "roles_starting": True,
                "public_ip": "203.0.113.10",
            }
            completed = subprocess.CompletedProcess(
                ["ssh"], 0, stdout=b"", stderr=b""
            )

            with patch(
                "senpai.launch.aws_backend._ssh", return_value=completed
            ) as ssh:
                _stop_remote_roles(run_dir, state)

            command = ssh.call_args.args[2]
            self.assertIn("k8s/docker.py", command)
            self.assertIn("terminate", command)
            self.assertIn("--run-root", command)

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
                root_device="/dev/sda1",
                volume_gib=250,
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
                patch("senpai.launch.aws_backend._preflight_roles"),
                patch("senpai.launch.aws_backend._prepare_remote_storage"),
                patch("senpai.launch.aws_backend._start_roles"),
            ):
                launch_aws(launch_args, [student()], plan)

            text = output.getvalue()
            self.assertIn(f"--state-root {launch_args.aws_state_root}", text)
            self.assertIn("--profile research", text)
            self.assertIn("status aws-r1", text)
            self.assertIn("terminate aws-r1", text)

    def test_provision_retries_capacity_across_subnets_with_persisted_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = {}
            plan = aws_plan(
                subnets=(
                    AwsSubnet("subnet-a", "vpc-123", "us-east-1a"),
                    AwsSubnet("subnet-b", "vpc-123", "us-east-1b"),
                ),
            )
            attempts = []

            def aws_json(_context, _service, operation, *arguments):
                if operation == "create-security-group":
                    return {"GroupId": "sg-123"}
                if operation == "authorize-security-group-ingress":
                    return {}
                if operation == "run-instances":
                    saved = json.loads((run_dir / "state.json").read_text())
                    self.assertTrue(saved["instance_launch_started"])
                    token = arguments[arguments.index("--client-token") + 1]
                    subnet = arguments[arguments.index("--subnet-id") + 1]
                    self.assertEqual(saved["client_token"], token)
                    self.assertEqual(saved["client_tokens"][-1], token)
                    self.assertEqual(saved["subnet_id"], subnet)
                    attempts.append((subnet, token))
                    if subnet == "subnet-a":
                        raise AwsCommandError(
                            "`aws ec2 run-instances` failed: "
                            "InsufficientInstanceCapacity"
                        )
                    return {"Instances": [{"InstanceId": "i-123"}]}
                if operation == "describe-instances":
                    return {
                        "Reservations": [
                            {"Instances": [{"PublicIpAddress": "203.0.113.10"}]}
                        ]
                    }
                self.fail(f"unexpected operation: {operation}")

            with (
                patch(
                    "senpai.launch.aws_backend._aws_raw",
                    side_effect=["private-key", ""],
                ),
                patch(
                    "senpai.launch.aws_backend._aws_json",
                    side_effect=aws_json,
                ),
            ):
                _provision(args(), plan, run_dir, state)

            self.assertEqual(
                [subnet for subnet, _token in attempts],
                ["subnet-a", "subnet-b"],
            )
            self.assertEqual(len({token for _subnet, token in attempts}), 2)
            self.assertEqual(state["client_tokens"], [token for _, token in attempts])
            self.assertEqual(state["capacity_failures"][0].split(":", 1)[0], "us-east-1a")
            self.assertEqual(state["instance_id"], "i-123")

    def test_provision_does_not_retry_unknown_run_instance_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = {}
            plan = aws_plan(
                subnets=(
                    AwsSubnet("subnet-a", "vpc-123", "us-east-1a"),
                    AwsSubnet("subnet-b", "vpc-123", "us-east-1b"),
                ),
            )
            attempts = []

            def aws_json(_context, _service, operation, *arguments):
                if operation == "create-security-group":
                    return {"GroupId": "sg-123"}
                if operation == "authorize-security-group-ingress":
                    return {}
                if operation == "run-instances":
                    attempts.append(arguments[arguments.index("--subnet-id") + 1])
                    raise AwsCommandError(
                        "`aws ec2 run-instances` failed: UnauthorizedOperation"
                    )
                self.fail(f"unexpected operation: {operation}")

            with (
                patch(
                    "senpai.launch.aws_backend._aws_raw",
                    return_value="private-key",
                ),
                patch(
                    "senpai.launch.aws_backend._aws_json",
                    side_effect=aws_json,
                ),
                self.assertRaisesRegex(AwsCommandError, "UnauthorizedOperation"),
            ):
                _provision(args(), plan, run_dir, state)

            self.assertEqual(attempts, ["subnet-a"])
            self.assertEqual(len(state["client_tokens"]), 1)

    def test_security_group_is_recovered_after_create_response_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = {
                "tag": "aws-r1",
                "vpc_id": "vpc-123",
                "region": "us-east-1",
                "profile": "research",
            }

            def aws_json(_context, _service, operation, *arguments):
                if operation == "create-security-group":
                    saved = json.loads((run_dir / "state.json").read_text())
                    self.assertEqual(saved["security_group_name"], state["security_group_name"])
                    raise AwsCommandError("connection lost after create response")
                if operation == "describe-security-groups":
                    self.assertIn(
                        f"Name=group-name,Values={state['security_group_name']}",
                        arguments,
                    )
                    self.assertIn("Name=vpc-id,Values=vpc-123", arguments)
                    self.assertIn("Name=tag:senpai:run,Values=aws-r1", arguments)
                    return {"SecurityGroups": [{"GroupId": "sg-recovered"}]}
                self.fail(f"unexpected operation: {operation}")

            def aws_raw(_context, _service, operation, *arguments):
                if operation == "create-key-pair":
                    return "private-key"
                self.assertIn(
                    operation,
                    {"delete-security-group", "delete-key-pair"},
                )
                return ""

            with (
                patch(
                    "senpai.launch.aws_backend._aws_raw",
                    side_effect=aws_raw,
                ) as aws,
                patch(
                    "senpai.launch.aws_backend._aws_json",
                    side_effect=aws_json,
                ),
            ):
                with self.assertRaisesRegex(AwsCommandError, "connection lost"):
                    _provision(args(), aws_plan(), run_dir, state)

                self.assertTrue(
                    state["security_group_name"].startswith("senpai-aws-r1-")
                )
                self.assertNotIn("security_group_id", state)
                errors = _cleanup(run_dir, state)

            self.assertEqual(errors, [])
            self.assertFalse(run_dir.exists())
            operations = [call.args[2] for call in aws.call_args_list]
            self.assertEqual(
                operations,
                ["create-key-pair", "delete-security-group", "delete-key-pair"],
            )

    def test_cleanup_preserves_state_while_security_group_outcome_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            state = {
                "phase": "provisioning",
                "region": "us-east-1",
                "profile": "research",
                "security_group_create_started": True,
                "security_group_name": "senpai-aws-r1-abcd1234",
                "tag": "aws-r1",
                "vpc_id": "vpc-123",
            }

            with (
                patch(
                    "senpai.launch.aws_backend._aws_json",
                    return_value={"SecurityGroups": []},
                ) as aws_json,
                patch("senpai.launch.aws_backend.time.sleep"),
            ):
                errors = _cleanup(run_dir, state)

            self.assertEqual(aws_json.call_count, 3)
            self.assertEqual(len(errors), 1)
            self.assertIn("outcome is still unknown", errors[0])
            self.assertTrue(run_dir.is_dir())
            saved = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(saved["phase"], "cleanup-failed")
            self.assertEqual(saved["cleanup_errors"], errors)

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

    def test_cleanup_continues_after_a_termination_waiter_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            state = {
                "region": "us-east-1",
                "profile": "research",
                "instance_id": "i-123",
                "security_group_id": "sg-123",
            }
            waiter_calls = 0

            def aws_raw(_context, _service, operation, *arguments):
                nonlocal waiter_calls
                if operation == "wait":
                    waiter_calls += 1
                    if waiter_calls == 1:
                        raise AwsCommandError(
                            "`aws ec2 wait` failed: Waiter InstanceTerminated "
                            "failed: Max attempts exceeded"
                        )
                return ""

            with patch(
                "senpai.launch.aws_backend._aws_raw",
                side_effect=aws_raw,
            ) as aws:
                errors = _cleanup(run_dir, state)

            self.assertEqual(errors, [])
            self.assertFalse(run_dir.exists())
            operations = [call.args[2] for call in aws.call_args_list]
            self.assertEqual(
                operations,
                [
                    "terminate-instances",
                    "wait",
                    "wait",
                    "delete-security-group",
                ],
            )

    def test_cleanup_bounds_termination_wait_and_preserves_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            state = {
                "region": "us-east-1",
                "profile": "research",
                "instance_id": "i-123",
                "security_group_id": "sg-123",
            }

            def aws_raw(_context, _service, operation, *arguments):
                if operation == "wait":
                    raise AwsCommandError(
                        "`aws ec2 wait` failed: Waiter InstanceTerminated "
                        "failed: Max attempts exceeded"
                    )
                return ""

            with patch(
                "senpai.launch.aws_backend._aws_raw",
                side_effect=aws_raw,
            ) as aws:
                errors = _cleanup(run_dir, state)

            self.assertEqual(len(errors), 1)
            self.assertIn("Max attempts exceeded", errors[0])
            operations = [call.args[2] for call in aws.call_args_list]
            self.assertEqual(
                operations,
                ["terminate-instances", "wait", "wait"],
            )
            saved = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(saved["phase"], "cleanup-failed")
            self.assertEqual(saved["cleanup_errors"], errors)

    def test_cleanup_does_not_retry_non_timeout_waiter_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            state = {
                "region": "us-east-1",
                "profile": "research",
                "instance_id": "i-123",
            }

            def aws_raw(_context, _service, operation, *arguments):
                if operation == "wait":
                    raise AwsCommandError(
                        "`aws ec2 wait` failed: UnauthorizedOperation"
                    )
                return ""

            with patch(
                "senpai.launch.aws_backend._aws_raw",
                side_effect=aws_raw,
            ) as aws:
                errors = _cleanup(run_dir, state)

            self.assertEqual(len(errors), 1)
            self.assertIn("UnauthorizedOperation", errors[0])
            operations = [call.args[2] for call in aws.call_args_list]
            self.assertEqual(operations, ["terminate-instances", "wait"])
            self.assertTrue((run_dir / "state.json").is_file())


if __name__ == "__main__":
    unittest.main()
