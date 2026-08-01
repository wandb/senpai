# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Provision one AWS GPU host and reuse Senpai's Docker launcher on it."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .specs import (
    CONTAINER_RESERVED_PATHS,
    RoleSpec,
    validate_identifier,
    validate_pvc_mount_path,
    validate_role_specs,
    validate_writable_parent,
)

ROOT = Path(__file__).resolve().parents[2]
DLAMI_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)
DLAMI_NAME = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"
REMOTE_SOURCE = "/home/ubuntu/senpai-source"
REMOTE_DATA = "/home/ubuntu/senpai-data"
REMOTE_RUN_ROOT = "/home/ubuntu/.senpai/runs"
DEFAULT_STATE_ROOT = "~/.senpai/aws"
GIB = 1024**3
GP3_MAX_GIB = 16_384
AWS_VOLUME_HEADROOM_GIB = 160
AWS_POST_UPLOAD_HEADROOM_GIB = 80
AWS_HOST_VCPU_HEADROOM = 4
AWS_HOST_MEMORY_HEADROOM_GIB = 16
AUTO_INSTANCE_TYPES = (
    ("g6e.8xlarge", 1, 32, 256),
    ("g5.24xlarge", 4, 96, 384),
    ("g5.48xlarge", 8, 192, 768),
)
CAPACITY_ERROR_CODES = (
    "InsufficientInstanceCapacity",
    "InsufficientHostCapacity",
    "InsufficientFreeAddressesInSubnet",
    "UnfulfillableCapacity",
)
AWS_TERMINATION_WAITER_ATTEMPTS = 2  # Two 10-minute AWS waiter windows.


@dataclass(frozen=True)
class AwsContext:
    region: str
    profile: str = ""


@dataclass(frozen=True)
class AwsRequirements:
    gpus: int
    vcpus: int
    memory_gib: int
    data_files: int = 0
    data_bytes: int = 0


@dataclass(frozen=True)
class AwsInstanceShape:
    instance_type: str
    gpus: int
    vcpus: int
    memory_gib: int
    architectures: tuple[str, ...] = ("x86_64",)

    def fits(self, requirements: AwsRequirements) -> bool:
        return (
            self.gpus >= requirements.gpus
            and self.vcpus >= requirements.vcpus
            and self.memory_gib >= requirements.memory_gib
        )


@dataclass(frozen=True)
class AwsSubnet:
    subnet_id: str
    vpc_id: str
    availability_zone: str


@dataclass(frozen=True)
class AwsLaunchPlan:
    """Read-only resolution of an AWS launch."""

    context: AwsContext
    account_id: str
    instance_type: str
    ami_id: str
    root_device: str
    volume_gib: int
    subnets: tuple[AwsSubnet, ...]
    ssh_cidr: str
    data_files: int = 0
    data_bytes: int = 0

    @property
    def subnet_id(self) -> str:
        return self.subnets[0].subnet_id

    @property
    def vpc_id(self) -> str:
        return self.subnets[0].vpc_id

    @property
    def availability_zone(self) -> str:
        return self.subnets[0].availability_zone


class AwsCommandError(RuntimeError):
    """An AWS CLI call failed."""


def _aws_cli_missing() -> AwsCommandError:
    return AwsCommandError(
        "AWS CLI is not installed. Install AWS CLI v2, then run "
        "`aws configure sso` and `aws sso login`."
    )


def _validate_tag(tag: str) -> None:
    validate_identifier("AWS tag", tag)


def _aws_command(context: AwsContext, *arguments: str) -> list[str]:
    command = ["aws"]
    if context.profile:
        command.extend(["--profile", context.profile])
    if context.region:
        command.extend(["--region", context.region])
    command.extend(arguments)
    return command


def _redact_aws_error(message: str) -> str:
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "EXA_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
    ):
        secret = os.environ.get(name, "")
        if secret:
            message = message.replace(secret, "<redacted>")
    return message


def _aws_raw(context: AwsContext, *arguments: str) -> str:
    command = _aws_command(context, *arguments)
    env = {**os.environ, "AWS_PAGER": "", "AWS_CLI_AUTO_PROMPT": "off"}
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, env=env, check=False
        )
    except FileNotFoundError as error:
        raise _aws_cli_missing() from error
    if result.returncode:
        detail = _redact_aws_error(result.stderr.strip())
        operation = shlex.join(["aws", *arguments[:2]])
        raise AwsCommandError(
            f"`{operation}` failed: {detail or '<no error detail>'}"
        )
    return result.stdout


def _aws_json(context: AwsContext, *arguments: str) -> dict:
    output = _aws_raw(context, *arguments, "--output", "json")
    return json.loads(output or "{}")


def _state_dir(state_root: str, tag: str) -> Path:
    _validate_tag(tag)
    root = Path(state_root).expanduser().resolve()
    candidate = root / tag
    if candidate.is_symlink():
        raise RuntimeError(f"AWS state path must not be a symlink: {candidate}")
    path = candidate.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"AWS state path escapes configured root: {path}")
    return path


def _save_state(run_dir: Path, state: dict) -> None:
    path = run_dir / "state.json"
    temporary = run_dir / "state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _load_state(tag: str, state_root: str) -> tuple[Path, dict]:
    run_dir = _state_dir(state_root, tag)
    path = run_dir / "state.json"
    if not path.is_file():
        raise RuntimeError(f"No AWS Senpai state found for {tag!r} at {path}")
    return run_dir, json.loads(path.read_text())


def _state_context(state: dict, profile: str = "") -> AwsContext:
    return AwsContext(region=state["region"], profile=profile or state.get("profile", ""))


def _recorded_instance_id(state: dict) -> str:
    instance_id = state.get("instance_id", "")
    if not instance_id:
        raise RuntimeError(
            f"AWS run {state.get('tag', '<unknown>')!r} has no recorded instance "
            f"yet (launcher phase: {state.get('phase', 'unknown')})"
        )
    return instance_id


def _resolve_region(configured: str, profile: str) -> str:
    region = (
        configured
        or os.environ.get("AWS_REGION", "")
        or os.environ.get("AWS_DEFAULT_REGION", "")
    )
    if region:
        return region
    context = AwsContext("", profile)
    try:
        result = subprocess.run(
            _aws_command(context, "configure", "get", "region"),
            capture_output=True,
            text=True,
            env={**os.environ, "AWS_PAGER": "", "AWS_CLI_AUTO_PROMPT": "off"},
            check=False,
        )
    except FileNotFoundError as error:
        raise _aws_cli_missing() from error
    region = result.stdout.strip()
    if result.returncode or not region:
        raise RuntimeError(
            "No AWS region is configured. Pass --aws_region or run "
            "`aws configure set region REGION --profile PROFILE`."
        )
    return region


def _check_account(context: AwsContext, expected: str | None = None) -> dict:
    try:
        identity = _aws_json(context, "sts", "get-caller-identity")
    except AwsCommandError as error:
        login = (
            shlex.join(["aws", "sso", "login", "--profile", context.profile])
            if context.profile
            else "`aws configure sso` + `aws sso login`, or set standard AWS "
            "credential environment variables"
        )
        raise AwsCommandError(
            f"AWS credential preflight failed.\n{error}\nAuthenticate with {login}."
        ) from error
    account = identity["Account"]
    if expected and account != expected:
        raise RuntimeError(
            f"Refusing AWS operation: current credentials are for account "
            f"{account}, but this run belongs to {expected}"
        )
    return identity


def _required_gpus(args, role_specs: list[RoleSpec]) -> int:
    students = sum(spec.role == "student" for spec in role_specs)
    return students * args.gpus_per_student


def _validate_aws_inputs(
    args,
    role_specs: list[RoleSpec],
    *,
    require_writable_state: bool = True,
) -> AwsRequirements:
    """Validate the local launch contract before invoking AWS."""
    validate_role_specs("AWS", args.tag, role_specs)
    run_dir = _state_dir(args.aws_state_root, args.tag)
    state_root = run_dir.parent
    if args.docker_student_gpu_ids:
        raise ValueError(
            "AWS assigns its dedicated GPUs automatically; remove "
            "--docker_student_gpu_ids"
        )
    data_files = data_bytes = 0
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        if not data_dir.is_dir():
            raise ValueError(f"AWS data directory does not exist: {data_dir}")
        source_root = ROOT.resolve()
        if data_dir in {Path(data_dir.anchor), Path.home().resolve()}:
            raise ValueError("AWS data directory must not be / or the home directory")
        if data_dir.is_relative_to(source_root) or source_root.is_relative_to(data_dir):
            raise ValueError("AWS data directory must not overlap the Senpai source")
        if data_dir.is_relative_to(state_root) or state_root.is_relative_to(data_dir):
            raise ValueError("AWS data directory must not overlap AWS lifecycle state")
        validate_pvc_mount_path(args.pvc_mount_path, CONTAINER_RESERVED_PATHS)
        data_files, data_bytes = _data_summary(data_dir)
    if args.start_gate_path:
        raise ValueError(
            "AWS does not mount an operator filesystem; remove --start_gate_path"
        )
    if args.gpus_per_student < 0:
        raise ValueError("--gpus_per_student must be non-negative")

    if (
        any(spec.role == "student" for spec in role_specs)
        and args.gpus_per_student < 1
    ):
        raise ValueError("AWS students require --gpus_per_student at least 1")

    if run_dir.is_relative_to(ROOT.resolve()):
        raise ValueError(
            "--aws_state_root must be outside the Senpai source checkout so "
            "lifecycle state and the SSH key cannot enter the source snapshot"
        )
    if state_root.exists() and not state_root.is_dir():
        raise ValueError(f"AWS state root is not a directory: {state_root}")
    if run_dir.exists() or run_dir.is_symlink():
        raise RuntimeError(
            f"AWS run state already exists at {run_dir}; use a new --tag or "
            f"`python k8s/aws.py terminate {args.tag}`"
        )
    if require_writable_state:
        validate_writable_parent(state_root, "AWS state root")

    if args.aws_volume_gib < 1:
        raise ValueError("--aws_volume_gib must be at least 1")
    if args.aws_volume_gib > GP3_MAX_GIB:
        raise ValueError(f"--aws_volume_gib cannot exceed {GP3_MAX_GIB:,} GiB")
    if args.aws_ttl_hours <= 0:
        raise ValueError("--aws_ttl_hours must be greater than 0")
    if args.aws_ready_timeout_s < 1:
        raise ValueError("--aws_ready_timeout_s must be at least 1")
    if args.aws_data_timeout_s < 1:
        raise ValueError("--aws_data_timeout_s must be at least 1")

    required_gpus = _required_gpus(args, role_specs)
    requirements = AwsRequirements(
        gpus=required_gpus,
        vcpus=required_gpus * args.cpu_per_gpu + AWS_HOST_VCPU_HEADROOM,
        memory_gib=(
            required_gpus * args.memory_gi_per_gpu + AWS_HOST_MEMORY_HEADROOM_GIB
        ),
        data_files=data_files,
        data_bytes=data_bytes,
    )
    required_volume_gib = (
        math.ceil(data_bytes / GIB) + AWS_VOLUME_HEADROOM_GIB if data_bytes else 0
    )
    if required_volume_gib > GP3_MAX_GIB:
        raise ValueError(
            f"AWS data needs a {required_volume_gib:,} GiB root volume including "
            f"headroom, beyond the gp3 limit of {GP3_MAX_GIB:,} GiB"
        )
    if not args.aws_instance_type:
        _automatic_instance_type(requirements)
    return requirements


def _automatic_instance_type(requirements: AwsRequirements) -> str:
    for instance_type, gpus, vcpus, memory_gib in AUTO_INSTANCE_TYPES:
        if AwsInstanceShape(instance_type, gpus, vcpus, memory_gib).fits(requirements):
            return instance_type
    raise ValueError(
        "AWS single-host automatic sizing cannot fit "
        f"{requirements.gpus} GPUs, {requirements.vcpus} vCPUs, and "
        f"{requirements.memory_gib} GiB RAM. Reduce the role resources or pass "
        "--aws_instance_type for a larger GPU host."
    )


def _instance_type_details(
    context: AwsContext, instance_type: str
) -> AwsInstanceShape:
    payload = _aws_json(
        context,
        "ec2",
        "describe-instance-types",
        "--instance-types",
        instance_type,
    )
    details = payload["InstanceTypes"][0]
    gpus = sum(
        gpu["Count"]
        for gpu in details.get("GpuInfo", {}).get("Gpus", [])
        if gpu.get("Manufacturer", "").lower() == "nvidia"
    )
    return AwsInstanceShape(
        instance_type=instance_type,
        gpus=gpus,
        vcpus=details["VCpuInfo"]["DefaultVCpus"],
        memory_gib=details["MemoryInfo"]["SizeInMiB"] // 1024,
        architectures=tuple(details["ProcessorInfo"]["SupportedArchitectures"]),
    )


def _resolve_ami(context: AwsContext, configured_ami: str) -> tuple[str, str, int]:
    if configured_ami:
        ami_id = configured_ami
    else:
        payload = _aws_json(
            context,
            "ssm",
            "get-parameter",
            "--name",
            DLAMI_PARAMETER,
        )
        ami_id = payload["Parameter"]["Value"]

    payload = _aws_json(context, "ec2", "describe-images", "--image-ids", ami_id)
    images = payload.get("Images", [])
    if not images:
        raise RuntimeError(f"AWS AMI {ami_id!r} is not available in {context.region}")
    image = images[0]
    if image["Architecture"] != "x86_64":
        raise RuntimeError(f"AWS AMI {ami_id} must be x86_64")
    if DLAMI_NAME not in image.get("Name", ""):
        raise RuntimeError(
            f"AWS AMI {ami_id} is not the supported Ubuntu 22.04 NVIDIA "
            "Deep Learning Base AMI. Omit --aws_ami_id to select it automatically."
        )
    root_device = image["RootDeviceName"]
    secondary_ebs = [
        mapping["DeviceName"]
        for mapping in image["BlockDeviceMappings"]
        if mapping["DeviceName"] != root_device and mapping.get("Ebs")
    ]
    if secondary_ebs:
        raise RuntimeError(
            f"AWS AMI {ami_id} has additional EBS devices "
            f"{', '.join(secondary_ebs)}. Use an AMI with one EBS root volume so "
            "the temporary Senpai host cannot leave storage behind."
        )
    root_mapping = next(
        mapping
        for mapping in image["BlockDeviceMappings"]
        if mapping["DeviceName"] == root_device
    )
    return ami_id, root_device, root_mapping["Ebs"].get("VolumeSize", 0)


def _public_route_tables(
    context: AwsContext,
) -> tuple[set[str], set[str], set[str]]:
    tables = _aws_json(context, "ec2", "describe-route-tables")["RouteTables"]
    public_subnets: set[str] = set()
    public_main_vpcs: set[str] = set()
    associated_subnets = {
        association["SubnetId"]
        for table in tables
        for association in table.get("Associations", [])
        if association.get("SubnetId")
    }
    for table in tables:
        public = any(
            route.get("State", "active") == "active"
            and route.get("GatewayId", "").startswith("igw-")
            and (
                route.get("DestinationCidrBlock") == "0.0.0.0/0"
            )
            for route in table.get("Routes", [])
        )
        if not public:
            continue
        for association in table.get("Associations", []):
            if association.get("SubnetId"):
                public_subnets.add(association["SubnetId"])
            if association.get("Main"):
                public_main_vpcs.add(table["VpcId"])
    return public_subnets, public_main_vpcs, associated_subnets


def _select_subnets(
    context: AwsContext,
    instance_type: str,
    configured_subnet: str,
) -> tuple[AwsSubnet, ...]:
    arguments = ["ec2", "describe-subnets"]
    if configured_subnet:
        arguments.extend(["--subnet-ids", configured_subnet])
    payload = _aws_json(context, *arguments)
    public_subnets, public_main_vpcs, associated_subnets = _public_route_tables(
        context
    )
    offered = {
        item["Location"]
        for item in _aws_json(
            context,
            "ec2",
            "describe-instance-type-offerings",
            "--location-type",
            "availability-zone",
            "--filters",
            f"Name=instance-type,Values={instance_type}",
        )["InstanceTypeOfferings"]
    }
    candidates = [
        subnet
        for subnet in payload["Subnets"]
        if subnet["State"] == "available"
        and not subnet.get("Ipv6Native", False)
        and subnet.get("AvailableIpAddressCount", 0) > 0
        and subnet["AvailabilityZone"] in offered
        and (
            subnet["SubnetId"] in public_subnets
            or (
                subnet["SubnetId"] not in associated_subnets
                and subnet["VpcId"] in public_main_vpcs
            )
        )
    ]
    if not candidates:
        target = f" {configured_subnet}" if configured_subnet else ""
        raise RuntimeError(
            f"No public subnet{target} offers {instance_type} in {context.region}. "
            "Pass --aws_subnet_id for a subnet with an internet-gateway route."
        )
    if not configured_subnet:
        candidates_by_vpc: dict[str, list[dict]] = {}
        for subnet in candidates:
            candidates_by_vpc.setdefault(subnet["VpcId"], []).append(subnet)

        def vpc_score(vpc_id: str) -> tuple[bool, int, int, str]:
            subnets = candidates_by_vpc[vpc_id]
            return (
                any(subnet.get("DefaultForAz", False) for subnet in subnets),
                len({subnet["AvailabilityZone"] for subnet in subnets}),
                sum(subnet.get("AvailableIpAddressCount", 0) for subnet in subnets),
                vpc_id,
            )

        candidates = candidates_by_vpc[max(candidates_by_vpc, key=vpc_score)]
    ordered = sorted(
        candidates,
        key=lambda subnet: (
            not subnet.get("DefaultForAz", False),
            -subnet.get("AvailableIpAddressCount", 0),
            subnet["AvailabilityZone"],
            subnet["SubnetId"],
        ),
    )
    return tuple(
        AwsSubnet(
            subnet_id=subnet["SubnetId"],
            vpc_id=subnet["VpcId"],
            availability_zone=subnet["AvailabilityZone"],
        )
        for subnet in ordered
    )


def _select_subnet(
    context: AwsContext,
    instance_type: str,
    configured_subnet: str,
) -> dict:
    """Compatibility wrapper returning the preferred subnet as an AWS payload."""
    subnet = _select_subnets(context, instance_type, configured_subnet)[0]
    return {
        "SubnetId": subnet.subnet_id,
        "VpcId": subnet.vpc_id,
        "AvailabilityZone": subnet.availability_zone,
    }


def _ssh_cidr(configured: str) -> str:
    if configured:
        network = ipaddress.ip_network(configured, strict=True)
    else:
        try:
            with urllib.request.urlopen(
                "https://checkip.amazonaws.com",
                timeout=10,
            ) as response:
                address = response.read().decode().strip()
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                "Could not discover your public IP. Pass --aws_ssh_cidr "
                "with your current IPv4 address followed by /32."
            ) from error
        network = ipaddress.ip_network(f"{address}/32", strict=True)
    if network.version != 4 or network.prefixlen != 32:
        raise ValueError("--aws_ssh_cidr must be one IPv4 address with a /32 prefix")
    return str(network)


def preflight_aws(args, role_specs: list[RoleSpec]) -> AwsLaunchPlan:
    """Resolve and validate AWS resources without changing the account."""
    requirements = _validate_aws_inputs(args, role_specs)
    if not shutil.which("ssh"):
        raise RuntimeError("OpenSSH is required to launch on AWS")
    _check_source_revision(args.repo_revision)

    profile = (
        args.aws_profile
        or os.environ.get("AWS_PROFILE", "")
        or os.environ.get("AWS_DEFAULT_PROFILE", "")
    )
    context = AwsContext(_resolve_region(args.aws_region, profile), profile)
    identity = _check_account(context)
    instance_type = args.aws_instance_type or _automatic_instance_type(requirements)
    shape = _instance_type_details(context, instance_type)
    if "x86_64" not in shape.architectures:
        raise RuntimeError(f"AWS instance type {instance_type} must support x86_64")
    if shape.gpus < 1:
        raise RuntimeError(f"AWS instance {instance_type} does not have an NVIDIA GPU")
    if not shape.fits(requirements):
        raise RuntimeError(
            f"AWS instance {instance_type} has {shape.gpus} GPUs, {shape.vcpus} "
            f"vCPUs, and {shape.memory_gib} GiB RAM; this launch needs "
            f"{requirements.gpus} GPUs, {requirements.vcpus} vCPUs, and "
            f"{requirements.memory_gib} GiB RAM"
        )
    ami_id, root_device, minimum_volume_gib = _resolve_ami(
        context,
        args.aws_ami_id,
    )
    subnets = _select_subnets(context, instance_type, args.aws_subnet_id)
    data_volume_gib = (
        math.ceil(requirements.data_bytes / GIB) + AWS_VOLUME_HEADROOM_GIB
        if requirements.data_bytes
        else 0
    )
    volume_gib = max(args.aws_volume_gib, minimum_volume_gib, data_volume_gib)
    if volume_gib > GP3_MAX_GIB:
        raise RuntimeError(
            f"AWS root volume resolves to {volume_gib:,} GiB, beyond the gp3 "
            f"limit of {GP3_MAX_GIB:,} GiB"
        )
    plan = AwsLaunchPlan(
        context=context,
        account_id=identity["Account"],
        instance_type=instance_type,
        ami_id=ami_id,
        root_device=root_device,
        volume_gib=volume_gib,
        subnets=subnets,
        ssh_cidr=_ssh_cidr(args.aws_ssh_cidr),
        data_files=requirements.data_files,
        data_bytes=requirements.data_bytes,
    )
    print(
        "AWS preflight OK — "
        f"account={plan.account_id}, region={context.region}, "
        f"instance={instance_type} ({shape.gpus} GPUs, {shape.vcpus} vCPUs, "
        f"{shape.memory_gib} GiB RAM), volume={plan.volume_gib} GiB, "
        f"subnets={len(plan.subnets)} across "
        f"{len({subnet.availability_zone for subnet in plan.subnets})} AZs, ami={ami_id}"
    )
    return plan


def _write_private_key(path: Path, material: str) -> None:
    path.write_text(material)
    path.chmod(0o600)


def _user_data(args) -> str:
    ttl_minutes = max(1, round(args.aws_ttl_hours * 60))
    return f"""#!/bin/bash
set -euxo pipefail
shutdown -h +{ttl_minutes} "Senpai AWS safety TTL reached"
if ! command -v docker >/dev/null; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
fi
systemctl enable --now docker
command -v nvidia-ctk
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
usermod -aG docker ubuntu
"""


def _ssh_base(
    run_dir: Path,
    state: dict,
    *,
    compression: bool = False,
) -> list[str]:
    command = [
        "ssh",
        "-i",
        str(run_dir / "id_ed25519"),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={run_dir / 'known_hosts'}",
        f"ubuntu@{state['public_ip']}",
    ]
    if compression:
        command.insert(1, "-C")
    return command


def _ssh(
    run_dir: Path,
    state: dict,
    command: str,
    *,
    input_bytes: bytes | None = None,
    input_file=None,
    timeout: float | None = None,
    check: bool = True,
    compression: bool = False,
    redactions: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            [*_ssh_base(run_dir, state, compression=compression), command],
            input=input_bytes,
            stdin=input_file,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"AWS host command timed out after {timeout:g}s"
        ) from error
    except OSError as error:
        raise RuntimeError(f"Could not run SSH: {error}") from error
    if check and result.returncode:
        stdout = result.stdout.decode(errors="replace").strip()
        stderr = result.stderr.decode(errors="replace").strip()
        details = []
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        detail = "\n".join(details) or "<no output>"
        for secret in redactions:
            if secret:
                detail = detail.replace(secret, "<redacted>")
        raise RuntimeError(
            "AWS host command failed:\n" + _redact_aws_error(detail)
        )
    return result


def _wait_for_ssh(run_dir: Path, state: dict, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            result = _ssh(run_dir, state, "true", timeout=15, check=False)
        except RuntimeError:
            result = None
        if result and result.returncode == 0:
            return
        time.sleep(5)
    raise RuntimeError(
        f"Timed out after {timeout_s:g}s waiting for SSH on {state['instance_id']}"
    )


def _wait_for_gpu(run_dir: Path, state: dict, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            result = _ssh(
                run_dir,
                state,
                "nvidia-smi --query-gpu=index --format=csv,noheader",
                timeout=15,
                check=False,
            )
        except RuntimeError as error:
            last_error = str(error)
        else:
            if result.returncode == 0 and result.stdout.strip():
                return
            last_error = (
                result.stderr.decode(errors="replace").strip()
                or result.stdout.decode(errors="replace").strip()
            )
        time.sleep(5)
    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(
        f"Timed out after {timeout_s:g}s waiting for the NVIDIA GPU on "
        f"{state['instance_id']}{detail}"
    )


def _check_source_revision(revision: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != revision:
        raise RuntimeError(
            f"AWS source checkout is {head}, but the images use {revision}"
        )
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError(
            "AWS source checkout must be clean so uploaded code exactly matches "
            "the immutable images"
        )


def _source_bundle(path: Path) -> None:
    """Bundle the exact checked-out commit so EC2 preserves image provenance."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "bundle", "create", str(path), "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Could not bundle Senpai source: {result.stderr.strip()}")


def _data_summary(source: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for root, directories, files in os.walk(source):
        root_path = Path(root)
        for name in directories:
            if (root_path / name).is_symlink():
                raise ValueError("AWS data directory must not contain symlinks")
        for name in files:
            path = root_path / name
            if path.is_symlink():
                raise ValueError("AWS data directory must not contain symlinks")
            count += 1
            size += path.stat().st_size
    if not count:
        raise ValueError(f"AWS data directory contains no files: {source}")
    return count, size


def _remote_payload(
    args,
    role_specs: list[RoleSpec],
    *,
    include_data: bool = True,
    include_secrets: bool = True,
) -> bytes:
    remote_args = dict(vars(args))
    remote_args.update(
        {
            "backend": "docker",
            "docker_run_root": REMOTE_RUN_ROOT,
            "data_dir": REMOTE_DATA if include_data and args.data_dir else "",
            "dry_run": False,
            "repo_url": REMOTE_SOURCE,
            "start_gate_path": "",
        }
    )
    roles = []
    for spec in role_specs:
        values = asdict(spec)
        values["env"]["REPO_URL"] = REMOTE_SOURCE
        if not include_secrets:
            values["secrets"] = {}
        roles.append(values)
    return json.dumps({"args": remote_args, "roles": roles}).encode()


def _instance_details(context: AwsContext, instance_id: str) -> dict:
    payload = _aws_json(
        context,
        "ec2",
        "describe-instances",
        "--instance-ids",
        instance_id,
    )
    return payload["Reservations"][0]["Instances"][0]


def _wait_for_public_ip(
    context: AwsContext,
    instance_id: str,
    timeout_s: float = 120,
) -> str:
    deadline = time.monotonic() + timeout_s
    while True:
        address = _instance_details(context, instance_id).get("PublicIpAddress", "")
        if address:
            return address
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"AWS instance {instance_id} did not receive a public IPv4 address"
            )
        time.sleep(min(2, remaining))


def _plan_subnets(plan) -> tuple[AwsSubnet, ...]:
    if getattr(plan, "subnets", None):
        return tuple(plan.subnets)
    return (
        AwsSubnet(
            subnet_id=plan.subnet_id,
            vpc_id=plan.vpc_id,
            availability_zone=getattr(plan, "availability_zone", ""),
        ),
    )


def _capacity_error(error: AwsCommandError) -> bool:
    return any(code in str(error) for code in CAPACITY_ERROR_CODES)


def _provision(args, plan: AwsLaunchPlan, run_dir: Path, state: dict) -> None:
    resource_token = uuid.uuid4().hex
    suffix = resource_token[:8]
    key_name = f"senpai-{args.tag}-{suffix}"
    security_group_name = f"senpai-{args.tag}-{suffix}"
    state.update(
        {
            "client_tokens": [],
            "failed_client_tokens": [],
            "instance_launch_started": False,
            "key_name": key_name,
            "security_group_name": security_group_name,
        }
    )
    _save_state(run_dir, state)
    state["key_create_started"] = True
    _save_state(run_dir, state)
    try:
        key_material = _aws_raw(
            plan.context,
            "ec2",
            "create-key-pair",
            "--key-name",
            key_name,
            "--key-type",
            "ed25519",
            "--query",
            "KeyMaterial",
            "--output",
            "text",
        )
    except AwsCommandError as error:
        if "InvalidKeyPair.Duplicate" in str(error):
            state["key_owned"] = False
            _save_state(run_dir, state)
        raise
    state["key_owned"] = True
    _save_state(run_dir, state)
    _write_private_key(run_dir / "id_ed25519", key_material)

    state["security_group_create_started"] = True
    _save_state(run_dir, state)
    payload = _aws_json(
        plan.context,
        "ec2",
        "create-security-group",
        "--group-name",
        security_group_name,
        "--description",
        f"Temporary SSH access for Senpai run {args.tag}",
        "--vpc-id",
        plan.vpc_id,
        "--tag-specifications",
        (
            "ResourceType=security-group,Tags="
            f"[{{Key=Name,Value={security_group_name}}},"
            f"{{Key=senpai:run,Value={args.tag}}}]"
        ),
    )
    state["security_group_id"] = payload["GroupId"]
    _save_state(run_dir, state)
    permissions = [
        {
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "IpRanges": [
                {
                    "CidrIp": plan.ssh_cidr,
                    "Description": f"Senpai {args.tag} operator",
                }
            ],
        }
    ]
    _aws_json(
        plan.context,
        "ec2",
        "authorize-security-group-ingress",
        "--group-id",
        state["security_group_id"],
        "--ip-permissions",
        json.dumps(permissions),
    )

    block_devices = [
        {
            "DeviceName": plan.root_device,
            "Ebs": {
                "DeleteOnTermination": True,
                "Encrypted": True,
                "VolumeSize": plan.volume_gib,
                "VolumeType": "gp3",
            },
        }
    ]
    payload = None
    capacity_failures = []
    for subnet in _plan_subnets(plan):
        client_token = uuid.uuid4().hex
        state["client_tokens"].append(client_token)
        state["client_token"] = client_token
        state["instance_launch_started"] = True
        state["subnet_id"] = subnet.subnet_id
        state["availability_zone"] = subnet.availability_zone
        _save_state(run_dir, state)
        try:
            payload = _aws_json(
                plan.context,
                "ec2",
                "run-instances",
                "--image-id",
                plan.ami_id,
                "--instance-type",
                plan.instance_type,
                "--count",
                "1",
                "--client-token",
                client_token,
                "--subnet-id",
                subnet.subnet_id,
                "--security-group-ids",
                state["security_group_id"],
                "--key-name",
                key_name,
                "--associate-public-ip-address",
                "--block-device-mappings",
                json.dumps(block_devices),
                "--metadata-options",
                "HttpTokens=required,HttpEndpoint=enabled",
                "--instance-initiated-shutdown-behavior",
                "terminate",
                "--user-data",
                _user_data(args),
                "--tag-specifications",
                (
                    "ResourceType=instance,Tags="
                    f"[{{Key=Name,Value=senpai-{args.tag}}},"
                    f"{{Key=senpai:run,Value={args.tag}}}]"
                ),
            )
            break
        except AwsCommandError as error:
            if not _capacity_error(error):
                raise
            capacity_failures.append(
                f"{subnet.availability_zone or subnet.subnet_id}: {error}"
            )
            state["failed_client_tokens"].append(client_token)
            state["capacity_failures"] = capacity_failures
            _save_state(run_dir, state)
            print(
                f"No {plan.instance_type} capacity in "
                f"{subnet.availability_zone or subnet.subnet_id}; trying the "
                "next public subnet.",
                flush=True,
            )
    if payload is None:
        raise RuntimeError(
            f"AWS could not place {plan.instance_type} in any of the "
            f"{len(_plan_subnets(plan))} eligible subnets in "
            f"{plan.context.region}. Try another region or instance type.\n"
            + "\n".join(capacity_failures)
        )
    state["instance_id"] = payload["Instances"][0]["InstanceId"]
    state["phase"] = "provisioning"
    _save_state(run_dir, state)
    _aws_raw(
        plan.context,
        "ec2",
        "wait",
        "instance-running",
        "--instance-ids",
        state["instance_id"],
    )
    state["public_ip"] = _wait_for_public_ip(plan.context, state["instance_id"])
    state["phase"] = "booting"
    _save_state(run_dir, state)


def _stream_directory(
    run_dir: Path,
    state: dict,
    source: Path,
    destination: str,
    timeout_s: float,
) -> None:
    """Stream one local directory to the fresh host without a second local copy."""
    expected = _data_summary(source)
    destination_arg = shlex.quote(destination)
    script = (
        "set -e; "
        f"install -d -m 0750 {destination_arg}; "
        f"tar -xf - -C {destination_arg}; "
        f"chmod -R g+rwX {destination_arg}; "
        f"find {destination_arg} -type d -exec chmod g+s {{}} +; "
        f"find {destination_arg} -type f -printf '%s\\n' | "
        "awk '{bytes += $1; count += 1} END {print count, bytes}'"
    )
    with tempfile.TemporaryFile() as archive_errors:
        archive = subprocess.Popen(
            ["tar", "--no-xattrs", "-C", str(source), "-cf", "-", "."],
            stdout=subprocess.PIPE,
            stderr=archive_errors,
            env={**os.environ, "COPYFILE_DISABLE": "1"},
        )
        assert archive.stdout is not None
        try:
            result = _ssh(
                run_dir,
                state,
                "bash -o pipefail -c " + shlex.quote(script),
                input_file=archive.stdout,
                timeout=timeout_s,
                compression=True,
            )
        finally:
            archive.stdout.close()
            archive.wait()
            archive_errors.seek(0)
            stderr = archive_errors.read().decode(errors="replace").strip()
    if archive.returncode:
        raise RuntimeError(f"Could not archive AWS data directory: {stderr}")
    try:
        actual = tuple(int(value) for value in result.stdout.split())
    except ValueError as error:
        raise RuntimeError("AWS host returned an invalid dataset summary") from error
    if len(actual) != 2:
        raise RuntimeError("AWS host returned an invalid dataset summary")
    if actual != expected:
        raise RuntimeError(
            "AWS dataset verification failed: "
            f"uploaded {actual[0]} files/{actual[1]} bytes, expected "
            f"{expected[0]} files/{expected[1]} bytes"
        )


def _prepare_host(args, run_dir: Path, state: dict) -> None:
    _wait_for_ssh(run_dir, state, args.aws_ready_timeout_s)
    _ssh(
        run_dir,
        state,
        "sudo cloud-init status --wait",
        timeout=args.aws_ready_timeout_s,
    )
    _wait_for_gpu(run_dir, state, args.aws_ready_timeout_s)

    bundle_path = run_dir / "source.bundle"
    _source_bundle(bundle_path)
    try:
        with bundle_path.open("rb") as bundle:
            _ssh(
                run_dir,
                state,
                (
                    "set -eu; umask 077; "
                    "bundle=$(mktemp /home/ubuntu/senpai.XXXXXX.bundle); "
                    "source_tmp=$(mktemp -d /home/ubuntu/senpai-source.XXXXXX); "
                    'trap \'rm -rf "$bundle" "$source_tmp"\' EXIT; '
                    f"test ! -e {REMOTE_SOURCE}; "
                    'cat > "$bundle"; git clone -q "$bundle" "$source_tmp"; '
                    f'test "$(git -C "$source_tmp" rev-parse HEAD)" = '
                    f"{shlex.quote(args.repo_revision)}; "
                    f"mv \"$source_tmp\" {REMOTE_SOURCE}"
                ),
                input_file=bundle,
                timeout=args.aws_ready_timeout_s,
            )
    finally:
        bundle_path.unlink(missing_ok=True)

def _invoke_remote(
    action: str,
    args,
    role_specs: list[RoleSpec],
    run_dir: Path,
    state: dict,
    *,
    include_data: bool,
    include_secrets: bool,
) -> str:
    command = (
        "umask 077; "
        "payload=$(mktemp /home/ubuntu/senpai-launch.XXXXXX.json); "
        "trap 'rm -f \"$payload\"' EXIT; "
        'cat > "$payload"; '
        f"cd {REMOTE_SOURCE}; "
        f'python3 -m senpai.launch.remote {shlex.quote(action)} "$payload"'
    )
    result = _ssh(
        run_dir,
        state,
        command,
        input_bytes=_remote_payload(
            args,
            role_specs,
            include_data=include_data,
            include_secrets=include_secrets,
        ),
        timeout=args.aws_ready_timeout_s + args.docker_ready_timeout_s,
        redactions=(
            tuple(
                secret
                for spec in role_specs
                for secret in spec.secrets.values()
                if secret
            )
            if include_secrets
            else ()
        ),
    )
    return result.stdout.decode(errors="replace").strip()


def _preflight_roles(
    args,
    role_specs: list[RoleSpec],
    run_dir: Path,
    state: dict,
    *,
    include_data: bool,
) -> None:
    output = _invoke_remote(
        "preflight",
        args,
        role_specs,
        run_dir,
        state,
        include_data=include_data,
        include_secrets=False,
    )
    if output:
        print(output)


def _check_remote_data_capacity(
    args,
    run_dir: Path,
    state: dict,
    data_bytes: int,
) -> None:
    result = _ssh(
        run_dir,
        state,
        "df -B1 --output=avail /home/ubuntu | tail -1",
        timeout=args.aws_ready_timeout_s,
    )
    try:
        available = int(result.stdout.strip())
    except ValueError as error:
        raise RuntimeError("AWS host returned invalid free-disk information") from error
    required = data_bytes + AWS_POST_UPLOAD_HEADROOM_GIB * GIB
    if available < required:
        raise RuntimeError(
            f"AWS host has {available / GIB:.1f} GiB free after pulling images, "
            f"but the dataset plus runtime headroom needs {required / GIB:.1f} "
            "GiB. Increase --aws_volume_gib."
        )


def _upload_data(args, run_dir: Path, state: dict, data_bytes: int) -> None:
    if not args.data_dir:
        return
    _check_remote_data_capacity(args, run_dir, state, data_bytes)
    data_dir = Path(args.data_dir).expanduser().resolve()
    print(f"Uploading dataset from {data_dir} to the AWS host.", flush=True)
    _stream_directory(
        run_dir,
        state,
        data_dir,
        REMOTE_DATA,
        args.aws_data_timeout_s,
    )
    print("Dataset upload complete.", flush=True)


def _start_roles(
    args,
    role_specs: list[RoleSpec],
    run_dir: Path,
    state: dict,
) -> None:
    output = _invoke_remote(
        "launch",
        args,
        role_specs,
        run_dir,
        state,
        include_data=True,
        include_secrets=True,
    )
    if output:
        print(output)


def _missing_resource(error: AwsCommandError) -> bool:
    return any(
        code in str(error)
        for code in (
            "InvalidInstanceID.NotFound",
            "InvalidGroup.NotFound",
            "InvalidKeyPair.NotFound",
        )
    )


def _wait_for_instances_terminated(
    context: AwsContext,
    instance_ids: list[str],
) -> None:
    for attempt in range(AWS_TERMINATION_WAITER_ATTEMPTS):
        try:
            _aws_raw(
                context,
                "ec2",
                "wait",
                "instance-terminated",
                "--instance-ids",
                *instance_ids,
            )
            return
        except AwsCommandError as error:
            timed_out = "Max attempts exceeded" in str(error)
            if not timed_out or attempt == AWS_TERMINATION_WAITER_ATTEMPTS - 1:
                raise
            print(
                "AWS instance termination is still in progress; "
                "continuing to wait.",
                flush=True,
            )


def _instances_for_client_token(
    context: AwsContext, client_token: str
) -> list[str]:
    payload = _aws_json(
        context,
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=client-token,Values={client_token}",
    )
    return [
        instance["InstanceId"]
        for reservation in payload.get("Reservations", [])
        for instance in reservation.get("Instances", [])
        if instance.get("State", {}).get("Name") != "terminated"
    ]


def _instance_for_client_token(context: AwsContext, client_token: str) -> str:
    instances = _instances_for_client_token(context, client_token)
    return instances[0] if instances else ""


def _recover_security_group(context: AwsContext, state: dict) -> str:
    if state.get("security_group_id") or not state.get("security_group_name"):
        return state.get("security_group_id", "")
    payload = _aws_json(
        context,
        "ec2",
        "describe-security-groups",
        "--filters",
        f"Name=group-name,Values={state['security_group_name']}",
        f"Name=vpc-id,Values={state['vpc_id']}",
        f"Name=tag:senpai:run,Values={state['tag']}",
    )
    groups = payload.get("SecurityGroups", [])
    return groups[0]["GroupId"] if groups else ""


def _stop_remote_roles(run_dir: Path, state: dict) -> None:
    """Flush supervisors and remove remote containers plus credential state."""
    if (
        not state.get("roles_starting")
        or not state.get("public_ip")
        or not (run_dir / "id_ed25519").is_file()
    ):
        return
    tag = state["tag"]
    _validate_tag(tag)
    try:
        result = _ssh(
            run_dir,
            state,
            shlex.join(
                [
                    "python3",
                    f"{REMOTE_SOURCE}/k8s/docker.py",
                    "terminate",
                    tag,
                    "--run-root",
                    REMOTE_RUN_ROOT,
                ]
            ),
            timeout=150,
            check=False,
        )
    except RuntimeError:
        result = None
    if result is None or result.returncode:
        print(
            "AWS cleanup could not gracefully stop every Senpai container; "
            "terminating the host.",
            flush=True,
        )


def _cleanup(
    run_dir: Path,
    state: dict,
    context: AwsContext | None = None,
) -> list[str]:
    context = context or _state_context(state)
    errors: list[str] = []
    if state.get("account_id"):
        try:
            _check_account(context, state["account_id"])
        except (AwsCommandError, RuntimeError) as error:
            errors.append(str(error))
            state["phase"] = "cleanup-failed"
            state["cleanup_errors"] = errors
            _save_state(run_dir, state)
            return errors

    instance_ids = [state["instance_id"]] if state.get("instance_id") else []
    client_tokens = state.get("client_tokens") or (
        [state["client_token"]] if state.get("client_token") else []
    )
    failed_client_tokens = set(state.get("failed_client_tokens", []))
    client_tokens = [
        token for token in client_tokens if token not in failed_client_tokens
    ]
    if not instance_ids and state.get("instance_launch_started"):
        try:
            for client_token in client_tokens:
                for attempt in range(12):
                    recovered = _instances_for_client_token(context, client_token)
                    if recovered:
                        instance_ids.extend(recovered)
                        break
                    if attempt < 11:
                        time.sleep(5)
            instance_ids = list(dict.fromkeys(instance_ids))
            if instance_ids:
                state["instance_id"] = instance_ids[0]
                state["instance_ids"] = instance_ids
                _save_state(run_dir, state)
        except AwsCommandError as error:
            errors.append(str(error))
    if instance_ids:
        _stop_remote_roles(run_dir, state)
        try:
            _aws_raw(
                context,
                "ec2",
                "terminate-instances",
                "--instance-ids",
                *instance_ids,
            )
            _wait_for_instances_terminated(context, instance_ids)
        except AwsCommandError as error:
            if not _missing_resource(error):
                errors.append(str(error))

    security_group_id = state.get("security_group_id", "")
    if (
        not security_group_id
        and state.get("security_group_name")
        and state.get("security_group_create_started")
    ):
        try:
            for attempt in range(3):
                security_group_id = _recover_security_group(context, state)
                if security_group_id:
                    state["security_group_id"] = security_group_id
                    _save_state(run_dir, state)
                    break
                if attempt < 2:
                    time.sleep(2)
        except AwsCommandError as error:
            errors.append(str(error))
        if not security_group_id and not errors:
            errors.append(
                "AWS security-group creation started, but its outcome is still "
                "unknown; retry cleanup after AWS resource discovery catches up."
            )
    if security_group_id and not errors:
        for attempt in range(6):
            try:
                _aws_raw(
                    context,
                    "ec2",
                    "delete-security-group",
                    "--group-id",
                    security_group_id,
                )
                break
            except AwsCommandError as error:
                if _missing_resource(error):
                    break
                if attempt == 5:
                    errors.append(str(error))
                else:
                    time.sleep(5)

    key_name = state.get("key_name")
    if key_name and state.get("key_owned") is not False:
        try:
            _aws_raw(context, "ec2", "delete-key-pair", "--key-name", key_name)
        except AwsCommandError as error:
            if not _missing_resource(error):
                errors.append(str(error))

    if errors:
        state["phase"] = "cleanup-failed"
        state["cleanup_errors"] = errors
        _save_state(run_dir, state)
    else:
        shutil.rmtree(run_dir)
    return errors


def _lifecycle_command(
    action: str,
    tag: str,
    state_root: str,
    profile: str,
) -> str:
    command = ["python3", "k8s/aws.py", action, tag]
    if Path(state_root).expanduser().resolve() != Path(
        DEFAULT_STATE_ROOT
    ).expanduser().resolve():
        command.extend(["--state-root", state_root])
    if profile:
        command.extend(["--profile", profile])
    return shlex.join(command)


def launch_aws(
    args,
    role_specs: list[RoleSpec],
    plan: AwsLaunchPlan | None = None,
    *,
    before_start: Callable[[], None] | None = None,
) -> None:
    """Provision an AWS host and launch the existing Docker role specs."""
    if args.dry_run:
        requirements = _validate_aws_inputs(
            args,
            role_specs,
            require_writable_state=False,
        )
        instance_type = args.aws_instance_type or _automatic_instance_type(requirements)
        roles = ", ".join(spec.key for spec in role_specs)
        print(f"AWS Senpai dry run: tag={args.tag}")
        print(
            f"Host: {instance_type} in {args.aws_region or '<AWS CLI region>'}; "
            f"encrypted at least {args.aws_volume_gib} GiB root volume; "
            f"SSH restricted to {args.aws_ssh_cidr or '<current public IP>/32'}"
        )
        print(f"Roles ({roles}) reuse the Docker launcher; credentials redacted.")
        print(f"Safety TTL: {args.aws_ttl_hours:g} hours")
        return

    plan = plan or preflight_aws(args, role_specs)
    run_dir = _state_dir(args.aws_state_root, args.tag)
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    state = {
        "account_id": plan.account_id,
        "ami_id": plan.ami_id,
        "availability_zone": plan.availability_zone,
        "created_at": int(time.time()),
        "instance_type": plan.instance_type,
        "phase": "creating",
        "profile": plan.context.profile,
        "region": plan.context.region,
        "ssh_cidr": plan.ssh_cidr,
        "subnet_id": plan.subnet_id,
        "tag": args.tag,
        "vpc_id": plan.vpc_id,
        "roles": [spec.key for spec in role_specs],
        "volume_gib": plan.volume_gib,
        "data_files": getattr(plan, "data_files", 0),
        "data_bytes": getattr(plan, "data_bytes", 0),
    }
    _save_state(run_dir, state)
    try:
        _provision(args, plan, run_dir, state)
        print(
            f"AWS host {state['instance_id']} is running; "
            "waiting for SSH, cloud-init, and the NVIDIA driver.",
            flush=True,
        )
        _prepare_host(args, run_dir, state)
        print("AWS host is ready; validating Senpai images and CUDA.", flush=True)
        state["phase"] = "validating-images"
        _save_state(run_dir, state)
        _preflight_roles(
            args,
            role_specs,
            run_dir,
            state,
            include_data=False,
        )
        state["phase"] = "uploading-data"
        _save_state(run_dir, state)
        data_bytes = getattr(plan, "data_bytes", 0)
        if args.data_dir and not data_bytes:
            _, data_bytes = _data_summary(Path(args.data_dir).expanduser().resolve())
        _upload_data(args, run_dir, state, data_bytes)
        if args.data_dir:
            state["phase"] = "validating-data"
            _save_state(run_dir, state)
            _preflight_roles(
                args,
                role_specs,
                run_dir,
                state,
                include_data=True,
            )
        if before_start is not None:
            state["phase"] = "preparing-github"
            _save_state(run_dir, state)
            before_start()
        state["phase"] = "starting-roles"
        state["roles_starting"] = True
        _save_state(run_dir, state)
        _start_roles(args, role_specs, run_dir, state)
    except BaseException as error:
        if detail := str(error).strip():
            print(f"AWS launch failed:\n{detail}", flush=True)
        print("Terminating the temporary host.", flush=True)
        cleanup_errors = _cleanup(run_dir, state)
        if cleanup_errors and isinstance(error, Exception):
            terminate = _lifecycle_command(
                "terminate",
                args.tag,
                args.aws_state_root,
                state.get("profile", ""),
            )
            raise RuntimeError(
                f"{error}\nAWS cleanup was incomplete; retry `{terminate}`.\n"
                + "\n".join(cleanup_errors)
            ) from error
        if cleanup_errors:
            print(
                "AWS cleanup was incomplete; lifecycle state was preserved:\n"
                + "\n".join(cleanup_errors),
                file=sys.stderr,
                flush=True,
            )
        raise

    state["phase"] = "running"
    _save_state(run_dir, state)
    print(
        f"\nAWS Senpai is running on {state['instance_id']} "
        f"({state['public_ip']}, {plan.instance_type})."
    )
    print(
        "Status:    " + _lifecycle_command(
            "status",
            args.tag,
            args.aws_state_root,
            state.get("profile", ""),
        )
    )
    print(
        "Logs:      " + _lifecycle_command(
            "logs",
            args.tag,
            args.aws_state_root,
            state.get("profile", ""),
        )
    )
    print(
        "Terminate: " + _lifecycle_command(
            "terminate",
            args.tag,
            args.aws_state_root,
            state.get("profile", ""),
        )
    )
    print(
        f"Safety: the instance will self-terminate after "
        f"{args.aws_ttl_hours:g} hours."
    )


def status_aws(
    tag: str,
    state_root: str = "~/.senpai/aws",
    *,
    profile: str = "",
) -> None:
    """Print EC2 and Docker state for one recorded run."""
    run_dir, state = _load_state(tag, state_root)
    context = _state_context(state, profile=profile)
    _check_account(context, state["account_id"])
    instance_id = _recorded_instance_id(state)
    instance = _instance_details(context, instance_id)
    instance_phase = instance["State"]["Name"]
    print(
        f"{tag}: launcher={state.get('phase', 'unknown')} "
        f"instance={instance_id} state={instance_phase} "
        f"type={state['instance_type']} region={state['region']}"
    )
    if (
        instance_phase == "running"
        and state.get("roles_starting")
        and state.get("public_ip")
        and (run_dir / "id_ed25519").is_file()
    ):
        result = _ssh(
            run_dir,
            state,
            shlex.join(
                [
                    "python3",
                    f"{REMOTE_SOURCE}/k8s/docker.py",
                    "status",
                    tag,
                    "--run-root",
                    REMOTE_RUN_ROOT,
                ]
            ),
            timeout=20,
        )
        output = result.stdout.decode(errors="replace").strip()
        if output:
            print(output)


def logs_aws(
    tag: str,
    state_root: str = "~/.senpai/aws",
    *,
    profile: str = "",
    role_key: str = "",
    tail: int = 200,
) -> None:
    """Print bounded logs from one role on a recorded AWS host."""
    if tail < 1:
        raise ValueError("AWS log tail must be at least 1")
    run_dir, state = _load_state(tag, state_root)
    context = _state_context(state, profile=profile)
    _check_account(context, state["account_id"])
    instance_id = _recorded_instance_id(state)
    instance = _instance_details(context, instance_id)
    if instance["State"]["Name"] != "running":
        raise RuntimeError(f"AWS instance {instance_id} is not running")
    listed = _ssh(
        run_dir,
        state,
        (
            "docker ps --all --format '{{.Names}}' --filter "
            + shlex.quote(f"label=com.wandb.senpai.run={tag}")
        ),
        timeout=20,
    )
    names = listed.stdout.decode().split()
    if not names:
        raise RuntimeError(f"No Senpai containers found for AWS run {tag!r}")
    expected = f"senpai-{tag}-{role_key}" if role_key else ""
    name = expected if expected in names else names[0] if not role_key else ""
    if not name:
        choices = ", ".join(value.removeprefix(f"senpai-{tag}-") for value in names)
        raise ValueError(f"Unknown AWS role {role_key!r}; choose one of: {choices}")
    result = _ssh(
        run_dir,
        state,
        f"docker logs --tail {tail} {shlex.quote(name)}",
        timeout=30,
    )
    print(result.stdout.decode(errors="replace"), end="")
    stderr = result.stderr.decode(errors="replace")
    if stderr:
        print(stderr, end="", file=sys.stderr)


def terminate_aws(
    tag: str,
    state_root: str = "~/.senpai/aws",
    *,
    profile: str = "",
) -> None:
    """Terminate only the resources recorded for one run."""
    run_dir, state = _load_state(tag, state_root)
    context = _state_context(state, profile=profile)
    _check_account(context, state["account_id"])
    errors = _cleanup(run_dir, state, context)
    if errors:
        raise RuntimeError(
            "AWS cleanup was incomplete; state was preserved for retry:\n"
            + "\n".join(errors)
        )
    print(f"Terminated AWS Senpai run {tag!r} and removed its ephemeral SSH key.")
