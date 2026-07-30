# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Provision one AWS GPU host and reuse Senpai's Docker launcher on it."""

from __future__ import annotations

import ipaddress
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .specs import (
    RoleSpec,
    validate_identifier,
    validate_role_specs,
    validate_writable_parent,
)

ROOT = Path(__file__).resolve().parents[2]
DLAMI_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)
SOURCE_BRANCH = "senpai-aws-source"
REMOTE_SOURCE = "/home/ubuntu/senpai-source"
REMOTE_RUN_ROOT = "/home/ubuntu/.senpai/runs"
DEFAULT_STATE_ROOT = "~/.senpai/aws"
AUTO_INSTANCE_TYPES = ((1, "g4dn.xlarge"), (4, "g4dn.12xlarge"), (8, "g5.48xlarge"))


@dataclass(frozen=True)
class AwsContext:
    region: str
    profile: str = ""


@dataclass(frozen=True)
class AwsLaunchPlan:
    """Read-only resolution of an AWS launch."""

    context: AwsContext
    account_id: str
    instance_type: str
    ami_id: str
    root_device: str
    volume_gib: int
    subnet_id: str
    vpc_id: str
    availability_zone: str
    ssh_cidr: str


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
        result = subprocess.run(command, capture_output=True, text=True, env=env)
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
) -> int:
    """Validate the local launch contract before invoking AWS."""
    validate_role_specs("AWS", args.tag, role_specs)
    if args.docker_student_gpu_ids:
        raise ValueError(
            "AWS assigns its dedicated GPUs automatically; remove "
            "--docker_student_gpu_ids"
        )
    if args.docker_data_dir:
        raise ValueError(
            "A fresh AWS host has no local dataset directory; remove "
            "--docker_data_dir and let the target package download its data"
        )
    if args.gpus_per_student < 0:
        raise ValueError("--gpus_per_student must be non-negative")

    if (
        any(spec.role == "student" for spec in role_specs)
        and args.gpus_per_student < 1
    ):
        raise ValueError("AWS students require --gpus_per_student at least 1")

    run_dir = _state_dir(args.aws_state_root, args.tag)
    state_root = run_dir.parent
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
    if args.aws_ttl_hours <= 0:
        raise ValueError("--aws_ttl_hours must be greater than 0")
    if args.aws_ready_timeout_s < 1:
        raise ValueError("--aws_ready_timeout_s must be at least 1")

    required_gpus = _required_gpus(args, role_specs)
    if not args.aws_instance_type:
        _automatic_instance_type(required_gpus)
    return required_gpus


def _automatic_instance_type(required_gpus: int) -> str:
    for capacity, instance_type in AUTO_INSTANCE_TYPES:
        if required_gpus <= capacity:
            return instance_type
    raise ValueError(
        f"AWS single-host launches support at most 8 GPUs, but this launch "
        f"requests {required_gpus}. Reduce --n_students or --gpus_per_student."
    )


def _instance_type_details(context: AwsContext, instance_type: str) -> tuple[int, list[str]]:
    payload = _aws_json(
        context,
        "ec2",
        "describe-instance-types",
        "--instance-types",
        instance_type,
    )
    details = payload["InstanceTypes"][0]
    gpus = sum(gpu["Count"] for gpu in details.get("GpuInfo", {}).get("Gpus", []))
    architectures = details["ProcessorInfo"]["SupportedArchitectures"]
    return gpus, architectures


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
    root_device = image["RootDeviceName"]
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


def _select_subnet(
    context: AwsContext,
    instance_type: str,
    configured_subnet: str,
) -> dict:
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
    candidate_vpcs = sorted({subnet["VpcId"] for subnet in candidates})
    if not configured_subnet and len(candidate_vpcs) > 1:
        default_subnets = [
            subnet for subnet in candidates if subnet.get("DefaultForAz", False)
        ]
        if len({subnet["VpcId"] for subnet in default_subnets}) == 1:
            candidates = default_subnets
        else:
            choices = []
            for subnet in sorted(
                candidates,
                key=lambda value: (
                    value["VpcId"],
                    value["AvailabilityZone"],
                    value["SubnetId"],
                ),
            )[:6]:
                name = next(
                    (
                        tag["Value"]
                        for tag in subnet.get("Tags", [])
                        if tag["Key"] == "Name"
                    ),
                    subnet["VpcId"],
                )
                choices.append(f"{subnet['SubnetId']} ({name})")
            raise RuntimeError(
                "Public subnets in multiple VPCs offer "
                f"{instance_type}. Pass --aws_subnet_id with one of: "
                + ", ".join(choices)
            )
    return min(
        candidates,
        key=lambda subnet: (
            not subnet.get("DefaultForAz", False),
            subnet["AvailabilityZone"],
            subnet["SubnetId"],
        ),
    )


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
    required_gpus = _validate_aws_inputs(args, role_specs)
    if not shutil.which("ssh"):
        raise RuntimeError("OpenSSH is required to launch on AWS")
    _source_names()

    profile = (
        args.aws_profile
        or os.environ.get("AWS_PROFILE", "")
        or os.environ.get("AWS_DEFAULT_PROFILE", "")
    )
    context = AwsContext(_resolve_region(args.aws_region, profile), profile)
    identity = _check_account(context)
    instance_type = args.aws_instance_type or _automatic_instance_type(required_gpus)
    gpu_count, architectures = _instance_type_details(context, instance_type)
    if "x86_64" not in architectures:
        raise RuntimeError(f"AWS instance type {instance_type} must support x86_64")
    if gpu_count < 1:
        raise RuntimeError(f"AWS instance {instance_type} does not have an NVIDIA GPU")
    if gpu_count < required_gpus:
        raise RuntimeError(
            f"AWS instance {instance_type} has {gpu_count} GPUs, but this launch "
            f"needs {required_gpus}"
        )
    ami_id, root_device, minimum_volume_gib = _resolve_ami(
        context,
        args.aws_ami_id,
    )
    subnet = _select_subnet(context, instance_type, args.aws_subnet_id)
    plan = AwsLaunchPlan(
        context=context,
        account_id=identity["Account"],
        instance_type=instance_type,
        ami_id=ami_id,
        root_device=root_device,
        volume_gib=max(args.aws_volume_gib, minimum_volume_gib),
        subnet_id=subnet["SubnetId"],
        vpc_id=subnet["VpcId"],
        availability_zone=subnet["AvailabilityZone"],
        ssh_cidr=_ssh_cidr(args.aws_ssh_cidr),
    )
    print(
        "AWS preflight OK — "
        f"account={plan.account_id}, region={context.region}, "
        f"instance={instance_type} ({gpu_count} GPUs), "
        f"subnet={plan.subnet_id} ({plan.availability_zone}), ami={ami_id}"
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


def _ssh_base(run_dir: Path, state: dict) -> list[str]:
    return [
        "ssh",
        "-i",
        str(run_dir / "id_ed25519"),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={run_dir / 'known_hosts'}",
        f"ubuntu@{state['public_ip']}",
    ]


def _ssh(
    run_dir: Path,
    state: dict,
    command: str,
    *,
    input_bytes: bytes | None = None,
    input_file=None,
    timeout: float | None = None,
    check: bool = True,
    redactions: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            [*_ssh_base(run_dir, state), command],
            input=input_bytes,
            stdin=input_file,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"AWS host command timed out after {timeout:g}s"
        ) from error
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


def _source_names() -> list[str]:
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    untracked_names = [
        name.decode(errors="surrogateescape")
        for name in untracked
        if name
    ]
    if untracked_names:
        preview = ", ".join(untracked_names[:5])
        suffix = " ..." if len(untracked_names) > 5 else ""
        raise RuntimeError(
            "AWS source upload refuses non-ignored untracked files. Stage intended "
            f"source files with `git add` or ignore local data: {preview}{suffix}"
        )

    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "-z",
        ],
        capture_output=True,
        check=True,
    )
    names = [
        name.decode(errors="surrogateescape")
        for name in result.stdout.split(b"\0")
        if name
    ]
    if not names:
        raise RuntimeError(f"No source files found in {ROOT}")
    return names


def _source_archive(path: Path) -> None:
    names = _source_names()
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe git source path: {name!r}")
            source = ROOT / relative
            if source.exists() or source.is_symlink():
                archive.add(source, arcname=name, recursive=False)


def _remote_payload(args, role_specs: list[RoleSpec]) -> bytes:
    remote_args = dict(vars(args))
    remote_args.update(
        {
            "backend": "docker",
            "docker_run_root": REMOTE_RUN_ROOT,
            "dry_run": False,
            "repo_url": REMOTE_SOURCE,
            "repo_branch": SOURCE_BRANCH,
        }
    )
    roles = []
    for spec in role_specs:
        values = asdict(spec)
        values["env"]["REPO_URL"] = REMOTE_SOURCE
        values["env"]["REPO_BRANCH"] = SOURCE_BRANCH
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


def _provision(args, plan: AwsLaunchPlan, run_dir: Path, state: dict) -> None:
    client_token = uuid.uuid4().hex
    suffix = client_token[:8]
    key_name = f"senpai-{args.tag}-{suffix}"
    state.update(
        {
            "client_token": client_token,
            "instance_launch_started": False,
            "key_name": key_name,
        }
    )
    _save_state(run_dir, state)
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
    _write_private_key(run_dir / "id_ed25519", key_material)

    security_group_name = f"senpai-{args.tag}-{suffix}"
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
    state["instance_launch_started"] = True
    _save_state(run_dir, state)
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
        plan.subnet_id,
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
    instance = _instance_details(plan.context, state["instance_id"])
    state["public_ip"] = instance["PublicIpAddress"]
    state["phase"] = "booting"
    _save_state(run_dir, state)


def _prepare_host(args, run_dir: Path, state: dict) -> None:
    _wait_for_ssh(run_dir, state, args.aws_ready_timeout_s)
    _ssh(
        run_dir,
        state,
        "sudo cloud-init status --wait",
        timeout=args.aws_ready_timeout_s,
    )
    _wait_for_gpu(run_dir, state, args.aws_ready_timeout_s)

    archive_path = run_dir / "source.tar.gz"
    _source_archive(archive_path)
    try:
        with archive_path.open("rb") as archive:
            _ssh(
                run_dir,
                state,
                (
                    f"umask 077; rm -rf {REMOTE_SOURCE}; mkdir {REMOTE_SOURCE}; "
                    f"tar -xzf - -C {REMOTE_SOURCE}"
                ),
                input_file=archive,
                timeout=args.aws_ready_timeout_s,
            )
    finally:
        archive_path.unlink(missing_ok=True)
    _ssh(
        run_dir,
        state,
        (
            f"cd {REMOTE_SOURCE}; git init -q -b {SOURCE_BRANCH}; git add -A; "
            "git -c user.name=senpai -c user.email=senpai@local "
            'commit -q -m "AWS launch source"'
        ),
        timeout=args.aws_ready_timeout_s,
    )


def _start_roles(
    args,
    role_specs: list[RoleSpec],
    run_dir: Path,
    state: dict,
) -> None:
    command = (
        "umask 077; "
        "payload=$(mktemp /home/ubuntu/senpai-launch.XXXXXX.json); "
        "trap 'rm -f \"$payload\"' EXIT; "
        'cat > "$payload"; '
        f"cd {REMOTE_SOURCE}; "
        'python3 -m senpai.launch.remote "$payload"'
    )
    result = _ssh(
        run_dir,
        state,
        command,
        input_bytes=_remote_payload(args, role_specs),
        timeout=args.aws_ready_timeout_s + args.docker_ready_timeout_s,
        redactions=tuple(
            secret
            for spec in role_specs
            for secret in spec.secrets.values()
            if secret
        ),
    )
    output = result.stdout.decode(errors="replace").strip()
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


def _instance_for_client_token(context: AwsContext, client_token: str) -> str:
    payload = _aws_json(
        context,
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=client-token,Values={client_token}",
    )
    return next(
        (
            instance["InstanceId"]
            for reservation in payload.get("Reservations", [])
            for instance in reservation.get("Instances", [])
        ),
        "",
    )


def _cleanup(
    run_dir: Path,
    state: dict,
    context: AwsContext | None = None,
) -> list[str]:
    context = context or _state_context(state)
    errors: list[str] = []

    instance_id = state.get("instance_id")
    if (
        not instance_id
        and state.get("instance_launch_started")
        and state.get("client_token")
    ):
        try:
            for attempt in range(3):
                instance_id = _instance_for_client_token(
                    context,
                    state["client_token"],
                )
                if instance_id:
                    state["instance_id"] = instance_id
                    _save_state(run_dir, state)
                    break
                if attempt < 2:
                    time.sleep(2)
        except AwsCommandError as error:
            errors.append(str(error))
    if instance_id:
        try:
            _aws_raw(
                context,
                "ec2",
                "terminate-instances",
                "--instance-ids",
                instance_id,
            )
            _aws_raw(
                context,
                "ec2",
                "wait",
                "instance-terminated",
                "--instance-ids",
                instance_id,
            )
        except AwsCommandError as error:
            if not _missing_resource(error):
                errors.append(str(error))

    security_group_id = state.get("security_group_id")
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
    if key_name:
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
    command = ["python", "k8s/aws.py", action, tag]
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
) -> None:
    """Provision an AWS host and launch the existing Docker role specs."""
    if args.dry_run:
        required_gpus = _validate_aws_inputs(
            args,
            role_specs,
            require_writable_state=False,
        )
        instance_type = args.aws_instance_type or _automatic_instance_type(required_gpus)
        roles = ", ".join(spec.key for spec in role_specs)
        print(f"AWS Senpai dry run: tag={args.tag}")
        print(
            f"Host: {instance_type} in {args.aws_region or '<AWS CLI region>'}; "
            f"encrypted {args.aws_volume_gib} GiB root volume; "
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
        print(
            "AWS host is ready; checking the Senpai image and starting roles. "
            "The first image pull can take several minutes.",
            flush=True,
        )
        _start_roles(args, role_specs, run_dir, state)
    except BaseException as error:
        print("AWS launch failed; terminating the temporary host.", flush=True)
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
    instance = _instance_details(context, state["instance_id"])
    phase = instance["State"]["Name"]
    print(
        f"{tag}: instance={state['instance_id']} state={phase} "
        f"type={state['instance_type']} region={state['region']}"
    )
    if phase == "running" and state.get("public_ip") and (run_dir / "id_ed25519").is_file():
        result = _ssh(
            run_dir,
            state,
            (
                "docker ps --all --format "
                + shlex.quote("{{.Names}}\t{{.Status}}")
                + " --filter "
                + shlex.quote(f"label=com.wandb.senpai.run={tag}")
            ),
            timeout=20,
        )
        output = result.stdout.decode(errors="replace").strip()
        if output:
            print(output)


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
