# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Run one native Senpai student on each existing EC2 Mac Dedicated Host."""

from __future__ import annotations

import json
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from senpai_agent.model_compatibility import (
    WANDB_GLM_52_MODEL,
    WANDB_GLM_52_TOKENIZER,
)

from .aws_backend import (
    AwsCommandError,
    AwsContext,
    _authorize_ssh_host,
    _aws_json,
    _aws_raw,
    _check_account,
    _check_source_revision,
    _missing_resource,
    _resolve_region,
    _save_state,
    _ssh_cidr,
    _ssh_host_key_console_script,
    _state_dir,
    _write_private_key,
)
from .specs import RoleSpec, validate_identifier, validate_role_specs

ROOT = Path(__file__).resolve().parents[2]
REMOTE_HOME = "/Users/ec2-user"
REMOTE_SOURCE = f"{REMOTE_HOME}/senpai-source"
REMOTE_VENV = f"{REMOTE_HOME}/.senpai/venv"
REMOTE_BROWSER_ROOT = f"{REMOTE_HOME}/.senpai/ms-playwright"
REMOTE_HF_HOME = f"{REMOTE_HOME}/.senpai/huggingface"
REMOTE_RUN_ROOT = f"{REMOTE_HOME}/.senpai/native"
REMOTE_SETUP_SCRIPT = "/tmp/senpai-setup.sh"
REMOTE_METAL_TOOLCHAIN_ARCHIVE = "/tmp/senpai-MetalToolchain.zip"
DEFAULT_MLXFAST_BUNDLE = "~/.local/share/mlxfast/mlxfast.js"
MAC_ROOT_IOPS = 10_000
MAC_ROOT_THROUGHPUT = 400
DEFAULT_AMI_PARAMETER = (
    "/aws/service/ec2-macos/tahoe/arm64_mac/latest/image_id"
)
MAC_ARCHITECTURE = "arm64_mac"
AWS_MAC_STATE_BACKEND = "aws-mac"
AWS_MAC_STATE_VERSION = 1
HOST_ID = re.compile(r"h-[0-9a-f]+")
SUBNET_ID = re.compile(r"subnet-[0-9a-f]+")
SECURITY_GROUP_ID = re.compile(r"sg-[0-9a-f]+")


@dataclass(frozen=True)
class AwsMacHost:
    host_id: str
    availability_zone: str
    subnet_id: str
    student: str


@dataclass(frozen=True)
class AwsMacPlan:
    context: AwsContext
    account_id: str
    ami_id: str
    root_device: str
    volume_gib: int
    security_group_id: str
    ssh_cidr: str
    hosts: tuple[AwsMacHost, ...]


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _subnet_map(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _csv(value):
        availability_zone, separator, subnet_id = item.partition("=")
        if not separator or not availability_zone or not SUBNET_ID.fullmatch(subnet_id):
            raise ValueError(
                "--aws_mac_subnet_ids must use AZ=subnet-id entries separated "
                "by commas"
            )
        if availability_zone in result:
            raise ValueError(
                f"--aws_mac_subnet_ids repeats {availability_zone!r}"
            )
        result[availability_zone] = subnet_id
    return result


def _student_specs(role_specs: list[RoleSpec]) -> list[RoleSpec]:
    return [spec for spec in role_specs if spec.role == "student"]


def _advisor_spec(role_specs: list[RoleSpec]) -> RoleSpec | None:
    return next((spec for spec in role_specs if spec.role == "advisor"), None)


def _metal_toolchain_archive(args) -> Path:
    configured = getattr(args, "aws_mac_metal_toolchain_archive", "")
    if not configured:
        raise ValueError("--aws_mac_metal_toolchain_archive is required")
    archive = Path(configured).expanduser().resolve()
    if not archive.is_file():
        raise RuntimeError(
            f"Prepared Metal toolchain archive was not found at {archive}"
        )
    try:
        with ZipFile(archive) as contents:
            bundle_roots = {
                parts[0]
                for name in contents.namelist()
                if (parts := PurePosixPath(name).parts)
                and parts[0].endswith(".exportedBundle")
            }
    except BadZipFile as error:
        raise RuntimeError(
            f"Prepared Metal toolchain archive is not a zip file: {archive}"
        ) from error
    if len(bundle_roots) != 1:
        raise RuntimeError(
            "Prepared Metal toolchain archive must contain exactly one "
            "top-level .exportedBundle"
        )
    return archive


def distribute_roles(
    role_specs: list[RoleSpec], hosts: tuple[AwsMacHost, ...]
) -> tuple[tuple[AwsMacHost, tuple[RoleSpec, ...]], ...]:
    """Map students one-to-one to hosts and co-locate the advisor on host zero."""
    students = _student_specs(role_specs)
    if len(students) != len(hosts):
        raise ValueError(
            f"AWS Mac needs one host per student: {len(students)} students, "
            f"{len(hosts)} hosts"
        )
    advisor = _advisor_spec(role_specs)
    groups = []
    for index, (host, student) in enumerate(zip(hosts, students, strict=True)):
        assigned = [student]
        if index == 0 and advisor is not None:
            assigned.append(advisor)
        groups.append((host, tuple(assigned)))
    return tuple(groups)


def _resolve_ami(context: AwsContext, configured: str) -> tuple[str, str, int]:
    ami_id = configured
    if not ami_id:
        payload = _aws_json(
            context,
            "ssm",
            "get-parameter",
            "--name",
            DEFAULT_AMI_PARAMETER,
        )
        ami_id = payload["Parameter"]["Value"]
    images = _aws_json(
        context,
        "ec2",
        "describe-images",
        "--image-ids",
        ami_id,
    ).get("Images", [])
    if len(images) != 1:
        raise RuntimeError(f"AWS Mac AMI {ami_id} was not found")
    image = images[0]
    if image.get("State") != "available" or image.get("Architecture") != MAC_ARCHITECTURE:
        raise RuntimeError(
            f"AWS Mac AMI {ami_id} must be an available {MAC_ARCHITECTURE} image"
        )
    root_device = image["RootDeviceName"]
    mapping = next(
        (
            item
            for item in image.get("BlockDeviceMappings", [])
            if item.get("DeviceName") == root_device and "Ebs" in item
        ),
        None,
    )
    if mapping is None:
        raise RuntimeError(f"AWS Mac AMI {ami_id} has no EBS root mapping")
    return ami_id, root_device, int(mapping["Ebs"]["VolumeSize"])


def _resolve_hosts(
    context: AwsContext,
    host_ids: tuple[str, ...],
    subnet_ids: dict[str, str],
    students: list[RoleSpec],
    instance_type: str,
) -> tuple[AwsMacHost, ...]:
    if not host_ids:
        raise ValueError("--aws_mac_host_ids is required")
    if any(not HOST_ID.fullmatch(host_id) for host_id in host_ids):
        raise ValueError("--aws_mac_host_ids contains an invalid Dedicated Host ID")
    if len(host_ids) != len(set(host_ids)):
        raise ValueError("--aws_mac_host_ids must be unique")
    if len(host_ids) != len(students):
        raise ValueError(
            f"AWS Mac needs one Dedicated Host per student: {len(students)} "
            f"students, {len(host_ids)} hosts"
        )
    payload = _aws_json(
        context,
        "ec2",
        "describe-hosts",
        "--host-ids",
        *host_ids,
    )
    by_id = {host["HostId"]: host for host in payload.get("Hosts", [])}
    missing = [host_id for host_id in host_ids if host_id not in by_id]
    if missing:
        raise RuntimeError("AWS Dedicated Hosts were not found: " + ", ".join(missing))

    hosts = []
    for host_id, student in zip(host_ids, students, strict=True):
        host = by_id[host_id]
        state = host.get("State")
        if state != "available":
            raise RuntimeError(f"AWS Dedicated Host {host_id} is {state}, not available")
        actual_type = host.get("HostProperties", {}).get("InstanceType")
        if actual_type != instance_type:
            raise RuntimeError(
                f"AWS Dedicated Host {host_id} is {actual_type}, expected "
                f"{instance_type}"
            )
        if host.get("Instances"):
            raise RuntimeError(f"AWS Dedicated Host {host_id} already has an instance")
        capacity = host.get("AvailableCapacity", {}).get(
            "AvailableInstanceCapacity", []
        )
        available = sum(
            int(item.get("AvailableCapacity", 0))
            for item in capacity
            if item.get("InstanceType") == instance_type
        )
        if available < 1:
            raise RuntimeError(f"AWS Dedicated Host {host_id} has no free capacity")
        availability_zone = host["AvailabilityZone"]
        subnet_id = subnet_ids.get(availability_zone, "")
        if not subnet_id:
            raise ValueError(
                f"--aws_mac_subnet_ids has no subnet for {availability_zone}"
            )
        hosts.append(
            AwsMacHost(
                host_id=host_id,
                availability_zone=availability_zone,
                subnet_id=subnet_id,
                student=student.name,
            )
        )
    return tuple(hosts)


def _validate_network(
    context: AwsContext,
    hosts: tuple[AwsMacHost, ...],
    security_group_id: str,
) -> None:
    if not SECURITY_GROUP_ID.fullmatch(security_group_id):
        raise ValueError("--aws_mac_security_group_id is required")
    requested = sorted({host.subnet_id for host in hosts})
    subnets = _aws_json(
        context,
        "ec2",
        "describe-subnets",
        "--subnet-ids",
        *requested,
    ).get("Subnets", [])
    by_id = {subnet["SubnetId"]: subnet for subnet in subnets}
    for host in hosts:
        subnet = by_id.get(host.subnet_id)
        if subnet is None:
            raise RuntimeError(f"AWS subnet {host.subnet_id} was not found")
        if subnet.get("State") != "available":
            raise RuntimeError(f"AWS subnet {host.subnet_id} is not available")
        if subnet.get("AvailabilityZone") != host.availability_zone:
            raise RuntimeError(
                f"AWS subnet {host.subnet_id} is in "
                f"{subnet.get('AvailabilityZone')}, not {host.availability_zone}"
            )
        if not subnet.get("MapPublicIpOnLaunch"):
            raise RuntimeError(
                f"AWS subnet {host.subnet_id} does not assign public IPv4 addresses"
            )
    vpcs = {subnet["VpcId"] for subnet in subnets}
    if len(vpcs) != 1:
        raise RuntimeError("AWS Mac subnets must belong to one VPC")
    groups = _aws_json(
        context,
        "ec2",
        "describe-security-groups",
        "--group-ids",
        security_group_id,
    ).get("SecurityGroups", [])
    if len(groups) != 1 or groups[0]["VpcId"] not in vpcs:
        raise RuntimeError(
            f"AWS security group {security_group_id} is not in the Mac subnet VPC"
        )


def preflight_aws_mac(args, role_specs: list[RoleSpec]) -> AwsMacPlan:
    """Resolve the exact existing Mac fleet without changing AWS or GitHub."""
    validate_role_specs("AWS Mac", args.tag, role_specs)
    validate_identifier("AWS Mac tag", args.tag)
    students = _student_specs(role_specs)
    if not students:
        raise ValueError("AWS Mac launches require at least one student")
    if args.gpus_per_student != 1:
        raise ValueError("AWS Mac requires exactly one physical Mac per student")
    if args.data_dir:
        raise ValueError("AWS Mac does not upload --data_dir; target setup owns data")
    if args.start_gate_path:
        raise ValueError("AWS Mac owns its distributed launch gates")
    if args.aws_volume_gib < 200:
        raise ValueError("AWS Mac root volumes must be at least 200 GiB")
    if args.aws_ttl_hours < 0:
        raise ValueError("--aws_ttl_hours must be non-negative for AWS Mac")

    run_dir = _state_dir(args.aws_state_root, args.tag)
    if run_dir.exists() or run_dir.is_symlink():
        raise RuntimeError(
            f"AWS Mac state already exists at {run_dir}; choose a new tag or "
            "terminate the recorded run"
        )
    _check_source_revision(args.senpai_repo_revision)
    if not shutil.which("ssh") or not shutil.which("ditto"):
        raise RuntimeError("AWS Mac launch requires OpenSSH and ditto")
    archive = (
        Path(args.aws_mac_xcode_archive).expanduser().resolve()
        if args.aws_mac_xcode_archive
        else None
    )
    if archive is not None:
        if not archive.is_file():
            raise RuntimeError(f"Prepared Xcode archive was not found at {archive}")
    else:
        xcode = Path(args.aws_mac_xcode_app).expanduser().resolve()
        if not (xcode / "Contents" / "Developer").is_dir():
            raise RuntimeError(f"Full local Xcode was not found at {xcode}")
    _metal_toolchain_archive(args)
    mlxfast_bundle = Path(
        getattr(args, "aws_mac_mlxfast_bundle", DEFAULT_MLXFAST_BUNDLE)
    ).expanduser().resolve()
    if not mlxfast_bundle.is_file():
        raise RuntimeError(
            f"Bundled MLXFast CLI was not found at {mlxfast_bundle}"
        )

    profile = (
        args.aws_profile
        or os.environ.get("AWS_PROFILE", "")
        or os.environ.get("AWS_DEFAULT_PROFILE", "")
    )
    context = AwsContext(_resolve_region(args.aws_region, profile), profile)
    identity = _check_account(context)
    instance_type = args.aws_instance_type or "mac-m4pro.metal"
    if instance_type != "mac-m4pro.metal":
        raise ValueError("AWS Mac currently supports mac-m4pro.metal")
    hosts = _resolve_hosts(
        context,
        _csv(args.aws_mac_host_ids),
        _subnet_map(args.aws_mac_subnet_ids),
        students,
        instance_type,
    )
    _validate_network(context, hosts, args.aws_mac_security_group_id)
    ami_id, root_device, minimum_volume = _resolve_ami(context, args.aws_ami_id)
    volume_gib = max(args.aws_volume_gib, minimum_volume)
    plan = AwsMacPlan(
        context=context,
        account_id=identity["Account"],
        ami_id=ami_id,
        root_device=root_device,
        volume_gib=volume_gib,
        security_group_id=args.aws_mac_security_group_id,
        ssh_cidr=_ssh_cidr(args.aws_ssh_cidr),
        hosts=hosts,
    )
    print(
        "AWS Mac preflight OK — "
        f"account={plan.account_id}, region={context.region}, "
        f"hosts={len(hosts)}, instance={instance_type}, ami={ami_id}, "
        f"volume={volume_gib} GiB, ssh={plan.ssh_cidr}"
    )
    return plan


def _key_name(tag: str) -> str:
    return f"senpai-{tag}-{uuid.uuid4().hex[:8]}"


def _authorize_ssh(plan: AwsMacPlan) -> None:
    permission = json.dumps(
        [
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [
                    {
                        "CidrIp": plan.ssh_cidr,
                        "Description": "Senpai AWS Mac operator",
                    }
                ],
            }
        ]
    )
    _aws_raw(
        plan.context,
        "ec2",
        "authorize-security-group-ingress",
        "--group-id",
        plan.security_group_id,
        "--ip-permissions",
        permission,
    )


def _revoke_ssh(context: AwsContext, state: dict) -> None:
    permission = json.dumps(
        [
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [
                    {
                        "CidrIp": state["ssh_cidr"],
                        "Description": "Senpai AWS Mac operator",
                    }
                ],
            }
        ]
    )
    _aws_raw(
        context,
        "ec2",
        "revoke-security-group-ingress",
        "--group-id",
        state["security_group_id"],
        "--ip-permissions",
        permission,
    )


def _user_data(ttl_hours: float) -> str:
    shutdown = ""
    if ttl_hours:
        minutes = max(1, round(ttl_hours * 60))
        shutdown = f"/sbin/shutdown -h +{minutes}\n"
    return f"""#!/bin/bash
set -eu
{shutdown}{_ssh_host_key_console_script()}
"""


def _run_instance(
    args,
    plan: AwsMacPlan,
    host: AwsMacHost,
    key_name: str,
    client_token: str,
) -> dict:
    name = f"{args.tag}-{host.student}"
    user_data = _user_data(args.aws_ttl_hours)
    shutdown_behavior = "terminate" if args.aws_ttl_hours else "stop"
    network = json.dumps(
        [
            {
                "DeviceIndex": 0,
                "SubnetId": host.subnet_id,
                "Groups": [plan.security_group_id],
                "AssociatePublicIpAddress": True,
                "DeleteOnTermination": True,
            }
        ]
    )
    placement = json.dumps({"HostId": host.host_id, "Tenancy": "host"})
    block = json.dumps(
        [
            {
                "DeviceName": plan.root_device,
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                    "Iops": MAC_ROOT_IOPS,
                    "Throughput": MAC_ROOT_THROUGHPUT,
                    "VolumeSize": plan.volume_gib,
                    "VolumeType": "gp3",
                },
            }
        ]
    )
    tags = json.dumps(
        [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": name},
                    {"Key": "Campaign", "Value": args.tag},
                    {"Key": "senpai:run", "Value": args.tag},
                    {"Key": "SenpaiRole", "Value": f"student-{host.student}"},
                ],
            }
        ]
    )
    payload = _aws_json(
        plan.context,
        "ec2",
        "run-instances",
        "--image-id",
        plan.ami_id,
        "--instance-type",
        args.aws_instance_type or "mac-m4pro.metal",
        "--key-name",
        key_name,
        "--network-interfaces",
        network,
        "--placement",
        placement,
        "--block-device-mappings",
        block,
        "--metadata-options",
        "HttpTokens=required,HttpEndpoint=enabled",
        "--instance-initiated-shutdown-behavior",
        shutdown_behavior,
        *(("--user-data", user_data) if user_data else ()),
        "--tag-specifications",
        tags,
        "--client-token",
        client_token,
        "--count",
        "1",
    )
    instance = payload["Instances"][0]
    return {
        **asdict(host),
        "client_token": client_token,
        "instance_id": instance["InstanceId"],
        "public_ip": "",
    }


def _instance(context: AwsContext, instance_id: str) -> dict:
    reservations = _aws_json(
        context,
        "ec2",
        "describe-instances",
        "--instance-ids",
        instance_id,
    ).get("Reservations", [])
    if not reservations or not reservations[0].get("Instances"):
        raise RuntimeError(f"AWS Mac instance {instance_id} was not found")
    return reservations[0]["Instances"][0]


def _wait_instance(
    context: AwsContext,
    instance_id: str,
    timeout_s: float,
) -> dict:
    deadline = time.monotonic() + timeout_s
    detail = "not visible to EC2 yet"
    while True:
        try:
            instance = _instance(context, instance_id)
        except AwsCommandError as error:
            if not _missing_resource(error):
                raise
            instance = {}
            detail = str(error)
        except RuntimeError as error:
            if "was not found" not in str(error):
                raise
            instance = {}
            detail = str(error)

        state = instance.get("State", {}).get("Name", "unknown")
        if state in {"shutting-down", "terminated", "stopping", "stopped"}:
            raise RuntimeError(
                f"AWS Mac instance {instance_id} entered terminal state {state}"
            )

        statuses = []
        if instance:
            try:
                statuses = _aws_json(
                    context,
                    "ec2",
                    "describe-instance-status",
                    "--include-all-instances",
                    "--instance-ids",
                    instance_id,
                ).get("InstanceStatuses", [])
            except AwsCommandError as error:
                if not _missing_resource(error):
                    raise
                detail = str(error)
        status = statuses[0] if statuses else {}
        instance_status = status.get("InstanceStatus", {}).get("Status", "initializing")
        system_status = status.get("SystemStatus", {}).get("Status", "initializing")
        public_ip = instance.get("PublicIpAddress", "")
        detail = (
            f"state={state}, instance-status={instance_status}, "
            f"system-status={system_status}, public-ip={'ready' if public_ip else 'missing'}"
        )
        if (
            state == "running"
            and instance_status == "ok"
            and system_status == "ok"
            and public_ip
        ):
            return instance

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Timed out after {timeout_s:g}s waiting for AWS Mac instance "
                f"{instance_id} ({detail})"
            )
        time.sleep(min(10, remaining))


def _ssh_base(run_dir: Path, node: dict) -> list[str]:
    return [
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
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={run_dir / 'known_hosts'}",
        f"ec2-user@{node['public_ip']}",
    ]


def _ssh(
    run_dir: Path,
    node: dict,
    command: str,
    *,
    input_file=None,
    input_bytes: bytes | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    stdin = (
        input_file
        if input_file is not None
        else None if input_bytes is not None else subprocess.DEVNULL
    )
    result = subprocess.run(
        [*_ssh_base(run_dir, node), command],
        stdin=stdin,
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode:
        stdout = result.stdout.decode(errors="replace").strip()
        stderr = result.stderr.decode(errors="replace").strip()
        detail = "\n".join(value for value in (stdout, stderr) if value)
        raise RuntimeError(
            f"AWS Mac command failed on {node['instance_id']}:\n"
            f"{detail or '<no output>'}"
        )
    return result


def _wait_ssh(run_dir: Path, node: dict, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        result = _ssh(
            run_dir,
            node,
            "test \"$(uname -s)\" = Darwin && test \"$(uname -m)\" = arm64",
            timeout=20,
            check=False,
        )
        if result.returncode == 0:
            return
        last_error = result.stderr.decode(errors="replace").strip()
        time.sleep(10)
    suffix = f": {last_error}" if last_error else ""
    raise RuntimeError(
        f"Timed out waiting for SSH on {node['instance_id']}{suffix}"
    )


def _xcode_archive(args, run_dir: Path) -> tuple[Path, bool]:
    if args.aws_mac_xcode_archive:
        return Path(args.aws_mac_xcode_archive).expanduser().resolve(), False
    archive = run_dir / "Xcode.zip"
    if archive.is_file():
        return archive, True
    source = Path(args.aws_mac_xcode_app).expanduser().resolve()
    result = subprocess.run(
        [
            "ditto",
            "-c",
            "-k",
            "--sequesterRsrc",
            "--keepParent",
            str(source),
            str(archive),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Could not archive Xcode: {result.stderr.strip()}")
    return archive, True


def _uses_glm(args) -> bool:
    models = (
        getattr(args, f"{profile}_model", "")
        for profile in ("advisor", "student", "smart", "fast", "frontier")
    )
    return any(model.lower() == WANDB_GLM_52_MODEL for model in models)


def _remote_setup_script(args) -> bytes:
    repo_url = shlex.quote(args.senpai_repo_url)
    revision = shlex.quote(args.senpai_repo_revision)
    tokenizer_setup = ""
    if _uses_glm(args):
        tokenizer_setup = f"""mkdir -p {REMOTE_HF_HOME}
chmod 0700 {REMOTE_HF_HOME}
HF_HOME={REMOTE_HF_HOME} {REMOTE_VENV}/bin/python -c 'from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained("{WANDB_GLM_52_TOKENIZER}"); assert tokenizer.chat_template'
HF_HOME={REMOTE_HF_HOME} HF_HUB_OFFLINE=1 {REMOTE_VENV}/bin/python -c 'from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained("{WANDB_GLM_52_TOKENIZER}", local_files_only=True); tokens = tokenizer.apply_chat_template([{{"role": "user", "content": "smoke"}}], tools=[{{"type": "function", "function": {{"name": "echo", "description": "echo text", "parameters": {{"type": "object", "properties": {{"text": {{"type": "string"}}}}, "required": ["text"]}}}}}}], tokenize=True, add_generation_prompt=True, enable_thinking=True, reasoning_effort="max"); token_ids = tokens.get("input_ids") if hasattr(tokens, "get") else tokens; token_count = int(token_ids.shape[-1]) if hasattr(token_ids, "shape") else len(token_ids[0]) if token_ids and isinstance(token_ids[0], (list, tuple)) else len(token_ids); assert token_count > 0'
HF_HOME={REMOTE_HF_HOME} HF_HUB_OFFLINE=1 {REMOTE_VENV}/bin/python -c 'from openhands.sdk import LLM; from pydantic import SecretStr; llm = LLM(model="{WANDB_GLM_52_MODEL}", api_key=SecretStr("smoke"), api_mode="chat", base_url="https://api.inference.wandb.ai/v1", custom_tokenizer="{WANDB_GLM_52_TOKENIZER}"); assert llm.has_chat_template_tokenizer()'
"""
    return f"""#!/bin/bash
set -euo pipefail
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
sudo /usr/bin/sntp -sS -t 10 169.254.169.123
if [ ! -d /Applications/Xcode.app ]; then
  sudo ditto -x -k /tmp/senpai-Xcode.zip /Applications
fi
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
sudo xcodebuild -runFirstLaunch
metal_root=$(mktemp -d /tmp/senpai-metal-toolchain.XXXXXX)
ditto -x -k {REMOTE_METAL_TOOLCHAIN_ARCHIVE} "$metal_root"
set -- "$metal_root"/*.exportedBundle
test "$#" -eq 1
test -d "$1"
xcodebuild -importComponent metalToolchain -importPath "$1"
rm -rf "$metal_root" {REMOTE_METAL_TOOLCHAIN_ARCHIVE} /tmp/senpai-Xcode.zip
if ! command -v brew >/dev/null; then
  NONINTERACTIVE=1 /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"
fi
eval \"$(/opt/homebrew/bin/brew shellenv)\"
brew install uv gh gettext cmake jq bun tmux
command -v jq >/dev/null
jq -n -e 'true' >/dev/null
sudo mkdir -p /usr/local/libexec /usr/local/bin
sudo install -m 0755 /tmp/senpai-mlxfast.js /usr/local/libexec/mlxfast.js
sudo tee /usr/local/bin/mlxfast >/dev/null <<'MLXFAST_WRAPPER'
#!/bin/sh
if [ -z "${{MLXFAST_API_URL:-}}" ]; then export MLXFAST_API_URL='https://api.mlx.fast'; fi
if [ -z "${{MLXFAST_BENCHMARK_REF:-}}" ]; then export MLXFAST_BENCHMARK_REF='eigenlabs/mlxfast-challenge'; fi
exec /opt/homebrew/bin/bun /usr/local/libexec/mlxfast.js "$@"
MLXFAST_WRAPPER
sudo chmod 0755 /usr/local/bin/mlxfast
rm -f /tmp/senpai-mlxfast.js
mlxfast version
test ! -e {REMOTE_SOURCE}
git clone --filter=blob:none {repo_url} {REMOTE_SOURCE}
git -C {REMOTE_SOURCE} checkout --detach {revision}
test \"$(git -C {REMOTE_SOURCE} rev-parse HEAD)\" = {revision}
uv python install 3.13
uv venv --python 3.13 {REMOTE_VENV}
cd {REMOTE_SOURCE}
uv export --locked --python 3.13 --no-dev --no-emit-project --format requirements.txt > /tmp/senpai-requirements.txt
uv pip install --python {REMOTE_VENV}/bin/python -r /tmp/senpai-requirements.txt
uv pip install --python {REMOTE_VENV}/bin/python --no-deps -e .
rm -f /tmp/senpai-requirements.txt
{tokenizer_setup}PLAYWRIGHT_BROWSERS_PATH={REMOTE_BROWSER_ROOT} uvx --from playwright==1.55.0 playwright install chromium --no-shell
chromium_path=$(find {REMOTE_BROWSER_ROOT} -type f -path '*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium' -print -quit)
test -x "$chromium_path"
sudo mkdir -p /usr/local/bin
sudo rm -f /usr/local/bin/chromium
printf '%s\\n' '#!/bin/sh' 'exec "'"$chromium_path"'" "$@"' | sudo tee /usr/local/bin/chromium >/dev/null
sudo chmod 0755 /usr/local/bin/chromium
/usr/local/bin/chromium --version
role_home=$(mktemp -d)
tmux_socket="senpai-smoke-$$"
cleanup_role_runtime() {{
  HOME="$role_home" tmux -L "$tmux_socket" kill-server >/dev/null 2>&1 || true
  rm -rf "$role_home"
}}
trap cleanup_role_runtime EXIT
HOME="$role_home" tmux -V
HOME="$role_home" tmux -L "$tmux_socket" -f /dev/null new-session -d -s smoke /bin/sleep 30
HOME="$role_home" tmux -L "$tmux_socket" has-session -t smoke
HOME="$role_home" tmux -L "$tmux_socket" kill-server
HOME="$role_home" {REMOTE_VENV}/bin/python {REMOTE_SOURCE}/scripts/senpai-browser-smoke-test.py
rm -rf "$role_home"
trap - EXIT
{REMOTE_VENV}/bin/python -c 'import openhands.sdk, weave_openhands'
xcrun -sdk macosx metal --version
""".encode()


def _prepare_node(args, run_dir: Path, node: dict, archive: Path) -> None:
    print(
        f"AWS Mac {node['instance_id']} ({node['student']}) waiting for SSH.",
        flush=True,
    )
    _wait_ssh(run_dir, node, args.aws_ready_timeout_s)
    print(
        f"AWS Mac {node['instance_id']} uploading Xcode and installing runtime.",
        flush=True,
    )
    with archive.open("rb") as source:
        _ssh(
            run_dir,
            node,
            "umask 077; cat > /tmp/senpai-Xcode.zip",
            input_file=source,
            timeout=args.aws_data_timeout_s,
        )
    metal_toolchain_archive = _metal_toolchain_archive(args)
    with metal_toolchain_archive.open("rb") as source:
        _ssh(
            run_dir,
            node,
            f"umask 077; cat > {REMOTE_METAL_TOOLCHAIN_ARCHIVE}",
            input_file=source,
            timeout=args.aws_data_timeout_s,
        )
    mlxfast_bundle = Path(
        getattr(args, "aws_mac_mlxfast_bundle", DEFAULT_MLXFAST_BUNDLE)
    ).expanduser().resolve()
    with mlxfast_bundle.open("rb") as source:
        _ssh(
            run_dir,
            node,
            "umask 077; cat > /tmp/senpai-mlxfast.js",
            input_file=source,
            timeout=args.aws_data_timeout_s,
        )
    _ssh(
        run_dir,
        node,
        f"set -eu; umask 077; cat > {REMOTE_SETUP_SCRIPT}; "
        f"chmod 0700 {REMOTE_SETUP_SCRIPT}",
        input_bytes=_remote_setup_script(args),
        timeout=args.aws_data_timeout_s,
    )
    _ssh(
        run_dir,
        node,
        "set -eu; "
        f"trap 'rm -f {REMOTE_SETUP_SCRIPT}' EXIT; "
        f"/bin/bash {REMOTE_SETUP_SCRIPT} </dev/null",
        timeout=args.aws_data_timeout_s,
    )
    _ssh(
        run_dir,
        node,
        f"{REMOTE_VENV}/bin/python -c "
        "'import openhands.sdk, weave_openhands'",
        timeout=60,
    )
    print(f"AWS Mac {node['instance_id']} runtime is ready.", flush=True)


def _native_payload(args, specs: tuple[RoleSpec, ...]) -> bytes:
    shared_hf_environment = {"HF_HOME": REMOTE_HF_HOME} if _uses_glm(args) else {}
    values = {
        "args": {
            "tag": args.tag,
            "native_run_root": REMOTE_RUN_ROOT,
            "senpai_repo_revision": args.senpai_repo_revision,
            "native_ready_timeout_s": args.native_ready_timeout_s,
        },
        "roles": [
            {
                **asdict(spec),
                "env": {**spec.env, **shared_hf_environment},
            }
            for spec in specs
        ],
    }
    return (json.dumps(values) + "\n").encode()


def _native_action(
    action: str,
    args,
    run_dir: Path,
    node: dict,
    specs: tuple[RoleSpec, ...],
) -> None:
    payload = _native_payload(args, specs)
    command = (
        "set -eu; umask 077; "
        "export PATH=/opt/homebrew/bin:/opt/homebrew/opt/gettext/bin:"
        "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin; "
        f"payload=$(mktemp {REMOTE_HOME}/senpai-native.XXXXXX.json); "
        'trap \'rm -f "$payload"\' EXIT; '
        'cat > "$payload"; '
        f"{REMOTE_VENV}/bin/python {REMOTE_SOURCE}/k8s/native.py "
        f"{shlex.quote(action)}-payload \"$payload\""
    )
    redactions = tuple(
        secret
        for spec in specs
        for secret in spec.secrets.values()
        if secret
    )
    try:
        _ssh(
            run_dir,
            node,
            command,
            input_bytes=payload,
            timeout=args.native_ready_timeout_s + 60,
        )
    except RuntimeError as error:
        detail = str(error)
        for secret in redactions:
            detail = detail.replace(secret, "<redacted>")
        raise RuntimeError(detail) from error


def _launchd_canary(run_dir: Path, node: dict) -> None:
    """Prove a system LaunchDaemon survives the SSH session that creates it."""
    label = f"com.wandb.senpai.canary.{uuid.uuid4().hex[:12]}"
    temporary = f"/tmp/{label}.plist"
    installed = f"/Library/LaunchDaemons/{label}.plist"
    payload = plistlib.dumps(
        {
            "Label": label,
            "ProgramArguments": [
                "/usr/bin/caffeinate",
                "-is",
                "/bin/sleep",
                "300",
            ],
            "UserName": "ec2-user",
            "GroupName": "staff",
            "RunAtLoad": True,
            "ProcessType": "Interactive",
        },
        sort_keys=True,
    )
    try:
        _ssh(
            run_dir,
            node,
            f"cat > {shlex.quote(temporary)}; "
            f"sudo -n install -o root -g wheel -m 0644 "
            f"{shlex.quote(temporary)} {shlex.quote(installed)}; "
            f"sudo -n launchctl bootstrap system {shlex.quote(installed)}; "
            f"rm -f {shlex.quote(temporary)}",
            input_bytes=payload,
            timeout=30,
        )
        _ssh(
            run_dir,
            node,
            "for attempt in 1 2 3 4 5 6 7 8 9 10; do "
            f"if sudo -n launchctl print system/{shlex.quote(label)} "
            "| grep -q 'state = running'; then exit 0; fi; "
            "sleep 1; done; exit 1",
            timeout=30,
        )
    finally:
        _ssh(
            run_dir,
            node,
            f"sudo -n launchctl bootout system/{shlex.quote(label)} "
            ">/dev/null 2>&1 || true; "
            f"sudo -n rm -f {shlex.quote(installed)} {shlex.quote(temporary)}",
            timeout=30,
            check=False,
        )


def _open_gate(args, run_dir: Path, node: dict) -> None:
    gate = f"{REMOTE_RUN_ROOT}/{args.tag}/launch-gate"
    _ssh(
        run_dir,
        node,
        f"mkdir -p {shlex.quote(str(Path(gate).parent))}; touch {shlex.quote(gate)}",
        timeout=20,
    )


def _run_parallel(label: str, actions: dict[str, Callable[[], None]]) -> None:
    with ThreadPoolExecutor(max_workers=len(actions)) as executor:
        futures = {executor.submit(action): name for name, action in actions.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(f"{label} failed for {name}: {error}") from error


def _context_from_state(state: dict, profile: str = "") -> AwsContext:
    return AwsContext(state["region"], profile or state.get("profile", ""))


def _is_legacy_mac_state(state: dict) -> bool:
    """Recognize only the complete Mac-only shape emitted before versioning."""
    security_group_id = state.get("security_group_id")
    ssh_authorized = state.get("ssh_authorized")
    return (
        state.get("instance_type") == "mac-m4pro.metal"
        and isinstance(state.get("nodes"), list)
        and isinstance(security_group_id, str)
        and SECURITY_GROUP_ID.fullmatch(security_group_id) is not None
        and isinstance(state.get("ssh_authorize_started"), bool)
        and "ssh_authorized" in state
        and (ssh_authorized is None or isinstance(ssh_authorized, bool))
        and not any(
            field in state
            for field in (
                "instance_id",
                "instance_ids",
                "roles",
                "security_group_name",
                "subnet_id",
                "vpc_id",
            )
        )
    )


def _load_lifecycle_state(tag: str, state_root: str) -> tuple[Path, dict]:
    run_dir = _state_dir(state_root, tag)
    state = json.loads((run_dir / "state.json").read_text())
    if not isinstance(state, dict):
        raise RuntimeError("AWS Mac lifecycle state must be a JSON object")
    if "backend" in state or "state_version" in state:
        backend = state.get("backend")
        if backend != AWS_MAC_STATE_BACKEND:
            raise RuntimeError(
                f"Recorded lifecycle state uses backend {backend!r}, not "
                f"{AWS_MAC_STATE_BACKEND!r}"
            )
        version = state.get("state_version")
        if isinstance(version, bool) or version != AWS_MAC_STATE_VERSION:
            raise RuntimeError(
                f"AWS Mac state version {version!r} is unsupported; expected "
                f"{AWS_MAC_STATE_VERSION}"
            )
    elif not _is_legacy_mac_state(state):
        raise RuntimeError(
            "Recorded state is not compatible with AWS Mac lifecycle; use "
            "k8s/aws.py for standard AWS runs"
        )
    if state.get("tag") != tag:
        raise RuntimeError(
            f"AWS Mac state records tag {state.get('tag')!r}, not {tag!r}"
        )
    return run_dir, state


def _missing_cleanup_resource(error: AwsCommandError) -> bool:
    return _missing_resource(error) or "InvalidPermission.NotFound" in str(error)


def _instances_for_node(context: AwsContext, state: dict, node: dict) -> list[str]:
    payload = _aws_json(
        context,
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=client-token,Values={node['client_token']}",
        f"Name=tag:senpai:run,Values={state['tag']}",
    )
    instances = [
        instance
        for reservation in payload.get("Reservations", [])
        for instance in reservation.get("Instances", [])
        if instance.get("State", {}).get("Name") != "terminated"
    ]
    wrong_hosts = [
        instance.get("InstanceId", "<unknown>")
        for instance in instances
        if instance.get("Placement", {}).get("HostId") != node["host_id"]
    ]
    if wrong_hosts:
        raise RuntimeError(
            f"Client token {node['client_token']} resolved outside Dedicated Host "
            f"{node['host_id']}: {', '.join(wrong_hosts)}"
        )
    if len(instances) > 1:
        raise RuntimeError(
            f"Client token {node['client_token']} resolved to multiple instances"
        )
    return [instance["InstanceId"] for instance in instances]


def _cleanup(run_dir: Path, state: dict, context: AwsContext | None = None) -> list[str]:
    errors: list[str] = []
    context = context or _context_from_state(state)
    for node in state.get("nodes", []):
        if (
            node.get("instance_id")
            or not node.get("client_token")
            or node.get("launch_failed")
        ):
            continue
        try:
            recovered: list[str] = []
            for attempt in range(12):
                recovered = _instances_for_node(context, state, node)
                if recovered:
                    node["instance_id"] = recovered[0]
                    _save_state(run_dir, state)
                    break
                if attempt < 11:
                    time.sleep(5)
            if not recovered:
                errors.append(
                    f"instance launch outcome is still unknown for "
                    f"{node['student']} (client token {node['client_token']}); "
                    "retry cleanup after EC2 discovery catches up"
                )
        except Exception as error:
            errors.append(f"recover instance for {node['student']}: {error}")

    instance_ids = [
        node["instance_id"]
        for node in state.get("nodes", [])
        if node.get("instance_id")
    ]
    for node in state.get("nodes", []):
        if node.get("public_ip") and (run_dir / "id_ed25519").is_file():
            try:
                manifest = f"{REMOTE_RUN_ROOT}/{state['tag']}/manifest.json"
                _ssh(
                    run_dir,
                    node,
                    f"if test -f {shlex.quote(manifest)}; then "
                    f"{REMOTE_VENV}/bin/python {REMOTE_SOURCE}/k8s/native.py "
                    f"terminate {shlex.quote(state['tag'])} --run-root "
                    f"{shlex.quote(REMOTE_RUN_ROOT)}; fi",
                    timeout=60,
                )
            except Exception as error:
                errors.append(f"native stop {node['instance_id']}: {error}")
    if instance_ids:
        instances_terminated = False
        try:
            _aws_raw(
                context,
                "ec2",
                "terminate-instances",
                "--instance-ids",
                *instance_ids,
            )
            instances_terminated = True
        except AwsCommandError as error:
            if _missing_cleanup_resource(error):
                instances_terminated = True
            else:
                errors.append(f"terminate instances: {error}")
        except Exception as error:
            errors.append(f"terminate instances: {error}")
        if instances_terminated:
            for node in state.get("nodes", []):
                if node.get("instance_id") in instance_ids:
                    node["instance_id"] = ""
            _save_state(run_dir, state)

    if state.get("key_name") and state.get("key_owned") is not False:
        key_deleted = False
        try:
            _aws_raw(
                context,
                "ec2",
                "delete-key-pair",
                "--key-name",
                state["key_name"],
            )
            key_deleted = True
        except AwsCommandError as error:
            if _missing_cleanup_resource(error):
                key_deleted = True
            else:
                errors.append(f"delete key pair: {error}")
        except Exception as error:
            errors.append(f"delete key pair: {error}")
        if key_deleted:
            state["key_name"] = ""
            state["key_create_started"] = False
            state["key_owned"] = False
            _save_state(run_dir, state)

    revoke_ssh = state.get("ssh_authorized") is True or (
        state.get("ssh_authorize_started")
        and state.get("ssh_authorized") is not False
    )
    if revoke_ssh:
        ssh_revoked = False
        try:
            _revoke_ssh(context, state)
            ssh_revoked = True
        except AwsCommandError as error:
            if _missing_cleanup_resource(error):
                ssh_revoked = True
            else:
                errors.append(f"revoke SSH ingress: {error}")
        except Exception as error:
            errors.append(f"revoke SSH ingress: {error}")
        if ssh_revoked:
            state["ssh_authorized"] = False
            state["ssh_authorize_started"] = False
            _save_state(run_dir, state)
    if errors:
        state["phase"] = "cleanup-failed"
        state["cleanup_errors"] = errors
        _save_state(run_dir, state)
    else:
        shutil.rmtree(run_dir)
    return errors


def _lifecycle_command(action: str, tag: str, state_root: str, profile: str) -> str:
    command = ["python3", "k8s/aws_mac.py", action, tag]
    if Path(state_root).expanduser().resolve() != Path("~/.senpai/aws").expanduser().resolve():
        command.extend(["--state-root", state_root])
    if profile:
        command.extend(["--profile", profile])
    return shlex.join(command)


def _launch_recorded_instance(
    args,
    plan: AwsMacPlan,
    run_dir: Path,
    state: dict,
    host: AwsMacHost,
    key_name: str,
) -> dict:
    existing_tokens = {node["client_token"] for node in state["nodes"]}
    client_token = uuid.uuid4().hex
    while client_token in existing_tokens:
        client_token = uuid.uuid4().hex
    node = {
        **asdict(host),
        "client_token": client_token,
        "instance_id": "",
        "public_ip": "",
    }
    state["nodes"].append(node)
    _save_state(run_dir, state)
    try:
        launched = _run_instance(args, plan, host, key_name, client_token)
    except AwsCommandError as error:
        if "An error occurred (" in str(error):
            node["launch_failed"] = True
            _save_state(run_dir, state)
        raise
    node.update(launched)
    _save_state(run_dir, state)
    return node


def _wait_recorded_instance(
    args,
    plan: AwsMacPlan,
    run_dir: Path,
    state: dict,
    node: dict,
) -> None:
    instance = _wait_instance(
        plan.context,
        node["instance_id"],
        args.aws_ready_timeout_s,
    )
    node["public_ip"] = instance["PublicIpAddress"]
    _save_state(run_dir, state)
    _authorize_ssh_host(
        plan.context,
        run_dir,
        node,
        timeout_s=args.aws_ready_timeout_s,
    )


def launch_aws_mac(
    args,
    role_specs: list[RoleSpec],
    plan: AwsMacPlan | None = None,
    *,
    before_start: Callable[[], None] | None = None,
) -> None:
    """Provision one macOS instance per existing host and start native roles."""
    if args.dry_run:
        hosts = _csv(args.aws_mac_host_ids)
        students = _student_specs(role_specs)
        if len(hosts) != len(students):
            raise ValueError("AWS Mac dry run still requires one host per student")
        print(
            f"AWS Mac dry run: {len(students)} students on {len(hosts)} existing "
            "Dedicated Hosts; advisor co-located on host zero."
        )
        return

    plan = plan or preflight_aws_mac(args, role_specs)
    groups = distribute_roles(role_specs, plan.hosts)
    run_dir = _state_dir(args.aws_state_root, args.tag)
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    state = {
        "account_id": plan.account_id,
        "ami_id": plan.ami_id,
        "backend": AWS_MAC_STATE_BACKEND,
        "created_at": int(time.time()),
        "instance_type": args.aws_instance_type or "mac-m4pro.metal",
        "key_name": "",
        "key_create_started": False,
        "key_owned": None,
        "nodes": [],
        "phase": "creating",
        "profile": plan.context.profile,
        "region": plan.context.region,
        "security_group_id": plan.security_group_id,
        "ssh_authorize_started": False,
        "ssh_authorized": None,
        "ssh_cidr": plan.ssh_cidr,
        "state_version": AWS_MAC_STATE_VERSION,
        "tag": args.tag,
        "volume_gib": plan.volume_gib,
    }
    _save_state(run_dir, state)
    try:
        key_name = _key_name(args.tag)
        state["key_name"] = key_name
        state["key_create_started"] = True
        _save_state(run_dir, state)
        try:
            key = _aws_json(
                plan.context,
                "ec2",
                "create-key-pair",
                "--key-name",
                key_name,
                "--key-type",
                "ed25519",
                "--key-format",
                "pem",
                "--tag-specifications",
                json.dumps(
                    [
                        {
                            "ResourceType": "key-pair",
                            "Tags": [
                                {"Key": "Campaign", "Value": args.tag},
                                {"Key": "senpai:run", "Value": args.tag},
                            ],
                        }
                    ]
                ),
            )
        except AwsCommandError as error:
            if "InvalidKeyPair.Duplicate" in str(error):
                state["key_owned"] = False
                _save_state(run_dir, state)
            raise
        state["key_owned"] = True
        _save_state(run_dir, state)
        _write_private_key(run_dir / "id_ed25519", key["KeyMaterial"])

        state["ssh_authorize_started"] = True
        _save_state(run_dir, state)
        try:
            _authorize_ssh(plan)
        except AwsCommandError as error:
            if "InvalidPermission.Duplicate" in str(error):
                state["ssh_authorize_started"] = False
                state["ssh_authorized"] = False
                _save_state(run_dir, state)
            else:
                raise
        else:
            state["ssh_authorized"] = True
            _save_state(run_dir, state)

        group_by_host = {host.host_id: specs for host, specs in groups}
        canary_host, *remaining_hosts = plan.hosts
        state["phase"] = "launching-canary"
        _save_state(run_dir, state)
        canary = _launch_recorded_instance(
            args,
            plan,
            run_dir,
            state,
            canary_host,
            key_name,
        )
        state["phase"] = "waiting-for-canary"
        _save_state(run_dir, state)
        _wait_recorded_instance(args, plan, run_dir, state, canary)

        state["phase"] = "preparing-canary"
        _save_state(run_dir, state)
        archive, remove_archive = _xcode_archive(args, run_dir)
        print(
            f"Preparing {canary['instance_id']} as the infrastructure canary "
            "before creating any remaining instances.",
            flush=True,
        )
        _prepare_node(args, run_dir, canary, archive)
        _native_action(
            "preflight",
            args,
            run_dir,
            canary,
            group_by_host[canary["host_id"]],
        )
        _launchd_canary(run_dir, canary)

        state["phase"] = "expanding-fleet"
        _save_state(run_dir, state)
        remaining = [
            _launch_recorded_instance(
                args,
                plan,
                run_dir,
                state,
                host,
                key_name,
            )
            for host in remaining_hosts
        ]
        if remaining:
            state["phase"] = "waiting-for-macos"
            _save_state(run_dir, state)
            for node in remaining:
                _wait_recorded_instance(args, plan, run_dir, state, node)

            state["phase"] = "preparing-hosts"
            _save_state(run_dir, state)
            _run_parallel(
                "Mac preparation",
                {
                    node["student"]: (
                        lambda node=node: _prepare_node(args, run_dir, node, archive)
                    )
                    for node in remaining
                },
            )
        if remove_archive:
            archive.unlink(missing_ok=True)

        state["phase"] = "native-preflight"
        _save_state(run_dir, state)
        if remaining:
            _run_parallel(
                "native preflight",
                {
                    node["student"]: (
                        lambda node=node: _native_action(
                            "preflight",
                            args,
                            run_dir,
                            node,
                            group_by_host[node["host_id"]],
                        )
                    )
                    for node in remaining
                },
            )
        if before_start is not None:
            state["phase"] = "preparing-github"
            _save_state(run_dir, state)
            before_start()

        state["phase"] = "starting-roles"
        _save_state(run_dir, state)
        _run_parallel(
            "native launch",
            {
                node["student"]: (
                    lambda node=node: _native_action(
                        "launch",
                        args,
                        run_dir,
                        node,
                        group_by_host[node["host_id"]],
                    )
                )
                for node in state["nodes"]
            },
        )
        _run_parallel(
            "launch gate",
            {
                node["student"]: (
                    lambda node=node: _open_gate(args, run_dir, node)
                )
                for node in state["nodes"]
            },
        )
    except BaseException as error:
        if detail := str(error).strip():
            print(f"AWS Mac launch failed:\n{detail}", flush=True)
        print("Terminating only instances created by this launch.", flush=True)
        cleanup_errors = _cleanup(run_dir, state)
        if cleanup_errors and isinstance(error, Exception):
            raise RuntimeError(
                f"{error}\nAWS Mac cleanup was incomplete:\n"
                + "\n".join(cleanup_errors)
            ) from error
        raise

    state["phase"] = "running"
    _save_state(run_dir, state)
    print(f"\nAWS Mac Senpai is running on {len(state['nodes'])} M4 Pro hosts.")
    for action in ("status", "logs", "terminate"):
        print(
            f"{action.title():10} "
            + _lifecycle_command(
                action,
                args.tag,
                args.aws_state_root,
                state.get("profile", ""),
            )
        )


def status_aws_mac(
    tag: str,
    state_root: str = "~/.senpai/aws",
    *,
    profile: str = "",
) -> None:
    run_dir, state = _load_lifecycle_state(tag, state_root)
    context = _context_from_state(state, profile)
    _check_account(context, state["account_id"])
    print(f"{tag}: launcher={state.get('phase', 'unknown')} region={state['region']}")
    for node in state.get("nodes", []):
        instance = _instance(context, node["instance_id"])
        phase = instance["State"]["Name"]
        print(
            f"  student-{node['student']}: instance={node['instance_id']} "
            f"host={node['host_id']} state={phase} ip={node.get('public_ip', '')}"
        )
        if phase == "running" and node.get("public_ip"):
            result = _ssh(
                run_dir,
                node,
                f"{REMOTE_VENV}/bin/python {REMOTE_SOURCE}/k8s/native.py "
                f"status {shlex.quote(tag)} --run-root {shlex.quote(REMOTE_RUN_ROOT)}",
                timeout=30,
                check=False,
            )
            output = result.stdout.decode(errors="replace").strip()
            if output:
                print("    " + output.replace("\n", "\n    "))


def _node_for_role(state: dict, role_key: str) -> dict:
    if role_key == "advisor":
        if not state.get("nodes"):
            raise RuntimeError("AWS Mac run has no nodes")
        return state["nodes"][0]
    student = role_key.removeprefix("student-")
    node = next(
        (item for item in state.get("nodes", []) if item["student"] == student),
        None,
    )
    if node is None:
        choices = ", ".join(
            ["advisor", *[f"student-{item['student']}" for item in state.get("nodes", [])]]
        )
        raise ValueError(f"Unknown AWS Mac role {role_key!r}; choose one of: {choices}")
    return node


def logs_aws_mac(
    tag: str,
    state_root: str = "~/.senpai/aws",
    *,
    profile: str = "",
    role_key: str = "advisor",
    tail: int = 200,
) -> None:
    if tail < 1:
        raise ValueError("AWS Mac log tail must be at least 1")
    run_dir, state = _load_lifecycle_state(tag, state_root)
    context = _context_from_state(state, profile)
    _check_account(context, state["account_id"])
    node = _node_for_role(state, role_key)
    result = _ssh(
        run_dir,
        node,
        f"{REMOTE_VENV}/bin/python {REMOTE_SOURCE}/k8s/native.py logs "
        f"{shlex.quote(tag)} --run-root {shlex.quote(REMOTE_RUN_ROOT)} "
        f"--role {shlex.quote(role_key)} --tail {tail}",
        timeout=30,
    )
    print(result.stdout.decode(errors="replace"), end="")


def terminate_aws_mac(
    tag: str,
    state_root: str = "~/.senpai/aws",
    *,
    profile: str = "",
) -> None:
    run_dir, state = _load_lifecycle_state(tag, state_root)
    context = _context_from_state(state, profile)
    _check_account(context, state["account_id"])
    errors = _cleanup(run_dir, state, context)
    if errors:
        raise RuntimeError(
            "AWS Mac cleanup was incomplete; state was preserved:\n"
            + "\n".join(errors)
        )
    print(
        f"Terminated AWS Mac Senpai run {tag!r}; existing Dedicated Hosts were "
        "not released."
    )
