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
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import yaml

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
    _wait_for_instances_terminated,
    _write_private_key,
)
from .specs import RoleSpec, validate_identifier, validate_role_specs

ROOT = Path(__file__).resolve().parents[2]
REMOTE_HOME = "/Users/ec2-user"
REMOTE_SOURCE = f"{REMOTE_HOME}/senpai-source"
REMOTE_VENV = f"{REMOTE_HOME}/.senpai/venv"
REMOTE_RUNNER_ROOT = f"{REMOTE_HOME}/.senpai/aws-mac-runners"
REMOTE_BROWSER_ROOT = f"{REMOTE_HOME}/.senpai/ms-playwright"
REMOTE_HF_HOME = f"{REMOTE_HOME}/.senpai/huggingface"
REMOTE_RUN_ROOT = f"{REMOTE_HOME}/.senpai/native"
REMOTE_SETUP_SCRIPT = "/tmp/senpai-setup.sh"
REMOTE_METAL_TOOLCHAIN_ARCHIVE = "/tmp/senpai-MetalToolchain.zip"
REMOTE_RUNNER_OWNER = ".senpai-owner"
DEFAULT_MLXFAST_BUNDLE = "~/.local/share/mlxfast/mlxfast.js"
MAC_ROOT_IOPS = 10_000
MAC_ROOT_THROUGHPUT = 400
DEFAULT_AMI_PARAMETER = (
    "/aws/service/ec2-macos/tahoe/arm64_mac/latest/image_id"
)
MAC_ARCHITECTURE = "arm64_mac"
AWS_MAC_STATE_BACKEND = "aws-mac"
AWS_MAC_STATE_VERSION = 3
AWS_MAC_COMPATIBLE_STATE_VERSIONS = frozenset({1, 2, AWS_MAC_STATE_VERSION})
HOST_ID = re.compile(r"h-[0-9a-f]+")
INSTANCE_ID = re.compile(r"i-[0-9a-f]+")
SUBNET_ID = re.compile(r"subnet-[0-9a-f]+")
SECURITY_GROUP_ID = re.compile(r"sg-[0-9a-f]+")
OWNERSHIP_TOKEN = re.compile(r"[0-9a-f]{32}")
SSH_EGRESS_VERIFY_ATTEMPTS = 5
SSH_EGRESS_VERIFY_DELAY_S = 1


@dataclass(frozen=True)
class AwsMacHost:
    host_id: str
    availability_zone: str
    subnet_id: str
    student: str
    instance_id: str = ""
    instance_ownership: str = "created"
    public_ip: str = ""
    security_group_ids: tuple[str, ...] = ()
    prior_native_run: str = ""


@dataclass(frozen=True)
class AwsMacAdoption:
    private_key_path: Path
    known_hosts_path: Path
    hosts: tuple[AwsMacHost, ...]


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
    vpc_id: str = ""
    bootstrap_mode: str = "fresh"
    private_key_path: Path | None = None
    known_hosts_path: Path | None = None


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


def _manifest_mapping(
    value: object,
    label: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{label} must be a mapping")
    missing = sorted(required - value.keys())
    unexpected = sorted(value.keys() - required - optional)
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {', '.join(unexpected)}")
    return value


def _access_file(value: object, label: str, *, private: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a file path")
    candidate = Path(value).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {candidate}")
    path = candidate.resolve()
    forbidden_mode = 0o077 if private else 0o022
    if path.stat().st_mode & forbidden_mode:
        requirement = (
            "accessible by group or other"
            if private
            else "writable by group or other"
        )
        raise RuntimeError(f"{label} must not be {requirement}: {path}")
    if not path.read_bytes():
        raise RuntimeError(f"{label} must not be empty: {path}")
    return path


def _load_adoption_manifest(
    path_value: str,
    students: list[RoleSpec],
) -> AwsMacAdoption:
    if not path_value:
        raise ValueError("--aws_mac_nodes_path is required in reuse mode")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"AWS Mac nodes manifest was not found at {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"Could not read AWS Mac nodes manifest {path}: {error}") from error
    manifest = _manifest_mapping(
        raw,
        "AWS Mac nodes manifest",
        required=frozenset({"schema_version", "access", "nodes"}),
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("AWS Mac nodes manifest schema_version must be 1")
    access = _manifest_mapping(
        manifest["access"],
        "AWS Mac manifest access",
        required=frozenset(
            {"private_key_path", "known_hosts_path", "ownership"}
        ),
    )
    if access["ownership"] != "external":
        raise ValueError("AWS Mac manifest access ownership must be 'external'")
    private_key_path = _access_file(
        access["private_key_path"],
        "AWS Mac adoption private key",
        private=True,
    )
    known_hosts_path = _access_file(
        access["known_hosts_path"],
        "AWS Mac adoption known_hosts",
        private=False,
    )

    nodes = manifest["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("AWS Mac nodes manifest nodes must be a non-empty list")
    by_student: dict[str, AwsMacHost] = {}
    instance_ids: set[str] = set()
    host_ids: set[str] = set()
    for index, raw_node in enumerate(nodes):
        label = f"AWS Mac manifest node {index}"
        node = _manifest_mapping(
            raw_node,
            label,
            required=frozenset(
                {"student", "source", "expect", "prior_native_run"}
            ),
        )
        student = node["student"]
        if not isinstance(student, str) or not student:
            raise ValueError(f"{label}.student must be a non-empty string")
        if student in by_student:
            raise ValueError(f"AWS Mac nodes manifest repeats student {student!r}")
        source = _manifest_mapping(
            node["source"],
            f"{label}.source",
            required=frozenset({"adopted_instance_id"}),
        )
        instance_id = source["adopted_instance_id"]
        if not isinstance(instance_id, str) or not INSTANCE_ID.fullmatch(instance_id):
            raise ValueError(f"{label} has an invalid adopted instance ID")
        if instance_id in instance_ids:
            raise ValueError(
                f"AWS Mac nodes manifest repeats instance {instance_id!r}"
            )
        expect = _manifest_mapping(
            node["expect"],
            f"{label}.expect",
            required=frozenset(
                {
                    "host_id",
                    "availability_zone",
                    "subnet_id",
                    "security_group_ids",
                }
            ),
        )
        host_id = expect["host_id"]
        subnet_id = expect["subnet_id"]
        availability_zone = expect["availability_zone"]
        security_group_ids = expect["security_group_ids"]
        if not isinstance(host_id, str) or not HOST_ID.fullmatch(host_id):
            raise ValueError(f"{label} has an invalid Dedicated Host ID")
        if host_id in host_ids:
            raise ValueError(f"AWS Mac nodes manifest repeats host {host_id!r}")
        if not isinstance(subnet_id, str) or not SUBNET_ID.fullmatch(subnet_id):
            raise ValueError(f"{label} has an invalid subnet ID")
        if not isinstance(availability_zone, str) or not availability_zone:
            raise ValueError(f"{label} has an invalid availability zone")
        if (
            not isinstance(security_group_ids, list)
            or not security_group_ids
            or any(
                not isinstance(group_id, str)
                or not SECURITY_GROUP_ID.fullmatch(group_id)
                for group_id in security_group_ids
            )
            or len(security_group_ids) != len(set(security_group_ids))
        ):
            raise ValueError(f"{label} has invalid security_group_ids")
        prior_native_run = node["prior_native_run"]
        if not isinstance(prior_native_run, str) or not prior_native_run:
            raise ValueError(f"{label}.prior_native_run must be a non-empty string")
        validate_identifier("AWS Mac prior native run", prior_native_run)
        instance_ids.add(instance_id)
        host_ids.add(host_id)
        by_student[student] = AwsMacHost(
            host_id=host_id,
            availability_zone=availability_zone,
            subnet_id=subnet_id,
            student=student,
            instance_id=instance_id,
            instance_ownership="adopted",
            security_group_ids=tuple(security_group_ids),
            prior_native_run=prior_native_run,
        )

    expected_students = [student.name for student in students]
    if set(by_student) != set(expected_students) or len(by_student) != len(
        expected_students
    ):
        missing = sorted(set(expected_students) - set(by_student))
        unexpected = sorted(set(by_student) - set(expected_students))
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            "AWS Mac nodes manifest students do not match this launch"
            + (": " + "; ".join(detail) if detail else "")
        )
    return AwsMacAdoption(
        private_key_path=private_key_path,
        known_hosts_path=known_hosts_path,
        hosts=tuple(by_student[name] for name in expected_students),
    )


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


def _resolve_adopted_hosts(
    context: AwsContext,
    configured: tuple[AwsMacHost, ...],
    instance_type: str,
) -> tuple[tuple[AwsMacHost, ...], str]:
    instance_ids = tuple(host.instance_id for host in configured)
    payload = _aws_json(
        context,
        "ec2",
        "describe-instances",
        "--instance-ids",
        *instance_ids,
    )
    instances = [
        instance
        for reservation in payload.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    by_id = {instance.get("InstanceId"): instance for instance in instances}
    if len(by_id) != len(instances):
        raise RuntimeError("AWS returned duplicate adopted instance descriptions")
    missing = [instance_id for instance_id in instance_ids if instance_id not in by_id]
    if missing:
        raise RuntimeError("AWS Mac instances were not found: " + ", ".join(missing))

    host_ids = tuple(host.host_id for host in configured)
    host_payload = _aws_json(
        context,
        "ec2",
        "describe-hosts",
        "--host-ids",
        *host_ids,
    )
    host_descriptions = {
        host.get("HostId"): host for host in host_payload.get("Hosts", [])
    }
    missing_hosts = [host_id for host_id in host_ids if host_id not in host_descriptions]
    if missing_hosts:
        raise RuntimeError(
            "AWS Dedicated Hosts were not found: " + ", ".join(missing_hosts)
        )

    subnet_ids = tuple(sorted({host.subnet_id for host in configured}))
    subnet_payload = _aws_json(
        context,
        "ec2",
        "describe-subnets",
        "--subnet-ids",
        *subnet_ids,
    )
    subnets = {
        subnet.get("SubnetId"): subnet for subnet in subnet_payload.get("Subnets", [])
    }
    missing_subnets = [subnet_id for subnet_id in subnet_ids if subnet_id not in subnets]
    if missing_subnets:
        raise RuntimeError("AWS subnets were not found: " + ", ".join(missing_subnets))

    group_ids = tuple(
        sorted(
            {
                group_id
                for host in configured
                for group_id in host.security_group_ids
            }
        )
    )
    group_payload = _aws_json(
        context,
        "ec2",
        "describe-security-groups",
        "--group-ids",
        *group_ids,
    )
    groups = {
        group.get("GroupId"): group
        for group in group_payload.get("SecurityGroups", [])
    }
    missing_groups = [group_id for group_id in group_ids if group_id not in groups]
    if missing_groups:
        raise RuntimeError(
            "AWS security groups were not found: " + ", ".join(missing_groups)
        )

    resolved = []
    vpc_ids: set[str] = set()
    for expected in configured:
        instance = by_id[expected.instance_id]
        public_ip, vpc_id = _validate_adopted_instance_snapshot(
            asdict(expected),
            instance,
            instance_type,
        )
        vpc_ids.add(vpc_id)

        dedicated_host = host_descriptions[expected.host_id]
        if dedicated_host.get("State") != "available":
            raise RuntimeError(
                f"AWS Dedicated Host {expected.host_id} is "
                f"{dedicated_host.get('State')}, not available"
            )
        if (
            dedicated_host.get("HostProperties", {}).get("InstanceType")
            != instance_type
        ):
            raise RuntimeError(
                f"AWS Dedicated Host {expected.host_id} has the wrong instance type"
            )
        if dedicated_host.get("AvailabilityZone") != expected.availability_zone:
            raise RuntimeError(
                f"AWS Dedicated Host {expected.host_id} is in the wrong availability zone"
            )
        hosted_instances = {
            item.get("InstanceId") for item in dedicated_host.get("Instances", [])
        }
        if hosted_instances != {expected.instance_id}:
            raise RuntimeError(
                f"AWS Dedicated Host {expected.host_id} does not contain exactly "
                f"{expected.instance_id}"
            )

        subnet = subnets[expected.subnet_id]
        if subnet.get("State") != "available":
            raise RuntimeError(f"AWS subnet {expected.subnet_id} is not available")
        if subnet.get("AvailabilityZone") != expected.availability_zone:
            raise RuntimeError(
                f"AWS subnet {expected.subnet_id} is in the wrong availability zone"
            )
        if subnet.get("VpcId") != vpc_id:
            raise RuntimeError(
                f"AWS subnet {expected.subnet_id} is not in instance VPC {vpc_id}"
            )
        wrong_group_vpcs = [
            group_id
            for group_id in expected.security_group_ids
            if groups[group_id].get("VpcId") != vpc_id
        ]
        if wrong_group_vpcs:
            raise RuntimeError(
                "AWS security groups are outside the adopted instance VPC: "
                + ", ".join(wrong_group_vpcs)
            )
        resolved.append(
            AwsMacHost(
                **{
                    **asdict(expected),
                    "public_ip": public_ip,
                }
            )
        )
    if len(vpc_ids) != 1:
        raise RuntimeError("Adopted AWS Mac instances must belong to one VPC")
    return tuple(resolved), next(iter(vpc_ids))


def _validate_adopted_instance_snapshot(
    expected: Mapping[str, object],
    instance: Mapping[str, object],
    instance_type: str,
    *,
    expected_vpc_id: str = "",
    require_public_ip_match: bool = False,
) -> tuple[str, str]:
    instance_id = expected["instance_id"]
    if instance.get("InstanceId") != instance_id:
        raise RuntimeError(
            f"AWS returned instance {instance.get('InstanceId')}, expected {instance_id}"
        )
    state = instance.get("State", {})
    state_name = state.get("Name") if isinstance(state, Mapping) else None
    if state_name != "running":
        raise RuntimeError(
            f"AWS Mac instance {instance_id} is {state_name}, not running"
        )
    if instance.get("InstanceType") != instance_type:
        raise RuntimeError(
            f"AWS Mac instance {instance_id} is {instance.get('InstanceType')}, "
            f"expected {instance_type}"
        )
    if instance.get("Architecture") != MAC_ARCHITECTURE:
        raise RuntimeError(
            f"AWS Mac instance {instance_id} is not {MAC_ARCHITECTURE}"
        )
    placement = instance.get("Placement", {})
    if not isinstance(placement, Mapping) or placement.get("Tenancy") != "host":
        raise RuntimeError(f"AWS Mac instance {instance_id} does not use host tenancy")
    actual_host = placement.get("HostId")
    if actual_host != expected["host_id"]:
        raise RuntimeError(
            f"AWS Mac instance {instance_id} is on host {actual_host}, "
            f"expected {expected['host_id']}"
        )
    actual_zone = placement.get("AvailabilityZone")
    if actual_zone != expected["availability_zone"]:
        raise RuntimeError(
            f"AWS Mac instance {instance_id} is in {actual_zone}, "
            f"expected {expected['availability_zone']}"
        )
    if instance.get("SubnetId") != expected["subnet_id"]:
        raise RuntimeError(
            f"AWS Mac instance {instance_id} uses subnet {instance.get('SubnetId')}, "
            f"expected {expected['subnet_id']}"
        )
    security_groups = instance.get("SecurityGroups", [])
    actual_groups = {
        item.get("GroupId")
        for item in security_groups
        if isinstance(item, Mapping)
    }
    if actual_groups != set(expected["security_group_ids"]):
        raise RuntimeError(
            f"AWS Mac instance {instance_id} security groups do not match the manifest"
        )
    public_ip = instance.get("PublicIpAddress")
    if not isinstance(public_ip, str) or not public_ip:
        raise RuntimeError(
            f"AWS Mac instance {instance_id} has no public IPv4 address"
        )
    if require_public_ip_match and public_ip != expected["public_ip"]:
        raise RuntimeError(
            f"AWS Mac instance {instance_id} public IP changed from "
            f"{expected['public_ip']} to {public_ip}"
        )
    vpc_id = instance.get("VpcId")
    if not isinstance(vpc_id, str) or not vpc_id:
        raise RuntimeError(f"AWS Mac instance {instance_id} has no VPC identity")
    if expected_vpc_id and vpc_id != expected_vpc_id:
        raise RuntimeError(
            f"AWS Mac instance {instance_id} moved from VPC {expected_vpc_id} "
            f"to {vpc_id}"
        )
    return public_ip, vpc_id


def _validate_network(
    context: AwsContext,
    hosts: tuple[AwsMacHost, ...],
    security_group_id: str,
) -> str:
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
    return next(iter(vpcs))


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
    if args.aws_ttl_hours < 0:
        raise ValueError("--aws_ttl_hours must be non-negative for AWS Mac")
    bootstrap_mode = getattr(args, "aws_mac_bootstrap_mode", "fresh")
    nodes_path = getattr(args, "aws_mac_nodes_path", "")
    if bootstrap_mode not in {"fresh", "reuse"}:
        raise ValueError("--aws_mac_bootstrap_mode must be fresh or reuse")
    if bootstrap_mode == "reuse":
        if not nodes_path:
            raise ValueError("--aws_mac_nodes_path is required in reuse mode")
        if args.aws_ttl_hours != 0:
            raise ValueError("Adopted AWS Mac fleets require --aws_ttl_hours 0")
        ambiguous = [
            name
            for name in (
                "aws_mac_host_ids",
                "aws_mac_subnet_ids",
                "aws_mac_security_group_id",
            )
            if getattr(args, name, "")
        ]
        if ambiguous:
            raise ValueError(
                "Reuse mode takes placement only from --aws_mac_nodes_path; remove "
                + ", ".join(f"--{name}" for name in ambiguous)
            )
    elif nodes_path:
        raise ValueError(
            "--aws_mac_nodes_path requires --aws_mac_bootstrap_mode reuse"
        )
    elif args.aws_volume_gib < 200:
        raise ValueError("AWS Mac root volumes must be at least 200 GiB")

    run_dir = _state_dir(args.aws_state_root, args.tag)
    if run_dir.exists() or run_dir.is_symlink():
        raise RuntimeError(
            f"AWS Mac state already exists at {run_dir}; choose a new tag or "
            "terminate the recorded run"
        )
    _check_source_revision(args.senpai_repo_revision)
    if not shutil.which("ssh"):
        raise RuntimeError("AWS Mac launch requires OpenSSH")
    submission_cli, submission_bundle, _ = _submission_cli(args)
    if not submission_bundle.is_file():
        raise RuntimeError(
            f"Bundled {submission_cli} CLI was not found at {submission_bundle}"
        )
    adoption = None
    if bootstrap_mode == "reuse":
        adoption = _load_adoption_manifest(nodes_path, students)
        repeated_tag = [
            host.student for host in adoption.hosts if host.prior_native_run == args.tag
        ]
        if repeated_tag:
            raise ValueError("The new AWS Mac tag must differ from every prior native run")
    else:
        if not shutil.which("ditto"):
            raise RuntimeError("Fresh AWS Mac launch requires ditto")
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
    if adoption is not None:
        hosts, vpc_id = _resolve_adopted_hosts(
            context,
            adoption.hosts,
            instance_type,
        )
        group_ids = {group for host in hosts for group in host.security_group_ids}
        plan = AwsMacPlan(
            context=context,
            account_id=identity["Account"],
            ami_id="",
            root_device="",
            volume_gib=0,
            security_group_id=next(iter(group_ids)) if len(group_ids) == 1 else "",
            ssh_cidr="",
            hosts=hosts,
            vpc_id=vpc_id,
            bootstrap_mode="reuse",
            private_key_path=adoption.private_key_path,
            known_hosts_path=adoption.known_hosts_path,
        )
        _preflight_adopted_access(plan, args.tag)
        print(
            "AWS Mac adoption preflight OK — "
            f"account={plan.account_id}, region={context.region}, "
            f"instances={len(hosts)}, instance={instance_type}, access=external"
        )
        return plan

    hosts = _resolve_hosts(
        context,
        _csv(args.aws_mac_host_ids),
        _subnet_map(args.aws_mac_subnet_ids),
        students,
        instance_type,
    )
    vpc_id = _validate_network(context, hosts, args.aws_mac_security_group_id)
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
        vpc_id=vpc_id,
        bootstrap_mode="fresh",
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


def _ssh_security_group_name(tag: str) -> str:
    return f"senpai-{tag}-ssh-{uuid.uuid4().hex[:8]}"


def _create_ssh_security_group(
    plan: AwsMacPlan,
    *,
    name: str,
    tag: str,
) -> str:
    if not plan.vpc_id:
        raise RuntimeError("AWS Mac plan has no VPC for its SSH security group")
    payload = _aws_json(
        plan.context,
        "ec2",
        "create-security-group",
        "--group-name",
        name,
        "--description",
        f"Senpai AWS Mac SSH for {tag}",
        "--vpc-id",
        plan.vpc_id,
        "--tag-specifications",
        json.dumps(
            [
                {
                    "ResourceType": "security-group",
                    "Tags": [
                        {"Key": "Campaign", "Value": tag},
                        {"Key": "senpai:run", "Value": tag},
                        {"Key": "SenpaiPurpose", "Value": "ssh"},
                    ],
                }
            ]
        ),
    )
    group_id = payload.get("GroupId", "")
    if not isinstance(group_id, str) or not SECURITY_GROUP_ID.fullmatch(group_id):
        raise RuntimeError("AWS did not return a valid SSH security group ID")
    return group_id


def _security_group_egress(
    context: AwsContext,
    group_id: str,
) -> list[dict]:
    payload = _aws_json(
        context,
        "ec2",
        "describe-security-groups",
        "--group-ids",
        group_id,
    )
    groups = payload.get("SecurityGroups", [])
    if len(groups) != 1 or groups[0].get("GroupId") != group_id:
        raise RuntimeError(f"AWS Mac SSH security group {group_id} is ambiguous")
    permissions = groups[0].get("IpPermissionsEgress")
    if not isinstance(permissions, list):
        raise RuntimeError(
            f"AWS Mac SSH security group {group_id} has invalid egress metadata"
        )
    if not all(isinstance(permission, dict) for permission in permissions):
        raise RuntimeError(
            f"AWS Mac SSH security group {group_id} has invalid egress permissions"
        )
    return permissions


def _harden_ssh_security_group(context: AwsContext, group_id: str) -> None:
    permissions = _security_group_egress(context, group_id)
    if not permissions:
        return
    revoke_error: Exception | None = None
    try:
        _aws_raw(
            context,
            "ec2",
            "revoke-security-group-egress",
            "--group-id",
            group_id,
            "--ip-permissions",
            json.dumps(permissions),
        )
    except Exception as error:  # Verification resolves response-loss ambiguity.
        revoke_error = error
    for attempt in range(SSH_EGRESS_VERIFY_ATTEMPTS):
        if not _security_group_egress(context, group_id):
            return
        if attempt + 1 < SSH_EGRESS_VERIFY_ATTEMPTS:
            time.sleep(SSH_EGRESS_VERIFY_DELAY_S)
    failure = RuntimeError(
        f"AWS Mac SSH security group {group_id} still permits egress"
    )
    if revoke_error is not None:
        raise failure from revoke_error
    raise failure


def _recover_ssh_security_group(context: AwsContext, state: dict) -> str:
    group_id = state.get("ssh_security_group_id", "")
    if group_id:
        if not SECURITY_GROUP_ID.fullmatch(group_id):
            raise RuntimeError("AWS Mac state has an invalid SSH security group ID")
        return group_id
    name = state.get("ssh_security_group_name", "")
    vpc_id = state.get("vpc_id", "")
    if not name or not vpc_id:
        return ""
    payload = _aws_json(
        context,
        "ec2",
        "describe-security-groups",
        "--filters",
        f"Name=group-name,Values={name}",
        f"Name=vpc-id,Values={vpc_id}",
        f"Name=tag:senpai:run,Values={state['tag']}",
        "Name=tag:SenpaiPurpose,Values=ssh",
    )
    groups = payload.get("SecurityGroups", [])
    if len(groups) > 1:
        raise RuntimeError(
            f"AWS Mac SSH security group name {name!r} resolved ambiguously"
        )
    if not groups:
        return ""
    recovered = groups[0].get("GroupId", "")
    if not isinstance(recovered, str) or not SECURITY_GROUP_ID.fullmatch(recovered):
        raise RuntimeError("Recovered AWS Mac SSH security group ID is invalid")
    return recovered


def _authorize_ssh(
    context: AwsContext,
    security_group_id: str,
    ssh_cidr: str,
) -> None:
    permission = json.dumps(
        [
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [
                    {
                        "CidrIp": ssh_cidr,
                        "Description": "Senpai AWS Mac operator",
                    }
                ],
            }
        ]
    )
    _aws_raw(
        context,
        "ec2",
        "authorize-security-group-ingress",
        "--group-id",
        security_group_id,
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
        state["ssh_security_group_id"],
        "--ip-permissions",
        permission,
    )


def _delete_ssh_security_group(context: AwsContext, group_id: str) -> None:
    _aws_raw(
        context,
        "ec2",
        "delete-security-group",
        "--group-id",
        group_id,
    )


def _security_group_has_ssh_rule(
    context: AwsContext,
    group_id: str,
    ssh_cidr: str,
) -> bool:
    if not SECURITY_GROUP_ID.fullmatch(group_id):
        raise RuntimeError("Legacy AWS Mac state has an invalid security group ID")
    payload = _aws_json(
        context,
        "ec2",
        "describe-security-groups",
        "--group-ids",
        group_id,
    )
    groups = payload.get("SecurityGroups", [])
    if not groups:
        return False
    if len(groups) != 1 or groups[0].get("GroupId") != group_id:
        raise RuntimeError(
            f"Legacy AWS Mac security group {group_id} resolved ambiguously"
        )
    return any(
        permission.get("IpProtocol") == "tcp"
        and permission.get("FromPort") == 22
        and permission.get("ToPort") == 22
        and any(
            item.get("CidrIp") == ssh_cidr
            for item in permission.get("IpRanges", [])
        )
        for permission in groups[0].get("IpPermissions", [])
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
    *,
    ssh_security_group_id: str,
) -> dict:
    if not SECURITY_GROUP_ID.fullmatch(ssh_security_group_id):
        raise RuntimeError("AWS Mac launch requires its campaign SSH security group")
    name = f"{args.tag}-{host.student}"
    user_data = _user_data(args.aws_ttl_hours)
    shutdown_behavior = "terminate" if args.aws_ttl_hours else "stop"
    network = json.dumps(
        [
            {
                "DeviceIndex": 0,
                "SubnetId": host.subnet_id,
                "Groups": [plan.security_group_id, ssh_security_group_id],
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


def _ssh_base_paths(
    private_key_path: Path,
    known_hosts_path: Path,
    node: dict,
) -> list[str]:
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-i",
        str(private_key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "IdentityAgent=none",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        f"ec2-user@{node['public_ip']}",
    ]


def _ssh_base(run_dir: Path, node: dict) -> list[str]:
    return _ssh_base_paths(
        run_dir / "id_ed25519",
        run_dir / "known_hosts",
        node,
    )


def _run_ssh(
    base: list[str],
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
        [*base, command],
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
    return _run_ssh(
        _ssh_base(run_dir, node),
        node,
        command,
        input_file=input_file,
        input_bytes=input_bytes,
        timeout=timeout,
        check=check,
    )


def _ssh_with_access(
    private_key_path: Path,
    known_hosts_path: Path,
    node: dict,
    command: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return _run_ssh(
        _ssh_base_paths(private_key_path, known_hosts_path, node),
        node,
        command,
        timeout=30,
        check=check,
    )


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


def _prior_manifest_labels(payload: bytes, prior_native_run: str) -> tuple[str, ...]:
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Prior native run {prior_native_run!r} has an invalid manifest"
        ) from error
    if not isinstance(manifest, dict):
        raise RuntimeError(
            f"Prior native run {prior_native_run!r} manifest is not an object"
        )
    if manifest.get("tag") != prior_native_run or manifest.get("domain") != "system":
        raise RuntimeError(
            f"Prior native run {prior_native_run!r} manifest identity is invalid"
        )
    roles = manifest.get("roles")
    if not isinstance(roles, list) or not roles:
        raise RuntimeError(f"Prior native run {prior_native_run!r} has no roles")
    prefix = f"com.wandb.senpai.{prior_native_run}."
    labels = []
    for role in roles:
        label = role.get("label") if isinstance(role, dict) else None
        if not isinstance(label, str) or not label.startswith(prefix):
            raise RuntimeError(
                f"Prior native run {prior_native_run!r} has an invalid role label"
            )
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise RuntimeError(
            f"Prior native run {prior_native_run!r} repeats a role label"
        )
    return tuple(labels)


def _verify_adopted_guest(
    node: dict,
    runner: Callable[..., subprocess.CompletedProcess],
    new_tag: str,
) -> None:
    expected = shlex.quote(node["instance_id"])
    identity = (
        "set -eu; "
        "export PATH=/opt/homebrew/bin:/opt/homebrew/opt/gettext/bin:"
        "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin; "
        'test "$(uname -s)" = Darwin; test "$(uname -m)" = arm64; '
        "sudo -n true; "
        "for command in brew git uv gh envsubst cmake jq bun tmux curl; do "
        'command -v "$command" >/dev/null; done; '
        "test -d /Applications/Xcode.app/Contents/Developer; "
        "xcrun -sdk macosx metal --version >/dev/null; "
        "test -x /usr/local/bin/chromium; "
        "token=$(curl -fsS --connect-timeout 5 -X PUT "
        "-H 'X-aws-ec2-metadata-token-ttl-seconds: 60' "
        "http://169.254.169.254/latest/api/token); "
        "actual=$(curl -fsS --connect-timeout 5 "
        "-H \"X-aws-ec2-metadata-token: $token\" "
        "http://169.254.169.254/latest/meta-data/instance-id); "
        f'test "$actual" = {expected}'
    )
    runner(identity)

    prior_native_run = node["prior_native_run"]
    manifest = f"{REMOTE_RUN_ROOT}/{prior_native_run}/manifest.json"
    result = runner(f"cat {shlex.quote(manifest)}")
    for label in _prior_manifest_labels(result.stdout, prior_native_run):
        status = runner(
            f"sudo -n launchctl print system/{shlex.quote(label)}",
            check=False,
        )
        if status.returncode == 0:
            raise RuntimeError(
                f"Prior native LaunchDaemon {label!r} is still loaded"
            )
        detail = (status.stdout + status.stderr).decode(
            "utf-8", errors="replace"
        ).lower()
        if not any(
            marker in detail
            for marker in (
                "could not find service",
                "service is not loaded",
                "no such process",
            )
        ):
            raise RuntimeError(
                f"Could not prove prior native LaunchDaemon {label!r} is unloaded"
            )

    validate_identifier("AWS Mac tag", new_tag)
    runner_root, _, _ = _remote_runtime_paths(new_tag, "reuse")
    native_root = f"{REMOTE_RUN_ROOT}/{new_tag}"
    runner(
        f"set -eu; test ! -e {shlex.quote(runner_root)}; "
        f"test ! -L {shlex.quote(runner_root)}; "
        f"test ! -e {shlex.quote(native_root)}; "
        f"test ! -L {shlex.quote(native_root)}"
    )


def _preflight_adopted_access(plan: AwsMacPlan, new_tag: str) -> None:
    if plan.private_key_path is None or plan.known_hosts_path is None:
        raise RuntimeError("AWS Mac adoption plan has no imported SSH access")
    for host in plan.hosts:
        node = asdict(host)

        def run(command: str, *, check: bool = True):
            return _ssh_with_access(
                plan.private_key_path,
                plan.known_hosts_path,
                node,
                command,
                check=check,
            )

        _verify_adopted_guest(node, run, new_tag)


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


def aws_mac_submission_token_env(args) -> str:
    return (
        "YUKON_API_TOKEN"
        if getattr(args, "aws_mac_yukon_bundle", "")
        else "MLXFAST_API_TOKEN"
    )


def _submission_cli(args) -> tuple[str, Path, str]:
    yukon_bundle = getattr(args, "aws_mac_yukon_bundle", "")
    if yukon_bundle:
        return (
            "yukon",
            Path(yukon_bundle).expanduser().resolve(),
            "/tmp/senpai-yukon.js",
        )
    return (
        "mlxfast",
        Path(
            getattr(args, "aws_mac_mlxfast_bundle", DEFAULT_MLXFAST_BUNDLE)
        ).expanduser().resolve(),
        "/tmp/senpai-mlxfast.js",
    )


def _submission_cli_install_script(args) -> str:
    name, _, staging_path = _submission_cli(args)
    if name == "yukon":
        return f"""sudo mkdir -p /usr/local/libexec /usr/local/bin
sudo install -m 0755 {staging_path} /usr/local/libexec/yukon.js
sudo tee /usr/local/bin/yukon >/dev/null <<'YUKON_WRAPPER'
#!/bin/sh
if [ -z "${{YUKON_API_URL:-}}" ]; then export YUKON_API_URL='https://api.yukon.org'; fi
exec /opt/homebrew/bin/bun /usr/local/libexec/yukon.js "$@"
YUKON_WRAPPER
sudo chmod 0755 /usr/local/bin/yukon
rm -f {staging_path}
yukon version
"""
    return f"""sudo mkdir -p /usr/local/libexec /usr/local/bin
sudo install -m 0755 {staging_path} /usr/local/libexec/mlxfast.js
sudo tee /usr/local/bin/mlxfast >/dev/null <<'MLXFAST_WRAPPER'
#!/bin/sh
if [ -z "${{MLXFAST_API_URL:-}}" ]; then export MLXFAST_API_URL='https://api.mlx.fast'; fi
if [ -z "${{MLXFAST_BENCHMARK_REF:-}}" ]; then export MLXFAST_BENCHMARK_REF='eigenlabs/mlxfast-challenge'; fi
exec /opt/homebrew/bin/bun /usr/local/libexec/mlxfast.js "$@"
MLXFAST_WRAPPER
sudo chmod 0755 /usr/local/bin/mlxfast
rm -f {staging_path}
mlxfast version
"""


def _remote_runtime_paths(tag: str, bootstrap_mode: str) -> tuple[str, str, str]:
    if bootstrap_mode == "fresh":
        return REMOTE_HOME, REMOTE_SOURCE, REMOTE_VENV
    if bootstrap_mode != "reuse":
        raise RuntimeError(f"Unknown AWS Mac bootstrap mode {bootstrap_mode!r}")
    validate_identifier("AWS Mac tag", tag)
    root = f"{REMOTE_RUNNER_ROOT}/{tag}"
    return root, f"{root}/source", f"{root}/venv"


def _runtime_paths_for_args(args) -> tuple[str, str, str]:
    return _remote_runtime_paths(
        args.tag,
        getattr(args, "aws_mac_bootstrap_mode", "fresh"),
    )


def _runtime_paths_for_state(state: dict) -> tuple[str, str, str]:
    return _remote_runtime_paths(
        state["tag"],
        state.get("bootstrap_mode", "fresh"),
    )


def _remote_setup_script(args) -> bytes:
    repo_url = shlex.quote(args.senpai_repo_url)
    revision = shlex.quote(args.senpai_repo_revision)
    submission_cli_setup = _submission_cli_install_script(args)
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
{submission_cli_setup}test ! -e {REMOTE_SOURCE}
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


def _remote_reuse_setup_script(args, ownership_token: str) -> bytes:
    if not OWNERSHIP_TOKEN.fullmatch(ownership_token):
        raise RuntimeError("AWS Mac reuse ownership token is invalid")
    runner_root, source_root, venv_root = _runtime_paths_for_args(args)
    owner_marker = f"{runner_root}/{REMOTE_RUNNER_OWNER}"
    repo_url = shlex.quote(args.senpai_repo_url)
    revision = shlex.quote(args.senpai_repo_revision)
    requirements = f"/tmp/senpai-requirements-{args.tag}.txt"
    submission_cli_setup = _submission_cli_install_script(args)
    tokenizer_setup = ""
    if _uses_glm(args):
        tokenizer_setup = f"""mkdir -p {REMOTE_HF_HOME}
chmod 0700 {REMOTE_HF_HOME}
HF_HOME={REMOTE_HF_HOME} {venv_root}/bin/python -c 'from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained("{WANDB_GLM_52_TOKENIZER}"); assert tokenizer.chat_template'
HF_HOME={REMOTE_HF_HOME} HF_HUB_OFFLINE=1 {venv_root}/bin/python -c 'from openhands.sdk import LLM; from pydantic import SecretStr; llm = LLM(model="{WANDB_GLM_52_MODEL}", api_key=SecretStr("smoke"), api_mode="chat", base_url="https://api.inference.wandb.ai/v1", custom_tokenizer="{WANDB_GLM_52_TOKENIZER}"); assert llm.has_chat_template_tokenizer()'
"""
    return f"""#!/bin/bash
set -euo pipefail
export PATH=/opt/homebrew/bin:/opt/homebrew/opt/gettext/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
sudo /usr/bin/sntp -sS -t 10 169.254.169.123
for command in brew git uv gh envsubst cmake jq bun tmux curl; do
  command -v "$command" >/dev/null
done
sudo -n true
test -d /Applications/Xcode.app/Contents/Developer
xcrun -sdk macosx metal --version
test -x /usr/local/bin/chromium
/usr/local/bin/chromium --version
mkdir -p {REMOTE_RUNNER_ROOT}
mkdir {runner_root}
chmod 0700 {runner_root}
printf '%s\n' {shlex.quote(ownership_token)} > {owner_marker}
chmod 0600 {owner_marker}
{submission_cli_setup}git clone --filter=blob:none {repo_url} {source_root}
git -C {source_root} checkout --detach {revision}
test "$(git -C {source_root} rev-parse HEAD)" = {revision}
uv python install 3.13
uv venv --python 3.13 {venv_root}
cd {source_root}
uv export --locked --python 3.13 --no-dev --no-emit-project --format requirements.txt > {requirements}
uv pip install --python {venv_root}/bin/python -r {requirements}
uv pip install --python {venv_root}/bin/python --no-deps -e .
rm -f {requirements}
{tokenizer_setup}role_home=$(mktemp -d)
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
HOME="$role_home" {venv_root}/bin/python {source_root}/scripts/senpai-browser-smoke-test.py
rm -rf "$role_home"
trap - EXIT
{venv_root}/bin/python -c 'import openhands.sdk, weave_openhands'
""".encode()


def _prepare_node(
    args,
    run_dir: Path,
    node: dict,
    archive: Path | None,
) -> None:
    print(
        f"AWS Mac {node['instance_id']} ({node['student']}) waiting for SSH.",
        flush=True,
    )
    _wait_ssh(run_dir, node, args.aws_ready_timeout_s)
    _, submission_bundle, staging_path = _submission_cli(args)
    with submission_bundle.open("rb") as source:
        _ssh(
            run_dir,
            node,
            f"umask 077; cat > {staging_path}",
            input_file=source,
            timeout=args.aws_data_timeout_s,
        )
    bootstrap_mode = getattr(args, "aws_mac_bootstrap_mode", "fresh")
    if bootstrap_mode == "reuse":
        print(
            f"AWS Mac {node['instance_id']} installing an isolated Senpai runtime.",
            flush=True,
        )
        setup_script = _remote_reuse_setup_script(
            args,
            node["runtime_ownership_token"],
        )
    else:
        if archive is None:
            raise RuntimeError("Fresh AWS Mac preparation requires an Xcode archive")
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
        setup_script = _remote_setup_script(args)
    _ssh(
        run_dir,
        node,
        f"set -eu; umask 077; cat > {REMOTE_SETUP_SCRIPT}; "
        f"chmod 0700 {REMOTE_SETUP_SCRIPT}",
        input_bytes=setup_script,
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
    _, _, venv_root = _runtime_paths_for_args(args)
    _ssh(
        run_dir,
        node,
        f"{venv_root}/bin/python -c "
        "'import openhands.sdk, weave_openhands'",
        timeout=60,
    )
    print(f"AWS Mac {node['instance_id']} runtime is ready.", flush=True)


def _native_payload(
    args,
    specs: tuple[RoleSpec, ...],
    *,
    ownership_token: str = "",
) -> bytes:
    shared_hf_environment = {"HF_HOME": REMOTE_HF_HOME} if _uses_glm(args) else {}
    values = {
        "args": {
            "tag": args.tag,
            "native_run_root": REMOTE_RUN_ROOT,
            "senpai_repo_revision": args.senpai_repo_revision,
            "native_ready_timeout_s": args.native_ready_timeout_s,
            "native_ownership_token": ownership_token,
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
    payload = _native_payload(
        args,
        specs,
        ownership_token=node.get("runtime_ownership_token", ""),
    )
    _, source_root, venv_root = _runtime_paths_for_args(args)
    command = (
        "set -eu; umask 077; "
        "export PATH=/opt/homebrew/bin:/opt/homebrew/opt/gettext/bin:"
        "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin; "
        f"payload=$(mktemp {REMOTE_HOME}/senpai-native.XXXXXX.json); "
        'trap \'rm -f "$payload"\' EXIT; '
        'cat > "$payload"; '
        f"{venv_root}/bin/python {source_root}/k8s/native.py "
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
        if (
            isinstance(version, bool)
            or version not in AWS_MAC_COMPATIBLE_STATE_VERSIONS
        ):
            raise RuntimeError(
                f"AWS Mac state version {version!r} is unsupported; expected "
                + " or ".join(
                    str(item) for item in sorted(AWS_MAC_COMPATIBLE_STATE_VERSIONS)
                )
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


def _instances_for_node(
    context: AwsContext,
    state: dict,
    node: dict,
) -> tuple[list[str], bool]:
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
    if not instances:
        return [], False
    instance = instances[0]
    if instance.get("State", {}).get("Name") == "terminated":
        return [], True
    return [instance["InstanceId"]], False


def _node_ownership(node: dict) -> str:
    ownership = node.get("instance_ownership")
    return ownership if ownership in {"created", "adopted"} else "unknown"


def _adopted_runtime_cleanup_command(
    state: dict,
    node: dict,
    runner_root: str,
    source_root: str,
    venv_root: str,
) -> str:
    ownership_token = node.get("runtime_ownership_token", "")
    if (
        not isinstance(ownership_token, str)
        or not OWNERSHIP_TOKEN.fullmatch(ownership_token)
        or ownership_token != state.get("runtime_ownership_token")
    ):
        raise RuntimeError(
            "adopted runtime ownership token is missing or inconsistent"
        )
    owner_marker = f"{runner_root}/{REMOTE_RUNNER_OWNER}"
    native_root = f"{REMOTE_RUN_ROOT}/{state['tag']}"
    manifest = f"{native_root}/manifest.json"
    return (
        "set -eu; "
        f"if test ! -e {shlex.quote(runner_root)} && "
        f"test ! -L {shlex.quote(runner_root)}; then "
        f"test ! -e {shlex.quote(native_root)}; "
        f"test ! -L {shlex.quote(native_root)}; exit 0; fi; "
        f"test -f {shlex.quote(owner_marker)}; "
        f"test ! -L {shlex.quote(owner_marker)}; "
        f'test "$(cat {shlex.quote(owner_marker)})" = '
        f"{shlex.quote(ownership_token)}; "
        f"if test -f {shlex.quote(manifest)}; then "
        f"{venv_root}/bin/python {source_root}/k8s/native.py "
        f"terminate {shlex.quote(state['tag'])} --run-root "
        f"{shlex.quote(REMOTE_RUN_ROOT)} --ownership-token "
        f"{shlex.quote(ownership_token)}; fi; "
        f"rm -rf {shlex.quote(runner_root)}"
    )


def _cleanup(run_dir: Path, state: dict, context: AwsContext | None = None) -> list[str]:
    errors: list[str] = []
    unknown_nodes = [
        node
        for node in state.get("nodes", [])
        if _node_ownership(node) == "unknown"
    ]
    if unknown_nodes:
        names = ", ".join(str(node.get("student", "<unknown>")) for node in unknown_nodes)
        errors.append(
            "instance ownership is missing or invalid for "
            f"{names}; preserving all AWS and remote resources for operator "
            "reconciliation"
        )
        state["phase"] = "cleanup-failed"
        state["cleanup_errors"] = errors
        _save_state(run_dir, state)
        return errors

    bootstrap_mode = state.get("bootstrap_mode", "fresh")
    if bootstrap_mode not in {"fresh", "reuse"} or (
        any(_node_ownership(node) == "adopted" for node in state.get("nodes", []))
        and bootstrap_mode != "reuse"
    ):
        errors.append(
            "AWS Mac bootstrap ownership is missing or invalid; preserving all "
            "AWS and remote resources for operator reconciliation"
        )
        state["phase"] = "cleanup-failed"
        state["cleanup_errors"] = errors
        _save_state(run_dir, state)
        return errors

    context = context or _context_from_state(state)
    for node in state.get("nodes", []):
        if (
            _node_ownership(node) != "created"
            or node.get("termination_confirmed")
            or node.get("instance_id")
            or not node.get("client_token")
            or node.get("launch_failed")
        ):
            continue
        try:
            recovered: list[str] = []
            resolved = False
            for attempt in range(12):
                recovered, already_terminated = _instances_for_node(
                    context,
                    state,
                    node,
                )
                if already_terminated:
                    node["termination_confirmed"] = True
                    _save_state(run_dir, state)
                    resolved = True
                    break
                if recovered:
                    node["instance_id"] = recovered[0]
                    _save_state(run_dir, state)
                    resolved = True
                    break
                if attempt < 11:
                    time.sleep(5)
            if not resolved:
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
        if _node_ownership(node) == "created" and node.get("instance_id")
    ]
    native_stop_errors: list[tuple[str, str]] = []
    runner_root, source_root, venv_root = _remote_runtime_paths(
        state["tag"],
        bootstrap_mode,
    )
    for node in state.get("nodes", []):
        if (
            _node_ownership(node) == "adopted"
            and node.get("runtime_cleanup_confirmed") is True
        ):
            continue
        instance_id = node.get("instance_id")
        if (
            instance_id
            and node.get("public_ip")
            and (run_dir / "id_ed25519").is_file()
        ):
            try:
                if _node_ownership(node) == "adopted":
                    instance = _instance(context, instance_id)
                    _validate_adopted_instance_snapshot(
                        node,
                        instance,
                        state.get("instance_type", "mac-m4pro.metal"),
                        expected_vpc_id=state.get("vpc_id", ""),
                        require_public_ip_match=True,
                    )
                if bootstrap_mode == "reuse":
                    command = _adopted_runtime_cleanup_command(
                        state,
                        node,
                        runner_root,
                        source_root,
                        venv_root,
                    )
                else:
                    manifest = f"{REMOTE_RUN_ROOT}/{state['tag']}/manifest.json"
                    command = (
                        f"if test -f {shlex.quote(manifest)}; then "
                        f"{venv_root}/bin/python {source_root}/k8s/native.py "
                        f"terminate {shlex.quote(state['tag'])} --run-root "
                        f"{shlex.quote(REMOTE_RUN_ROOT)}; fi"
                    )
                _ssh(
                    run_dir,
                    node,
                    command,
                    timeout=60,
                )
                if _node_ownership(node) == "adopted":
                    node["runtime_cleanup_confirmed"] = True
                    _save_state(run_dir, state)
            except Exception as error:
                native_stop_errors.append(
                    (instance_id, f"native stop {instance_id}: {error}")
                )
        elif instance_id and _node_ownership(node) == "adopted":
            native_stop_errors.append(
                (
                    instance_id,
                    f"native stop {instance_id}: adopted instance access is "
                    "incomplete; preserving lifecycle state",
                )
            )
    terminated_ids: set[str] = set()
    for instance_id in instance_ids:
        try:
            _aws_raw(
                context,
                "ec2",
                "terminate-instances",
                "--instance-ids",
                instance_id,
            )
        except AwsCommandError as error:
            if _missing_cleanup_resource(error):
                terminated_ids.add(instance_id)
            else:
                errors.append(f"terminate instance {instance_id}: {error}")
        except Exception as error:
            errors.append(f"terminate instance {instance_id}: {error}")
        else:
            try:
                _wait_for_instances_terminated(context, [instance_id])
                terminated_ids.add(instance_id)
            except AwsCommandError as error:
                if _missing_cleanup_resource(error):
                    terminated_ids.add(instance_id)
                else:
                    errors.append(
                        f"wait for instance {instance_id} termination: {error}"
                    )
            except Exception as error:
                errors.append(
                    f"wait for instance {instance_id} termination: {error}"
                )
    if terminated_ids:
        for node in state.get("nodes", []):
            if node.get("instance_id") in terminated_ids:
                node["instance_id"] = ""
                node["termination_confirmed"] = True
        _save_state(run_dir, state)
    errors.extend(
        message
        for instance_id, message in native_stop_errors
        if instance_id not in terminated_ids
    )

    unresolved_nodes = [
        node
        for node in state.get("nodes", [])
        if _node_ownership(node) == "created"
        and (
            node.get("instance_id")
            or (
                node.get("client_token")
                and not node.get("launch_failed")
                and not node.get("termination_confirmed")
            )
        )
    ]
    if unresolved_nodes:
        if not errors:
            errors.append(
                "AWS Mac instance cleanup is unresolved; retry before removing "
                "campaign access resources"
            )
        state["phase"] = "cleanup-failed"
        state["cleanup_errors"] = errors
        _save_state(run_dir, state)
        return errors

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
    state_version = state.get("state_version")
    if not isinstance(state_version, int) or state_version < 2:
        if revoke_ssh:
            group_id = state.get("security_group_id", "")
            ssh_cidr = state.get("ssh_cidr", "")
            try:
                rule_exists = _security_group_has_ssh_rule(
                    context,
                    group_id,
                    ssh_cidr,
                )
            except AwsCommandError as error:
                if _missing_cleanup_resource(error):
                    rule_exists = False
                else:
                    errors.append(f"inspect legacy shared SSH rule: {error}")
                    rule_exists = None
            except Exception as error:
                errors.append(f"inspect legacy shared SSH rule: {error}")
                rule_exists = None
            if rule_exists:
                errors.append(
                    "legacy shared SSH rule was not revoked automatically; "
                    f"remove {ssh_cidr} from {group_id} only when safe, then "
                    "rerun terminate"
                )
            elif rule_exists is False:
                state["ssh_authorized"] = False
                state["ssh_authorize_started"] = False
                _save_state(run_dir, state)
    else:
        group_id = state.get("ssh_security_group_id", "")
        try:
            if (
                not group_id
                and state.get("ssh_security_group_create_started")
                and state.get("ssh_security_group_owned") is not False
            ):
                group_id = _recover_ssh_security_group(context, state)
                if not group_id:
                    raise RuntimeError(
                        "AWS Mac SSH security group creation outcome is still "
                        "unknown; retry cleanup after EC2 discovery catches up"
                    )
                state["ssh_security_group_id"] = group_id
                state["ssh_security_group_owned"] = True
                _save_state(run_dir, state)
        except Exception as error:
            errors.append(f"recover SSH security group: {error}")

        if group_id and state.get("ssh_security_group_owned") is not True:
            errors.append(
                f"SSH security group {group_id} is not recorded as campaign-owned"
            )
        elif group_id:
            revoke_error = ""
            if revoke_ssh:
                try:
                    _revoke_ssh(context, state)
                except AwsCommandError as error:
                    if not _missing_cleanup_resource(error):
                        revoke_error = f"revoke SSH ingress: {error}"
                except Exception as error:
                    revoke_error = f"revoke SSH ingress: {error}"
                else:
                    state["ssh_authorized"] = False
                    state["ssh_authorize_started"] = False
                    _save_state(run_dir, state)

            group_deleted = False
            try:
                _delete_ssh_security_group(context, group_id)
                group_deleted = True
            except AwsCommandError as error:
                if _missing_cleanup_resource(error):
                    group_deleted = True
                else:
                    if revoke_error:
                        errors.append(revoke_error)
                    errors.append(f"delete SSH security group: {error}")
            except Exception as error:
                if revoke_error:
                    errors.append(revoke_error)
                errors.append(f"delete SSH security group: {error}")
            if group_deleted:
                state["ssh_authorized"] = False
                state["ssh_authorize_started"] = False
                state["ssh_security_group_id"] = ""
                state["ssh_security_group_create_started"] = False
                state["ssh_security_group_egress_hardened"] = False
                state["ssh_security_group_egress_hardening_started"] = False
                state["ssh_security_group_owned"] = False
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
    if state.get("ssh_security_group_egress_hardened") is not True:
        raise RuntimeError(
            "AWS Mac SSH security group egress was not verified before launch"
        )
    existing_tokens = {node["client_token"] for node in state["nodes"]}
    client_token = uuid.uuid4().hex
    while client_token in existing_tokens:
        client_token = uuid.uuid4().hex
    node = {
        **asdict(host),
        "client_token": client_token,
        "instance_id": "",
        "instance_ownership": "created",
        "public_ip": "",
    }
    state["nodes"].append(node)
    _save_state(run_dir, state)
    try:
        launched = _run_instance(
            args,
            plan,
            host,
            key_name,
            client_token,
            ssh_security_group_id=state["ssh_security_group_id"],
        )
    except AwsCommandError as error:
        if "An error occurred (" in str(error):
            node["launch_failed"] = True
            _save_state(run_dir, state)
        raise
    node.update(launched)
    _save_state(run_dir, state)
    return node


def _copy_adoption_access(plan: AwsMacPlan, run_dir: Path) -> None:
    if plan.private_key_path is None or plan.known_hosts_path is None:
        raise RuntimeError("AWS Mac adoption plan has no external SSH access")
    for source, name in (
        (plan.private_key_path, "id_ed25519"),
        (plan.known_hosts_path, "known_hosts"),
    ):
        destination = run_dir / name
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"AWS Mac access destination already exists: {destination}")
        shutil.copyfile(source, destination, follow_symlinks=False)
        destination.chmod(0o600)


def _record_adopted_nodes(
    run_dir: Path,
    state: dict,
    hosts: tuple[AwsMacHost, ...],
    ownership_token: str,
) -> list[dict]:
    if not OWNERSHIP_TOKEN.fullmatch(ownership_token):
        raise RuntimeError("AWS Mac reuse ownership token is invalid")
    nodes = [
        {
            **asdict(host),
            "client_token": "",
            "instance_ownership": "adopted",
            "runtime_ownership_token": ownership_token,
            "termination_confirmed": False,
        }
        for host in hosts
    ]
    state["nodes"] = nodes
    _save_state(run_dir, state)
    return nodes


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
    if node.get("instance_ownership") == "adopted":
        _validate_adopted_instance_snapshot(
            node,
            instance,
            args.aws_instance_type or "mac-m4pro.metal",
            expected_vpc_id=plan.vpc_id,
            require_public_ip_match=True,
        )
        _wait_ssh(run_dir, node, args.aws_ready_timeout_s)

        def run(command: str, *, check: bool = True):
            return _ssh(run_dir, node, command, timeout=30, check=check)

        _verify_adopted_guest(node, run, args.tag)
    else:
        node["public_ip"] = instance["PublicIpAddress"]
        _save_state(run_dir, state)
        _authorize_ssh_host(
            plan.context,
            run_dir,
            node,
            timeout_s=args.aws_ready_timeout_s,
        )


def _create_fresh_access(
    args,
    plan: AwsMacPlan,
    run_dir: Path,
    state: dict,
) -> str:
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

    ssh_security_group_name = _ssh_security_group_name(args.tag)
    state["ssh_security_group_name"] = ssh_security_group_name
    state["ssh_security_group_create_started"] = True
    _save_state(run_dir, state)
    try:
        ssh_security_group_id = _create_ssh_security_group(
            plan,
            name=ssh_security_group_name,
            tag=args.tag,
        )
    except AwsCommandError as error:
        if "An error occurred (" in str(error):
            state["ssh_security_group_create_started"] = False
            state["ssh_security_group_owned"] = False
            _save_state(run_dir, state)
        raise
    state["ssh_security_group_id"] = ssh_security_group_id
    state["ssh_security_group_owned"] = True
    _save_state(run_dir, state)

    state["ssh_security_group_egress_hardening_started"] = True
    _save_state(run_dir, state)
    _harden_ssh_security_group(plan.context, ssh_security_group_id)
    state["ssh_security_group_egress_hardened"] = True
    _save_state(run_dir, state)

    state["ssh_authorize_started"] = True
    _save_state(run_dir, state)
    try:
        _authorize_ssh(
            plan.context,
            ssh_security_group_id,
            plan.ssh_cidr,
        )
    except AwsCommandError as error:
        if "InvalidPermission.Duplicate" in str(error):
            state["ssh_authorize_started"] = False
            state["ssh_authorized"] = True
            _save_state(run_dir, state)
        else:
            raise
    else:
        state["ssh_authorized"] = True
        _save_state(run_dir, state)
    return key_name


def launch_aws_mac(
    args,
    role_specs: list[RoleSpec],
    plan: AwsMacPlan | None = None,
    *,
    before_start: Callable[[], None] | None = None,
) -> None:
    """Create or adopt one Mac instance per student and start native roles."""
    if args.dry_run:
        students = _student_specs(role_specs)
        if getattr(args, "aws_mac_bootstrap_mode", "fresh") == "reuse":
            hosts = _load_adoption_manifest(args.aws_mac_nodes_path, students).hosts
        else:
            hosts = _csv(args.aws_mac_host_ids)
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
    runtime_ownership_token = (
        uuid.uuid4().hex if plan.bootstrap_mode == "reuse" else ""
    )
    state = {
        "account_id": plan.account_id,
        "ami_id": plan.ami_id,
        "backend": AWS_MAC_STATE_BACKEND,
        "bootstrap_mode": plan.bootstrap_mode,
        "created_at": int(time.time()),
        "instance_type": args.aws_instance_type or "mac-m4pro.metal",
        "key_name": "",
        "key_create_started": False,
        "key_owned": False if plan.bootstrap_mode == "reuse" else None,
        "nodes": [],
        "phase": "creating",
        "profile": plan.context.profile,
        "region": plan.context.region,
        "runtime_ownership_token": runtime_ownership_token,
        "security_group_id": plan.security_group_id,
        "ssh_security_group_create_started": False,
        "ssh_security_group_egress_hardened": False,
        "ssh_security_group_egress_hardening_started": False,
        "ssh_security_group_id": "",
        "ssh_security_group_name": "",
        "ssh_security_group_owned": (
            False if plan.bootstrap_mode == "reuse" else None
        ),
        "ssh_authorize_started": False,
        "ssh_authorized": False if plan.bootstrap_mode == "reuse" else None,
        "ssh_cidr": plan.ssh_cidr,
        "state_version": AWS_MAC_STATE_VERSION,
        "tag": args.tag,
        "vpc_id": plan.vpc_id,
        "volume_gib": plan.volume_gib,
    }
    _save_state(run_dir, state)
    try:
        group_by_host = {host.host_id: specs for host, specs in groups}
        if plan.bootstrap_mode == "reuse":
            state["phase"] = "adopting-fleet"
            _save_state(run_dir, state)
            _copy_adoption_access(plan, run_dir)
            canary, *remaining = _record_adopted_nodes(
                run_dir,
                state,
                plan.hosts,
                runtime_ownership_token,
            )
        else:
            key_name = _create_fresh_access(args, plan, run_dir, state)
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
            remaining = []

        state["phase"] = "waiting-for-canary"
        _save_state(run_dir, state)
        _wait_recorded_instance(args, plan, run_dir, state, canary)

        state["phase"] = "preparing-canary"
        _save_state(run_dir, state)
        if plan.bootstrap_mode == "reuse":
            archive = None
            remove_archive = False
        else:
            archive, remove_archive = _xcode_archive(args, run_dir)
        print(
            f"Preparing {canary['instance_id']} as the infrastructure canary "
            "before preparing the remaining fleet.",
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
        if plan.bootstrap_mode == "fresh":
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
        if remove_archive and archive is not None:
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
        if plan.bootstrap_mode == "reuse":
            print(
                "Cleaning up the new run; adopted instances will be preserved.",
                flush=True,
            )
        else:
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
    _, source_root, venv_root = _runtime_paths_for_state(state)
    print(f"{tag}: launcher={state.get('phase', 'unknown')} region={state['region']}")
    for node in state.get("nodes", []):
        instance_id = node.get("instance_id", "")
        if instance_id:
            instance = _instance(context, instance_id)
            phase = instance["State"]["Name"]
            if _node_ownership(node) == "adopted":
                current_host = instance.get("Placement", {}).get("HostId", "")
                current_ip = instance.get("PublicIpAddress", "")
                if current_host != node.get("host_id"):
                    raise RuntimeError(
                        f"Adopted instance {instance_id} moved away from its "
                        "recorded Dedicated Host"
                    )
                if phase == "running" and current_ip != node.get("public_ip"):
                    raise RuntimeError(
                        f"Adopted instance {instance_id} public IP changed; "
                        "authenticate and update trusted host access before SSH"
                    )
        else:
            phase = (
                "terminated"
                if node.get("termination_confirmed")
                else "unresolved"
            )
        print(
            f"  student-{node['student']}: instance={instance_id or '-'} "
            f"host={node['host_id']} state={phase} ip={node.get('public_ip', '')}"
        )
        if phase == "running" and node.get("public_ip"):
            result = _ssh(
                run_dir,
                node,
                f"{venv_root}/bin/python {source_root}/k8s/native.py "
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
    if node.get("termination_confirmed"):
        raise RuntimeError(f"AWS Mac role {role_key!r} is terminated; no logs remain")
    if not node.get("instance_id"):
        raise RuntimeError(
            f"AWS Mac role {role_key!r} has an unresolved instance; logs are "
            "unavailable until lifecycle cleanup resolves it"
        )
    if _node_ownership(node) == "adopted":
        instance = _instance(context, node["instance_id"])
        if instance.get("State", {}).get("Name") != "running":
            raise RuntimeError(
                f"AWS Mac role {role_key!r} instance is not running"
            )
        if instance.get("Placement", {}).get("HostId") != node.get("host_id"):
            raise RuntimeError(
                f"Adopted instance {node['instance_id']} moved away from its "
                "recorded Dedicated Host"
            )
        if instance.get("PublicIpAddress", "") != node.get("public_ip"):
            raise RuntimeError(
                f"Adopted instance {node['instance_id']} public IP changed; "
                "authenticate and update trusted host access before SSH"
            )
    _, source_root, venv_root = _runtime_paths_for_state(state)
    result = _ssh(
        run_dir,
        node,
        f"{venv_root}/bin/python {source_root}/k8s/native.py logs "
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
    if state.get("bootstrap_mode") == "reuse":
        print(
            f"Stopped AWS Mac Senpai run {tag!r}; adopted instances and "
            "Dedicated Hosts were preserved."
        )
    else:
        print(
            f"Terminated AWS Mac Senpai run {tag!r}; existing Dedicated Hosts "
            "were not released."
        )
