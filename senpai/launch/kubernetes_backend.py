# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Kubernetes backend for launch-scoped configuration and credentials."""

import base64
import subprocess
import sys
from pathlib import Path

from .specs import RoleSpec

ROOT = Path(__file__).resolve().parents[2]
STUDENT_TEMPLATE = ROOT / "k8s" / "student-deployment.yaml"
ADVISOR_TEMPLATE = ROOT / "k8s" / "advisor-deployment.yaml"


def _kubectl_get_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["kubectl", "get", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def existing_student_names(tag: str) -> list[str]:
    return _kubectl_get_lines(
        "deployments",
        "-l",
        f"app=senpai,role=student,research-tag={tag}",
        "-o",
        'jsonpath={range .items[*]}{.metadata.labels.student}{"\\n"}{end}',
    )


def existing_deployment_names(tag: str) -> list[str]:
    return _kubectl_get_lines(
        "deployments",
        "-l",
        f"app=senpai,research-tag={tag}",
        "-o",
        "name",
    )


def render_template(template: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    return template


def render_configmap(
    name: str, labels: dict[str, str], data: dict[str, str]
) -> str:
    lines = [
        "apiVersion: v1",
        "kind: ConfigMap",
        "metadata:",
        f"  name: {name}",
        "  labels:",
    ]
    lines.extend(f"    {key}: {value}" for key, value in labels.items())
    lines.append("data:")
    lines.extend(f'  {key}: "{value}"' for key, value in data.items())
    return "\n".join(lines)


def render_launch_secret(tag: str, secrets: dict[str, str]) -> str:
    manifest = (
        "apiVersion: v1\n"
        "kind: Secret\n"
        "metadata:\n"
        f"  name: senpai-launch-secrets-{tag}\n"
        "  labels:\n"
        "    app: senpai\n"
        f"    research-tag: {tag}\n"
        "type: Opaque\n"
        "data:\n"
    )
    for name, value in sorted(secrets.items()):
        encoded = base64.b64encode(value.encode()).decode()
        manifest += f"  {name}: {encoded}\n"
    return manifest


def kubectl_apply(manifest: str, name: str) -> None:
    print(f"Launching: {name}")
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: could not apply {name}: {result.stderr.strip()}")
    print(f"  {result.stdout.strip()}")


def render_student(args, spec: RoleSpec, secret_name: str) -> str:
    configmap_name = f"senpai-config-student-{args.tag}-{spec.name}"
    deployment_name = f"senpai-{args.tag}-{spec.name}"
    cpu = args.cpu_per_gpu * args.gpus_per_student
    memory_gi = args.memory_gi_per_gpu * args.gpus_per_student
    configmap = render_configmap(
        name=configmap_name,
        labels={
            "app": "senpai",
            "role": "student",
            "research-tag": args.tag,
        },
        data=spec.env,
    )
    deployment = render_template(
        STUDENT_TEMPLATE.read_text(),
        {
            "STUDENT_DEPLOYMENT_NAME": deployment_name,
            "STUDENT_CONFIGMAP_NAME": configmap_name,
            "STUDENT_NAME": spec.name,
            "RESEARCH_TAG": args.tag,
            "IMAGE": args.image,
            "ADVISOR_BRANCH": args.advisor_branch,
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "LAUNCH_SECRET_NAME": secret_name,
            "STUDENT_CPU": str(cpu),
            "STUDENT_MEMORY": f"{memory_gi}Gi",
            "GPUS_PER_STUDENT": str(args.gpus_per_student),
        },
    )
    return configmap + "\n---\n" + deployment


def render_advisor(args, spec: RoleSpec, secret_name: str) -> str:
    configmap_name = f"senpai-config-advisor-{args.tag}"
    deployment_name = f"senpai-advisor-{args.tag}"
    configmap = render_configmap(
        name=configmap_name,
        labels={
            "app": "senpai",
            "role": "advisor",
            "research-tag": args.tag,
        },
        data=spec.env,
    )
    deployment = render_template(
        ADVISOR_TEMPLATE.read_text(),
        {
            "ADVISOR_DEPLOYMENT_NAME": deployment_name,
            "ADVISOR_CONFIGMAP_NAME": configmap_name,
            "RESEARCH_TAG": args.tag,
            "IMAGE": args.image,
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "LAUNCH_SECRET_NAME": secret_name,
        },
    )
    return configmap + "\n---\n" + deployment


def _warn_existing_workers(tag: str, deployments: list[str]) -> None:
    if not deployments:
        return
    print(
        f"\nWARNING: research tag {tag!r} already has {len(deployments)} deployment(s)."
    )
    print(
        "Existing pods keep their current credentials until they restart; "
        "Senpai will not restart them automatically because that could interrupt "
        "long-running training."
    )
    print("When it is safe to interrupt the jobs, load the new credentials with:")
    print(f"  kubectl rollout restart deployment -l research-tag={tag}")


def launch_kubernetes(
    args, role_specs: list[RoleSpec], secrets: dict[str, str]
) -> None:
    secret_name = f"senpai-launch-secrets-{args.tag}"
    if args.dry_run:
        print(f"--- Secret: {secret_name} ---")
        print(render_launch_secret(args.tag, secrets))
        print()
    else:
        _warn_existing_workers(args.tag, existing_deployment_names(args.tag))
        kubectl_apply(
            render_launch_secret(args.tag, secrets),
            f"secret {secret_name}",
        )

    students = [spec for spec in role_specs if spec.role == "student"]
    for spec in students:
        manifest = render_student(args, spec, secret_name)
        if args.dry_run:
            print(f"--- Student: {spec.name} ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, f"student {spec.name}")

    advisor = next((spec for spec in role_specs if spec.role == "advisor"), None)
    if advisor:
        manifest = render_advisor(args, advisor, secret_name)
        if args.dry_run:
            print("--- Advisor ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, "advisor")

    if args.dry_run:
        return

    student_names = [spec.name for spec in students]
    print(f"\nLaunched {len(students)} students: {', '.join(student_names)}")
    if advisor:
        print("Launched advisor pod")
    print("\nMonitor:")
    print(f"  kubectl get deployments -l research-tag={args.tag}")
    if advisor:
        print(f"  kubectl get deployment senpai-advisor-{args.tag}")
    if students:
        print(f"  kubectl logs -f deployment/senpai-{args.tag}-{students[0].name}")
    print("\nStop:")
    print(f"  kubectl delete deployments,configmaps,secrets -l research-tag={args.tag}")
