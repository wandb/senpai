# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Kubernetes backend for launch-scoped configuration and credentials."""

from pathlib import Path

from k8s.launch_helpers import (
    existing_deployment_names,
    kubectl_apply,
    render_configmap,
    render_launch_secret,
    render_template,
)

from .specs import RoleSpec

ROOT = Path(__file__).resolve().parents[2]
STUDENT_TEMPLATE = ROOT / "k8s" / "student-deployment.yaml"
ADVISOR_TEMPLATE = ROOT / "k8s" / "advisor-deployment.yaml"


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
