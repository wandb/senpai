# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Render the credentialed executor sidecar and training RBAC for one student.

Students that train in separate worker pods never hold Kubernetes credentials.
A sidecar owns the service-account token and brokers workload requests over a
Unix socket shared with the student container.
"""

import yaml

from launch_helpers import yaml_list_insertion

EXECUTOR_SOCKET_DIR = "/var/run/senpai-kubernetes"
EXECUTOR_SOCKET_PATH = f"{EXECUTOR_SOCKET_DIR}/executor.sock"
EXECUTOR_STATE_PATH = "/var/lib/senpai-executor/reservation.json"


def training_access_name(tag: str, student_name: str) -> str:
    """ServiceAccount, Role, and RoleBinding name for one student's training."""
    return f"senpai-training-{tag}-{student_name}"


def executor_socket_mount() -> str:
    return yaml_list_insertion(
        [{"name": "executor-socket", "mountPath": EXECUTOR_SOCKET_DIR}], 8
    )


def executor_container(image: str, secret_name: str, configmap_name: str) -> str:
    return yaml_list_insertion(
        [
            {
                "name": "kubernetes-executor",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "envFrom": [{"configMapRef": {"name": configmap_name}}],
                "env": [
                    {"name": "SENPAI_LAUNCH_SECRET_NAME", "value": secret_name},
                    {
                        "name": "SENPAI_POD_NAME",
                        "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                    },
                    {
                        "name": "SENPAI_POD_UID",
                        "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
                    },
                ],
                "resources": {
                    "requests": {"cpu": "250m", "memory": "256Mi"},
                    "limits": {"cpu": "1", "memory": "1Gi"},
                },
                "readinessProbe": {
                    "exec": {
                        "command": [
                            "/bin/sh",
                            "-c",
                            'test -S "$SENPAI_KUBERNETES_EXECUTOR_SOCKET"',
                        ]
                    },
                    "periodSeconds": 2,
                    "timeoutSeconds": 1,
                    "failureThreshold": 60,
                },
                "volumeMounts": [
                    {"name": "executor-socket", "mountPath": EXECUTOR_SOCKET_DIR},
                    {"name": "executor-state", "mountPath": "/var/lib/senpai-executor"},
                    {
                        "name": "executor-token",
                        "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount",
                        "readOnly": True,
                    },
                ],
            }
        ],
        6,
    )


def executor_volumes() -> str:
    return yaml_list_insertion(
        [
            {"name": "executor-socket", "emptyDir": {}},
            {"name": "executor-state", "emptyDir": {}},
            {
                "name": "executor-token",
                "projected": {
                    "defaultMode": 0o440,
                    "sources": [
                        {
                            "serviceAccountToken": {
                                "path": "token",
                                "expirationSeconds": 3600,
                            }
                        },
                        {
                            "configMap": {
                                "name": "kube-root-ca.crt",
                                "items": [{"key": "ca.crt", "path": "ca.crt"}],
                            }
                        },
                    ],
                },
            },
        ],
        6,
    )


def student_training_access(student_name: str, tag: str, namespace: str) -> str:
    """ServiceAccount, Role, and RoleBinding that scope one student's workloads."""
    name = training_access_name(tag, student_name)
    metadata = {
        "name": name,
        "namespace": namespace,
        "labels": {"app": "senpai", "role": "student", "research-tag": tag},
    }
    documents = [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata,
            "automountServiceAccountToken": False,
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata,
            "rules": [
                {
                    "apiGroups": ["batch"],
                    "resources": ["jobs"],
                    "verbs": ["create", "get", "patch", "delete"],
                },
                {
                    "apiGroups": ["kubeflow.org"],
                    "resources": ["mpijobs"],
                    "verbs": ["create", "get", "patch", "delete"],
                },
                {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]},
                {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
                {"apiGroups": [""], "resources": ["events"], "verbs": ["list"]},
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": metadata,
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": name,
            },
            "subjects": [
                {"kind": "ServiceAccount", "name": name, "namespace": namespace}
            ],
        },
    ]
    return "\n---\n".join(
        yaml.safe_dump(document, sort_keys=False).rstrip() for document in documents
    )
