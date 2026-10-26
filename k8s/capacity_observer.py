"""Render one credential-isolated capacity observer and its snapshot ConfigMap."""

import hashlib
import json

import yaml

from launch_helpers import yaml_list_insertion

CAPACITY_SNAPSHOT_DIR = "/var/run/senpai-capacity"
CAPACITY_SNAPSHOT_PATH = f"{CAPACITY_SNAPSHOT_DIR}/snapshot.json"


def capacity_configmap_name(tag: str) -> str:
    return f"senpai-capacity-{tag}"


def capacity_snapshot_mount() -> str:
    """Read-only snapshot mount for an advisor or student container."""
    return yaml_list_insertion(
        [{"name": "capacity-snapshot", "mountPath": CAPACITY_SNAPSHOT_DIR, "readOnly": True}],
        8,
    )


def capacity_snapshot_volume(tag: str) -> str:
    return yaml_list_insertion(
        [
            {
                "name": "capacity-snapshot",
                "configMap": {
                    "name": capacity_configmap_name(tag),
                    "items": [{"key": "snapshot.json", "path": "snapshot.json"}],
                },
            }
        ],
        6,
    )



def render_capacity_observer(
    *,
    tag: str,
    namespace: str,
    image: str,
    image_pull_secrets: list[str],
    revision: str,
    config: dict,
    node_selector: dict[str, str],
) -> str:
    name = capacity_configmap_name(tag)
    scope_hash = hashlib.sha256(f"{namespace}/{tag}".encode()).hexdigest()[:16]
    cluster_name = f"senpai-capacity-{scope_hash}"
    labels = {
        "app": "senpai-capacity-observer",
        "role": "capacity-observer",
        "research-tag": tag,
        "senpai.wandb.com/namespace": namespace,
    }
    metadata = {"name": name, "namespace": namespace, "labels": labels}
    subject = {"kind": "ServiceAccount", "name": name, "namespace": namespace}
    documents = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": metadata,
            "data": {"snapshot.json": "{}"},
        },
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata,
            "automountServiceAccountToken": False,
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": cluster_name, "labels": labels},
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["nodes", "pods"],
                    "verbs": ["list"],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": cluster_name, "labels": labels},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": cluster_name,
            },
            "subjects": [subject],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata,
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["configmaps"],
                    "resourceNames": [name],
                    "verbs": ["get", "update"],
                }
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
            "subjects": [subject],
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": metadata,
            "spec": {
                "replicas": 1,
                "strategy": {"type": "Recreate"},
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "serviceAccountName": name,
                        "automountServiceAccountToken": False,
                        "imagePullSecrets": [{"name": name} for name in image_pull_secrets],
                        "nodeSelector": node_selector,
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 10001,
                            "runAsGroup": 10001,
                            "fsGroup": 10001,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "containers": [
                            {
                                "name": "capacity-observer",
                                "image": image,
                                "command": ["python", "-m", "senpai_agent.cluster_capacity"],
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "readOnlyRootFilesystem": True,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "env": [
                                    {"name": "SENPAI_REPO_REVISION", "value": revision},
                                    {"name": "SENPAI_CAPACITY_NAMESPACE", "value": namespace},
                                    {"name": "SENPAI_CAPACITY_CONFIGMAP", "value": name},
                                    {"name": "SENPAI_CAPACITY_CONFIG", "value": json.dumps(config)},
                                ],
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "128Mi"},
                                    "limits": {"cpu": "1", "memory": "512Mi"},
                                },
                                "volumeMounts": [
                                    {
                                        "name": "observer-token",
                                        "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount",
                                        "readOnly": True,
                                    }
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "observer-token",
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
                            }
                        ],
                    },
                },
            },
        },
    ]
    return "\n---\n".join(
        yaml.safe_dump(document, sort_keys=False).rstrip() for document in documents
    )
