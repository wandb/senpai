"""Render one credential-isolated capacity observer and its snapshot ConfigMap."""

import hashlib
import json

import yaml


def render_capacity_observer(
    *,
    tag: str,
    namespace: str,
    image: str,
    revision: str,
    config: dict,
    node_selector: dict[str, str],
) -> str:
    name = f"senpai-capacity-{tag}"
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
