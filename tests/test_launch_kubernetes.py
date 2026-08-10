import json
import re
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from launch_test_support import launch, launch_args, launch_helpers


@pytest.mark.parametrize(
    ("updates", "students"),
    [
        ({"tag": "../escape"}, ["fern"]),
        ({"tag": "Uppercase"}, ["fern"]),
        ({"tag": "bad\nkind: Secret"}, ["fern"]),
        ({"namespace": "bad_namespace"}, ["fern"]),
        ({"pvc_claim_name": "bad/claim"}, ["fern"]),
        ({"supervisor_state_pvc_claim_name": "../state"}, ["fern"]),
        ({"pvc_mount_path": "/mnt/data\n- name: injected"}, ["fern"]),
        ({}, ["bad/student"]),
        ({"tag": "x" * 50}, ["student-name"]),
    ],
)
def test_kubernetes_input_boundary_rejects_unsafe_values_before_kubectl(
    updates,
    students,
    monkeypatch,
):
    args = launch_args(**updates)
    kubectl = monkeypatch.setattr(
        launch_helpers.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("kubectl must not run"),
    )

    with pytest.raises(ValueError):
        launch.validate_kubernetes_inputs(args, students)

    assert kubectl is None


def test_configmap_values_round_trip_yaml_metacharacters():
    rendered = launch_helpers.render_configmap(
        "safe-name",
        {"research-tag": 'quote" slash\\ newline\n'},
        {
            "VALUE": 'quote" slash\\ newline\nkind: Secret',
            "PLAIN": "normal",
        },
    )

    parsed = yaml.safe_load(rendered)
    assert parsed["metadata"]["labels"]["research-tag"] == (
        'quote" slash\\ newline\n'
    )
    assert parsed["data"]["VALUE"] == 'quote" slash\\ newline\nkind: Secret'
    assert parsed["data"]["PLAIN"] == "normal"


def test_rendered_image_value_cannot_inject_a_yaml_document():
    hostile = 'ghcr.io/example/image:tag"\n---\nkind: Secret'
    args = launch_args(advisor_image=hostile)
    secret = launch_helpers.render_launch_secret(
        args.tag,
        "github",
        "exa",
        "wandb",
        openai_api_key="openai",
    )
    rendered = launch.render_advisor(
        (launch.ROOT / "k8s/advisor-deployment.yaml").read_text(),
        args.tag,
        ["fern"],
        "secret",
        secret,
        args,
    )
    documents = list(yaml.safe_load_all(rendered))

    assert len(documents) == 2
    assert documents[1]["spec"]["template"]["spec"]["containers"][0]["image"] == hostile


def test_kubectl_apply_raises_with_the_resource_and_error_detail(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(
            args=argv,
            returncode=1,
            stdout="",
            stderr="forbidden",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="advisor service.*forbidden"):
        launch_helpers.kubectl_apply(
            "kind: Service",
            "advisor service",
            kube_context="gpu-cluster",
            namespace="research",
        )

    assert captured["argv"] == [
        "kubectl",
        "--context",
        "gpu-cluster",
        "--namespace",
        "research",
        "apply",
        "-f",
        "-",
    ]
    assert captured["kwargs"]["input"] == "kind: Service"


def test_student_discovery_uses_the_requested_cluster_scope(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="fern\nfrieren\n",
            stderr="",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    names = launch_helpers.existing_student_names(
        "track-a",
        kube_context="gpu-cluster",
        namespace="research",
    )

    assert names == ["fern", "frieren"]
    assert captured["argv"][:5] == [
        "kubectl",
        "--context",
        "gpu-cluster",
        "--namespace",
        "research",
    ]
    assert "app=senpai,role=student,research-tag=track-a" in captured["argv"]


def test_advisor_discovery_uses_the_exact_campaign_scope(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="senpai-advisor-track-a\n",
            stderr="",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    names = launch_helpers.existing_advisor_deployments(
        "track-a",
        kube_context="gpu-cluster",
        namespace="research",
    )

    assert names == ["senpai-advisor-track-a"]
    assert "app=senpai,role=advisor,research-tag=track-a" in captured["argv"]


def test_role_revision_discovery_is_exact_and_reports_missing_annotations(
    monkeypatch,
):
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=(
                '{"items":['
                '{"metadata":{"name":"senpai-track-a-fern",'
                '"labels":{"student":"fern"},'
                '"annotations":{"senpai.wandb.com/source-revision":"abc"}}},'
                '{"metadata":{"name":"senpai-track-a-frieren",'
                '"labels":{"student":"frieren"}}}'
                "]}"
            ),
            stderr="",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    metadata = launch_helpers.existing_role_metadata(
        "track-a",
        "student",
        kube_context="gpu-cluster",
        namespace="research",
    )

    assert metadata == {
        "fern": {"senpai.wandb.com/source-revision": "abc"},
        "frieren": {},
    }
    assert "app=senpai,role=student,research-tag=track-a" in captured["argv"]
    assert captured["argv"][-2:] == ["-o", "json"]


def test_kubectl_default_scope_omits_an_empty_context():
    assert launch_helpers.kubectl_command("apply", "-f", "-") == [
        "kubectl",
        "--namespace",
        "default",
        "apply",
        "-f",
        "-",
    ]


def test_content_addressed_names_fit_one_kubernetes_dns_label():
    base = "senpai-supervisor-secrets-" + "long-campaign-" * 5

    first = launch_helpers.content_addressed_name(base, {"token": "release-a"})
    second = launch_helpers.content_addressed_name(base, {"token": "release-b"})

    assert len(first) <= 63
    assert re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", first)
    assert first != second
    assert first.rsplit("-", 1)[1].isalnum()
    assert len(first.rsplit("-", 1)[1]) == 12


def bypass_external_preflight(monkeypatch):
    for name, value in (
        ("resolve_github_token", "github"),
        ("resolve_anthropic_api_key", "anthropic"),
        ("resolve_openai_api_key", "openai"),
        ("resolve_exa_api_key", "exa"),
        ("resolve_wandb_api_key", "wandb"),
    ):
        monkeypatch.setattr(launch, name, lambda _path, value=value: value)
    for name in (
        "preflight_check_target_repo_access",
        "preflight_check_student_name_availability",
        "preflight_check_anthropic_api_key",
        "preflight_check_openai_api_key",
        "preflight_check_exa_api_key",
        "preflight_check_wandb_api_key",
        "preflight_check_wandb_inference",
        "ensure_advisor_branch",
        "ensure_target_repo_labels",
    ):
        monkeypatch.setattr(launch, name, lambda *_args: None)
    monkeypatch.setattr(
        launch,
        "preflight_check_target_repo_branch",
        lambda *_args: "main",
    )
    monkeypatch.setattr(
        launch,
        "existing_operational_supervisors",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        launch,
        "kubernetes_pvc_metadata",
        lambda *_args, **_kwargs: {
            "metadata": {
                "annotations": {"senpai.wandb.com/sqlite-safe": "true"}
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "volumeMode": "Filesystem",
            },
            "status": {"phase": "Bound"},
        },
    )


def bypass_supervisor_rollback_snapshot(monkeypatch):
    monkeypatch.setattr(
        launch.SupervisorRollback,
        "capture",
        lambda **_kwargs: SimpleNamespace(
            commit=lambda: None,
            manual_restore_argv=lambda: ["python", "restore"],
            restore=lambda **_kwargs: None,
        ),
    )


def supervisor_launch_args(**overrides):
    overrides.setdefault("supervisor_network_policy_enforced", True)
    overrides.setdefault(
        "supervisor_state_pvc_claim_name",
        "senpai-supervisor-state-test-track",
    )
    return launch_args(
        namespace="senpai-test-track",
        supervisor_dedicated_namespace=True,
        **overrides,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"supervisor_state_pvc_claim_name": ""}, "state_pvc_claim_name"),
        (
            {"supervisor_state_pvc_claim_name": "new-pvc"},
            "separate from --pvc_claim_name",
        ),
    ),
)
def test_supervisor_requires_dedicated_acknowledged_state_storage(
    monkeypatch, overrides, message
):
    args = supervisor_launch_args(operational_supervisor=True, **overrides)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)

    with pytest.raises(SystemExit, match=message):
        launch.main()


def test_supervisor_requires_sqlite_safe_annotation_on_the_live_state_claim(
    monkeypatch,
):
    args = supervisor_launch_args(operational_supervisor=True)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "kubernetes_pvc_metadata",
        lambda *_args, **_kwargs: {
            "metadata": {"annotations": {}},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "volumeMode": "Filesystem",
            },
            "status": {"phase": "Bound"},
        },
    )

    with pytest.raises(SystemExit, match="sqlite-safe=true"):
        launch.main()


def test_supervisor_rechecks_live_storage_and_inventory_before_mutation(
    monkeypatch,
):
    args = supervisor_launch_args(
        advisor=True,
        operational_supervisor=True,
        names="",
        n_students=0,
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    bypass_supervisor_rollback_snapshot(monkeypatch)
    checks = {"pvc": 0, "advisors": 0}

    def pvc(*_args, **_kwargs):
        checks["pvc"] += 1
        return {
            "metadata": {
                "annotations": {"senpai.wandb.com/sqlite-safe": "true"}
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "volumeMode": "Filesystem",
            },
            "status": {"phase": "Bound"},
        }

    def advisors(*_args, **_kwargs):
        checks["advisors"] += 1
        return []

    monkeypatch.setattr(launch, "kubernetes_pvc_metadata", pvc)
    monkeypatch.setattr(launch, "existing_advisor_deployments", advisors)
    monkeypatch.setattr(
        launch,
        "existing_role_metadata",
        lambda *_args, **_kwargs: {},
    )
    mutation_checks = []
    monkeypatch.setattr(
        launch,
        "ensure_advisor_branch",
        lambda *_args, **_kwargs: mutation_checks.append(dict(checks)),
    )
    monkeypatch.setattr(launch, "kubectl_apply", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        launch,
        "kubectl_rollout_status",
        lambda *_args, **_kwargs: None,
    )

    launch.main()

    assert mutation_checks == [{"pvc": 2, "advisors": 2}]


def test_supervisor_preflight_only_checks_kubernetes_storage_and_inventory(
    monkeypatch,
):
    args = supervisor_launch_args(
        advisor=True,
        operational_supervisor=True,
        names="",
        n_students=0,
        preflight_only=True,
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    checked = []
    monkeypatch.setattr(
        launch,
        "kubernetes_pvc_metadata",
        lambda *_args, **_kwargs: checked.append("pvc")
        or {
            "metadata": {
                "annotations": {"senpai.wandb.com/sqlite-safe": "true"}
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "volumeMode": "Filesystem",
            },
            "status": {"phase": "Bound"},
        },
    )
    monkeypatch.setattr(
        launch,
        "existing_advisor_deployments",
        lambda *_args, **_kwargs: checked.append("advisor-inventory") or [],
    )
    monkeypatch.setattr(
        launch,
        "existing_role_metadata",
        lambda *_args, **_kwargs: checked.append("student-inventory") or {},
    )
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: pytest.fail("preflight mutated Kubernetes"),
    )

    launch.main()

    assert checked == ["pvc", "advisor-inventory", "student-inventory"]


def compatible_existing_campaign(monkeypatch, args, students=("fern",)):
    monkeypatch.setattr(
        launch,
        "kubernetes_pvc_metadata",
        lambda *_args, **_kwargs: {
            "metadata": {
                "annotations": {"senpai.wandb.com/sqlite-safe": "true"}
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "volumeMode": "Filesystem",
            },
            "status": {"phase": "Bound"},
        },
    )
    monkeypatch.setattr(
        launch,
        "existing_advisor_deployments",
        lambda *_args, **_kwargs: [f"senpai-advisor-{args.tag}"],
    )

    def metadata(_tag, role, **_kwargs):
        if role == "advisor":
            return {
                f"senpai-advisor-{args.tag}": {
                    "senpai.wandb.com/source-revision": args.repo_revision,
                    "senpai.wandb.com/advisor-branch": args.advisor_branch,
                    "senpai.wandb.com/student-names": ",".join(students),
                    "senpai.wandb.com/management-protocol": launch.MANAGEMENT_PROTOCOL_VERSION,
                    "senpai.wandb.com/repair-protocol": launch.REPAIR_PROTOCOL_VERSION,
                }
            }
        return {
            student: {
                "senpai.wandb.com/source-revision": args.repo_revision,
                "senpai.wandb.com/advisor-branch": args.advisor_branch,
                "senpai.wandb.com/management-protocol": launch.MANAGEMENT_PROTOCOL_VERSION,
                "senpai.wandb.com/repair-protocol": launch.REPAIR_PROTOCOL_VERSION,
            }
            for student in students
        }

    monkeypatch.setattr(launch, "existing_role_metadata", metadata)


def test_supervisor_only_upgrade_rejects_retained_roles_without_repair_protocol(
    monkeypatch,
):
    args = supervisor_launch_args(
        advisor=False,
        operational_supervisor=True,
        names="",
        n_students=0,
    )
    args.supervisor_network_policy_enforced = True
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    compatible_existing_campaign(monkeypatch, args)
    original = launch.existing_role_metadata

    def missing_protocol(tag, role, **kwargs):
        records = original(tag, role, **kwargs)
        return {
            name: {
                key: value
                for key, value in record.items()
                if key != "senpai.wandb.com/repair-protocol"
            }
            for name, record in records.items()
        }

    monkeypatch.setattr(launch, "existing_role_metadata", missing_protocol)
    mutations = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: mutations.append("apply"),
    )

    with pytest.raises(SystemExit, match="repair protocol"):
        launch.main()

    assert mutations == []


def test_supervisor_requires_an_explicit_network_policy_enforcement_invariant(
    monkeypatch,
):
    args = supervisor_launch_args(operational_supervisor=True)
    args.supervisor_network_policy_enforced = False
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)

    with pytest.raises(SystemExit, match="network policy enforcement"):
        launch.main()


def successful_kubectl(calls):
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="" if "get" in argv else "applied",
            stderr="",
        )

    return run


@pytest.mark.parametrize(
    "overrides",
    (
        {"namespace": "default", "supervisor_dedicated_namespace": True},
        {"namespace": "", "supervisor_dedicated_namespace": True},
        {"namespace": "shared-research", "supervisor_dedicated_namespace": False},
    ),
)
def test_supervisor_requires_explicit_campaign_dedicated_namespace(
    monkeypatch,
    overrides,
):
    args = launch_args(operational_supervisor=True, **overrides)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)

    with pytest.raises(SystemExit, match="campaign-dedicated"):
        launch.main()


def test_incremental_supervisor_requires_one_existing_exact_tag_advisor(monkeypatch):
    args = supervisor_launch_args(advisor=False, operational_supervisor=True)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_advisor_deployments",
        lambda *_args, **_kwargs: [],
    )
    mutations = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: mutations.append("apply"),
    )

    with pytest.raises(SystemExit, match="exactly one existing advisor Deployment"):
        launch.main()

    assert mutations == []


def test_incremental_supervisor_allows_a_compatible_older_advisor_revision(
    monkeypatch,
):
    args = supervisor_launch_args(
        advisor=False,
        operational_supervisor=True,
        names="",
        n_students=0,
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    bypass_supervisor_rollback_snapshot(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_advisor_deployments",
        lambda *_args, **_kwargs: ["senpai-advisor-test-track"],
    )
    monkeypatch.setattr(
        launch,
        "existing_role_metadata",
        lambda _tag, role, **_kwargs: (
            {
                "senpai-advisor-test-track": {
                    "senpai.wandb.com/source-revision": "b" * 40,
                    "senpai.wandb.com/advisor-branch": args.advisor_branch,
                    "senpai.wandb.com/student-names": "fern",
                    "senpai.wandb.com/management-protocol": launch.MANAGEMENT_PROTOCOL_VERSION,
                    "senpai.wandb.com/repair-protocol": launch.REPAIR_PROTOCOL_VERSION,
                }
            }
            if role == "advisor"
            else {
                "fern": {
                    "senpai.wandb.com/source-revision": "b" * 40,
                    "senpai.wandb.com/advisor-branch": args.advisor_branch,
                    "senpai.wandb.com/management-protocol": launch.MANAGEMENT_PROTOCOL_VERSION,
                    "senpai.wandb.com/repair-protocol": launch.REPAIR_PROTOCOL_VERSION,
                }
            }
        ),
    )
    calls = []
    monkeypatch.setattr(launch_helpers.subprocess, "run", successful_kubectl(calls))

    launch.main()

    applied = [
        document
        for argv, kwargs in calls
        if "apply" in argv
        for document in yaml.safe_load_all(kwargs["input"])
    ]
    assert not any(
        document["kind"] == "Deployment"
        and document["metadata"]["name"].startswith("senpai-advisor-")
        for document in applied
    )
    supervisor = next(
        document
        for document in applied
        if document["kind"] == "Deployment"
        and document["metadata"]["name"] == "senpai-supervisor-test-track"
    )
    assert supervisor["metadata"]["annotations"][
        "senpai.wandb.com/management-protocol"
    ] == launch.MANAGEMENT_PROTOCOL_VERSION


def test_role_replacement_cannot_silently_strip_live_supervisor_capability(
    monkeypatch,
):
    args = supervisor_launch_args(operational_supervisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_operational_supervisors",
        lambda *_args, **_kwargs: ["senpai-supervisor-test-track"],
    )
    mutations = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: mutations.append("apply"),
    )

    with pytest.raises(SystemExit, match="--operational_supervisor.*repair capability"):
        launch.main()

    assert mutations == []


def test_incremental_supervisor_cannot_change_the_advisor_student_inventory(
    monkeypatch,
):
    args = supervisor_launch_args(advisor=False, operational_supervisor=True)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_advisor_deployments",
        lambda *_args, **_kwargs: ["senpai-advisor-test-track"],
    )

    def metadata(_tag, role, **_kwargs):
        if role == "advisor":
            return {
                "senpai-advisor-test-track": {
                    "senpai.wandb.com/source-revision": args.repo_revision,
                    "senpai.wandb.com/advisor-branch": args.advisor_branch,
                    "senpai.wandb.com/student-names": "fern",
                    "senpai.wandb.com/management-protocol": launch.MANAGEMENT_PROTOCOL_VERSION,
                    "senpai.wandb.com/repair-protocol": launch.REPAIR_PROTOCOL_VERSION,
                }
            }
        return {
            "frieren": {
                "senpai.wandb.com/source-revision": args.repo_revision,
                "senpai.wandb.com/advisor-branch": args.advisor_branch,
                "senpai.wandb.com/management-protocol": launch.MANAGEMENT_PROTOCOL_VERSION,
                "senpai.wandb.com/repair-protocol": launch.REPAIR_PROTOCOL_VERSION,
            }
        }

    monkeypatch.setattr(launch, "existing_role_metadata", metadata)
    mutations = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: mutations.append("apply"),
    )

    with pytest.raises(SystemExit, match="would change.*student inventory"):
        launch.main()

    assert mutations == []


def test_supervisor_only_launch_preserves_role_secret_and_uses_immutable_bundle(
    monkeypatch,
    capsys,
):
    args = supervisor_launch_args(
        advisor=False,
        operational_supervisor=True,
        names="",
        n_students=0,
        advisor_model="openai/gpt-5.6-sol",
        smart_model="anthropic/claude-opus-4-8",
        fast_model="anthropic/claude-sonnet-4-6",
        frontier_model="anthropic/claude-opus-4-8",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    compatible_existing_campaign(monkeypatch, args)
    bypass_supervisor_rollback_snapshot(monkeypatch)
    calls = []
    monkeypatch.setattr(launch_helpers.subprocess, "run", successful_kubectl(calls))

    launch.main()

    applied = [kwargs["input"] for argv, kwargs in calls if "apply" in argv]
    assert yaml.safe_load(applied[0])["kind"] == "NetworkPolicy"
    documents = [
        document
        for manifest in applied
        for document in yaml.safe_load_all(manifest)
    ]
    resources = {
        (document["kind"], document["metadata"]["name"]): document
        for document in documents
    }
    assert ("Secret", "senpai-launch-secrets-test-track") not in resources
    assert not any(kind == "Deployment" and "fern" in name for kind, name in resources)

    supervisor_secret_name = next(
        name
        for kind, name in resources
        if kind == "Secret" and name.startswith("senpai-supervisor-secrets-test-track-")
    )
    supervisor_config_name = next(
        name
        for kind, name in resources
        if kind == "ConfigMap" and name.startswith("senpai-config-supervisor-test-track-")
    )
    assert resources[("Secret", supervisor_secret_name)]["immutable"] is True
    assert resources[("ConfigMap", supervisor_config_name)]["immutable"] is True
    assert set(resources[("Secret", supervisor_secret_name)]["data"]) == {
        "github-token",
        "wandb-api-key",
        "openai-api-key",
    }

    deployment = resources[("Deployment", "senpai-supervisor-test-track")]
    pod = deployment["spec"]["template"]["spec"]
    source = next(
        container
        for container in pod["initContainers"]
        if container["name"] == "source"
    )
    control = next(
        container
        for container in pod["containers"]
        if container["name"] == "supervisor-control"
    )
    assert source["env"][0]["valueFrom"]["secretKeyRef"]["name"] == (
        supervisor_secret_name
    )
    assert {
        env["valueFrom"]["secretKeyRef"]["name"]
        for env in control["env"]
        if "valueFrom" in env
    } == {supervisor_secret_name}
    assert control["envFrom"][0]["configMapRef"]["name"] == (
        supervisor_config_name
    )

    assert [
        argv
        for argv, _kwargs in calls
        if "rollout" in argv and "status" in argv
    ] == [
        [
            "kubectl",
            "--namespace",
            "senpai-test-track",
            "rollout",
            "status",
            "deployment/senpai-supervisor-test-track",
            "--timeout=900s",
        ]
    ]
    output = capsys.readouterr().out
    assert "Disable only the operational supervisor" in output
    assert (
        "kubectl --namespace senpai-test-track delete "
        "deployment/senpai-supervisor-test-track"
    ) in output
    assert "advisor/student pods and host capacity stay running" in output


def test_supervisor_only_upgrade_skips_role_credentials_and_repo_setup(monkeypatch):
    """An incremental supervisor upgrade must not depend on or mutate role setup."""

    args = supervisor_launch_args(
        advisor=False,
        operational_supervisor=True,
        names="",
        n_students=0,
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    compatible_existing_campaign(monkeypatch, args)
    bypass_supervisor_rollback_snapshot(monkeypatch)

    resolved = []
    checked = []
    monkeypatch.setattr(
        launch,
        "resolve_github_token",
        lambda _path: resolved.append("github") or "github",
    )
    monkeypatch.setattr(
        launch,
        "resolve_openai_api_key",
        lambda _path: resolved.append("openai") or "openai",
    )
    monkeypatch.setattr(
        launch,
        "resolve_wandb_api_key",
        lambda _path: resolved.append("wandb") or "wandb",
    )
    for name in ("resolve_exa_api_key", "resolve_optional_secret"):
        monkeypatch.setattr(
            launch,
            name,
            lambda *_args, name=name: pytest.fail(
                f"supervisor-only upgrade called {name}"
            ),
        )

    monkeypatch.setattr(
        launch,
        "preflight_check_target_repo_access",
        lambda *_args: checked.append("github"),
    )
    monkeypatch.setattr(
        launch,
        "preflight_check_openai_api_key",
        lambda *_args: checked.append("openai"),
    )
    monkeypatch.setattr(
        launch,
        "preflight_check_wandb_api_key",
        lambda *_args: checked.append("wandb"),
    )
    for name in (
        "preflight_check_target_repo_branch",
        "preflight_check_student_name_availability",
        "preflight_check_exa_api_key",
        "ensure_advisor_branch",
        "ensure_target_repo_labels",
    ):
        monkeypatch.setattr(
            launch,
            name,
            lambda *_args, name=name: pytest.fail(
                f"supervisor-only upgrade called {name}"
            ),
        )
    monkeypatch.setattr(launch, "kubectl_apply", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        launch,
        "kubectl_rollout_status",
        lambda *_args, **_kwargs: None,
    )

    launch.main()

    assert resolved == ["github", "openai", "wandb"]
    assert checked == ["github", "openai", "wandb"]


def test_supervisor_bundle_names_change_without_overwriting_prior_release(
    monkeypatch,
):
    args = supervisor_launch_args(
        advisor=False,
        operational_supervisor=True,
        names="",
        n_students=0,
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    compatible_existing_campaign(monkeypatch, args)
    bypass_supervisor_rollback_snapshot(monkeypatch)
    model_key = {"value": "openai-release-a"}
    monkeypatch.setattr(
        launch,
        "resolve_openai_api_key",
        lambda _path: model_key["value"],
    )
    calls = []
    monkeypatch.setattr(launch_helpers.subprocess, "run", successful_kubectl(calls))

    launch.main()
    first_count = len(calls)
    model_key["value"] = "openai-release-b"
    launch.main()

    def applied_documents(records):
        return [
            document
            for argv, kwargs in records
            if "apply" in argv
            for document in yaml.safe_load_all(kwargs["input"])
        ]

    def bundle_names(documents):
        return {
            document["metadata"]["name"]
            for document in documents
            if document["kind"] in {"Secret", "ConfigMap"}
        }

    def deployment_bundle(documents):
        deployment = next(
            document for document in documents if document["kind"] == "Deployment"
        )
        pod = deployment["spec"]["template"]["spec"]
        return {
            pod["containers"][0]["env"][0]["valueFrom"]["secretKeyRef"]["name"],
            pod["containers"][0]["envFrom"][0]["configMapRef"]["name"],
        }

    first_documents = applied_documents(calls[:first_count])
    second_documents = applied_documents(calls[first_count:])
    first = bundle_names(first_documents)
    second = bundle_names(second_documents)
    assert next(name for name in first if "supervisor-secrets" in name) != next(
        name for name in second if "supervisor-secrets" in name
    )
    assert next(name for name in first if "config-supervisor" in name) == next(
        name for name in second if "config-supervisor" in name
    )
    assert deployment_bundle(first_documents) == first
    assert deployment_bundle(second_documents) == second
    assert not any("delete" in argv for argv, _kwargs in calls)


def test_supervisor_rollout_failure_restores_the_exact_previous_release(
    monkeypatch,
    capsys,
    tmp_path,
):
    args = supervisor_launch_args(
        advisor=False,
        operational_supervisor=True,
        names="",
        n_students=0,
        kube_context="gpu-cluster",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    compatible_existing_campaign(monkeypatch, args)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    rollout_attempts = 0
    previous = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "senpai-supervisor-test-track",
            "namespace": "senpai-test-track",
            "resourceVersion": "before-upgrade",
        },
        "spec": {"replicas": 1},
    }

    def run(argv, **kwargs):
        nonlocal rollout_attempts
        if "get" in argv:
            if "deployment.apps/senpai-supervisor-test-track" in argv:
                current = dict(previous)
                current["metadata"] = dict(previous["metadata"])
                current["metadata"]["resourceVersion"] = "current-version"
                return subprocess.CompletedProcess(
                    args=argv,
                    returncode=0,
                    stdout=json.dumps(current),
                    stderr="",
                )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if "rollout" in argv and "status" in argv:
            rollout_attempts += 1
            if rollout_attempts > 1:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout="restored",
                    stderr="",
                )
            return subprocess.CompletedProcess(
                args=argv,
                returncode=1,
                stdout="",
                stderr="deployment exceeded its progress deadline",
            )
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="applied",
            stderr="",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    with pytest.raises(SystemExit, match="operational supervisor rollout failed"):
        launch.main()

    output = capsys.readouterr()
    assert "deployment exceeded its progress deadline" in output.err
    assert "Automatic rollback restored the prior mutable resources" in output.err
    assert "persistent SQLite state was never rolled back" in output.err
    assert "Rollback bundle retained at" in output.err
    assert "Manual recovery:" in output.err
    assert rollout_attempts == 2
    assert list(tmp_path.rglob("*.json"))


def test_supervisor_apply_failure_removes_every_new_mutable_resource(
    monkeypatch,
    capsys,
    tmp_path,
):
    args = supervisor_launch_args(
        advisor=False,
        operational_supervisor=True,
        names="",
        n_students=0,
        kube_context="gpu-cluster",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    compatible_existing_campaign(monkeypatch, args)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "get" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if (
            "apply" in argv
            and "kind: ConfigMap" in kwargs.get("input", "")
            and "kind: Deployment" in kwargs.get("input", "")
        ):
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr="admission denied the supervisor Deployment",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="restored", stderr="")

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    with pytest.raises(SystemExit, match="operational supervisor launch failed"):
        launch.main()

    deleted = [argv for argv, _kwargs in calls if "delete" in argv]
    first_apply = next(
        index
        for index, (argv, _kwargs) in enumerate(calls)
        if "apply" in argv
    )
    assert all("get" in argv for argv, _kwargs in calls[:first_apply])
    assert first_apply == 5
    assert [argv[argv.index("delete") + 1] for argv in deleted] == [
        "networkpolicy.networking.k8s.io/senpai-supervisor-egress-test-track",
        "serviceaccount/senpai-supervisor-test-track",
        "role.rbac.authorization.k8s.io/senpai-supervisor-test-track",
        "rolebinding.rbac.authorization.k8s.io/senpai-supervisor-test-track",
        "deployment.apps/senpai-supervisor-test-track",
    ]
    assert all(
        argv[:5]
        == [
            "kubectl",
            "--context",
            "gpu-cluster",
            "--namespace",
            "senpai-test-track",
        ]
        for argv in deleted
    )
    output = capsys.readouterr()
    assert "admission denied the supervisor Deployment" in output.err
    assert "persistent SQLite state was never rolled back" in output.err
    assert list(tmp_path.rglob("*.json"))


def test_successful_supervisor_rollout_removes_the_rollback_bundle(
    monkeypatch,
    tmp_path,
):
    args = supervisor_launch_args(
        advisor=False,
        operational_supervisor=True,
        names="",
        n_students=0,
        kube_context="gpu-cluster",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    compatible_existing_campaign(monkeypatch, args)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="" if "get" in argv else "ready",
            stderr="",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    launch.main()

    assert not list(tmp_path.rglob("*.json"))
    first_apply = next(
        index for index, (argv, _kwargs) in enumerate(calls) if "apply" in argv
    )
    assert first_apply == 5
    assert all("get" in argv for argv, _kwargs in calls[:first_apply])


def test_supervisor_rejects_an_extra_exact_tag_advisor(monkeypatch):
    args = supervisor_launch_args(advisor=True, operational_supervisor=True)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_advisor_deployments",
        lambda *_args, **_kwargs: ["senpai-advisor-test-track", "stray-advisor"],
    )
    mutations = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: mutations.append("apply"),
    )

    with pytest.raises(SystemExit, match="remain alongside"):
        launch.main()

    assert mutations == []


def test_supervisor_rejects_unreplaced_students_without_compatible_protocols(
    monkeypatch,
):
    args = supervisor_launch_args(advisor=True, operational_supervisor=True)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_advisor_deployments",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        launch,
        "existing_role_metadata",
        lambda _tag, role, **_kwargs: (
            {
                "legacy-student": {
                    "senpai.wandb.com/source-revision": "b" * 40,
                    "senpai.wandb.com/advisor-branch": args.advisor_branch,
                }
            }
            if role == "student"
            else {}
        ),
    )
    mutations = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: mutations.append("apply"),
    )

    with pytest.raises(SystemExit, match="legacy-student"):
        launch.main()

    assert mutations == []


def test_wandb_gateway_uses_the_wandb_key_for_openai_compatible_inference(
    monkeypatch,
):
    model = "wandb/zai-org/GLM-5.2"
    args = launch_args(
        advisor_model=model,
        advisor_reasoning_effort="max",
        student_model=model,
        student_reasoning_effort="max",
        smart_model=model,
        smart_reasoning_effort="max",
        fast_model=model,
        fast_reasoning_effort="max",
        frontier_model=model,
        frontier_reasoning_effort="max",
        wandb_entity="research-team",
        wandb_project="mlxfast",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(launch, "resolve_wandb_api_key", lambda _path: "wandb-key")
    monkeypatch.setattr(
        launch,
        "resolve_openai_api_key",
        lambda _path: pytest.fail("W&B inference must not resolve an OpenAI key"),
    )
    checked = []
    monkeypatch.setattr(
        launch,
        "preflight_check_wandb_inference",
        lambda key, entity, project: checked.append((key, entity, project)),
    )
    monkeypatch.setattr(launch, "kubectl_apply", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        launch,
        "existing_student_names",
        lambda *_args, **_kwargs: [],
    )

    launch.main()

    assert checked == [("wandb-key", "research-team", "mlxfast")]


@pytest.mark.parametrize(
    ("model", "expected_provider"),
    [
        ("anthropic/claude-opus-4-8", "anthropic"),
        ("openai/gpt-5.6-sol", "openai"),
    ],
)
def test_launch_resolves_and_preflights_only_referenced_model_providers(
    monkeypatch, model, expected_provider
):
    args = launch_args(
        advisor=False,
        advisor_model=model,
        student_model=model,
        smart_model=model,
        fast_model=model,
        frontier_model=model,
        frontier_reasoning_effort="xhigh",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    resolved = []
    checked = []

    for provider in ("anthropic", "openai"):
        monkeypatch.setattr(
            launch,
            f"resolve_{provider}_api_key",
            lambda _path, provider=provider: resolved.append(provider)
            or f"{provider}-key",
        )
        monkeypatch.setattr(
            launch,
            f"preflight_check_{provider}_api_key",
            lambda _key, provider=provider: checked.append(provider),
        )
    monkeypatch.setattr(launch, "kubectl_apply", lambda *_args, **_kwargs: None)

    launch.main()

    assert resolved == [expected_provider]
    assert checked == [expected_provider]


def test_students_only_launch_ignores_the_inactive_advisor_provider(
    monkeypatch,
):
    args = launch_args(
        advisor=False,
        advisor_model="openai/gpt-5.6-sol",
        advisor_reasoning_effort="max",
        student_model="anthropic/claude-opus-4-8",
        smart_model="anthropic/claude-opus-4-8",
        fast_model="anthropic/claude-haiku-4-5",
        frontier_model="anthropic/claude-opus-4-8",
        frontier_reasoning_effort="xhigh",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    resolved = []

    for provider in ("anthropic", "openai"):
        monkeypatch.setattr(
            launch,
            f"resolve_{provider}_api_key",
            lambda _path, provider=provider: resolved.append(provider)
            or f"{provider}-key",
        )
    monkeypatch.setattr(launch, "kubectl_apply", lambda *_args, **_kwargs: None)

    launch.main()

    assert resolved == ["anthropic"]


def test_advisor_only_launch_ignores_the_inactive_student_provider(monkeypatch):
    args = launch_args(
        names="",
        n_students=0,
        advisor=True,
        student_image="",
        advisor_model="anthropic/claude-opus-4-8",
        student_model="openai/gpt-5.6-sol",
        student_reasoning_effort="max",
        smart_model="anthropic/claude-opus-4-8",
        fast_model="anthropic/claude-haiku-4-5",
        frontier_model="anthropic/claude-opus-4-8",
        frontier_reasoning_effort="xhigh",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(launch, "existing_student_names", lambda *_args, **_kwargs: [])
    resolved = []

    for provider in ("anthropic", "openai"):
        monkeypatch.setattr(
            launch,
            f"resolve_{provider}_api_key",
            lambda _path, provider=provider: resolved.append(provider)
            or f"{provider}-key",
        )
    monkeypatch.setattr(launch, "kubectl_apply", lambda *_args, **_kwargs: None)

    launch.main()

    assert resolved == ["anthropic"]


def test_launch_uses_one_scope_for_apply_discovery_and_handoff_commands(
    monkeypatch,
    capsys,
):
    args = launch_args(
        tag="scope-test",
        kube_context="gpu-cluster",
        namespace="research",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)

    discovery = []

    def existing(tag, *, kube_context, namespace):
        discovery.append((tag, kube_context, namespace))
        return []

    monkeypatch.setattr(launch, "existing_student_names", existing)
    applies = []

    def apply(_manifest, name, *, kube_context, namespace):
        applies.append((name, kube_context, namespace))

    monkeypatch.setattr(launch, "kubectl_apply", apply)

    launch.main()

    assert discovery == [("scope-test", "gpu-cluster", "research")]
    assert applies == [
        ("secret senpai-launch-secrets-scope-test", "gpu-cluster", "research"),
        ("student fern", "gpu-cluster", "research"),
        ("advisor", "gpu-cluster", "research"),
    ]
    prefix = "kubectl --context gpu-cluster --namespace research"
    handoff_commands = [
        line.strip()
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("kubectl ")
    ]
    assert len(handoff_commands) == 4
    assert all(command.startswith(prefix) for command in handoff_commands)


def test_assignment_collision_stops_before_launch_mutation(monkeypatch):
    args = launch_args(student_prefix="acceptance")
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)

    checked = []

    def reject(_repo, _token, students, advisor_branch):
        checked.extend(students)
        assert advisor_branch == "schmidhuber"
        raise SystemExit("active assignment")

    monkeypatch.setattr(launch, "preflight_check_student_name_availability", reject)
    mutations = []
    monkeypatch.setattr(
        launch,
        "ensure_advisor_branch",
        lambda *_args: mutations.append("branch"),
    )
    monkeypatch.setattr(
        launch,
        "ensure_target_repo_labels",
        lambda *_args: mutations.append("labels"),
    )
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: mutations.append("kubernetes"),
    )

    with pytest.raises(SystemExit, match="active assignment"):
        launch.main()

    assert checked == ["acceptance-fern"]
    assert mutations == []
