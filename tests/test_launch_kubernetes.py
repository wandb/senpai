import re
import subprocess

import pytest
import yaml

from launch_test_support import launch, launch_args, launch_helpers


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


def supervisor_launch_args(**overrides):
    return launch_args(
        namespace="senpai-test-track",
        supervisor_dedicated_namespace=True,
        **overrides,
    )


def compatible_existing_campaign(monkeypatch, args, students=("fern",)):
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
                }
            }
        return {
            student: {
                "senpai.wandb.com/source-revision": args.repo_revision,
                "senpai.wandb.com/advisor-branch": args.advisor_branch,
            }
            for student in students
        }

    monkeypatch.setattr(launch, "existing_role_metadata", metadata)


def successful_kubectl(calls):
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="applied",
            stderr="",
        )

    return run


@pytest.mark.parametrize(
    "overrides",
    (
        {"namespace": "default", "supervisor_dedicated_namespace": True},
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


def test_incremental_supervisor_requires_the_same_advisor_source_revision(
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
    monkeypatch.setattr(
        launch,
        "existing_role_metadata",
        lambda _tag, role, **_kwargs: (
            {
                "senpai-advisor-test-track": {
                    "senpai.wandb.com/source-revision": "b" * 40,
                    "senpai.wandb.com/advisor-branch": args.advisor_branch,
                    "senpai.wandb.com/student-names": "fern",
                }
            }
            if role == "advisor"
            else {}
        ),
    )
    mutations = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: mutations.append("apply"),
    )

    with pytest.raises(SystemExit, match="does not match the requested Senpai"):
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
                }
            }
        return {
            "frieren": {
                "senpai.wandb.com/source-revision": args.repo_revision,
                "senpai.wandb.com/advisor-branch": args.advisor_branch,
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
    calls = []
    monkeypatch.setattr(launch_helpers.subprocess, "run", successful_kubectl(calls))

    launch.main()

    applied = [kwargs["input"] for argv, kwargs in calls if "apply" in argv]
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
    assert pod["initContainers"][0]["env"][0]["valueFrom"]["secretKeyRef"][
        "name"
    ] == supervisor_secret_name
    assert {
        env["valueFrom"]["secretKeyRef"]["name"]
        for env in pod["containers"][0]["env"]
    } == {supervisor_secret_name}
    assert pod["containers"][0]["envFrom"][0]["configMapRef"]["name"] == (
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


def test_supervisor_rollout_failure_surfaces_exact_rollback_command(
    monkeypatch,
    capsys,
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

    def run(argv, **kwargs):
        if "rollout" in argv and "status" in argv:
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
    assert (
        "kubectl --context gpu-cluster --namespace senpai-test-track rollout undo "
        "deployment/senpai-supervisor-test-track"
    ) in output.err


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


def test_supervisor_rejects_unreplaced_students_on_an_old_revision(monkeypatch):
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
