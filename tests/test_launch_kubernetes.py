import base64
import json
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


def test_kubectl_default_scope_omits_an_empty_context():
    assert launch_helpers.kubectl_command("apply", "-f", "-") == [
        "kubectl",
        "--namespace",
        "default",
        "apply",
        "-f",
        "-",
    ]


def bypass_external_preflight(monkeypatch):
    monkeypatch.setattr(
        launch,
        "resolve_github_token",
        lambda _path, _custom_secret_env_names: "github",
    )
    for name, value in (
        ("resolve_anthropic_api_key", "anthropic"),
        ("resolve_openai_api_key", "openai"),
        ("resolve_exa_api_key", "exa"),
        ("resolve_wandb_api_key", "wandb"),
        ("resolve_wandb_inference_api_key", "wandb-inference"),
    ):
        monkeypatch.setattr(launch, name, lambda _path, value=value: value)
    monkeypatch.setattr(
        launch,
        "resolve_student_wandb_api_keys",
        lambda _path, names: {name: f"wandb-training-{name}" for name in names},
    )
    monkeypatch.setattr(
        launch,
        "preflight_check_wandb_api_key",
        lambda key: f"viewer-{key}",
    )
    for name in (
        "existing_controller_wandb_viewers",
        "existing_wandb_viewer_owners",
    ):
        monkeypatch.setattr(launch, name, lambda *_args, **_kwargs: {})
    for name in (
        "preflight_check_target_repo_access",
        "preflight_check_student_name_availability",
        "preflight_check_anthropic_api_key",
        "preflight_check_openai_api_key",
        "preflight_check_exa_api_key",
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


def test_preflight_resolves_custom_secrets(monkeypatch):
    args = launch_args(
        preflight_only=True,
        custom_secret_env_names=["HF_TOKEN", "DATASET_LICENSE_KEY"],
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    resolved = []
    monkeypatch.setattr(
        launch,
        "resolve_custom_secrets",
        lambda path, names: resolved.append((path, names)) or {},
    )

    launch.main()

    assert resolved == [
        (launch.DOTENV_PATH, ["HF_TOKEN", "DATASET_LICENSE_KEY"])
    ]


def test_launch_reports_invalid_custom_secret_names_without_a_traceback(monkeypatch):
    args = launch_args(custom_secret_env_names=["NOT-VALID"])
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)

    with pytest.raises(SystemExit, match="^ERROR: invalid custom secret"):
        launch.main()


def test_dry_run_never_reads_custom_secret_values(monkeypatch, capsys):
    args = launch_args(dry_run=True, custom_secret_env_names=["HF_TOKEN"])
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(
        launch,
        "resolve_custom_secrets",
        lambda *_args: pytest.fail("dry-run must not resolve custom secrets"),
    )

    launch.main()

    output = capsys.readouterr().out
    encoded_placeholder = base64.b64encode(b"<REDACTED_HF_TOKEN>").decode()
    assert f"HF_TOKEN: {encoded_placeholder}" in output


def test_wandb_gateway_uses_its_dedicated_key_for_openai_compatible_inference(
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
    monkeypatch.setattr(
        launch, "resolve_wandb_api_key", lambda _path: "wandb-key"
    )
    monkeypatch.setattr(
        launch,
        "resolve_openai_api_key",
        lambda _path: pytest.fail(
            "W&B inference must not resolve an OpenAI key"
        ),
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

    assert checked == [("wandb-inference", "research-team", "mlxfast")]


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
    assert applies[0][0].startswith("secret senpai-launch-secrets-scope-test-")
    assert applies == [
        (applies[0][0], "gpu-cluster", "research"),
        ("W&B writer secret for student fern", "gpu-cluster", "research"),
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


def test_preflight_rejects_wandb_keys_for_the_same_viewer(monkeypatch):
    args = launch_args(preflight_only=True)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "preflight_check_wandb_api_key",
        lambda _api_key: "shared-viewer",
    )

    with pytest.raises(
        SystemExit, match="controller.*student 'fern'.*same viewer"
    ):
        launch.main()


def test_partial_update_cannot_change_an_active_controller_viewer(monkeypatch):
    args = launch_args(advisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_controller_wandb_viewers",
        lambda *_args, **_kwargs: {"student/frieren": {"different-viewer"}},
    )
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: pytest.fail(
            "preflight must fail before mutation"
        ),
    )

    with pytest.raises(SystemExit, match="frieren.*complete fleet"):
        launch.main()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tag": "Uppercase"}, "--tag must be a lowercase"),
        ({"names": "not_valid"}, "student name must be a lowercase"),
        ({"tag": "t" * 64}, "at most 63"),
    ],
)
def test_launch_rejects_invalid_kubernetes_labels_before_preflight(
    monkeypatch, overrides, message
):
    args = launch_args(dry_run=True, **overrides)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)

    with pytest.raises(SystemExit, match=message):
        launch.main()


def role_resource(
    *,
    tag="test-track",
    role="student",
    student="fern",
    kind="Deployment",
    controller="viewer-wandb",
    writer="viewer-wandb-training-fern",
    inference="",
    phase="Running",
):
    labels = {"app": "senpai", "research-tag": tag, "role": role}
    if role == "student":
        labels["student"] = student
    annotations = {
        "senpai.wandb.com/controller-wandb-viewer": base64.b64encode(
            controller.encode()
        ).decode(),
        "senpai.wandb.com/inference-wandb-viewer": base64.b64encode(
            inference.encode()
        ).decode(),
    }
    if role == "student":
        annotations["senpai.wandb.com/wandb-viewer"] = base64.b64encode(
            writer.encode()
        ).decode()
    resource = {"kind": kind, "metadata": {"labels": labels}}
    if kind == "Deployment":
        resource["spec"] = {
            "template": {"metadata": {"annotations": annotations}}
        }
    else:
        resource["metadata"]["annotations"] = annotations
        resource["metadata"]["deletionTimestamp"] = "2026-09-25T00:00:00Z"
        resource["status"] = {"phase": phase}
    return resource


def launch_with_existing_roles(monkeypatch, resources, **overrides):
    args = launch_args(advisor=False, **overrides)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_controller_wandb_viewers",
        launch_helpers.existing_controller_wandb_viewers,
    )
    monkeypatch.setattr(
        launch,
        "existing_wandb_viewer_owners",
        launch_helpers.existing_wandb_viewer_owners,
    )
    scans = []

    def run(argv, **_kwargs):
        scans.append(argv)
        selector = argv[argv.index("-l") + 1]
        selected = resources
        if "research-tag=" in selector:
            tag = selector.split("research-tag=", 1)[1]
            selected = [
                item
                for item in resources
                if item["metadata"]["labels"].get("research-tag") == tag
            ]
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"items": selected}), ""
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)
    applied = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda manifest, _description, **_kwargs: applied.extend(
            yaml.safe_load_all(manifest)
        ),
    )
    return applied, scans


@pytest.mark.parametrize(
    "existing_owner", ["controller", "writer", "inference"]
)
@pytest.mark.parametrize("kind", ["Deployment", "Pod"])
def test_launch_rejects_viewer_reuse_across_tags_before_apply(
    monkeypatch, existing_owner, kind
):
    resource = role_resource(
        tag="older-track",
        kind=kind,
        controller="old-controller",
        writer="old-writer",
        inference="old-inference",
        **{},
    )
    annotations = (
        resource["metadata"]["annotations"]
        if kind == "Pod"
        else resource["spec"]["template"]["metadata"]["annotations"]
    )
    annotation = {
        "controller": "controller-wandb-viewer",
        "writer": "wandb-viewer",
        "inference": "inference-wandb-viewer",
    }[existing_owner]
    annotations[f"senpai.wandb.com/{annotation}"] = base64.b64encode(
        b"viewer-wandb-training-fern"
    ).decode()
    applied, scans = launch_with_existing_roles(monkeypatch, [resource])
    with pytest.raises(
        SystemExit, match="older-track.*test-track.*same viewer"
    ):
        launch.main()
    assert applied == []
    assert any("app=senpai" in command for command in scans)


@pytest.mark.parametrize("phase", ["Succeeded", "Failed"])
def test_terminal_pods_release_identity_ownership(monkeypatch, phase):
    resource = role_resource(tag="older-track", kind="Pod", phase=phase)
    applied, _scans = launch_with_existing_roles(monkeypatch, [resource])
    launch.main()
    assert any(item["kind"] == "Deployment" for item in applied)


@pytest.mark.parametrize(
    "annotation",
    ["controller-wandb-viewer", "inference-wandb-viewer", "wandb-viewer"],
)
@pytest.mark.parametrize("value", [None, "not base64!"])
def test_legacy_namespace_roles_block_launch_until_migrated(
    monkeypatch, annotation, value
):
    resource = role_resource(tag="legacy-track")
    annotations = resource["spec"]["template"]["metadata"]["annotations"]
    if value is None:
        annotations.pop(f"senpai.wandb.com/{annotation}")
    else:
        annotations[f"senpai.wandb.com/{annotation}"] = value
    applied, _scans = launch_with_existing_roles(monkeypatch, [resource])
    with pytest.raises(SystemExit, match="every active legacy role"):
        launch.main()
    assert applied == []


def test_same_student_rotation_keeps_old_live_writer_reserved_and_secret_immutable(
    monkeypatch,
):
    old_pod = role_resource(kind="Pod", writer="old-viewer")
    applied, _scans = launch_with_existing_roles(monkeypatch, [old_pod])
    launch.main()
    secrets = {
        item["metadata"]["name"]: item
        for item in applied
        if item["kind"] == "Secret"
    }
    deployment = next(item for item in applied if item["kind"] == "Deployment")
    template = deployment["spec"]["template"]
    refs = {
        item["name"]: item["valueFrom"]["secretKeyRef"]
        for item in template["spec"]["containers"][0]["env"]
    }
    writer_ref = refs["SENPAI_WANDB_TRAINING_API_KEY"]
    shared_ref = refs["WANDB_API_KEY"]
    assert writer_ref["name"] != shared_ref["name"]
    assert all(secret["immutable"] is True for secret in secrets.values())
    assert (
        base64.b64decode(
            secrets[writer_ref["name"]]["data"][writer_ref["key"]]
        ).decode()
        == "wandb-training-fern"
    )
    assert (
        base64.b64decode(
            secrets[shared_ref["name"]]["data"][shared_ref["key"]]
        ).decode()
        == "wandb"
    )
    assert (
        base64.b64decode(
            template["metadata"]["annotations"]["senpai.wandb.com/wandb-viewer"]
        ).decode()
        == "viewer-wandb-training-fern"
    )

    # During rollout, the old terminating writer remains owned by fern.
    applied.clear()
    monkeypatch.setattr(
        launch,
        "resolve_student_wandb_api_keys",
        lambda _path, _names: {"fern": "new-key"},
    )
    monkeypatch.setattr(
        launch,
        "preflight_check_wandb_api_key",
        lambda key: "old-viewer" if key == "wandb" else "new-writer",
    )
    with pytest.raises(
        SystemExit, match="student 'fern'.*controller.*same viewer"
    ):
        launch.main()
    assert applied == []


@pytest.mark.parametrize("credential_owner", ["controller", "W&B Inference"])
def test_controller_and_inference_viewers_belong_to_one_tag(
    monkeypatch, credential_owner
):
    overrides = {}
    if credential_owner == "controller":
        resource = role_resource(tag="older-track", role="advisor")
    else:
        resource = role_resource(
            tag="older-track",
            role="advisor",
            controller="older-controller",
            inference="viewer-wandb-inference",
        )
        overrides = {
            "student_model": "wandb/zai-org/GLM-5.2",
            "student_reasoning_effort": "max",
        }
    applied, _scans = launch_with_existing_roles(
        monkeypatch, [resource], **overrides
    )

    with pytest.raises(
        SystemExit,
        match=(
            f"tag 'older-track' {credential_owner} and "
            f"tag 'test-track' {credential_owner}.*same viewer"
        ),
    ):
        launch.main()

    assert applied == []


def test_relaunch_accepts_the_same_controller_inference_and_writer_owners(
    monkeypatch,
):
    resources = [
        role_resource(kind=kind, inference="viewer-wandb-inference")
        for kind in ("Deployment", "Pod")
    ]
    applied, _scans = launch_with_existing_roles(
        monkeypatch,
        resources,
        student_model="wandb/zai-org/GLM-5.2",
        student_reasoning_effort="max",
    )

    launch.main()

    assert [
        item["metadata"]["name"]
        for item in applied
        if item["kind"] == "Deployment"
    ] == ["senpai-test-track-fern"]
