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


@pytest.mark.parametrize(
    ("bindings", "expected", "error"),
    [
        ([], None, None),
        ([("Deployment", "one", ""), ("Pod", "one", "Running")], "one", None),
        ([("Deployment", "one", ""), ("Pod", "old", "Failed")], "one", None),
        ([("Pod", "old", "Succeeded")], None, None),
        (
            [("Deployment", "one", ""), ("Pod", "two", "Running")],
            None,
            "different program snapshots",
        ),
        (
            [("Deployment", None, "")],
            None,
            "lacks a valid program context binding",
        ),
        ([("Pod", "one", "Terminating")], "one", None),
    ],
)
def test_program_binding_accounts_for_desired_roles_and_live_pods(
    monkeypatch, bindings, expected, error
):
    resources = []
    for kind, name, phase in bindings:
        metadata = {
            "annotations": {"senpai.wandb.com/program-context-secret": name}
        }
        resource = {
            "kind": kind, "metadata": metadata, "status": {"phase": phase}
        }
        if kind == "Deployment":
            resource["spec"] = {"template": {"metadata": metadata}}
        if phase == "Terminating":
            metadata["deletionTimestamp"] = "2026-09-25T00:00:00Z"
            resource["status"]["phase"] = "Running"
        resources.append(resource)

    def run(argv, **_kwargs):
        assert "app=senpai,research-tag=track-a" in argv
        assert argv[:5] == [
            "kubectl", "--context", "cluster", "--namespace", "research"
        ]
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"items": resources}), ""
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)
    if error:
        with pytest.raises(RuntimeError, match=error):
            launch_helpers.existing_program_context_secret(
                "track-a", kube_context="cluster", namespace="research"
            )
    else:
        assert launch_helpers.existing_program_context_secret(
            "track-a", kube_context="cluster", namespace="research"
        ) == expected


@pytest.mark.parametrize(
    "corruption",
    [None, "mutable", "tag", "role", "name", "content", "base64", "empty"],
)
def test_reused_program_secret_verifies_ownership_immutability_and_content(
    monkeypatch, corruption
):
    program = launch.ProgramSystemPrompt("program.md", "a" * 40, "Launch policy.")
    encoded = launch.encode_program_system_prompt(program)
    name, manifest = launch_helpers.render_program_context_secret(
        "track-a", encoded
    )
    document = yaml.safe_load(manifest)
    if corruption == "mutable":
        document["immutable"] = False
    elif corruption == "tag":
        document["metadata"]["labels"]["research-tag"] = "another-track"
    elif corruption == "role":
        document["metadata"]["labels"]["senpai.wandb.com/secret-role"] = (
            "credentials"
        )
    elif corruption == "name":
        document["metadata"]["name"] = "another-name"
    elif corruption == "content":
        document["data"]["program-context"] = base64.b64encode(
            b"different-payload"
        ).decode()
    elif corruption == "base64":
        document["data"]["program-context"] = "%%%"
    elif corruption == "empty":
        document["data"]["program-context"] = ""
    monkeypatch.setattr(
        launch_helpers.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 0, json.dumps(document), ""
        ),
    )

    if corruption:
        with pytest.raises(RuntimeError, match="bound program context Secret"):
            launch_helpers.read_program_context_secret(name, "track-a")
    else:
        assert launch_helpers.read_program_context_secret(
            name, "track-a"
        ) == encoded


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
        launch, "existing_program_context_secret", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        launch,
        "load_launch_program_snapshot",
        lambda *_args: launch.ProgramSystemPrompt(
            "program.md", "a" * 40, "Test launch research policy."
        ),
    )


def test_incremental_launch_reuses_original_snapshot_for_both_roles(monkeypatch):
    args = launch_args()
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    bound = launch.ProgramSystemPrompt(
        "program.md", "b" * 40, "Test launch research policy."
    )
    bound_name, _manifest = launch_helpers.render_program_context_secret(
        args.tag, launch.encode_program_system_prompt(bound)
    )
    monkeypatch.setattr(
        launch, "existing_program_context_secret", lambda *_args, **_kwargs: bound_name
    )
    monkeypatch.setattr(
        launch, "read_program_context_secret",
        lambda *_args, **_kwargs: launch.encode_program_system_prompt(bound),
    )
    monkeypatch.setattr(
        launch, "existing_student_names", lambda *_args, **_kwargs: []
    )
    applied = []
    monkeypatch.setattr(
        launch, "kubectl_apply",
        lambda manifest, description, **_kwargs: applied.append((description, manifest)),
    )

    launch.main()

    assert [description for description, _manifest in applied] == [
        "secret senpai-launch-secrets-test-track",
        f"program context secret {bound_name}",
        "student fern",
        "advisor",
    ]
    for _description, manifest in applied[2:]:
        configmap, deployment = list(yaml.safe_load_all(manifest))
        assert configmap["data"]["SENPAI_PROGRAM_SOURCE_COMMIT"] == "b" * 40
        assert configmap["data"]["SENPAI_PROGRAM_CONTENT_SHA256"] == (
            bound.content_sha256
        )
        assert deployment["spec"]["template"]["metadata"]["annotations"][
            "senpai.wandb.com/program-context-secret"
        ] == bound_name


@pytest.mark.parametrize(
    "path,content",
    [
        ("program.md", "Old policy."),
        ("nested/program.md", "Test launch research policy."),
    ],
)
def test_incremental_launch_rejects_policy_or_path_drift_before_apply(
    monkeypatch, path, content
):
    args = launch_args(advisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    bound = launch.ProgramSystemPrompt(path, "b" * 40, content)
    monkeypatch.setattr(
        launch, "existing_program_context_secret",
        lambda *_args, **_kwargs: "bound-secret",
    )
    monkeypatch.setattr(
        launch, "read_program_context_secret",
        lambda *_args, **_kwargs: launch.encode_program_system_prompt(bound),
    )
    monkeypatch.setattr(
        launch, "kubectl_apply",
        lambda *_args, **_kwargs: pytest.fail("policy drift must fail before apply"),
    )

    with pytest.raises(SystemExit, match="^ERROR: program.md changed.*new tag"):
        launch.main()


@pytest.mark.parametrize("failure", ["invalid", "missing"])
def test_invalid_bound_program_reports_a_clean_launch_error(monkeypatch, failure):
    args = launch_args()
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch, "existing_program_context_secret",
        lambda *_args, **_kwargs: "bound-secret",
    )

    def read(*_args, **_kwargs):
        if failure == "missing":
            raise subprocess.CalledProcessError(
                1, ["kubectl", "get", "secret", "bound-secret"]
            )
        return "not-a-snapshot"

    monkeypatch.setattr(launch, "read_program_context_secret", read)
    monkeypatch.setattr(
        launch, "kubectl_apply",
        lambda *_args, **_kwargs: pytest.fail("invalid binding must fail before apply"),
    )

    with pytest.raises(SystemExit, match="^ERROR:"):
        launch.main()


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
    assert len(applies) == 4
    assert all(
        (context, namespace) == ("gpu-cluster", "research")
        for _description, context, namespace in applies
    )
    descriptions = [description for description, _context, _namespace in applies]
    assert descriptions[0] == "secret senpai-launch-secrets-scope-test"
    assert descriptions[1].startswith(
        "program context secret senpai-program-context-scope-test-"
    )
    assert descriptions[2:] == ["student fern", "advisor"]
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
