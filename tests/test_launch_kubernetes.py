import base64
import json
import subprocess
import urllib.error

import pytest
import yaml

from git_workflow_support import commit_file, git, repository
from launch_test_support import launch, launch_args, launch_helpers


@pytest.fixture(autouse=True)
def block_unmocked_launch_writes(monkeypatch):
    def fail(*_args, **_kwargs):
        pytest.fail("launch.main test attempted an unmocked Kubernetes write")

    monkeypatch.setattr(launch, "kubectl_apply", fail)
    monkeypatch.setattr(launch, "kubectl_create", fail, raising=False)


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


@pytest.mark.parametrize("advisor_exists", [False, True])
def test_program_capture_uses_base_only_when_exact_advisor_ref_is_absent(
    tmp_path, monkeypatch, advisor_exists
):
    workspace, remote, _head = repository(tmp_path)
    base_commit = commit_file(workspace, "program.md", "Base policy.", "base policy")
    git(workspace, "push", "origin", f"{base_commit}:refs/heads/main")
    advisor_commit = commit_file(
        workspace, "program.md", "Advisor policy.", "advisor policy"
    )
    # A similarly named branch must not suppress the base-branch fallback.
    ref = "research" if advisor_exists else "research-extra"
    git(workspace, "push", "origin", f"{advisor_commit}:refs/heads/{ref}")
    monkeypatch.setattr(launch, "github_repository_url", lambda _repo: str(remote))

    snapshot = launch.load_launch_program_snapshot(
        "https://github.com/acme/widgets.git", "research", "", "github", "main"
    )

    assert snapshot.source_commit == (advisor_commit if advisor_exists else base_commit)
    assert snapshot.content == ("Advisor policy." if advisor_exists else "Base policy.")


def test_program_capture_does_not_fallback_after_a_failed_ref_lookup(monkeypatch):
    calls = []

    def fail(*_args, **_kwargs):
        calls.append(_args)
        raise subprocess.CalledProcessError(128, ["git", "ls-remote"])

    monkeypatch.setattr(launch, "run_git", fail)
    with pytest.raises(subprocess.CalledProcessError):
        launch.load_launch_program_snapshot(
            "https://github.com/acme/widgets.git", "research", "", "github", "main"
        )
    assert len(calls) == 1


@pytest.mark.parametrize("advisor_head", [None, "a" * 40, "b" * 40, 403])
def test_advisor_branch_preserves_the_captured_commit(monkeypatch, advisor_head):
    monkeypatch.setattr(
        launch_helpers, "preflight_check_target_repo_branch", lambda *_args: "main"
    )
    writes = []

    def github_api(path, _token, *, method="GET", data=None):
        if method == "POST":
            writes.append(json.loads(data))
            return {}
        if path.endswith("/branches/research"):
            if advisor_head is None or advisor_head == 403:
                code = 404 if advisor_head is None else 403
                raise urllib.error.HTTPError(path, code, "unavailable", {}, None)
            return {"commit": {"sha": advisor_head}}
        assert path.endswith("/branches/main")
        return {"commit": {"sha": "c" * 40}}

    monkeypatch.setattr(launch_helpers, "_github_api", github_api)

    def ensure():
        launch_helpers.ensure_advisor_branch(
            "https://github.com/acme/widgets.git", "github", "main", "research", "a" * 40
        )

    if advisor_head == "b" * 40:
        with pytest.raises(SystemExit, match="changed after the program snapshot"):
            ensure()
    elif advisor_head == 403:
        with pytest.raises(urllib.error.HTTPError):
            ensure()
    else:
        ensure()
    assert writes == (
        [{"ref": "refs/heads/research", "sha": "a" * 40}]
        if advisor_head is None else []
    )


def test_kubectl_create_is_create_only_and_reports_the_resource(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(
            args=argv,
            returncode=1,
            stdout="",
            stderr="already exists",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="student Deployment.*already exists"):
        launch_helpers.kubectl_create(
            "kind: Deployment",
            "student Deployment",
            kube_context="gpu-cluster",
            namespace="research",
        )

    assert captured["argv"] == [
        "kubectl",
        "--context",
        "gpu-cluster",
        "--namespace",
        "research",
        "create",
        "-f",
        "-",
    ]
    assert captured["kwargs"]["input"] == "kind: Deployment"


def test_single_node_slot_scan_does_not_query_an_absent_mpijob_api(monkeypatch):
    commands = []

    def run(argv, **_kwargs):
        commands.append(argv)
        arguments = argv[argv.index("default") + 1 :]
        if arguments[:2] == ["get", "deployment"]:
            stdout = ""
        elif arguments[:2] in (["get", "pods"], ["get", "jobs"]):
            stdout = json.dumps({"items": []})
        elif arguments[0] == "api-resources":
            stdout = "jobs.batch\n"
        else:
            pytest.fail(f"unexpected kubectl command: {arguments}")
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    launch_helpers.ensure_new_student_slot(
        "senpai-track-fern",
        tag="track",
        student_name="fern",
        require_mpijob=False,
    )

    assert not any("mpijobs" in command for command in commands)


def test_multinode_slot_scan_requires_the_mpijob_api(monkeypatch):
    def run(argv, **_kwargs):
        arguments = argv[argv.index("default") + 1 :]
        if arguments[:2] == ["get", "deployment"]:
            stdout = ""
        elif arguments[:2] in (["get", "pods"], ["get", "jobs"]):
            stdout = json.dumps({"items": []})
        elif arguments[0] == "api-resources":
            stdout = "jobs.batch\n"
        else:
            pytest.fail(f"unexpected kubectl command: {arguments}")
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="does not provide the MPIJob API"):
        launch_helpers.ensure_new_student_slot(
            "senpai-track-fern",
            tag="track",
            student_name="fern",
            require_mpijob=True,
        )


def test_single_node_slot_scan_rejects_an_orphan_mpijob_when_the_api_exists(
    monkeypatch,
):
    responses = iter(
        [
            "",
            json.dumps({"items": []}),
            json.dumps({"items": []}),
            "mpijobs.kubeflow.org\n",
            json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "active-mpi-training"},
                            "status": {
                                "conditions": [
                                    {"type": "Running", "status": "True"}
                                ]
                            },
                        }
                    ]
                }
            ),
        ]
    )

    def run(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, next(responses), "")

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="MPIJob active-mpi-training"):
        launch_helpers.ensure_new_student_slot(
            "senpai-track-fern",
            tag="track",
            student_name="fern",
            require_mpijob=False,
        )


def test_slot_scan_rejects_existing_controller_objects(monkeypatch):
    responses = iter(
        [
            "",
            json.dumps({"items": [{"metadata": {"name": "stale-controller"}}]}),
        ]
    )

    def run(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, next(responses), "")

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="still has controller Pod objects"):
        launch_helpers.ensure_new_student_slot(
            "senpai-track-fern",
            tag="track",
            student_name="fern",
            require_mpijob=False,
        )


def test_slot_scan_rejects_an_existing_student_deployment(monkeypatch):
    monkeypatch.setattr(
        launch_helpers.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(
                {
                    "kind": "Deployment",
                    "metadata": {"name": "senpai-track-fern"},
                }
            ),
            "",
        ),
    )

    with pytest.raises(RuntimeError, match="will not replace a running controller"):
        launch_helpers.ensure_new_student_slot(
            "senpai-track-fern",
            tag="track",
            student_name="fern",
            require_mpijob=False,
        )


def test_slot_scan_rejects_nonterminal_labeled_jobs(monkeypatch):
    responses = iter(
        [
            "",
            json.dumps({"items": []}),
            json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "active-training"},
                            "status": {"conditions": []},
                        },
                        {
                            "metadata": {"name": "finished-training"},
                            "status": {
                                "conditions": [
                                    {"type": "Complete", "status": "True"}
                                ]
                            },
                        },
                    ]
                }
            ),
            "jobs.batch\n",
        ]
    )

    def run(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, next(responses), "")

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="Job active-training"):
        launch_helpers.ensure_new_student_slot(
            "senpai-track-fern",
            tag="track",
            student_name="fern",
            require_mpijob=False,
        )


def test_slot_scan_fails_closed_when_kubernetes_cannot_be_read(monkeypatch):
    monkeypatch.setattr(
        launch_helpers.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 1, "", "forbidden"
        ),
    )

    with pytest.raises(RuntimeError, match="safety check failed: forbidden"):
        launch_helpers.ensure_new_student_slot(
            "senpai-track-fern",
            tag="track",
            student_name="fern",
            require_mpijob=False,
        )


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
        launch,
        "ensure_new_student_slot",
        lambda *_args, **_kwargs: None,
        raising=False,
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
    branch_commits = []
    monkeypatch.setattr(
        launch, "ensure_advisor_branch",
        lambda _repo, _token, _base, _advisor, commit: branch_commits.append(commit),
    )
    writes = []
    monkeypatch.setattr(
        launch, "kubectl_apply",
        lambda manifest, description, **_kwargs: writes.append(("apply", description, manifest)),
    )
    monkeypatch.setattr(
        launch, "kubectl_create",
        lambda manifest, description, **_kwargs: writes.append(("create", description, manifest)),
    )

    launch.main()

    assert branch_commits == ["a" * 40]
    assert [(operation, description) for operation, description, _ in writes] == [
        ("create", "secret senpai-launch-secrets-test-track"),
        ("apply", f"program context secret {bound_name}"),
        ("create", "student fern ConfigMap senpai-config-student-test-track-fern"),
        ("create", "student fern Deployment senpai-test-track-fern"),
        ("apply", "advisor"),
    ]
    resources = [document for _, _, manifest in writes[2:] for document in yaml.safe_load_all(manifest)]
    for resource in resources:
        if resource["kind"] == "ConfigMap":
            assert resource["data"]["SENPAI_PROGRAM_SOURCE_COMMIT"] == "b" * 40
            assert resource["data"]["SENPAI_PROGRAM_CONTENT_SHA256"] == bound.content_sha256
        elif resource["kind"] == "Deployment":
            assert resource["spec"]["template"]["metadata"]["annotations"][
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
    monkeypatch.setattr(launch, "kubectl_create", lambda *_args, **_kwargs: None)
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
    monkeypatch.setattr(launch, "kubectl_create", lambda *_args, **_kwargs: None)

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
    monkeypatch.setattr(launch, "kubectl_create", lambda *_args, **_kwargs: None)

    launch.main()

    assert resolved == ["anthropic"]


def test_launch_uses_one_scope_for_create_discovery_and_handoff_commands(
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
    mutations = []

    def apply(_manifest, name, *, kube_context, namespace):
        mutations.append(("apply", name, kube_context, namespace))

    def create(_manifest, name, *, kube_context, namespace):
        mutations.append(("create", name, kube_context, namespace))

    monkeypatch.setattr(launch, "kubectl_apply", apply)
    monkeypatch.setattr(launch, "kubectl_create", create, raising=False)

    launch.main()

    assert discovery == [("scope-test", "gpu-cluster", "research")]
    assert len(mutations) == 5
    assert mutations[1][0] == "apply"
    assert mutations[1][1].startswith("program context secret senpai-program-context-scope-test-")
    assert mutations[1][2:] == ("gpu-cluster", "research")
    assert [mutations[0], *mutations[2:]] == [
        (
            "create",
            "secret senpai-launch-secrets-scope-test",
            "gpu-cluster",
            "research",
        ),
        (
            "create",
            "student fern ConfigMap senpai-config-student-scope-test-fern",
            "gpu-cluster",
            "research",
        ),
        (
            "create",
            "student fern Deployment senpai-scope-test-fern",
            "gpu-cluster",
            "research",
        ),
        ("apply", "advisor", "gpu-cluster", "research"),
    ]
    prefix = "kubectl --context gpu-cluster --namespace research"
    handoff_commands = [
        line.strip()
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("kubectl ")
    ]
    assert len(handoff_commands) == 6
    assert all(command.startswith(prefix) for command in handoff_commands)
    assert not any("mpijobs" in command for command in handoff_commands)


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
    monkeypatch.setattr(
        launch,
        "kubectl_create",
        lambda *_args, **_kwargs: mutations.append("kubernetes"),
        raising=False,
    )

    with pytest.raises(SystemExit, match="active assignment"):
        launch.main()

    assert checked == ["acceptance-fern"]
    assert mutations == []


def test_every_student_is_scanned_before_the_atomic_tag_reservation(monkeypatch):
    args = launch_args(advisor=False, names="fern,frieren")
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    events = []

    monkeypatch.setattr(
        launch,
        "ensure_new_student_slot",
        lambda deployment, *, student_name, **_kwargs: events.append(
            ("scan", student_name, deployment)
        ),
    )
    monkeypatch.setattr(
        launch,
        "kubectl_create",
        lambda manifest, name, **_kwargs: events.append(
            ("create", name, yaml.safe_load(manifest)["kind"])
        ),
        raising=False,
    )
    monkeypatch.setattr(
        launch,
        "ensure_advisor_branch",
        lambda *_args: events.append(("github", "branch")),
    )
    monkeypatch.setattr(
        launch,
        "ensure_target_repo_labels",
        lambda *_args: events.append(("github", "labels")),
    )

    program_writes = []
    monkeypatch.setattr(
        launch, "kubectl_apply",
        lambda manifest, description, **_kwargs: program_writes.append((description, yaml.safe_load(manifest))),
    )
    launch.main()
    assert len(program_writes) == 1
    assert program_writes[0][0].startswith("program context secret ")
    assert program_writes[0][1]["immutable"] is True

    assert events[:5] == [
        ("scan", "fern", "senpai-test-track-fern"),
        ("scan", "frieren", "senpai-test-track-frieren"),
        ("create", "secret senpai-launch-secrets-test-track", "Secret"),
        ("github", "branch"),
        ("github", "labels"),
    ]


def test_student_launch_reservation_is_immutable(monkeypatch):
    args = launch_args(advisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    created = []

    def create(manifest, name, **_kwargs):
        created.append((name, yaml.safe_load(manifest)))

    monkeypatch.setattr(launch, "kubectl_create", create)

    program_writes = []
    monkeypatch.setattr(
        launch, "kubectl_apply",
        lambda manifest, description, **_kwargs: program_writes.append((description, yaml.safe_load(manifest))),
    )
    launch.main()
    assert len(program_writes) == 1
    assert program_writes[0][0].startswith("program context secret ")
    assert program_writes[0][1]["immutable"] is True

    reservation_name, reservation = created[0]
    assert reservation_name == "secret senpai-launch-secrets-test-track"
    assert reservation["immutable"] is True


def test_reordered_student_manifest_still_creates_the_deployment_last(monkeypatch):
    args = launch_args(advisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    render_student = launch.render_student

    def render_deployment_first(*render_args, **render_kwargs):
        documents = list(
            yaml.safe_load_all(render_student(*render_args, **render_kwargs))
        )
        documents.sort(key=lambda document: document["kind"] != "Deployment")
        return yaml.safe_dump_all(documents, sort_keys=False).rstrip()

    monkeypatch.setattr(launch, "render_student", render_deployment_first)
    created = []
    monkeypatch.setattr(
        launch,
        "kubectl_create",
        lambda manifest, _name, **_kwargs: created.append(
            yaml.safe_load(manifest)["kind"]
        ),
    )

    program_writes = []
    monkeypatch.setattr(
        launch, "kubectl_apply",
        lambda manifest, description, **_kwargs: program_writes.append((description, yaml.safe_load(manifest))),
    )
    launch.main()
    assert len(program_writes) == 1
    assert program_writes[0][0].startswith("program context secret ")
    assert program_writes[0][1]["immutable"] is True

    assert created == ["Secret", "ConfigMap", "Deployment"]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            "kind: ConfigMap\nmetadata:\n  name: config-only\n",
            "exactly one Deployment; found 0",
        ),
        (
            "kind: Deployment\nmetadata:\n  name: first\n"
            "---\nkind: Deployment\nmetadata:\n  name: second\n",
            "exactly one Deployment; found 2",
        ),
        ("kind: [\n", "invalid YAML"),
    ],
)
def test_invalid_student_manifest_fails_before_reservation_or_github_writes(
    monkeypatch,
    manifest,
    message,
):
    args = launch_args(advisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(launch, "render_student", lambda *_args, **_kwargs: manifest)
    writes = []
    monkeypatch.setattr(
        launch,
        "kubectl_create",
        lambda *_args, **_kwargs: writes.append("kubernetes"),
    )
    monkeypatch.setattr(
        launch,
        "ensure_advisor_branch",
        lambda *_args: writes.append("github"),
    )

    with pytest.raises(RuntimeError, match=message):
        launch.main()

    assert writes == []


def test_dry_run_prints_the_original_validated_student_manifest(monkeypatch, capsys):
    manifest = (
        "# deployment intentionally rendered first\n"
        "kind: Deployment\nmetadata:\n  name: controller\n"
        "---\n"
        "# preserve this dry-run comment\n"
        "kind: ConfigMap\nmetadata:\n  name: config\n"
    )
    args = launch_args(advisor=False, dry_run=True)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(launch, "render_student", lambda *_args, **_kwargs: manifest)

    launch.main()

    assert f"--- Student: fern ---\n{manifest}\n" in capsys.readouterr().out


def test_multinode_student_resources_are_created_in_dependency_order(monkeypatch):
    args = launch_args(
        advisor=False,
        nodes_per_student=2,
        executor_image=f"ghcr.io/wandb/senpai-executor@sha256:{'b' * 64}",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    created = []

    def create(manifest, _name, **_kwargs):
        document = yaml.safe_load(manifest)
        created.append((document["kind"], document["metadata"]["name"]))

    monkeypatch.setattr(launch, "kubectl_create", create, raising=False)

    program_writes = []
    monkeypatch.setattr(
        launch, "kubectl_apply",
        lambda manifest, description, **_kwargs: program_writes.append((description, yaml.safe_load(manifest))),
    )
    launch.main()
    assert len(program_writes) == 1
    assert program_writes[0][0].startswith("program context secret ")
    assert program_writes[0][1]["immutable"] is True

    assert [kind for kind, _name in created] == [
        "Secret",
        "ConfigMap",
        "ServiceAccount",
        "Role",
        "RoleBinding",
        "Deployment",
    ]


def test_reservation_conflict_stops_before_github_or_controller_mutation(monkeypatch):
    args = launch_args(advisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    mutations = []

    def create(_manifest, name, **_kwargs):
        mutations.append(name)
        raise RuntimeError("kubectl create failed: already exists")

    monkeypatch.setattr(launch, "kubectl_create", create, raising=False)
    monkeypatch.setattr(
        launch,
        "ensure_advisor_branch",
        lambda *_args: mutations.append("github branch"),
    )
    monkeypatch.setattr(
        launch,
        "ensure_target_repo_labels",
        lambda *_args: mutations.append("github labels"),
    )

    with pytest.raises(RuntimeError, match="stale reservation.*delete secret"):
        launch.main()

    assert mutations == ["secret senpai-launch-secrets-test-track"]


def test_student_scan_failure_stops_before_every_mutation(monkeypatch):
    args = launch_args(advisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    mutations = []
    monkeypatch.setattr(
        launch,
        "ensure_new_student_slot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("active Job")),
    )
    monkeypatch.setattr(
        launch,
        "kubectl_create",
        lambda *_args, **_kwargs: mutations.append("kubernetes"),
        raising=False,
    )
    monkeypatch.setattr(
        launch,
        "ensure_advisor_branch",
        lambda *_args: mutations.append("github branch"),
    )

    with pytest.raises(RuntimeError, match="active Job"):
        launch.main()

    assert mutations == []


def test_advisor_only_launch_keeps_apply_semantics(monkeypatch):
    args = launch_args(advisor=True, names="", n_students=0)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "ensure_new_student_slot",
        lambda *_args, **_kwargs: pytest.fail(
            "advisor-only launch has no student slot"
        ),
    )
    monkeypatch.setattr(
        launch,
        "kubectl_create",
        lambda *_args, **_kwargs: pytest.fail("advisor-only launch stays apply-based"),
        raising=False,
    )
    monkeypatch.setattr(launch, "existing_student_names", lambda *_args, **_kwargs: [])
    applied = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda _manifest, name, **_kwargs: applied.append(name),
    )

    launch.main()

    assert len(applied) == 3
    assert applied[0] == "secret senpai-launch-secrets-test-track"
    assert applied[1].startswith("program context secret senpai-program-context-test-track-")
    assert applied[2] == "advisor"


def test_dry_run_does_not_scan_or_reserve_student_slots(monkeypatch):
    args = launch_args(advisor=False, dry_run=True)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(
        launch,
        "ensure_new_student_slot",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not query Kubernetes"),
        raising=False,
    )
    monkeypatch.setattr(
        launch,
        "kubectl_create",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not reserve a tag"),
        raising=False,
    )

    launch.main()
