import base64
import json
import subprocess

import pytest
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


def test_existing_student_viewers_include_desired_and_live_bindings(monkeypatch):
    encoded = base64.b64encode(b"viewer-existing").decode()
    rotated = base64.b64encode(b"viewer-rotated").decode()

    def run(argv, **_kwargs):
        assert argv[-2:] == ["-o", "json"]
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(
                {
                    "items": [
                        {
                            "kind": "Deployment",
                            "metadata": {
                                "labels": {
                                    "role": "student",
                                    "student": "frieren",
                                }
                            },
                            "spec": {
                                "template": {
                                    "metadata": {
                                        "annotations": {
                                            "senpai.wandb.com/wandb-viewer": encoded
                                        }
                                    }
                                }
                            },
                        },
                        {
                            "kind": "Pod",
                            "metadata": {
                                "labels": {
                                    "role": "student",
                                    "student": "frieren",
                                },
                                "annotations": {
                                    "senpai.wandb.com/wandb-viewer": rotated
                                },
                            },
                        },
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    assert launch_helpers.existing_student_wandb_viewers("track-a") == {
        "frieren": {"viewer-existing", "viewer-rotated"}
    }


def test_legacy_viewer_bindings_require_a_new_tag(monkeypatch):
    payload = {
        "items": [
            {
                "kind": "Deployment",
                "metadata": {
                    "labels": {"role": "student", "student": "frieren"}
                },
                "spec": {"template": {"metadata": {"annotations": {}}}},
            }
        ]
    }
    monkeypatch.setattr(
        launch_helpers.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    with pytest.raises(SystemExit, match="new launch tag"):
        launch_helpers.existing_student_wandb_viewers("track-a")


def test_existing_controller_viewers_are_read_from_role_deployments(monkeypatch):
    encoded = base64.b64encode(b"viewer-advisor").decode()
    payload = {
        "items": [
            {
                "kind": "Deployment",
                "metadata": {"labels": {"role": "advisor"}},
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "senpai.wandb.com/controller-wandb-viewer": encoded
                            }
                        }
                    }
                }
            }
        ]
    }
    monkeypatch.setattr(
        launch_helpers.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    assert launch_helpers.existing_controller_wandb_viewers("track-a") == {
        "advisor": {"viewer-advisor"}
    }


def test_existing_inference_viewers_preserve_explicit_none(monkeypatch):
    encoded = base64.b64encode(b"viewer-inference").decode()
    payload = {
        "items": [
            {
                "kind": "Deployment",
                "metadata": {"labels": {"role": "advisor"}},
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "senpai.wandb.com/inference-wandb-viewer": encoded
                            }
                        }
                    }
                },
            },
            {
                "kind": "Pod",
                "metadata": {
                    "labels": {"role": "student", "student": "fern"},
                    "annotations": {
                        "senpai.wandb.com/inference-wandb-viewer": ""
                    },
                },
            },
        ]
    }
    monkeypatch.setattr(
        launch_helpers.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    assert launch_helpers.existing_inference_wandb_viewers("track-a") == {
        "advisor": {"viewer-inference"},
        "student/fern": {""},
    }


def test_namespace_viewer_owners_include_every_launch_tag(monkeypatch):
    def encoded(value: str) -> str:
        return base64.b64encode(value.encode()).decode()

    payload = {
        "items": [
            {
                "kind": "Deployment",
                "metadata": {
                    "labels": {
                        "research-tag": "track-a",
                        "role": "student",
                        "student": "fern",
                    }
                },
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "senpai.wandb.com/controller-wandb-viewer": encoded(
                                    "controller-a"
                                ),
                                "senpai.wandb.com/inference-wandb-viewer": "",
                                "senpai.wandb.com/wandb-viewer": encoded("writer-a"),
                            }
                        }
                    }
                },
            },
            {
                "kind": "Pod",
                "metadata": {
                    "labels": {
                        "research-tag": "track-b",
                        "role": "advisor",
                    },
                    "annotations": {
                        "senpai.wandb.com/controller-wandb-viewer": encoded(
                            "controller-b"
                        ),
                        "senpai.wandb.com/inference-wandb-viewer": encoded(
                            "inference-b"
                        ),
                    },
                },
            },
        ]
    }
    captured = {}

    def run(argv, **_kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    assert launch_helpers.existing_wandb_viewer_owners() == {
        "tag 'track-a' controller": {"controller-a"},
        "tag 'track-a' student 'fern'": {"writer-a"},
        "tag 'track-b' controller": {"controller-b"},
        "tag 'track-b' W&B Inference": {"inference-b"},
    }
    assert "app=senpai" in captured["argv"]
    assert not any("research-tag=" in value for value in captured["argv"])


def test_existing_roles_bind_one_program_context_secret(monkeypatch):
    name = "senpai-program-context-track-a-deadbeef"
    payload = {
        "items": [
            {
                "kind": "Deployment",
                "metadata": {"labels": {"role": "advisor"}},
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "senpai.program.com/context-secret": name
                            }
                        }
                    }
                },
            },
            {
                "kind": "Pod",
                "metadata": {
                    "labels": {"role": "student", "student": "fern"},
                    "annotations": {
                        "senpai.program.com/context-secret": name
                    },
                },
            },
        ]
    }
    monkeypatch.setattr(
        launch_helpers.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    assert launch_helpers.existing_program_context_secret("track-a") == name


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
        lambda _path, names: {
            name: f"wandb-training-{index}"
            for index, name in enumerate(names)
        },
    )
    monkeypatch.setattr(
        launch,
        "existing_controller_wandb_viewers",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        launch,
        "existing_wandb_viewer_owners",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        launch,
        "existing_program_context_secret",
        lambda *_args, **_kwargs: None,
    )
    for name in (
        "preflight_check_target_repo_access",
        "preflight_check_student_name_availability",
        "preflight_check_anthropic_api_key",
        "preflight_check_openai_api_key",
        "preflight_check_exa_api_key",
        "preflight_check_wandb_inference",
        "ensure_target_repo_labels",
    ):
        monkeypatch.setattr(launch, name, lambda *_args: None)
    monkeypatch.setattr(
        launch,
        "preflight_check_wandb_api_key",
        lambda api_key: f"viewer-{api_key}",
    )
    monkeypatch.setattr(launch, "ensure_advisor_branch", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        launch,
        "load_launch_program_snapshot",
        lambda *_args: launch.ProgramSystemPrompt(
            program_path="program.md",
            source_commit="a" * 40,
            content="Test launch research policy.",
        ),
    )
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
    monkeypatch.setattr(
        launch,
        "existing_wandb_viewer_owners",
        lambda *_args, **_kwargs: pytest.fail(
            "credential-only preflight must not require Kubernetes"
        ),
    )

    launch.main()

    assert resolved == [
        (launch.DOTENV_PATH, ["HF_TOKEN", "DATASET_LICENSE_KEY"])
    ]


def test_preflight_rejects_wandb_keys_for_the_same_viewer(monkeypatch):
    args = launch_args(preflight_only=True)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "preflight_check_wandb_api_key",
        lambda _api_key: "shared-viewer",
    )

    with pytest.raises(SystemExit, match="controller.*student 'fern'.*same viewer"):
        launch.main()


def test_incremental_launch_rejects_an_existing_student_viewer_reuse(monkeypatch):
    args = launch_args()
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_wandb_viewer_owners",
        lambda *_args, **_kwargs: {
            "tag 'older-track' student 'frieren'": {"viewer-wandb-training-0"}
        },
    )
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: pytest.fail("preflight must fail before mutation"),
    )

    with pytest.raises(SystemExit, match="frieren.*fern.*same viewer"):
        launch.main()


def test_incremental_launch_reserves_active_inference_viewers(monkeypatch):
    args = launch_args()
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(
        launch,
        "existing_wandb_viewer_owners",
        lambda *_args, **_kwargs: {
            "tag 'older-track' W&B Inference": {"viewer-wandb-training-0"}
        },
    )
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: pytest.fail("preflight must fail before mutation"),
    )

    with pytest.raises(SystemExit, match="W&B Inference.*student 'fern'"):
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
        lambda *_args, **_kwargs: pytest.fail("preflight must fail before mutation"),
    )

    with pytest.raises(SystemExit, match="frieren.*complete fleet"):
        launch.main()


def test_incremental_launch_reuses_the_tags_original_program_snapshot(monkeypatch):
    args = launch_args(advisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    bound = launch.ProgramSystemPrompt(
        program_path="program.md",
        source_commit="b" * 40,
        content="Test launch research policy.",
    )
    bound_name, _manifest = launch_helpers.render_program_context_secret(
        args.tag,
        launch.encode_program_system_prompt(bound),
    )
    monkeypatch.setattr(
        launch,
        "existing_program_context_secret",
        lambda *_args, **_kwargs: bound_name,
    )
    monkeypatch.setattr(
        launch,
        "read_program_context_secret",
        lambda *_args, **_kwargs: launch.encode_program_system_prompt(bound),
    )
    applied = []
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda manifest, description, **_kwargs: applied.append(
            (description, manifest)
        ),
    )

    launch.main()

    student_manifest = next(
        manifest for description, manifest in applied if description == "student fern"
    )
    assert f'SENPAI_PROGRAM_SOURCE_COMMIT: "{"b" * 40}"' in student_manifest
    assert f'senpai.program.com/context-secret: "{bound_name}"' in student_manifest


def test_incremental_launch_rejects_program_policy_drift(monkeypatch):
    args = launch_args(advisor=False)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    bound = launch.ProgramSystemPrompt(
        program_path="program.md",
        source_commit="b" * 40,
        content="Previous launch policy.",
    )
    monkeypatch.setattr(
        launch,
        "existing_program_context_secret",
        lambda *_args, **_kwargs: "senpai-program-context-existing",
    )
    monkeypatch.setattr(
        launch,
        "read_program_context_secret",
        lambda *_args, **_kwargs: launch.encode_program_system_prompt(bound),
    )
    monkeypatch.setattr(
        launch,
        "kubectl_apply",
        lambda *_args, **_kwargs: pytest.fail("policy drift must fail before apply"),
    )

    with pytest.raises(SystemExit, match="program.md changed.*new tag"):
        launch.main()


def test_launch_reports_invalid_custom_secret_names_without_a_traceback(monkeypatch):
    args = launch_args(custom_secret_env_names=["NOT-VALID"])
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)

    with pytest.raises(SystemExit, match="^ERROR: invalid custom secret"):
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


def test_launch_accepts_dotted_kubernetes_names(monkeypatch, capsys):
    args = launch_args(dry_run=True, tag="track.v2", names="team.fern")
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)

    launch.main()

    output = capsys.readouterr().out
    assert "research-tag: track.v2" in output
    assert "student: team.fern" in output


def test_monitor_commands_use_bounded_resource_names(monkeypatch, capsys):
    args = launch_args(tag="t" * 63)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    bypass_external_preflight(monkeypatch)
    monkeypatch.setattr(launch, "kubectl_apply", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        launch,
        "existing_student_names",
        lambda *_args, **_kwargs: [],
    )

    launch.main()

    output = capsys.readouterr().out
    advisor_name = launch_helpers.kubernetes_resource_name(
        f"senpai-advisor-{args.tag}"
    )
    student_name = launch_helpers.kubernetes_resource_name(
        f"senpai-{args.tag}-fern"
    )
    assert f"get deployment {advisor_name}" in output
    assert f"deployment/{student_name}" in output


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


def test_wandb_gateway_uses_a_dedicated_key_for_openai_compatible_inference(
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
        launch,
        "resolve_wandb_api_key",
        lambda _path: "wandb-controller-key",
    )
    monkeypatch.setattr(
        launch,
        "resolve_wandb_inference_api_key",
        lambda _path: "wandb-inference-key",
    )
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

    assert checked == [("wandb-inference-key", "research-team", "mlxfast")]


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
    secret_name, _secret = launch.render_launch_secret(
        "scope-test",
        "github",
        "exa",
        "wandb",
        openai_api_key="openai",
        custom_secrets={},
    )
    program = launch.ProgramSystemPrompt(
        program_path="program.md",
        source_commit="a" * 40,
        content="Test launch research policy.",
    )
    program_name, _program_secret = launch.render_program_context_secret(
        "scope-test",
        launch.encode_program_system_prompt(program),
    )
    assert applies == [
        (f"secret {secret_name}", "gpu-cluster", "research"),
        (
            f"program context secret {program_name}",
            "gpu-cluster",
            "research",
        ),
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
