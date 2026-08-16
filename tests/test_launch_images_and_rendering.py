import base64
import re

import pytest
import yaml

from launch_test_support import (
    ADVISOR_IMAGE,
    REVISION,
    STUDENT_IMAGE,
    launch,
    launch_args,
    launch_helpers,
    render_role,
    run_launch,
)


def test_default_config_exposes_every_model_profile_and_effort():
    config = yaml.safe_load(launch.SENPAI_CONFIG.read_text())

    assert {
        name
        for profile in ("advisor", "student", "smart", "fast", "frontier")
        for name in (f"{profile}_model", f"{profile}_reasoning_effort")
    } <= set(config)
    assert {
        "advisor_model": "openai/gpt-5.6-sol",
        "advisor_reasoning_effort": "xhigh",
        "student_model": "openai/gpt-5.6-sol",
        "student_reasoning_effort": "xhigh",
        "smart_model": "openai/gpt-5.6-sol",
        "smart_reasoning_effort": "xhigh",
        "fast_model": "openai/gpt-5.6-luna",
        "fast_reasoning_effort": "high",
        "frontier_model": "openai/gpt-5.6-sol",
        "frontier_reasoning_effort": "max",
        "local_condenser_max_events": 0,
        "local_condenser_max_tokens": 0,
        "local_condenser_target_events": 0,
    }.items() <= config.items()
    assert config["program_path"] == ""
    assert config["senpai_repo_url"] == "https://github.com/wandb/senpai.git"
    assert config["senpai_repo_revision"] == ""
    assert "repo_url" not in config
    assert "repo_revision" not in config


@pytest.mark.parametrize(
    "image",
    [
        f"ghcr.io/wandb/senpai:sha-{'a' * 40}",
        f"ghcr.io/wandb/senpai@sha256:{'b' * 64}",
        f"registry.example:5000/team/senpai@sha256:{'c' * 64}",
    ],
)
def test_image_reference_accepts_only_full_source_sha_tags_or_digests(image):
    assert launch_helpers.is_immutable_image_reference(image)


@pytest.mark.parametrize(
    "image",
    [
        "",
        "ghcr.io/wandb/senpai:latest",
        "ghcr.io/wandb/senpai:sha-deadbeef",
        f"ghcr.io/wandb/senpai:sha-{'A' * 40}",
        f"ghcr.io/wandb/senpai@sha256:{'b' * 63}",
    ],
)
def test_image_reference_rejects_mutable_or_incomplete_pins(image):
    assert not launch_helpers.is_immutable_image_reference(image)


def test_source_revision_is_derived_from_a_full_sha_tag():
    image = f"ghcr.io/wandb/senpai:sha-{REVISION}"

    assert launch_helpers.source_revision_for_image(image) == REVISION


def test_digest_image_requires_an_explicit_source_revision():
    image = f"ghcr.io/wandb/senpai@sha256:{'b' * 64}"

    assert launch_helpers.source_revision_for_image(image, REVISION) == REVISION
    with pytest.raises(
        ValueError, match="require an explicit senpai_repo_revision"
    ):
        launch_helpers.source_revision_for_image(image)


def test_explicit_revision_must_match_the_source_sha_tag():
    image = f"ghcr.io/wandb/senpai:sha-{REVISION}"

    with pytest.raises(ValueError, match="does not match"):
        launch_helpers.source_revision_for_image(image, "b" * 40)


def test_dry_run_binds_each_role_image_to_the_derived_source_revision():
    result = run_launch(
        "--advisor",
        "--advisor_image",
        ADVISOR_IMAGE,
        "--student_image",
        STUDENT_IMAGE,
    )

    assert result.returncode == 0, result.stderr
    rendered_yaml = re.sub(r"^--- .+ ---$", "---", result.stdout, flags=re.MULTILINE)
    documents = [
        document
        for document in yaml.safe_load_all(rendered_yaml)
        if isinstance(document, dict)
    ]
    deployments = {
        document["metadata"]["labels"]["role"]: document
        for document in documents
        if document.get("kind") == "Deployment"
    }
    assert {
        role: deployment["spec"]["template"]["spec"]["containers"][0]["image"]
        for role, deployment in deployments.items()
    } == {"advisor": ADVISOR_IMAGE, "student": STUDENT_IMAGE}
    assert {
        document["data"]["SENPAI_REPO_REVISION"]
        for document in documents
        if document.get("kind") == "ConfigMap"
    } == {REVISION}


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_runner_repository_is_explicit_in_every_role_configmap(role):
    args = launch_args(
        senpai_repo_url="https://github.com/example/senpai-fork.git"
    )

    configmap, _deployment, _secret = render_role(role, args)
    config = yaml.safe_load(configmap)["data"]

    assert config["SENPAI_REPO_URL"] == args.senpai_repo_url
    assert config["SENPAI_REPO_REVISION"] == REVISION
    assert config["TARGET_REPO_URL"] == args.target_repo_url


def test_launch_rejects_role_images_from_different_source_revisions():
    result = run_launch(
        "--advisor",
        "--advisor_image",
        ADVISOR_IMAGE,
        "--student_image",
        f"ghcr.io/wandb/senpai-student:sha-{'b' * 40}",
    )

    assert result.returncode != 0
    assert "same source revision" in result.stderr


def test_launch_rejects_an_unsafe_local_condenser_event_cap():
    result = run_launch(
        "--advisor",
        "--advisor_image",
        ADVISOR_IMAGE,
        "--student_image",
        STUDENT_IMAGE,
        "--local_condenser_max_events",
        "11",
    )

    assert result.returncode != 0
    assert "--local_condenser_max_events must be 0 or at least 12" in result.stderr


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_launch_rejects_a_mutable_image_for_either_role(role):
    images = {"advisor": ADVISOR_IMAGE, "student": STUDENT_IMAGE}
    images[role] = f"ghcr.io/wandb/senpai-{role}:latest"

    result = run_launch(
        "--advisor",
        "--advisor_image",
        images["advisor"],
        "--student_image",
        images["student"],
    )

    assert result.returncode != 0
    assert f"--{role}_image must be an immutable digest" in result.stderr


def test_student_only_launch_validates_only_the_student_image():
    args = launch_args(advisor=False, advisor_image="")

    launch.resolve_runner_revision(args, has_students=True)

    assert args.senpai_repo_revision == REVISION


def test_advisor_only_launch_validates_only_the_advisor_image():
    args = launch_args(names="", n_students=0, student_image="")

    launch.resolve_runner_revision(args, has_students=False)

    assert args.senpai_repo_revision == REVISION


def test_launch_help_keeps_descriptions_with_their_options():
    result = run_launch("--help")
    help_text = " ".join(result.stdout.split())

    assert result.returncode == 0, result.stderr
    for option_help in (
        "--senpai_repo_url str public read-only runner source",
        "--senpai_repo_revision str exact runner commit",
        "--advisor_image str advisor source-SHA tag or image digest — REQUIRED",
        "--student_image str student source-SHA tag or image digest — REQUIRED",
        "--human_issues, --nohuman_issues bool allow human GitHub issue triage",
        "--advisor, --noadvisor bool also deploy the advisor pod",
        "--extra_instructions str shared operator instructions",
        "--timeout_minutes float training run wall-clock limit",
        "--max_epochs int maximum training epochs",
        "--poll_interval_s int default advisor/student outer-loop sleep",
        "--dry_run, --nodry_run bool render manifests only",
        "--preflight_only, --nopreflight_only bool validate credentials/access only",
    ):
        assert option_help in help_text


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_role_bootstrap_verifies_both_checkout_and_image_source_revision(role):
    _configmap, deployment, _secret = render_role(role)
    command = yaml.safe_load(deployment)["spec"]["template"]["spec"]["containers"][
        0
    ]["args"][0]

    assert (
        'fetch --depth 1 "$SENPAI_REPO_URL" "$SENPAI_REPO_REVISION"' in command
    )
    assert 'test "$(git rev-parse HEAD)" = "$SENPAI_REPO_REVISION"' in command
    assert (
        'test "$SENPAI_IMAGE_REVISION" = "$SENPAI_REPO_REVISION"' in command
    )


@pytest.mark.parametrize(
    "gate",
    ["relative/start-gate", "/tmp/start-gate", "/mnt/shared/../escape"],
)
def test_start_gate_must_be_a_normalized_path_beneath_the_shared_pvc(gate):
    args = launch_args(pvc_mount_path="/mnt/shared", start_gate_path=gate)

    with pytest.raises(SystemExit, match="shared PVC"):
        launch.validate_timing_args(args)


def test_start_gate_is_rendered_when_it_is_beneath_the_shared_pvc():
    args = launch_args(
        pvc_mount_path="/mnt/shared",
        start_gate_path="/mnt/shared/gates/start",
    )

    launch.validate_timing_args(args)
    configmap, _deployment, _secret = render_role("student", args)

    assert yaml.safe_load(configmap)["data"]["SENPAI_START_GATE_PATH"] == (
        "/mnt/shared/gates/start"
    )


@pytest.mark.parametrize(
    "path",
    ["/program.md", "../program.md", "senpai/../program.md", "policy.md"],
)
def test_launch_rejects_a_program_path_outside_the_target_repo(path):
    result = run_launch("--program_path", path)

    assert result.returncode != 0
    assert "--program_path" in result.stderr


def test_launch_secret_contains_each_credential_and_both_roles_reference_it():
    expected_values = {
        "github-token": "github",
        "openai-api-key": "openai",
        "exa-api-key": "exa",
        "wandb-api-key": "wandb",
    }

    _configmap, _deployment, secret = render_role("advisor")
    secret_document = yaml.safe_load(secret)
    assert {
        key: base64.b64decode(value).decode()
        for key, value in secret_document["data"].items()
    } == expected_values

    for role in ("advisor", "student"):
        _configmap, deployment, _secret = render_role(role)
        container = yaml.safe_load(deployment)["spec"]["template"]["spec"][
            "containers"
        ][0]
        references = {
            item["valueFrom"]["secretKeyRef"]["key"]: item["valueFrom"][
                "secretKeyRef"
            ]["name"]
            for item in container["env"]
        }
        assert references == {
            key: "senpai-launch-secrets-test-track"
            for key in (*expected_values, "hf-token")
        }
        hf_reference = next(
            item["valueFrom"]["secretKeyRef"]
            for item in container["env"]
            if item["name"] == "HF_TOKEN"
        )
        assert hf_reference["optional"] is True


def test_launch_secret_includes_hugging_face_token_only_when_configured():
    with_token = yaml.safe_load(
        launch_helpers.render_launch_secret(
            "track",
            "github",
            "exa",
            "wandb",
            hf_token="hugging-face",
        )
    )["data"]
    without_token = yaml.safe_load(
        launch_helpers.render_launch_secret("track", "github", "exa", "wandb")
    )["data"]

    assert base64.b64decode(with_token["hf-token"]).decode() == "hugging-face"
    assert "hf-token" not in without_token


def test_role_model_configuration_preserves_the_configured_efforts():
    args = launch_args(
        advisor_model="openai/gpt-5.6-sol",
        advisor_reasoning_effort="max",
        student_model="anthropic/claude-opus-4-8",
        student_reasoning_effort="medium",
        smart_model="anthropic/claude-sonnet-4-6",
        smart_reasoning_effort="high",
        fast_model="openai/gpt-5.6-mini",
        fast_reasoning_effort="none",
        frontier_model="openai/gpt-5.6-sol",
        frontier_reasoning_effort="max",
        local_condenser_max_events=180,
        local_condenser_max_tokens=180_000,
        local_condenser_target_events=40,
    )

    advisor_config, _deployment, _secret = render_role("advisor", args)
    student_config, _deployment, _secret = render_role("student", args)
    advisor = yaml.safe_load(advisor_config)["data"]
    student = yaml.safe_load(student_config)["data"]

    assert advisor["SENPAI_OPENHANDS_MODEL"] == args.advisor_model
    assert advisor["SENPAI_OPENHANDS_REASONING_EFFORT"] == "max"
    assert student["SENPAI_OPENHANDS_MODEL"] == args.student_model
    assert student["SENPAI_OPENHANDS_REASONING_EFFORT"] == "medium"
    for config in (advisor, student):
        assert config["SENPAI_OPENHANDS_SMART_MODEL"] == args.smart_model
        assert config["SENPAI_OPENHANDS_SMART_REASONING_EFFORT"] == "high"
        assert config["SENPAI_OPENHANDS_FAST_MODEL"] == args.fast_model
        assert config["SENPAI_OPENHANDS_FAST_REASONING_EFFORT"] == "none"
        assert config["SENPAI_OPENHANDS_FRONTIER_MODEL"] == args.frontier_model
        assert config["SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT"] == "max"
        assert config["SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_EVENTS"] == "180"
        assert config["SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_TOKENS"] == "180000"
        assert config["SENPAI_OPENHANDS_LOCAL_CONDENSER_TARGET_EVENTS"] == "40"


def test_openai_ultra_launch_value_is_rejected():
    args = launch_args(
        frontier_model="openai/gpt-5.6-sol",
        frontier_reasoning_effort="ultra",
    )

    with pytest.raises(SystemExit, match="must be one of"):
        launch.validate_model_config(args)


def test_launch_accepts_anthropic_max_for_every_model_profile():
    args = launch_args(
        advisor_model="anthropic/claude-fable-5",
        advisor_reasoning_effort="max",
        student_model="anthropic/claude-opus-5",
        student_reasoning_effort="max",
        smart_model="anthropic/claude-opus-5",
        smart_reasoning_effort="max",
        fast_model="anthropic/claude-sonnet-5",
        fast_reasoning_effort="max",
        frontier_model="anthropic/claude-fable-5",
        frontier_reasoning_effort="max",
    )

    launch.validate_model_config(args)


def test_wandb_gateway_is_rendered_for_every_role():
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

    for role in ("advisor", "student"):
        configmap, _deployment, _secret = render_role(role, args)
        config = yaml.safe_load(configmap)["data"]
        assert config["SENPAI_OPENHANDS_MODEL"] == model
        assert config["WANDB_ENTITY"] == "research-team"
        assert config["WANDB_PROJECT"] == "mlxfast"

    launch.validate_model_config(args)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"advisor_reasoning_effort": "extreme"}, "must be one of"),
        (
            {
                "advisor_model": "anthropic/claude-opus-4-8",
                "advisor_reasoning_effort": "ultra",
            },
            "must be one of",
        ),
        (
            {
                "advisor_model": "openai/gpt-5.60",
                "advisor_reasoning_effort": "max",
            },
            "unsupported for",
        ),
    ],
)
def test_launch_rejects_unsupported_reasoning_effort(overrides, message):
    with pytest.raises(SystemExit, match=message):
        launch.validate_model_config(launch_args(**overrides))


def test_launch_accepts_documented_anthropic_extended_efforts():
    launch.validate_model_config(
        launch_args(
            advisor_model="anthropic/claude-opus-5",
            advisor_reasoning_effort="xhigh",
            student_model="anthropic/claude-opus-5",
            student_reasoning_effort="xhigh",
            smart_model="anthropic/claude-opus-5",
            smart_reasoning_effort="xhigh",
            fast_model="anthropic/claude-sonnet-5",
            fast_reasoning_effort="high",
            frontier_model="anthropic/claude-fable-5",
            frontier_reasoning_effort="max",
        )
    )


@pytest.mark.parametrize(
    ("model", "provider_env", "secret_key"),
    [
        ("anthropic/claude-opus-4-8", "ANTHROPIC_API_KEY", "anthropic-api-key"),
        ("openai/gpt-5.6-sol", "OPENAI_API_KEY", "openai-api-key"),
        ("wandb/zai-org/GLM-5.2", "WANDB_API_KEY", "wandb-api-key"),
    ],
)
def test_roles_mount_only_the_provider_used_by_their_models(
    model, provider_env, secret_key
):
    args = launch_args(
        advisor_model=model,
        student_model=model,
        smart_model=model,
        fast_model=model,
        frontier_model=model,
    )

    _configmap, deployment, secret = render_role("advisor", args)
    secret_keys = set(yaml.safe_load(secret)["data"])
    environment = yaml.safe_load(deployment)["spec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    environment_names = [item["name"] for item in environment]

    assert secret_keys == {"github-token", secret_key, "exa-api-key", "wandb-api-key"}
    assert len(environment_names) == len(set(environment_names))
    assert set(environment_names) == {
        "GITHUB_TOKEN",
        provider_env,
        "EXA_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
    }


def test_role_mounts_include_its_main_model_and_shared_profiles_only():
    args = launch_args(
        advisor_model="anthropic/claude-opus-4-8",
        student_model="openai/gpt-5.6",
        smart_model="anthropic/claude-opus-4-8",
        fast_model="anthropic/claude-haiku-4-5",
        frontier_model="anthropic/claude-opus-4-8",
        frontier_reasoning_effort="xhigh",
    )

    mounted = {}
    for role in ("advisor", "student"):
        _configmap, deployment, _secret = render_role(role, args)
        environment = yaml.safe_load(deployment)["spec"]["template"]["spec"][
            "containers"
        ][0]["env"]
        mounted[role] = {item["name"] for item in environment}

    common = {
        "GITHUB_TOKEN",
        "ANTHROPIC_API_KEY",
        "EXA_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
    }
    assert mounted["advisor"] == common
    assert mounted["student"] == common | {"OPENAI_API_KEY"}


def test_pod_template_hash_covers_complete_config_and_secret_content():
    config = "kind: ConfigMap\ndata:\n  POLL_INTERVAL: '60'\n"
    secret = launch_helpers.render_launch_secret(
        "track",
        "github",
        "exa",
        "wandb",
        anthropic_api_key="anthropic",
        openai_api_key="openai",
    )

    first = launch_helpers.pod_template_hash(config, secret)

    assert first == launch_helpers.pod_template_hash(config, secret)
    assert first != launch_helpers.pod_template_hash(
        config.replace("'60'", "'120'"), secret
    )
    assert first != launch_helpers.pod_template_hash(
        config, secret.replace("d2FuZGI=", "bmV3LXdhbmRi")
    )


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_rendered_role_annotation_matches_its_effective_content_hash(role):
    configmap, deployment, secret = render_role(role)

    annotation = yaml.safe_load(deployment)["spec"]["template"]["metadata"][
        "annotations"
    ]["senpai.wandb.com/content-hash"]

    assert annotation == launch_helpers.pod_template_hash(configmap, secret)
