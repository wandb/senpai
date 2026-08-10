import yaml

from launch_test_support import (
    ROOT,
    launch,
    launch_args,
    launch_helpers,
)


def rendered_supervisor(args=None):
    args = launch_args() if args is None else args
    provider = launch.model_provider(args.advisor_model)
    provider_secret_name = (
        None if provider == "wandb" else launch.MODEL_PROVIDERS[provider][1]
    )
    secret_name, secret = launch_helpers.render_supervisor_secret(
        args.tag,
        "github",
        "wandb",
        provider_secret_name=provider_secret_name,
        provider_api_key=None if provider == "wandb" else provider,
    )
    template = (ROOT / "k8s" / "operational-supervisor-deployment.yaml").read_text()
    manifest = launch.render_operational_supervisor(
        template,
        args.tag,
        ["fern", "frieren"],
        secret_name,
        secret,
        args,
    )
    return list(yaml.safe_load_all(manifest)), secret


def test_supervisor_is_opt_in_with_fifteen_minute_and_six_hour_defaults():
    config = yaml.safe_load(launch.SENPAI_CONFIG.read_text())

    assert config["operational_supervisor"] is False
    assert config["supervisor_dedicated_namespace"] is False
    assert config["supervisor_interval_s"] == 15 * 60
    assert config["supervisor_research_interval_s"] == 6 * 60 * 60
    assert config["supervisor_ready_timeout_s"] == 15 * 60


def test_supervisor_has_a_control_plane_specific_harness():
    entrypoint = (ROOT / "k8s" / "entrypoint-operational-supervisor.sh").read_text()
    harness = (
        ROOT / "system_instructions" / "OPERATIONAL_SUPERVISOR_HARNESS.md"
    ).read_text()

    assert "OPERATIONAL_SUPERVISOR_HARNESS.md" in entrypoint
    assert "native `terminal`" in harness
    assert "arbitrary shell, Git, `gh`, and `kubectl`" in harness
    assert "GitHub credentials are not" in harness
    assert "target checkout" in harness
    assert "program.md" not in harness
    assert "spawn_agents" not in harness


def test_student_runs_receive_explicit_campaign_and_student_scope():
    args = launch_args(tag="campaign-a")
    secret = launch_helpers.render_launch_secret(
        args.tag, "github", "exa", "wandb", openai_api_key="openai"
    )
    manifest = launch.render_student(
        (ROOT / "k8s" / "student-deployment.yaml").read_text(),
        "fern",
        args.tag,
        "secret",
        secret,
        args,
    )
    config = yaml.safe_load(manifest.split("\n---\n", 1)[0])["data"]

    assert config["SENPAI_WANDB_SCOPE"] == "campaign-a"
    assert "WANDB_RUN_GROUP" not in config
    assert config["WANDB_JOB_TYPE"] == "fern"
    assert "senpai:campaign-a" in config["WANDB_TAGS"]
    assert "senpai-student:fern" in config["WANDB_TAGS"]


def test_supervisor_is_separate_with_namespace_scoped_pod_rbac():
    documents, _secret = rendered_supervisor()
    by_kind = {document["kind"]: document for document in documents}
    deployment = by_kind["Deployment"]
    role = by_kind["Role"]

    assert deployment["metadata"]["labels"] == {
        "app": "senpai",
        "role": "supervisor",
        "research-tag": "test-track",
    }
    assert deployment["metadata"]["annotations"] == {
        "senpai.wandb.com/source-revision": "a" * 40,
        "senpai.wandb.com/advisor-branch": launch_args().advisor_branch,
    }
    assert deployment["spec"]["template"]["spec"]["serviceAccountName"] == (
        "senpai-supervisor-test-track"
    )
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]},
        {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
        {"apiGroups": [""], "resources": ["pods/exec"], "verbs": ["create"]},
    ]


def test_supervisor_runtime_and_instructions_are_mounted_read_only():
    documents, _secret = rendered_supervisor()
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    initializer = pod["initContainers"][0]
    supervisor = pod["containers"][0]

    assert initializer["name"] == "source"
    assert initializer["volumeMounts"] == [
        {"name": "runtime", "mountPath": "/workspace/senpai"}
    ]
    assert {mount["name"]: mount for mount in supervisor["volumeMounts"]}[
        "runtime"
    ] == {
        "name": "runtime",
        "mountPath": "/workspace/senpai",
        "readOnly": True,
    }
    assert {volume["name"]: volume for volume in pod["volumes"]}["runtime"] == {
        "name": "runtime",
        "emptyDir": {},
    }
    init_script = initializer["args"][0]
    assert "remote get-url origin" in init_script
    assert "remote set-url origin" in init_script


def test_supervisor_config_carries_exact_campaign_inventory_and_cadence():
    documents, _secret = rendered_supervisor(
        launch_args(
            tag="campaign-a",
            advisor_branch="campaign-a-advisor",
            supervisor_interval_s=901,
            supervisor_research_interval_s=21601,
            supervisor_action_cooldown_s=1801,
        )
    )
    config = documents[0]["data"]

    assert config["RESEARCH_TAG"] == "campaign-a"
    assert config["ADVISOR_BRANCH"] == "campaign-a-advisor"
    assert config["STUDENT_NAMES"] == "fern,frieren"
    assert config["SENPAI_WANDB_SCOPE"] == "campaign-a"
    assert config["SENPAI_SUPERVISOR_INTERVAL_SECONDS"] == "901"
    assert config["SENPAI_SUPERVISOR_RESEARCH_INTERVAL_SECONDS"] == "21601"
    assert config["SENPAI_SUPERVISOR_ACTION_COOLDOWN_SECONDS"] == "1801"


def test_supervisor_mounts_only_github_wandb_and_model_credentials():
    documents, _secret = rendered_supervisor()
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    names = {
        item["name"]
        for item in deployment["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    }

    assert names == {"GITHUB_TOKEN", "WANDB_API_KEY", "OPENAI_API_KEY"}
    assert "EXA_API_KEY" not in names


def test_supervisor_hands_off_and_unsets_credentials_before_python():
    handoff_script = (
        ROOT / "k8s" / "handoff-operational-supervisor-secrets.sh"
    ).read_text()
    entrypoint = (ROOT / "k8s" / "entrypoint-operational-supervisor.sh").read_text()

    assert "mktemp -d /tmp/senpai-supervisor-secrets.XXXXXX" in handoff_script
    for name in (
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        assert name in handoff_script
    handoff = handoff_script.index("export SENPAI_SUPERVISOR_SECRET_DIR")
    scrub = handoff_script.index("unset GITHUB_TOKEN")
    assert handoff < scrub
    assert entrypoint.index("handoff_operational_supervisor_secrets") < (
        entrypoint.index("exec python -m senpai_agent.operational_supervisor run")
    )


def test_supervisor_mounts_only_its_primary_model_provider():
    documents, _secret = rendered_supervisor(
        launch_args(
            advisor_model="openai/gpt-5.6-sol",
            frontier_model="anthropic/claude-fable-5",
        )
    )
    config = documents[0]["data"]
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    names = {
        item["name"]
        for item in deployment["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    }

    assert "SENPAI_OPENHANDS_FRONTIER_MODEL" not in config
    assert names == {"GITHUB_TOKEN", "WANDB_API_KEY", "OPENAI_API_KEY"}
