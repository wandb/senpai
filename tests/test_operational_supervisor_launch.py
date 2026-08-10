import yaml

from launch_test_support import (
    ROOT,
    launch,
    launch_args,
    launch_helpers,
)


def rendered_supervisor(args=None):
    args = launch_args() if args is None else args
    secret_name = f"senpai-launch-secrets-{args.tag}"
    providers = launch.deployed_model_providers(args)
    secret = launch_helpers.render_launch_secret(
        args.tag,
        "github",
        "exa",
        "wandb",
        anthropic_api_key="anthropic" if "anthropic" in providers else None,
        openai_api_key="openai" if "openai" in providers else None,
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


def test_supervisor_has_a_control_plane_specific_harness():
    entrypoint = (ROOT / "k8s" / "entrypoint-operational-supervisor.sh").read_text()
    harness = (
        ROOT / "system_instructions" / "OPERATIONAL_SUPERVISOR_HARNESS.md"
    ).read_text()

    assert "OPERATIONAL_SUPERVISOR_HARNESS.md" in entrypoint
    assert "native `terminal`" in harness
    assert "arbitrary shell and Git" in harness
    assert "senpai-role-shell" in harness
    assert "fixed secret-free `repair` sidecar" in harness
    assert "GitHub credentials are not" in harness
    assert "role workspace" in harness
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
    pod = deployment["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["shareProcessNamespace"] is False
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]},
        {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
        {"apiGroups": [""], "resources": ["pods/exec"], "verbs": ["create"]},
    ]


def test_supervisor_runtime_is_atomically_prepared_then_read_only():
    documents, _secret = rendered_supervisor()
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    state_initializer, source_initializer = pod["initContainers"]
    supervisor = next(
        container for container in pod["containers"]
        if container["name"] == "supervisor-control"
    )
    shell = next(
        container for container in pod["containers"]
        if container["name"] == "supervisor-shell"
    )

    assert state_initializer["name"] == "state-directory"
    assert state_initializer["volumeMounts"] == [
        {"name": "state", "mountPath": "/state-root"}
    ]
    assert source_initializer["name"] == "source"
    assert source_initializer["volumeMounts"] == [
        {"name": "runtime", "mountPath": "/workspace"}
    ]
    assert {entry["name"] for entry in source_initializer["env"]} == {
        "GITHUB_TOKEN"
    }
    assert {mount["name"]: mount for mount in supervisor["volumeMounts"]}[
        "runtime"
    ] == {
        "name": "runtime",
        "mountPath": "/workspace",
        "readOnly": True,
    }
    assert {mount["name"]: mount for mount in shell["volumeMounts"]}["runtime"][
        "readOnly"
    ] is True
    assert {volume["name"]: volume for volume in pod["volumes"]}["runtime"] == {
        "name": "runtime",
        "emptyDir": {},
    }
    assert "env" not in state_initializer
    assert "envFrom" not in state_initializer
    source_script = source_initializer["args"][0]
    assert "remote get-url origin" in source_script
    assert "remote set-url origin" in source_script
    assert "mv /workspace/.source /workspace/senpai" in source_script


def test_supervisor_control_and_shell_have_disjoint_ambient_authority():
    documents, _secret = rendered_supervisor()
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    containers = {container["name"]: container for container in pod["containers"]}
    control = containers["supervisor-control"]
    shell = containers["supervisor-shell"]
    volumes = {volume["name"]: volume for volume in pod["volumes"]}

    credential_names = {entry["name"] for entry in control["env"]}
    assert credential_names == {"GITHUB_TOKEN", "WANDB_API_KEY", "OPENAI_API_KEY"}
    assert "envFrom" not in shell
    assert all(
        fragment not in entry["name"]
        for entry in shell.get("env", [])
        for fragment in ("TOKEN", "KEY", "SECRET", "CREDENTIAL")
    )
    assert shell["securityContext"]["readOnlyRootFilesystem"] is True

    control_mounts = {mount["name"]: mount for mount in control["volumeMounts"]}
    shell_mounts = {mount["name"]: mount for mount in shell["volumeMounts"]}
    assert control_mounts["service-account"]["mountPath"] == (
        "/var/run/secrets/kubernetes.io/serviceaccount"
    )
    assert "service-account" not in shell_mounts
    assert "state" not in shell_mounts
    assert control_mounts["terminal-socket"]["readOnly"] is True
    assert "readOnly" not in control_mounts["repair-socket"]
    assert "readOnly" not in shell_mounts["terminal-socket"]
    assert shell_mounts["repair-socket"]["readOnly"] is True
    assert control_mounts["state"]["subPath"] == "test-track/operational-supervisor"
    assert set(shell_mounts) == {
        "runtime",
        "terminal-socket",
        "repair-socket",
        "shell-workspace",
        "shell-home",
        "shell-tmp",
    }
    token_projection = volumes["service-account"]["projected"]["sources"][0][
        "serviceAccountToken"
    ]
    assert token_projection["expirationSeconds"] == 3600
    assert "audience" not in token_projection


def test_supervisor_health_recovers_unlinked_terminal_and_repair_sockets():
    documents, _secret = rendered_supervisor()
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    containers = {
        container["name"]: container
        for container in deployment["spec"]["template"]["spec"]["containers"]
    }
    control = containers["supervisor-control"]
    shell = containers["supervisor-shell"]

    for probe_name in ("startupProbe", "livenessProbe"):
        control_health = control[probe_name]["exec"]["command"][-1]
        assert "senpai_agent.isolated_terminal health" in control_health
        assert "--socket @senpai-isolated-terminal" in control_health
        assert "test -S /run/senpai-repair/repair.sock" in control_health
        shell_health = shell[probe_name]["exec"]["command"]
        assert shell_health[:4] == [
            "/opt/senpai-venv/bin/python",
            "-I",
            "-m",
            "senpai_agent.isolated_terminal",
        ]
        assert shell_health[-3:] == [
            "health",
            "--socket",
            "@senpai-isolated-terminal",
        ]


def test_advisor_and_students_have_secret_free_exact_role_repair_sidecars():
    args = launch_args(tag="campaign-a")
    for role in ("advisor", "student"):
        secret = launch_helpers.render_launch_secret(
            args.tag, "github", "exa", "wandb", openai_api_key="openai"
        )
        template = (ROOT / "k8s" / f"{role}-deployment.yaml").read_text()
        if role == "advisor":
            manifest = launch.render_advisor(
                template, args.tag, ["fern"], "secret", secret, args
            )
        else:
            manifest = launch.render_student(
                template, "fern", args.tag, "secret", secret, args
            )
        deployment = yaml.safe_load(manifest.split("\n---\n", 1)[1])
        pod = deployment["spec"]["template"]["spec"]
        containers = {container["name"]: container for container in pod["containers"]}
        main = containers[role]
        repair = containers["repair"]
        main_mounts = {mount["name"] for mount in main["volumeMounts"]}
        repair_mounts = {mount["name"] for mount in repair["volumeMounts"]}

        assert pod["automountServiceAccountToken"] is False
        assert pod["shareProcessNamespace"] is False
        assert "env" not in repair
        assert "envFrom" not in repair
        assert repair["securityContext"]["readOnlyRootFilesystem"] is True
        assert repair_mounts == {"workspace", "state", "repair-scratch", "repair-tmp"}
        assert {"workspace", "state"} <= main_mounts
        assert "dataset" not in repair_mounts
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text()
        assert (
            "senpai_agent/repair_executor.py "
            "/usr/local/bin/senpai-repair-executor"
        ) in dockerfile


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
        for item in next(
            container
            for container in deployment["spec"]["template"]["spec"]["containers"]
            if container["name"] == "supervisor-control"
        )["env"]
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
        for item in next(
            container
            for container in deployment["spec"]["template"]["spec"]["containers"]
            if container["name"] == "supervisor-control"
        )["env"]
    }

    assert "SENPAI_OPENHANDS_FRONTIER_MODEL" not in config
    assert names == {"GITHUB_TOKEN", "WANDB_API_KEY", "OPENAI_API_KEY"}
