import yaml

from launch_test_support import (
    ROOT,
    launch,
    launch_args,
    launch_helpers,
    render_role,
)


def rendered_supervisor(args=None):
    args = launch_args(operational_supervisor=True) if args is None else args
    args.supervisor_state_pvc_claim_name = "senpai-supervisor-state-test-track"
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
    assert config["supervisor_network_policy_enforced"] is False
    assert config["supervisor_state_pvc_claim_name"] == ""
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
        "senpai.wandb.com/management-protocol": launch.MANAGEMENT_PROTOCOL_VERSION,
        "senpai.wandb.com/repair-protocol": launch.REPAIR_PROTOCOL_VERSION,
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
    initializers = {
        container["name"]: container for container in pod["initContainers"]
    }
    state_initializer = initializers["state-directory"]
    source_initializer = initializers["source"]
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
    assert 'chmod 0700 "/state-root/test-track/operational-supervisor"' in (
        state_initializer["args"][0]
    )
    assert source_initializer["name"] == "source"
    assert source_initializer["volumeMounts"] == [
        {"name": "runtime", "mountPath": "/workspace"},
        {"name": "source-tmp", "mountPath": "/tmp"},
    ]
    assert next(
        volume
        for volume in pod["volumes"]
        if volume["name"] == "source-tmp"
    )["emptyDir"]["sizeLimit"] == "1Gi"
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
    assert {volume["name"]: volume for volume in pod["volumes"]}["state"] == {
        "name": "state",
        "persistentVolumeClaim": {
            "claimName": "senpai-supervisor-state-test-track"
        },
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
    assert credential_names == {
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "KUBECONFIG",
    }
    assert "envFrom" not in shell
    assert all(
        fragment not in entry["name"]
        for entry in shell.get("env", [])
        for fragment in ("TOKEN", "KEY", "SECRET", "CREDENTIAL")
    )
    assert shell["securityContext"]["readOnlyRootFilesystem"] is True
    assert control["securityContext"]["readOnlyRootFilesystem"] is True
    assert shell["resources"]["requests"]["ephemeral-storage"] == "1Gi"
    assert shell["resources"]["limits"]["ephemeral-storage"] == "20Gi"

    control_mounts = {mount["name"]: mount for mount in control["volumeMounts"]}
    shell_mounts = {mount["name"]: mount for mount in shell["volumeMounts"]}
    assert control_mounts["service-account"]["mountPath"] == (
        "/var/run/secrets/kubernetes.io/serviceaccount"
    )
    assert "service-account" not in shell_mounts
    assert "state" not in shell_mounts
    assert control_mounts["kubeconfig"] == {
        "name": "kubeconfig",
        "mountPath": "/var/run/senpai-kubeconfig",
        "readOnly": True,
    }
    assert "kubeconfig" not in shell_mounts
    assert next(
        item["value"] for item in control["env"] if item["name"] == "KUBECONFIG"
    ) == "/var/run/senpai-kubeconfig/config"
    assert not any(item["name"] == "KUBECONFIG" for item in shell.get("env", []))
    assert control_mounts["control-home"]["mountPath"] == "/home/senpai"
    assert control_mounts["control-tmp"]["mountPath"] == "/tmp"
    assert "readOnly" not in control_mounts["repair-socket"]
    assert shell_mounts["repair-socket"]["readOnly"] is True
    assert control_mounts["state"]["subPath"] == "test-track/operational-supervisor"
    assert set(shell_mounts) == {
        "runtime",
        "repair-socket",
        "shell-workspace",
        "shell-tmp",
    }
    token_projection = volumes["service-account"]["projected"]["sources"][0][
        "serviceAccountToken"
    ]
    assert token_projection["expirationSeconds"] == 3600
    assert "audience" not in token_projection
    assert volumes["shell-workspace"]["emptyDir"]["sizeLimit"] == "8Gi"
    assert volumes["shell-tmp"]["emptyDir"]["sizeLimit"] == "8Gi"
    assert volumes["control-home"]["emptyDir"]["sizeLimit"] == "4Gi"
    assert volumes["control-tmp"]["emptyDir"]["sizeLimit"] == "8Gi"


def test_supervisor_prepares_a_tokenfile_kubeconfig_for_control_only():
    documents, _secret = rendered_supervisor()
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    initializer = next(
        container
        for container in pod["initContainers"]
        if container["name"] == "kubeconfig"
    )
    script = initializer["args"][0]

    assert "tokenFile: /var/run/secrets/kubernetes.io/serviceaccount/token" in script
    assert (
        "certificate-authority: "
        "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    ) in script
    assert "KUBERNETES_SERVICE_HOST" in script
    assert "KUBERNETES_SERVICE_PORT_HTTPS" in script
    assert {mount["name"] for mount in initializer["volumeMounts"]} == {
        "service-account",
        "kubeconfig",
    }
    assert pod["volumes"][-1] == {"name": "kubeconfig", "emptyDir": {}}


def test_supervisor_capable_pods_guard_metadata_before_loading_credentials():
    documents, _secret = rendered_supervisor()
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    guard = pod["initContainers"][0]

    assert guard["name"] == "metadata-egress-guard"
    assert guard["command"] == ["/usr/local/bin/senpai-metadata-egress-guard"]
    assert "env" not in guard and "envFrom" not in guard

    args = launch_args(operational_supervisor=True)
    _config, manifest, _secret = render_role("advisor", args)
    role = yaml.safe_load(manifest)["spec"]["template"]["spec"]
    role_guard = role["initContainers"][0]
    assert role_guard["name"] == "metadata-egress-guard"
    assert "env" not in role_guard and "envFrom" not in role_guard


def test_supervisor_health_uses_authoritative_terminal_and_repair_checks():
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
        assert "@senpai-isolated-terminal" in control_health
        assert "test -S /run/senpai-repair/repair.sock" in control_health
        assert shell[probe_name]["exec"]["command"] == [
            "/opt/senpai-venv/bin/python",
            "-I",
            "-m",
            "senpai_agent.isolated_terminal",
            "health",
            "--socket",
            "@senpai-isolated-terminal",
        ]


def test_supervised_roles_have_protocol_bound_repair_of_the_target_workspace():
    args = launch_args(tag="campaign-a", operational_supervisor=True)
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
        config = yaml.safe_load(manifest.split("\n---\n", 1)[0])["data"]
        containers = {container["name"]: container for container in pod["containers"]}
        main = containers[role]
        repair = containers["repair"]
        main_mounts = {mount["name"]: mount for mount in main["volumeMounts"]}
        repair_mounts = {mount["name"]: mount for mount in repair["volumeMounts"]}

        assert pod["automountServiceAccountToken"] is False
        assert pod["shareProcessNamespace"] is False
        assert deployment["metadata"]["annotations"][
            "senpai.wandb.com/repair-protocol"
        ] == launch.REPAIR_PROTOCOL_VERSION
        assert deployment["metadata"]["annotations"][
            "senpai.wandb.com/management-protocol"
        ] == launch.MANAGEMENT_PROTOCOL_VERSION
        assert "env" not in repair
        assert "envFrom" not in repair
        assert repair["securityContext"]["readOnlyRootFilesystem"] is True
        assert repair["resources"]["requests"]["ephemeral-storage"] == "256Mi"
        assert repair["resources"]["limits"]["ephemeral-storage"] == "12Gi"
        assert set(repair_mounts) == {
            "target-workspace",
            "state",
            "repair-scratch",
            "repair-tmp",
            "repair-executor-socket",
        }
        assert "repair-executor-socket" not in main_mounts
        assert repair_mounts["repair-executor-socket"]["mountPath"] == (
            "/run/senpai-repair-executor"
        )
        assert all(mount["mountPath"] != "/home/senpai" for mount in repair_mounts.values())
        assert repair_mounts["target-workspace"]["mountPath"] == "/repair/workspace"
        assert "runner" not in repair_mounts
        assert main_mounts["runner"] == {
            "name": "runner",
            "mountPath": "/workspace/senpai",
            "readOnly": True,
        }
        assert main_mounts["target-workspace"]["mountPath"] == "/workspace/target"
        assert config["SENPAI_WORKDIR"] == "/workspace/senpai"
        assert config["SENPAI_TARGET_WORKDIR"] == "/workspace/target"
        assert config["SENPAI_SKIP_EDITABLE_INSTALL"] == "1"
        source = next(item for item in pod["initContainers"] if item["name"] == "source")
        assert source["volumeMounts"] == [
            {"name": "runner", "mountPath": "/workspace/senpai"}
        ]
        assert "git init /workspace/senpai" in source["args"][0]
        assert "git init /workspace/senpai" not in main["args"][0]
        if role == "advisor":
            assert config["SENPAI_IMMUTABLE_ADVISOR_GUIDANCE_FILE"] == (
                "/workspace/senpai/.senpai/ADVISOR.md"
            )
            assert "envsubst" in source["args"][0]
            assert '"$SENPAI_IMMUTABLE_ADVISOR_GUIDANCE_FILE"' in source["args"][0]
        assert repair["command"] == [
            "/opt/senpai-venv/bin/python",
            "-I",
            "/usr/local/bin/senpai-repair-executor",
            "serve",
            "--socket",
            "/run/senpai-repair-executor/executor.sock",
        ]
        expected_health = [
            "/opt/senpai-venv/bin/python",
            "-I",
            "/usr/local/bin/senpai-repair-executor",
            "health",
            "--socket",
            "/run/senpai-repair-executor/executor.sock",
        ]
        assert repair["startupProbe"]["exec"]["command"] == expected_health
        assert repair["livenessProbe"]["exec"]["command"] == expected_health
        assert "dataset" not in repair_mounts
        volumes = {volume["name"]: volume for volume in pod["volumes"]}
        assert volumes["repair-scratch"]["emptyDir"]["sizeLimit"] == "4Gi"
        assert volumes["repair-tmp"]["emptyDir"]["sizeLimit"] == "8Gi"
        assert volumes["repair-executor-socket"]["emptyDir"] == {
            "medium": "Memory",
            "sizeLimit": "1Mi",
        }
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text()
        assert (
            "senpai_agent/repair_executor.py "
            "/usr/local/bin/senpai-repair-executor"
        ) in dockerfile

    advisor_dockerfile = (ROOT / "Dockerfile.advisor").read_text()
    role_shell = (ROOT / "k8s" / "senpai-role-shell").read_text()
    assert "k8s/senpai-role-shell /usr/local/bin/senpai-role-shell" in (
        advisor_dockerfile
    )
    assert "/opt/senpai-venv/bin/python -I -m" in role_shell


def test_unsupervised_roles_do_not_expose_repair_execution_or_policy_labels():
    args = launch_args(tag="campaign-a", operational_supervisor=False)
    for role in ("advisor", "student"):
        secret = launch_helpers.render_launch_secret(
            args.tag, "github", "exa", "wandb", openai_api_key="openai"
        )
        template = (ROOT / "k8s" / f"{role}-deployment.yaml").read_text()
        manifest = (
            launch.render_advisor(template, args.tag, ["fern"], "secret", secret, args)
            if role == "advisor"
            else launch.render_student(
                template, "fern", args.tag, "secret", secret, args
            )
        )
        deployment = yaml.safe_load(manifest.split("\n---\n", 1)[1])
        pod = deployment["spec"]["template"]
        config = yaml.safe_load(manifest.split("\n---\n", 1)[0])["data"]

        assert {item["name"] for item in pod["spec"]["containers"]} == {role}
        assert "initContainers" not in pod["spec"]
        main = pod["spec"]["containers"][0]
        mounts = {mount["name"]: mount for mount in main["volumeMounts"]}
        assert mounts["workspace"] == {
            "name": "workspace",
            "mountPath": "/workspace",
        }
        assert {volume["name"] for volume in pod["spec"]["volumes"]} >= {
            "workspace",
            "state",
            "dataset",
        }
        assert "runner" not in {volume["name"] for volume in pod["spec"]["volumes"]}
        assert "SENPAI_TARGET_WORKDIR" not in config
        assert not any(
            volume["name"].startswith("repair-")
            for volume in pod["spec"]["volumes"]
        )
        assert "senpai.wandb.com/repair-protocol" not in deployment["metadata"].get(
            "annotations", {}
        )
        assert "senpai-supervisor-access" not in pod["metadata"]["labels"]


def test_advisor_clone_modes_always_use_the_explicit_target_workdir():
    entrypoint = (ROOT / "k8s" / "entrypoint-advisor.sh").read_text()

    assert 'repo) git clone "$TARGET_REPO_URL" "$TARGET_WORKDIR" ;;' in entrypoint
    assert 'git clone "$TARGET_REPO_URL" "$PROBLEM_DIR"' not in entrypoint
    assert entrypoint.count('"$TARGET_REPO_URL" "$TARGET_WORKDIR"') == 3


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
        if "valueFrom" in item
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
        entrypoint.index(
            "exec /usr/local/bin/senpai-run-controller operational-supervisor"
        )
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
        if "valueFrom" in item
    }

    assert "SENPAI_OPENHANDS_FRONTIER_MODEL" not in config
    assert names == {"GITHUB_TOKEN", "WANDB_API_KEY", "OPENAI_API_KEY"}
