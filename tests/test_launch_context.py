import base64

import pytest
import yaml

from launch_test_support import launch, launch_args, render_role
from senpai_agent.launch_context import (
    INSTRUCTIONS_ROOT,
    LAUNCH_CONTEXT_ENV,
    PLACEHOLDER,
    render_launch_context,
    render_role_prompt,
)
from senpai_agent.program_context import ProgramSystemPrompt
from senpai_agent.system_instructions import SenpaiSystemInstructions


def test_default_fleet_is_four_students_with_one_gpu_each():
    args = launch_args()

    assert args.n_students == 4
    assert args.gpus_per_student == 1
    assert args.program_path == ""
    assert args.timeout_minutes == 30
    assert args.max_epochs == 50


@pytest.mark.parametrize("backend", ["kubernetes", "docker", "aws"])
def test_launch_context_records_resolved_runtime_facts(backend):
    args = launch_args(
        tag="foil-run",
        advisor_branch="research-v2",
        target_repo_branch="main",
        gpus_per_student=3,
        timeout_minutes=12.5,
        max_epochs=7,
    )

    context = render_launch_context(
        backend=backend,
        role="advisor",
        github_repo="example/problem",
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        gpus_per_student=args.gpus_per_student,
        timeout_minutes=args.timeout_minutes,
        max_epochs=args.max_epochs,
        tag=args.tag,
        advisor_branch=args.advisor_branch,
        target_base=args.target_repo_branch,
        students=["fern", "frieren"],
    )

    assert "resolved by the Senpai launcher" in context
    assert "override conflicting compute or run-limit claims" in context
    assert f"Compute backend: `{backend}`" in context
    assert "Role: `advisor`" in context
    assert "GitHub repository: `example/problem`" in context
    assert "W&B project: `wandb-applied-ai-team/senpai-v1`" in context
    assert "Visible GPUs per student: `3`" in context
    assert (
        "Hard limits for each training run: `12.5` minutes wall-clock and `7` epochs"
        in context
    )
    assert "research tag `foil-run`" in context
    assert "advisor branch `research-v2`" in context
    assert "base branch `main`" in context
    assert "fern, frieren" in context
    assert "{{" not in context


def test_launch_context_limits_each_role_to_its_assigned_students():
    args = launch_args(tag="bounded", advisor_branch="research")

    advisor = render_launch_context(
        backend="kubernetes",
        role="advisor",
        github_repo="example/problem",
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        gpus_per_student=args.gpus_per_student,
        timeout_minutes=args.timeout_minutes,
        max_epochs=args.max_epochs,
        tag=args.tag,
        advisor_branch=args.advisor_branch,
        target_base=args.target_repo_branch,
        students=["fern", "stark"],
    )
    student = render_launch_context(
        backend="kubernetes",
        role="student",
        github_repo="example/problem",
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        gpus_per_student=args.gpus_per_student,
        timeout_minutes=args.timeout_minutes,
        max_epochs=args.max_epochs,
        tag=args.tag,
        advisor_branch=args.advisor_branch,
        target_base=args.target_repo_branch,
        students=["stark"],
    )

    assert "Role: `advisor`" in advisor
    assert "Role: `student`" in student
    assert "fern, stark" in advisor
    assert "fern" not in student
    assert "stark" in student


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_each_role_receives_authoritative_launch_context(role):
    args = launch_args(
        advisor_branch="research",
        gpus_per_student=2,
        timeout_minutes=20,
        max_epochs=9,
        extra_instructions="Prefer small, measurable experiments.",
    )

    configmap, _deployment, _secret = render_role(role, args)
    data = yaml.safe_load(configmap)["data"]
    context = base64.b64decode(
        data[LAUNCH_CONTEXT_ENV], validate=True
    ).decode()
    operator = base64.b64decode(
        data["EXTRA_INSTRUCTIONS_B64"], validate=True
    ).decode()

    assert "Compute backend: `kubernetes`" in context
    assert f"Role: `{role}`" in context
    assert "GitHub repository: `example/problem`" in context
    assert "Advisor branch: `research`" in context
    assert "W&B project: `wandb-applied-ai-team/senpai-v1`" in context
    assert "Students in scope: `fern`" in context
    assert "Visible GPUs per student: `2`" in context
    assert (
        "Hard limits for each training run: `20` minutes wall-clock and `9` epochs"
        in context
    )
    if role == "student":
        assert data["SENPAI_TIMEOUT_MINUTES"] == "20"
        assert data["SENPAI_MAX_EPOCHS"] == "9"
    else:
        assert "SENPAI_TIMEOUT_MINUTES" not in data
        assert "SENPAI_MAX_EPOCHS" not in data
    assert "Prefer small, measurable experiments." not in context
    assert operator == "Prefer small, measurable experiments."


def test_long_multiline_operator_instructions_remain_mutable_literal_text():
    instructions = (
        "program.md can be found in senpai/program.md\n\n"
        "Campaign authority:\n"
        + "- Every role may submit a distinct validated candidate.\n" * 20
    )
    args = launch_args(extra_instructions=instructions)

    configmap, _deployment, _secret = render_role("advisor", args)
    data = yaml.safe_load(configmap)["data"]
    context = base64.b64decode(
        data[LAUNCH_CONTEXT_ENV], validate=True
    ).decode()
    operator = base64.b64decode(
        data["EXTRA_INSTRUCTIONS_B64"], validate=True
    ).decode()

    assert instructions.rstrip() not in context
    assert operator == instructions.rstrip()


def test_launch_context_source_is_combined():
    root = launch.ROOT / "system_instructions"

    assert (root / "SENPAI-LAUNCH-CONTEXT.md").is_file()
    assert not (root / "SENPAI-LAUNCH-RUNTIME.md").exists()
    assert not (root / "SENPAI-LAUNCH-ISOLATION.md").exists()


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_each_role_receives_the_configured_program_path(role):
    configmap, _deployment, _secret = render_role(
        role,
        launch_args(program_path="senpai/program.md"),
    )

    assert yaml.safe_load(configmap)["data"]["SENPAI_PROGRAM_PATH"] == (
        "senpai/program.md"
    )


@pytest.mark.parametrize(
    ("role", "template", "students", "role_identity", "other_identity"),
    [
        (
            "advisor",
            "ADVISOR.md",
            ["fern", "frieren"],
            "Students in scope: `fern, frieren`",
            "stark",
        ),
        (
            "student",
            "STUDENT.md",
            ["stark"],
            "Students in scope: `stark`",
            "fern",
        ),
    ],
)
def test_launch_context_owns_role_scoped_runtime_identity(
    role,
    template,
    students,
    role_identity,
    other_identity,
):
    env = {
        "GH_REPO": "acme/widgets",
        "ADVISOR_BRANCH": "research",
        "WANDB_ENTITY": "acme",
        "WANDB_PROJECT": "cfd",
        "STUDENT_NAMES": "fern,frieren",
        "STUDENT_NAME": "stark",
        "GPUS_PER_STUDENT": "2",
        "WANDB_API_KEY": "wandb-secret-sentinel",
        "GITHUB_TOKEN": "github-secret-sentinel",
        "EXTRA_INSTRUCTIONS_B64": "mutable-operator-sentinel",
    }

    role_prompt = render_role_prompt(INSTRUCTIONS_ROOT / template, role, env)
    launch_context = launch.build_launch_context(
        launch_args(
            advisor_branch="research",
            gpus_per_student=2,
            target_repo_url="https://github.com/acme/widgets.git",
            wandb_entity="acme",
            wandb_project="cfd",
        ),
        "test-track",
        students,
        backend="kubernetes",
        role=role,
    )
    system_prompt = SenpaiSystemInstructions(
        harness="Harness.",
        role=role_prompt,
        program=ProgramSystemPrompt(
            program_path="program.md",
            prompt="# program.md - program.md\n\nProgramme.",
        ),
        launch=launch_context,
    ).prompt

    assert "## Runtime identity" not in role_prompt
    assert "GitHub repository:" not in role_prompt
    assert "W&B project:" not in role_prompt
    assert "## Runtime identity" in system_prompt
    assert f"Role: `{role}`" in system_prompt
    assert "GitHub repository: `acme/widgets`" in system_prompt
    assert "Advisor branch: `research`" in system_prompt
    assert "W&B project: `acme/cfd`" in system_prompt
    assert role_identity in system_prompt
    assert other_identity not in system_prompt
    assert PLACEHOLDER.search(system_prompt) is None
    for excluded in (
        "WANDB_API_KEY",
        "wandb-secret-sentinel",
        "GITHUB_TOKEN",
        "github-secret-sentinel",
        "EXTRA_INSTRUCTIONS_B64",
        "mutable-operator-sentinel",
    ):
        assert excluded not in system_prompt


def test_advisor_role_prompt_retains_the_gpu_placeholder_value(tmp_path):
    template = tmp_path / "ADVISOR.md"
    template.write_text("GPUs: {{GPUS_PER_STUDENT}}\n")

    with pytest.raises(
        ValueError,
        match="Missing ADVISOR.md values: GPUS_PER_STUDENT",
    ):
        render_role_prompt(template, "advisor", {})

    assert render_role_prompt(
        template,
        "advisor",
        {"GPUS_PER_STUDENT": "2"},
    ) == "GPUs: 2"


def test_role_prompt_never_renders_a_secret_placeholder(tmp_path):
    template = tmp_path / "ADVISOR.md"
    template.write_text("Token: {{GITHUB_TOKEN}}\n")

    with pytest.raises(ValueError, match="Missing ADVISOR.md values: GITHUB_TOKEN"):
        render_role_prompt(
            template,
            "advisor",
            {"GITHUB_TOKEN": "github-secret-sentinel"},
        )


def test_role_prompt_rejects_an_unmapped_placeholder_containing_a_digit(tmp_path):
    template = tmp_path / "STUDENT.md"
    template.write_text("Value: {{VALUE2}}\n")

    with pytest.raises(ValueError, match="Missing STUDENT.md values: VALUE2"):
        render_role_prompt(template, "student", {})


def test_role_prompt_does_not_render_placeholders_introduced_by_values(tmp_path):
    template = tmp_path / "ADVISOR.md"
    template.write_text("GPUs: {{GPUS_PER_STUDENT}}\n")

    rendered = render_role_prompt(
        template,
        "advisor",
        {
            "GPUS_PER_STUDENT": "{{WANDB_PROJECT}}",
            "WANDB_PROJECT": "must-not-be-rendered",
        },
    )

    assert rendered == "GPUs: {{WANDB_PROJECT}}"
