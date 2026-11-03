"""Rules the advisor role instructions must keep stating."""

from pathlib import Path

ADVISOR = Path(__file__).resolve().parents[1] / "system_instructions" / "ADVISOR.md"


def test_critique_briefs_must_not_anchor_the_critic():
    """Both critiques behind a three-day idle decision were told the verdict."""

    text = ADVISOR.read_text(encoding="utf-8")
    section = text.split("### Critique and ideation briefs", 1)[1].split("\n## ", 1)[0]

    assert "must not state your own verdict" in section
    assert "must not hand the critic an exit" in section
    assert "confirm the verdict" in section
    assert "a confident NONE is a valid answer" in section
    assert "not independent evidence" in section


def test_sustained_no_keep_runs_two_differently_briefed_critics():
    text = ADVISOR.read_text(encoding="utf-8")
    section = text.split("### Critique and ideation briefs", 1)[1].split("\n## ", 1)[0]

    assert "no keep for a sustained period" in section
    assert "ignoring your dead-ends table" in section
    assert "best long-shot regardless of any bar" in section
    assert "access to public code" in section


def test_idle_policy_keeps_students_assigned_and_does_not_wait_for_the_operator():
    """Four students idled about 230 GPU-hours after a verdict of "no lane"."""

    text = ADVISOR.read_text(encoding="utf-8")
    section = text.split("## Idle policy", 1)[1].split("\n## ", 1)[0]

    assert "never left without an assignment while GPUs are allocated" in section
    assert "`create_human_issue`" in section
    assert "widening the research focus" in section
    assert "best public or library alternative" in section
    assert "kill criterion" in section
    assert "Do not wait for the answer" in section
    assert "every idle student has an exploratory assignment" in section


def test_advisor_instructions_stay_target_agnostic():
    text = ADVISOR.read_text(encoding="utf-8")

    for word in ("kernel", "CUDA", "GPU-hours", "leaderboard", "H200"):
        assert word not in text, word
