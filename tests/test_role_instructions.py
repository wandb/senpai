"""Rules the advisor role instructions must keep stating."""

from pathlib import Path

ADVISOR = Path(__file__).resolve().parents[1] / "system_instructions" / "ADVISOR.md"


def section(text: str, heading: str) -> str:
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def test_critique_briefs_leave_the_conclusion_to_the_critic():
    """Both critiques behind a three-day idle decision were told the verdict."""

    text = ADVISOR.read_text(encoding="utf-8")
    critique = section(text, "### Critique and ideation briefs")

    assert "let it reach its own conclusion" in critique
    assert "Keep your own verdict and any suggested answer out of the brief" in critique
    assert "not independent evidence" in critique


def test_sustained_no_improvement_briefs_two_critics_differently():
    text = ADVISOR.read_text(encoding="utf-8")
    critique = section(text, "### Critique and ideation briefs")

    assert "brief two critics in different ways" in critique
    assert "without your list of failed ideas" in critique
    assert "most promising long shot" in critique
    assert "access to public code" in critique


def test_students_always_have_an_experiment():
    """Four students idled for days after a verdict that no candidate qualified."""

    text = ADVISOR.read_text(encoding="utf-8")
    policy = section(text, "## Always run experiments")

    assert "has an assignment at all times" in policy
    assert "bounded exploratory experiments" in policy
    assert "time box and a stopping rule" in policy
    assert "best public or library alternative" in policy
    assert "idle student costs as much as a busy one" in policy
    for word in ("lane", "issue"):
        assert word not in policy, word


def test_advisor_instructions_stay_target_agnostic():
    text = ADVISOR.read_text(encoding="utf-8")

    for word in ("kernel", "CUDA", "GPU-hours", "leaderboard", "H200"):
        assert word not in text, word
