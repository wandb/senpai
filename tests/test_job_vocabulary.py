import re
from pathlib import Path

import senpai_agent


LEGACY_SUPERVISION_NAMES = re.compile(
    r"\b(?:"
    r"run_training|get_training_status|cancel_training|monitor_training|"
    r"training_monitor|training-monitor|senpai_training|training_id|"
    r"training_runtime|MONITOR_TRAINING|"
    r"close_training_runtimes|training_result_paths|"
    r"(?:Run|Get|Cancel|Monitor)Training\w*|"
    r"Training(?:State|Spec|Result|Supervisor|Monitor|Status)\w*"
    r")\b"
)


def test_supervised_process_api_uses_one_job_vocabulary():
    root = Path(senpai_agent.__file__).parent.parent
    paths = [
        root / "README.md",
        root / "SPEC.md",
        *(
            path
            for path in sorted((root / "senpai_agent").glob("*.py"))
            if path.name != "persisted_conversation_migration.py"
        ),
        *sorted((root / ".agents" / "skills").glob("*/SKILL.md")),
        *sorted((root / "system_instructions").glob("*.md")),
        *sorted((root / "plugins" / "senpai" / "skills").glob("*/SKILL.md")),
    ]
    violations = {
        str(path.relative_to(root)): sorted(set(LEGACY_SUPERVISION_NAMES.findall(text)))
        for path in paths
        if (text := path.read_text(encoding="utf-8"))
        and LEGACY_SUPERVISION_NAMES.search(text)
    }

    assert not (root / "senpai_agent" / "training.py").exists()
    assert violations == {}
