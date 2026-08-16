import json
from pathlib import Path
from uuid import UUID

import pytest

from senpai_agent.state import StartedConversationLedger


def test_started_conversation_ledger_is_durable_and_idempotent(tmp_path: Path) -> None:
    first = UUID("00000000-0000-0000-0000-000000000001")
    second = UUID("00000000-0000-0000-0000-000000000002")
    path = tmp_path / "started-conversations.json"
    ledger = StartedConversationLedger(path)

    ledger.mark_started(second)
    ledger.mark_started(first)
    ledger.mark_started(second)

    restored = StartedConversationLedger(path)
    assert restored.has_started(first)
    assert restored.has_started(second)
    assert json.loads(path.read_text()) == [str(first), str(second)]


def test_started_conversation_ledger_rejects_invalid_state(tmp_path: Path) -> None:
    path = tmp_path / "started-conversations.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid conversation ledger"):
        StartedConversationLedger(path).has_started(UUID(int=1))
