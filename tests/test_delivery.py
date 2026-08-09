from pathlib import Path
from uuid import UUID

from senpai_agent.delivery import PendingDeliveryLedger


def test_pending_delivery_attempt_survives_restart_until_completion(tmp_path: Path):
    path = tmp_path / "pending-deliveries.json"
    conversation_id = UUID("00000000-0000-0000-0000-000000000017")

    first = PendingDeliveryLedger(path).claim(conversation_id, ("event:first",))
    reopened = PendingDeliveryLedger(path)
    retried = reopened.claim(
        conversation_id,
        ("event:first", "event:second"),
    )

    assert retried["event:first"] == first["event:first"]
    assert retried["event:second"] != first["event:first"]

    reopened.complete(conversation_id, tuple(retried))
    next_attempt = PendingDeliveryLedger(path).claim(
        conversation_id,
        ("event:first",),
    )
    assert next_attempt["event:first"] != first["event:first"]
