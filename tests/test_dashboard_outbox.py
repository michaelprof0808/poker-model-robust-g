import os
import sqlite3
from unittest.mock import Mock

import pytest

from poker44.validator.reporting.client import DashboardReportingClient


def event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "created_at": "2026-07-16T12:00:00Z",
        "event_type": "validation_round_started",
    }


def test_dashboard_events_survive_delivery_failure_and_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("POKER44_DASHBOARD_REPORT_URL", "http://dashboard/events")
    outbox = tmp_path / "dashboard.sqlite3"
    monkeypatch.setenv("POKER44_DASHBOARD_OUTBOX_PATH", str(outbox))
    client = DashboardReportingClient(wallet=object())
    assert os.stat(outbox).st_mode & 0o777 == 0o600
    client._deliver = Mock(side_effect=RuntimeError("offline"))

    with pytest.raises(RuntimeError, match="offline"):
        client.send(event("event-1"))
    with sqlite3.connect(outbox) as connection:
        assert connection.execute(
            "SELECT event_id,attempts FROM dashboard_event_outbox"
        ).fetchall() == [("event-1", 1)]

    client._deliver = Mock()
    client.send(event("event-2"))
    with sqlite3.connect(outbox) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM dashboard_event_outbox"
        ).fetchone() == (0,)
    assert client._deliver.call_count == 2
