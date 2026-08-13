"""Best-effort signed event delivery to the read-only dashboard backend."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any
from urllib.request import Request, urlopen

from poker44.utils.runtime_info import build_signed_runtime_request


class DashboardReportingClient:
    def __init__(self, wallet: Any):
        self.wallet = wallet
        self.url = os.getenv(
            "POKER44_DASHBOARD_REPORT_URL",
            "https://api.poker44.net/api/v1/validator-events",
        ).strip()
        self.timeout = float(os.getenv("POKER44_DASHBOARD_REPORT_TIMEOUT_SECONDS", "5"))
        self.outbox_path = os.path.expanduser(
            os.getenv(
                "POKER44_DASHBOARD_OUTBOX_PATH",
                "~/.poker44/dashboard-report-outbox.sqlite3",
            )
        )
        os.makedirs(os.path.dirname(self.outbox_path) or ".", exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dashboard_event_outbox (
                    event_id TEXT PRIMARY KEY,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT
                )
                """
            )
        os.chmod(self.outbox_path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.outbox_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _deliver(self, event: dict[str, Any]) -> None:
        body = json.dumps(event, sort_keys=True).encode("utf-8")
        signed = build_signed_runtime_request(
            wallet=self.wallet, url=self.url, payload=event, method="POST"
        )
        request = Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "content-type": "application/json",
                "x-validator-hotkey": signed["hotkey_ss58"],
                "x-validator-signature": signed["signature_hex"],
                "x-validator-nonce": signed["nonce"],
                "x-validator-timestamp": str(signed["timestamp"]),
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            response.read()

    def send(self, event: dict[str, Any]) -> None:
        if not self.url:
            return
        serialized = json.dumps(event, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO dashboard_event_outbox(
                    event_id,event_json,created_at
                ) VALUES(?,?,?)
                """,
                (event["event_id"], serialized, event["created_at"]),
            )
            # Persist before attempting network delivery so an exception cannot
            # roll the newly queued event back out of the outbox.
            connection.commit()
            pending = connection.execute(
                """
                SELECT event_id,event_json FROM dashboard_event_outbox
                ORDER BY created_at,event_id LIMIT 100
                """
            ).fetchall()
            for event_id, event_json in pending:
                try:
                    self._deliver(json.loads(event_json))
                    connection.execute(
                        "DELETE FROM dashboard_event_outbox WHERE event_id=?",
                        (event_id,),
                    )
                    connection.commit()
                except Exception as exc:
                    connection.execute(
                        """
                        UPDATE dashboard_event_outbox
                        SET attempts=attempts+1,last_error=? WHERE event_id=?
                        """,
                        (str(exc)[:1000], event_id),
                    )
                    connection.commit()
                    raise
