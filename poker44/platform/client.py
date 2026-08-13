"""Signed HTTP client for evaluation windows and session leases."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from poker44.platform.models import SessionLease
from poker44.utils.runtime_info import build_signed_runtime_request


class SubnetDataHttpError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"subnet data http_{status}: {detail}")
        self.status = status


@dataclass(frozen=True)
class SubnetDataConfig:
    base_url: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "SubnetDataConfig":
        base_url = os.getenv("POKER44_SUBNET_DATA_URL", "https://api.poker44.net")
        return cls(
            base_url=base_url.strip().rstrip("/"),
            timeout_seconds=float(
                os.getenv("POKER44_SUBNET_DATA_TIMEOUT_SECONDS", "30")
            ),
        )


class SubnetDataClient:
    def __init__(self, config: SubnetDataConfig, wallet: Any):
        self.config = config
        self.wallet = wallet

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        url = f"{self.config.base_url}{path}"
        body = None
        if payload is not None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
        signed = build_signed_runtime_request(
            wallet=self.wallet,
            url=url,
            payload=payload,
            method=method,
        )
        request = Request(
            url,
            data=body,
            method=method,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "x-validator-hotkey": signed["hotkey_ss58"],
                "x-validator-signature": signed["signature_hex"],
                "x-validator-nonce": signed["nonce"],
                "x-validator-timestamp": str(signed["timestamp"]),
            },
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise SubnetDataHttpError(exc.code, detail) from exc
        if isinstance(decoded, dict) and decoded.get("success") is True:
            return decoded.get("data")
        return decoded

    def current_window(self) -> dict[str, Any] | None:
        result = self._request("GET", "/api/v1/evaluation/windows/current")
        return result if isinstance(result, dict) else None

    def acquire_lease(self, window_id: str) -> SessionLease:
        result = self._request(
            "POST",
            "/api/v1/evaluation/leases",
            {"window_id": window_id},
        )
        return SessionLease.from_payload(result)

    def complete_lease(self, lease_id: str, round_id: str) -> None:
        self._request(
            "POST",
            f"/api/v1/evaluation/leases/{lease_id}/complete",
            {"round_id": round_id},
        )

    def claim_evaluation_run(self, window_id: str, round_id: str) -> dict[str, Any]:
        result = self._request(
            "POST",
            "/api/v1/evaluation/evaluation-runs/claim",
            {"window_id": window_id, "round_id": round_id},
        )
        if not isinstance(result, dict) or not str(result.get("state") or ""):
            raise RuntimeError("subnet data returned an invalid evaluation run")
        return result

    def advance_evaluation_run(
        self,
        window_id: str,
        round_id: str,
        state: str,
        evidence: dict[str, Any] | None = None,
    ) -> bool:
        try:
            self._request(
                "POST",
                "/api/v1/evaluation/evaluation-runs/advance",
                {
                    "window_id": window_id,
                    "round_id": round_id,
                    "state": state,
                    "evidence": evidence or {},
                },
            )
            return True
        except SubnetDataHttpError as exc:
            if exc.status == 409:
                return False
            raise
