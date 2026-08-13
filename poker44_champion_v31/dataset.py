"""Parse and integrity-check a downloaded Poker44 benchmark release.

The download response contains either top-level ``items`` (the live v3.1
contract) or legacy pre-release ``dataset.items``. Each item carries a schema-v4.1
``payload`` (the model input) plus a separate supervised ``label``. Parsing is
strict: it rejects schema violations, label / labelName mismatch, duplicate item
IDs, outer ``itemId`` vs ``payload.item_id`` disagreement, and any leaked
ground-truth field inside a payload.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from poker44.contracts import find_forbidden, validate_v4_micro_session

__all__ = [
    "DatasetError",
    "ParsedRelease",
    "ReleaseRecord",
    "behavior_key",
    "parse_release",
]

_LABEL_NAMES = {0: "human", 1: "bot"}

# Behavioral fields that define an actor's strategic line. Transport identity
# (item_id / window_id / schema_version) and the raw decision_number *value* are
# deliberately excluded so the same session under a different id/window/numbering
# canonicalizes identically for cross-release leakage detection.
_BEHAVIOR_KEYS = (
    "phase",
    "position_group",
    "pressure",
    "action_type",
    "size_bucket",
    "is_all_in",
)


def behavior_key(record: ReleaseRecord) -> tuple:
    """Order-preserving canonical key of the behavioral decision payload only."""
    decisions = record.payload.get("decisions") or []
    ordered = sorted(
        (d for d in decisions if isinstance(d, dict)),
        key=lambda d: int(d["decision_number"]),
    )
    return tuple(
        tuple((k, bool(d[k]) if k == "is_all_in" else d.get(k)) for k in _BEHAVIOR_KEYS)
        for d in ordered
    )


class DatasetError(ValueError):
    """Raised when a release fails integrity or schema checks."""


@dataclass(frozen=True)
class ReleaseRecord:
    item_id: str
    payload: dict[str, Any]
    label: int
    label_name: str


@dataclass(frozen=True)
class ParsedRelease:
    release_id: str
    release_version: str
    dataset_hash: str
    records: list[ReleaseRecord] = field(default_factory=list)

    @property
    def item_count(self) -> int:
        return len(self.records)

    @property
    def class_counts(self) -> dict[int, int]:
        counts = Counter(r.label for r in self.records)
        return {label: counts.get(label, 0) for label in (0, 1)}

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [r.payload for r in self.records]

    @property
    def labels(self) -> list[int]:
        return [r.label for r in self.records]


def parse_release(data: Any) -> ParsedRelease:
    if not isinstance(data, dict):
        raise DatasetError("release payload must be a JSON object")
    dataset_obj = data.get("dataset")
    nested_items = dataset_obj.get("items") if isinstance(dataset_obj, dict) else None
    flat_items = data.get("items")
    if isinstance(flat_items, list) and isinstance(nested_items, list):
        raise DatasetError("release payload must not contain both items and dataset.items")
    items = flat_items if isinstance(flat_items, list) else nested_items
    if not isinstance(items, list):
        raise DatasetError("release payload must contain an items list")

    records: list[ReleaseRecord] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        record = _parse_item(item, index)
        if record.item_id in seen_ids:
            raise DatasetError(f"duplicate item id {record.item_id!r}")
        seen_ids.add(record.item_id)
        records.append(record)

    if not records:
        raise DatasetError("release dataset is empty; no items to ingest")
    present = {r.label for r in records}
    if present != {0, 1}:
        raise DatasetError(
            f"release must contain both classes (human and bot); found labels {sorted(present)}"
        )

    return ParsedRelease(
        release_id=str(data.get("releaseId") or ""),
        release_version=str(data.get("releaseVersion") or ""),
        dataset_hash=str(data.get("datasetHash") or ""),
        records=records,
    )


def _parse_item(item: Any, index: int) -> ReleaseRecord:
    if not isinstance(item, dict):
        raise DatasetError(f"items[{index}] must be an object")
    payload = item.get("payload")
    if not isinstance(payload, dict):
        raise DatasetError(f"items[{index}] has no payload object")
    # Reject leaked ground-truth fields anywhere inside the payload.
    leaked = find_forbidden(payload, f"items[{index}]")
    if leaked:
        raise DatasetError(
            f"items[{index}] payload contains ground-truth fields: {sorted(leaked)}"
        )
    # Exact schema-v4.1 field types/limits (stricter than the shared validator,
    # which coerces via str() and omits the 128-char maxima). Run BEFORE the
    # shared validator so numeric schema_version=4.1 and numeric/overlong ids are
    # rejected with precise messages instead of being coerced through.
    if payload.get("schema_version") != "4.1":
        raise DatasetError(
            f"items[{index}] schema_version must be the exact string '4.1'"
        )
    for key in ("item_id", "window_id"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise DatasetError(
                f"items[{index}] payload {key} must be a non-empty string of <= 128 chars"
            )

    # Strict schema-v4.1 validation of the model input.
    try:
        validate_v4_micro_session(payload, index)
    except ValueError as exc:
        raise DatasetError(f"items[{index}] failed schema validation: {exc}") from exc

    # Champion ingestion is deliberately stricter than the shared reference
    # validator: reject boolean / non-positive / duplicate decision_number even
    # though the upstream contract accepts them.
    seen_numbers: set[int] = set()
    for d_index, decision in enumerate(payload["decisions"]):
        number = decision.get("decision_number")
        if isinstance(number, bool) or not isinstance(number, int):
            raise DatasetError(
                f"items[{index}].decisions[{d_index}] decision_number must be a non-boolean integer"
            )
        if number < 1:
            raise DatasetError(
                f"items[{index}].decisions[{d_index}] decision_number must be >= 1"
            )
        if number in seen_numbers:
            raise DatasetError(
                f"items[{index}] has duplicate decision_number {number}"
            )
        seen_numbers.add(number)

    # Exact-match, coercion-free label. Booleans must not pass as 0/1.
    label = item.get("label")
    if isinstance(label, bool) or label not in (0, 1):
        raise DatasetError(f"items[{index}] label must be integer 0 or 1")
    label = int(label)

    # labelName is mandatory and must match exactly (no str() coercion).
    label_name = item.get("labelName")
    if not isinstance(label_name, str) or label_name != _LABEL_NAMES[label]:
        raise DatasetError(
            f"items[{index}] labelName {label_name!r} must equal {_LABEL_NAMES[label]!r}"
        )

    # Outer itemId is mandatory, a non-empty string, and exactly equal to the
    # payload item_id -- again with no str() coercion.
    outer_id = item.get("itemId")
    if not isinstance(outer_id, str) or not outer_id.strip():
        raise DatasetError(f"items[{index}] outer itemId must be a non-empty string")
    payload_item_id = payload.get("item_id")
    if outer_id != payload_item_id:
        raise DatasetError(
            f"items[{index}] outer itemId {outer_id!r} disagrees with "
            f"payload.item_id {payload_item_id!r}"
        )
    return ReleaseRecord(
        item_id=outer_id,
        payload=payload,
        label=label,
        label_name=label_name,
    )
