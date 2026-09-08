from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_or_jsonl(path: str | Path) -> Any:
    """Load either a regular JSON document or a JSONL file.

    Supported inputs:
    - one EvaluationRequest JSON object
    - one dataset-record JSON object (contains ``request`` and ``gold``)
    - a JSON array of dataset records (e.g. *.pretty.json)
    - a JSONL file with one dataset record per line
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    try:
        return json.loads(text)
    except json.JSONDecodeError as json_error:
        records: list[Any] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as line_error:
                raise ValueError(
                    f"{path} はJSONまたはJSONLとして解釈できません "
                    f"(line {line_no}: {line_error.msg})"
                ) from json_error
        if not records:
            raise ValueError(f"{path} に評価データがありません") from json_error
        return records


def as_dataset_records(payload: Any) -> list[dict] | None:
    """Return dataset records when payload is dataset-shaped, otherwise None."""
    if isinstance(payload, dict) and "request" in payload and "gold" in payload:
        return [payload]

    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(item, dict) and "request" in item and "gold" in item for item in payload):
            return payload

    return None


def load_dataset_records(path: str | Path) -> list[dict]:
    payload = load_json_or_jsonl(path)
    records = as_dataset_records(payload)
    if records is None:
        raise ValueError(
            "データセット形式ではありません。各事例に 'request' と 'gold' が必要です。"
        )
    return records
