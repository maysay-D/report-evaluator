from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .schemas import Rubric


_SENT_END = re.compile(r"[^。！？!?\n]+[。！？!?]?|[^\n]+$")


def split_sentences_with_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for m in _SENT_END.finditer(text):
        raw = m.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        start = m.start() + left_trim
        end = m.end() - right_trim
        spans.append((start, end, text[start:end]))
    return spans


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", s).casefold()


def char_ngrams(text: str, n: int) -> list[str]:
    text = normalize_text(text)
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def repeat_ratio(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    repeated_occurrences = sum(max(0, count - 1) for count in counts.values())
    return repeated_occurrences / len(items)


def weighted_overall(
    redundancy: float,
    consistency: float,
    coverage: float,
    rubric: Rubric,
) -> tuple[float, float]:
    weights = np.asarray(
        [rubric.redundancy_weight, rubric.consistency_weight, rubric.coverage_weight],
        dtype=float,
    )
    if float(weights.sum()) <= 0:
        weights[:] = 1.0
    weights /= weights.sum()
    overall_100 = float(np.dot([redundancy, consistency, coverage], weights))
    overall_5 = 1.0 + 4.0 * overall_100 / 100.0
    return round(overall_100, 2), round(overall_5, 2)


def score100_to_5(score: float) -> float:
    return 1.0 + 4.0 * float(score) / 100.0
