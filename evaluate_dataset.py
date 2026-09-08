from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from report_evaluator import BaselineEvaluator, EvaluationRequest
from report_evaluator.common import split_sentences_with_spans, score100_to_5


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, pred: set, gold: set) -> None:
        self.tp += len(pred & gold)
        self.fp += len(pred - gold)
        self.fn += len(gold - pred)

    def metrics(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def build_evaluator(name: str):
    if name == "baseline":
        return BaselineEvaluator()
    if name == "improved":
        from report_evaluator.improved import ImprovedEvaluator
        return ImprovedEvaluator()
    raise ValueError(name)


def safe_spearman(gold: list[float], pred: list[float]) -> float | None:
    if len(gold) < 2 or len(set(gold)) < 2 or len(set(pred)) < 2:
        return None
    value = spearmanr(gold, pred).statistic
    return None if np.isnan(value) else round(float(value), 4)


def score_metrics(gold: list[float], pred: list[float]) -> dict:
    if not gold:
        return {"n": 0, "mae": None, "rmse": None, "spearman": None}
    g = np.asarray(gold, dtype=float)
    p = np.asarray(pred, dtype=float)
    return {
        "n": len(gold),
        "mae": round(float(np.mean(np.abs(g - p))), 4),
        "rmse": round(float(np.sqrt(np.mean((g - p) ** 2))), 4),
        "spearman": safe_spearman(gold, pred),
    }


def evaluate_system(name: str, records: list[dict]) -> dict:
    evaluator = build_evaluator(name)
    red = Counts()
    con = Counts()
    kw = Counts()
    score_pairs: dict[str, tuple[list[float], list[float]]] = {
        "redundancy": ([], []),
        "consistency": ([], []),
        "coverage": ([], []),
        "overall": ([], []),
    }
    errors = []

    for item in records:
        case_id = str(item.get("case_id", "unknown"))
        req = EvaluationRequest.model_validate(item["request"])
        gold = item["gold"]
        result = evaluator.evaluate(req)
        sentence_spans = split_sentences_with_spans(req.report.text)
        sentences = [s[2] for s in sentence_spans]

        pred_red = {h.sentence_index for h in result.highlights if h.type == "redundancy" and h.sentence_index is not None}
        pred_con = {h.sentence_index for h in result.highlights if h.type == "contradiction" and h.sentence_index is not None}
        gold_red = set(gold.get("redundancy_sentences", []))
        gold_con = set(gold.get("contradiction_sentences", []))
        red.add(pred_red, gold_red)
        con.add(pred_con, gold_con)

        required = set(req.assignment.required_keywords)
        pred_kw = {k.keyword for k in result.keywords if k.required and k.matched}
        gold_kw = set(gold.get("matched_required_keywords", [])) & required
        kw.add(pred_kw, gold_kw)

        for axis, result_score in [
            ("redundancy", result.redundancy_score),
            ("consistency", result.consistency_score),
            ("coverage", result.coverage_score),
            ("overall", result.overall_score_100),
        ]:
            key = f"{axis}_score_5"
            if key in gold and gold[key] is not None:
                score_pairs[axis][0].append(float(gold[key]))
                score_pairs[axis][1].append(score100_to_5(result_score))

        for error_type, pred_set, gold_set in [
            ("redundancy", pred_red, gold_red),
            ("contradiction", pred_con, gold_con),
        ]:
            for idx in sorted(pred_set - gold_set):
                errors.append({
                    "case_id": case_id,
                    "category": error_type,
                    "error": "false_positive",
                    "sentence_index": idx,
                    "sentence": sentences[idx] if 0 <= idx < len(sentences) else None,
                })
            for idx in sorted(gold_set - pred_set):
                errors.append({
                    "case_id": case_id,
                    "category": error_type,
                    "error": "false_negative",
                    "sentence_index": idx,
                    "sentence": sentences[idx] if 0 <= idx < len(sentences) else None,
                })

        for keyword in sorted(pred_kw - gold_kw):
            errors.append({"case_id": case_id, "category": "keyword", "error": "false_positive", "keyword": keyword})
        for keyword in sorted(gold_kw - pred_kw):
            errors.append({"case_id": case_id, "category": "keyword", "error": "false_negative", "keyword": keyword})

        if "overall_score_5" in gold:
            predicted = score100_to_5(result.overall_score_100)
            diff = predicted - float(gold["overall_score_5"])
            if abs(diff) >= 1.0:
                errors.append({
                    "case_id": case_id,
                    "category": "overall_score",
                    "error": "large_score_error",
                    "gold": gold["overall_score_5"],
                    "predicted": round(predicted, 3),
                    "difference": round(diff, 3),
                })

    return {
        "system": name,
        "n_cases": len(records),
        "detection": {
            "redundancy": red.metrics(),
            "contradiction": con.metrics(),
            "required_keyword": kw.metrics(),
        },
        "score_agreement_1_to_5": {
            axis: score_metrics(gold, pred) for axis, (gold, pred) in score_pairs.items()
        },
        "error_analysis": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="1行1事例のJSONL正解データ")
    parser.add_argument("--systems", nargs="+", choices=["baseline", "improved"], default=["baseline"])
    parser.add_argument("--output", default="evaluation_result.json")
    args = parser.parse_args()

    records = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    output = {name: evaluate_system(name, records) for name in args.systems}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
