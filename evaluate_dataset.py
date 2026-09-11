from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from report_evaluator import BaselineEvaluator, EvaluationRequest
from report_evaluator.common import split_sentences_with_spans, score100_to_5
from report_evaluator.dataset_io import load_dataset_records


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, pred: set, gold: set, universe: set | None = None) -> None:
        if universe is not None:
            if not (pred | gold) <= universe:
                raise ValueError("正解または予測に評価対象外の文番号・キーワードがあります")
            self.tn += len(universe - (pred | gold))
        self.tp += len(pred & gold)
        self.fp += len(pred - gold)
        self.fn += len(gold - pred)

    def metrics(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "accuracy": round((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn), 4)
            if self.tp + self.tn + self.fp + self.fn else None,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
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
    case_results = []

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
        universe = set(range(len(sentences)))
        red.add(pred_red, gold_red, universe)
        con.add(pred_con, gold_con, universe)

        required = set(req.assignment.required_keywords)
        pred_kw = {k.keyword for k in result.keywords if k.required and k.matched}
        gold_kw = set(gold.get("matched_required_keywords", [])) & required
        kw.add(pred_kw, gold_kw, required)

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

        predicted_scores_5 = {
            "redundancy": round(score100_to_5(result.redundancy_score), 3),
            "consistency": round(score100_to_5(result.consistency_score), 3),
            "coverage": round(score100_to_5(result.coverage_score), 3),
            "overall": round(float(result.overall_score_5), 3),
        }
        gold_scores_5 = {
            "redundancy": float(gold["redundancy_score_5"]) if gold.get("redundancy_score_5") is not None else None,
            "consistency": float(gold["consistency_score_5"]) if gold.get("consistency_score_5") is not None else None,
            "coverage": float(gold["coverage_score_5"]) if gold.get("coverage_score_5") is not None else None,
            "overall": float(gold["overall_score_5"]) if gold.get("overall_score_5") is not None else None,
        }
        score_errors_5 = {
            axis: (round(predicted_scores_5[axis] - gold_scores_5[axis], 3) if gold_scores_5[axis] is not None else None)
            for axis in predicted_scores_5
        }

        case_results.append({
            "case_id": case_id,
            "topic": item.get("dataset_meta", {}).get("topic"),
            "category": item.get("dataset_meta", {}).get("category"),
            "pattern": item.get("dataset_meta", {}).get("pattern"),
            "difficulty": item.get("dataset_meta", {}).get("difficulty"),
            "gold_score_5": gold_scores_5,
            "predicted_score_5": predicted_scores_5,
            "score_error_5": score_errors_5,
            "predicted_score_100": {
                "redundancy": round(float(result.redundancy_score), 2),
                "consistency": round(float(result.consistency_score), 2),
                "coverage": round(float(result.coverage_score), 2),
                "overall": round(float(result.overall_score_100), 2),
            },
            "detected": {
                "redundancy_sentences": sorted(pred_red),
                "contradiction_sentences": sorted(pred_con),
                "matched_required_keywords": sorted(pred_kw),
            },
            "gold_detection": {
                "redundancy_sentences": sorted(gold_red),
                "contradiction_sentences": sorted(gold_con),
                "matched_required_keywords": sorted(gold_kw),
            },
        })

        if gold_scores_5["overall"] is not None:
            diff = predicted_scores_5["overall"] - gold_scores_5["overall"]
            if abs(diff) >= 1.0:
                errors.append({
                    "case_id": case_id,
                    "category": "overall_score",
                    "error": "large_score_error",
                    "gold": gold_scores_5["overall"],
                    "predicted": predicted_scores_5["overall"],
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
        "case_results": case_results,
        "error_analysis": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="正解データ（JSONL / JSON配列の両方に対応）")
    parser.add_argument("--systems", nargs="+", choices=["baseline", "improved"], default=["baseline"])
    parser.add_argument("--output", default="evaluation_result.json")
    args = parser.parse_args()

    records = load_dataset_records(args.dataset)

    output = {name: evaluate_system(name, records) for name in args.systems}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
