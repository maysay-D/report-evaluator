from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_evaluator import BaselineEvaluator, EvaluationRequest
from report_evaluator.dataset_io import as_dataset_records, load_json_or_jsonl


AXES = ("overall", "redundancy", "consistency", "coverage")


def compare_single_request(payload: dict) -> dict:
    req = EvaluationRequest.model_validate(payload)

    baseline = BaselineEvaluator().evaluate(req)
    from report_evaluator.improved import ImprovedEvaluator
    improved = ImprovedEvaluator().evaluate(req)

    return {
        "mode": "single_report",
        "baseline": baseline.model_dump(),
        "improved": improved.model_dump(),
        "delta_improved_minus_baseline": {
            "overall_score_100": round(improved.overall_score_100 - baseline.overall_score_100, 2),
            "redundancy_score": round(improved.redundancy_score - baseline.redundancy_score, 2),
            "consistency_score": round(improved.consistency_score - baseline.consistency_score, 2),
            "coverage_score": round(improved.coverage_score - baseline.coverage_score, 2),
        },
    }


def _merge_case_results(baseline_eval: dict, improved_eval: dict) -> list[dict]:
    baseline_by_id = {row["case_id"]: row for row in baseline_eval.get("case_results", [])}
    improved_by_id = {row["case_id"]: row for row in improved_eval.get("case_results", [])}
    case_ids = list(baseline_by_id)
    for case_id in improved_by_id:
        if case_id not in baseline_by_id:
            case_ids.append(case_id)

    merged = []
    for case_id in case_ids:
        b = baseline_by_id.get(case_id)
        i = improved_by_id.get(case_id)
        meta = b or i or {}
        merged.append({
            "case_id": case_id,
            "topic": meta.get("topic"),
            "category": meta.get("category"),
            "pattern": meta.get("pattern"),
            "difficulty": meta.get("difficulty"),
            "gold_score_5": meta.get("gold_score_5"),
            "baseline": b,
            "improved": i,
            "improved_minus_baseline_5": {
                axis: round(
                    i["predicted_score_5"][axis] - b["predicted_score_5"][axis], 3
                )
                for axis in AXES
            } if b and i else None,
        })
    return merged


def compare_dataset(records: list[dict]) -> dict:
    from evaluate_dataset import evaluate_system

    baseline_eval = evaluate_system("baseline", records)
    improved_eval = evaluate_system("improved", records)
    return {
        "mode": "dataset",
        "n_cases": len(records),
        "cases": _merge_case_results(baseline_eval, improved_eval),
        "summary": {
            "baseline": {
                "detection": baseline_eval["detection"],
                "score_agreement_1_to_5": baseline_eval["score_agreement_1_to_5"],
                "error_analysis": baseline_eval["error_analysis"],
            },
            "improved": {
                "detection": improved_eval["detection"],
                "score_agreement_1_to_5": improved_eval["score_agreement_1_to_5"],
                "error_analysis": improved_eval["error_analysis"],
            },
        },
    }


def _fmt(value: float | None, width: int = 4) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:.2f}".rjust(width)


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "   -  "
    return f"{value:+.2f}".rjust(6)


def print_dataset_comparison(result: dict) -> None:
    cases = result.get("cases", [])
    print(f"\nDataset comparison: {len(cases)} cases")
    print("Scores are on a 1-5 scale. R=redundancy, C=consistency, K=coverage.")
    print("Δ = predicted overall - gold overall.\n")

    header = (
        f"{'CASE':<14} {'PATTERN':<27} "
        f"{'GOLD O/R/C/K':<22} "
        f"{'BASE O/R/C/K':<22} {'ΔB':>6} "
        f"{'IMPR O/R/C/K':<22} {'ΔI':>6}"
    )
    print(header)
    print("-" * len(header))

    for row in cases:
        gold = row.get("gold_score_5") or {}
        b = (row.get("baseline") or {}).get("predicted_score_5", {})
        i = (row.get("improved") or {}).get("predicted_score_5", {})
        be = (row.get("baseline") or {}).get("score_error_5", {})
        ie = (row.get("improved") or {}).get("score_error_5", {})

        def compact(scores: dict) -> str:
            return "/".join(_fmt(scores.get(axis)) for axis in AXES)

        print(
            f"{row['case_id']:<14} "
            f"{str(row.get('pattern') or '-')[:27]:<27} "
            f"{compact(gold):<22} "
            f"{compact(b):<22} {_fmt_delta(be.get('overall')):>6} "
            f"{compact(i):<22} {_fmt_delta(ie.get('overall')):>6}"
        )

    print("\nAggregate score agreement (MAE, lower is better):")
    summary = result.get("summary", {})
    for system in ("baseline", "improved"):
        metrics = summary.get(system, {}).get("score_agreement_1_to_5", {})
        parts = []
        for axis in AXES:
            mae = metrics.get(axis, {}).get("mae")
            parts.append(f"{axis}={mae if mae is not None else '-'}")
        print(f"  {system:<8}: " + ", ".join(parts))


def print_single_comparison(result: dict) -> None:
    baseline = result["baseline"]
    improved = result["improved"]
    print("\nSingle report comparison (1-5 overall, 0-100 axis scores)")
    print(
        f"Baseline: overall={baseline['overall_score_5']:.2f}/5, "
        f"R={baseline['redundancy_score']:.2f}, "
        f"C={baseline['consistency_score']:.2f}, "
        f"K={baseline['coverage_score']:.2f}"
    )
    print(
        f"Improved: overall={improved['overall_score_5']:.2f}/5, "
        f"R={improved['redundancy_score']:.2f}, "
        f"C={improved['consistency_score']:.2f}, "
        f"K={improved['coverage_score']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ベースラインと改善版を比較します。1件の評価対象JSON、"
            "JSON配列データセット、JSONLデータセットを自動判別します。"
        )
    )
    parser.add_argument("input", help="評価対象JSON / JSON配列データセット / JSONL")
    parser.add_argument("--output", default="comparison_result.json")
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="人間向け表に加えて、保存するJSON全体も標準出力へ表示する",
    )
    args = parser.parse_args()

    payload = load_json_or_jsonl(args.input)
    records = as_dataset_records(payload)

    if records is not None:
        result = compare_dataset(records)
    elif isinstance(payload, dict):
        result = compare_single_request(payload)
    else:
        raise ValueError(
            "入力形式を判定できません。1件のEvaluationRequestか、"
            "'request' と 'gold' を持つデータセットを指定してください。"
        )

    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if result["mode"] == "dataset":
        print_dataset_comparison(result)
    else:
        print_single_comparison(result)
    print(f"\nFull JSON saved to: {args.output}")

    if args.json_stdout:
        print("\n--- JSON ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
