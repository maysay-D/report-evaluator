from __future__ import annotations

import argparse
import json

from report_evaluator import BaselineEvaluator, EvaluationRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="評価対象JSON")
    parser.add_argument("--output", default="comparison_result.json")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        req = EvaluationRequest.model_validate(json.load(f))

    baseline = BaselineEvaluator().evaluate(req)
    from report_evaluator.improved import ImprovedEvaluator
    improved = ImprovedEvaluator().evaluate(req)

    result = {
        "baseline": baseline.model_dump(),
        "improved": improved.model_dump(),
        "delta_improved_minus_baseline": {
            "overall_score_100": round(improved.overall_score_100 - baseline.overall_score_100, 2),
            "redundancy_score": round(improved.redundancy_score - baseline.redundancy_score, 2),
            "consistency_score": round(improved.consistency_score - baseline.consistency_score, 2),
            "coverage_score": round(improved.coverage_score - baseline.coverage_score, 2),
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
