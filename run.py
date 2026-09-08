from __future__ import annotations

import argparse
import json

from report_evaluator import BaselineEvaluator, EvaluationRequest


def build_evaluator(name: str):
    if name == "baseline":
        return BaselineEvaluator()
    if name == "improved":
        from report_evaluator.improved import ImprovedEvaluator
        return ImprovedEvaluator()
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="評価対象JSON")
    parser.add_argument("--system", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--output", help="出力JSON。省略時は標準出力")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        req = EvaluationRequest.model_validate(json.load(f))

    result = build_evaluator(args.system).evaluate(req)
    text = result.model_dump_json(indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
