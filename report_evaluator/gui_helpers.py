"""Pure presentation helpers, independent of the desktop toolkit."""
from __future__ import annotations

import re

from .schemas import EvaluationRequest


def parse_keywords(value: str) -> list[str]:
    return list(dict.fromkeys(k.strip() for k in re.split(r"[,、，\n]+", value) if k.strip()))


def make_request(text: str, prompt: str, required: str, recommended: str) -> EvaluationRequest:
    if not text.strip():
        raise ValueError("レポート本文を入力してください。")
    if not prompt.strip():
        raise ValueError("レポートの問題文を入力してください。")
    required_words = parse_keywords(required)
    return EvaluationRequest.model_validate({
        "report": {"text": text},
        "assignment": {
            "prompt": prompt,
            "required_keywords": required_words,
            "recommended_keywords": [k for k in parse_keywords(recommended) if k not in required_words],
        },
    })


def format_metrics(result: dict) -> str:
    def display(value):
        return "算出不可" if value is None else str(value)
    lines = [f"方式: {result['system']} / 評価件数: {result['n_cases']}",
             "検出指標は0〜1。Accuracyは全文／全必須キーワードを母集団として集計します。", ""]
    for key, label in [("redundancy", "冗長文"), ("contradiction", "矛盾候補"), ("required_keyword", "必須キーワード")]:
        metrics = result["detection"][key]
        lines.append(label)
        lines.append("  " + "   |   ".join(f"{name}: {display(metrics.get(field))}" for field, name in
                     [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy")]))
    lines.extend(["", "人手採点との一致（1〜5点）"])
    for key, label in [("redundancy", "冗長性"), ("consistency", "論理整合性"), ("coverage", "要求網羅度"), ("overall", "総合")]:
        m = result["score_agreement_1_to_5"][key]
        lines.append(f"{label}: n={m['n']} / Spearman={display(m['spearman'])} / MAE={display(m['mae'])} / RMSE={display(m['rmse'])}")
    lines.extend(["", "Spearmanは採点が2件未満、またはいずれかの採点が一定の場合は算出できません。",
                  "Precision・Recall・F1の分母が0の場合は既存評価処理に従い0と表示します。",
                  "同梱データでの結果は動作確認用です。性能の報告には人手で確認した正解データを使用してください。"])
    return "\n".join(lines)
