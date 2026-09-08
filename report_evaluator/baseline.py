from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .common import (
    char_ngrams,
    normalize_text,
    repeat_ratio,
    split_sentences_with_spans,
    weighted_overall,
)
from .schemas import EvaluationRequest, EvaluationResponse, Highlight, KeywordResult


NEGATION_PATTERNS = (
    "ない", "ません", "ではない", "でない", "不可能", "できない", "否定", "異ならない"
)


class BaselineEvaluator:
    """表層情報中心の比較用ベースライン。

    - 冗長性: 文字 n-gram 反復 + TF-IDF 文類似度
    - 論理整合性: 類似した隣接文で否定極性が反転しているかという単純ルール
    - 要求網羅度: 必須/推奨キーワードの完全一致のみ
    """

    def evaluate(self, req: EvaluationRequest) -> EvaluationResponse:
        text = req.report.text
        rubric = req.assignment.rubric
        spans = split_sentences_with_spans(text)
        sentences = [s[2] for s in spans]

        redundancy, red_hl, red_diag = self._redundancy(
            text, sentences, spans, rubric.baseline_sentence_similarity_threshold
        )
        consistency, con_hl, con_diag = self._consistency(
            sentences, spans, rubric.baseline_contradiction_similarity_threshold
        )
        coverage, kw_results, kw_hl, cov_diag = self._coverage(
            text,
            req.assignment.required_keywords,
            req.assignment.recommended_keywords,
        )

        overall_100, overall_5 = weighted_overall(
            redundancy, consistency, coverage, rubric
        )
        missing = [r.keyword for r in kw_results if r.required and not r.matched]

        return EvaluationResponse(
            system="baseline",
            overall_score_100=overall_100,
            overall_score_5=overall_5,
            redundancy_score=round(redundancy, 2),
            consistency_score=round(consistency, 2),
            coverage_score=round(coverage, 2),
            highlights=sorted(red_hl + con_hl + kw_hl, key=lambda h: (h.start, h.end, h.type)),
            keywords=kw_results,
            missing_required_keywords=missing,
            named_entities=[],
            diagnostics={
                "redundancy": red_diag,
                "consistency": con_diag,
                "coverage": cov_diag,
                "method_note": "表層的な比較手法。意味表現やNLIは使用しない。",
            },
        )

    @staticmethod
    def _tfidf_similarity_matrix(sentences: list[str]) -> np.ndarray:
        if len(sentences) < 2:
            return np.eye(max(1, len(sentences)))
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        x = vec.fit_transform(sentences)
        return cosine_similarity(x)

    def _redundancy(self, text, sentences, spans, threshold):
        bi = repeat_ratio(char_ngrams(text, 2))
        tri = repeat_ratio(char_ngrams(text, 3))
        lexical_penalty = min(1.0, 0.6 * bi + 0.4 * tri)

        highlights: list[Highlight] = []
        duplicate_pairs = []
        pair_penalties = []
        if len(sentences) >= 2:
            sims = self._tfidf_similarity_matrix(sentences)
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    sim = float(sims[i, j])
                    if sim >= threshold:
                        p = (sim - threshold) / max(1e-6, 1.0 - threshold)
                        pair_penalties.append(p)
                        duplicate_pairs.append({"i": i, "j": j, "tfidf_similarity": round(sim, 4)})
                        start, end, sent = spans[j]
                        highlights.append(
                            Highlight(
                                start=start,
                                end=end,
                                type="redundancy",
                                reason=f"TF-IDF上で文{i+1}と類似 ({sim:.2f})",
                                confidence=min(1.0, sim),
                                sentence_index=j,
                                text=sent,
                            )
                        )

        sentence_penalty = min(1.0, sum(pair_penalties) / max(1, len(sentences) - 1)) if pair_penalties else 0.0
        penalty = min(1.0, 0.45 * lexical_penalty + 0.55 * sentence_penalty)
        return 100.0 * (1.0 - penalty), highlights, {
            "bigram_repeat_ratio": round(bi, 4),
            "trigram_repeat_ratio": round(tri, 4),
            "tfidf_duplicate_pairs": duplicate_pairs,
        }

    @staticmethod
    def _has_negation(sentence: str) -> bool:
        return any(p in sentence for p in NEGATION_PATTERNS)

    def _consistency(self, sentences, spans, threshold):
        if len(sentences) < 2:
            return 100.0, [], {"candidate_pairs": []}
        sims = self._tfidf_similarity_matrix(sentences)
        contradictions = []
        highlights: list[Highlight] = []
        for i in range(1, len(sentences)):
            sim = float(sims[i - 1, i])
            neg_flip = self._has_negation(sentences[i - 1]) != self._has_negation(sentences[i])
            if sim >= threshold and neg_flip:
                contradictions.append({
                    "i": i - 1,
                    "j": i,
                    "tfidf_similarity": round(sim, 4),
                    "negation_flip": True,
                })
                start, end, sent = spans[i]
                confidence = min(1.0, 0.5 + 0.5 * sim)
                highlights.append(
                    Highlight(
                        start=start,
                        end=end,
                        type="contradiction",
                        reason=f"類似する隣接文で否定極性が反転 (TF-IDF={sim:.2f})",
                        confidence=confidence,
                        sentence_index=i,
                        text=sent,
                    )
                )
        penalty = len(contradictions) / max(1, len(sentences) - 1)
        return 100.0 * (1.0 - min(1.0, penalty)), highlights, {
            "candidate_pairs": contradictions,
            "note": "論理矛盾を意味理解せず、類似度と否定語だけで近似するベースライン。",
        }

    def _coverage(self, text, required, recommended):
        normalized_text = normalize_text(text)
        all_keywords = [(k, True) for k in required] + [(k, False) for k in recommended]
        if not all_keywords:
            return 100.0, [], [], {"exact_only": True}

        results: list[KeywordResult] = []
        highlights: list[Highlight] = []
        credits = 0.0
        total_weight = 0.0
        for kw, is_required in all_keywords:
            matched = normalize_text(kw) in normalized_text
            weight = 2.0 if is_required else 1.0
            total_weight += weight
            if matched:
                credits += weight
                for m in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
                    highlights.append(
                        Highlight(
                            start=m.start(),
                            end=m.end(),
                            type="keyword",
                            reason=f"完全一致キーワード: {kw}",
                            confidence=1.0,
                            text=m.group(0),
                        )
                    )
            results.append(
                KeywordResult(
                    keyword=kw,
                    required=is_required,
                    matched=matched,
                    match_type="exact" if matched else "missing",
                    confidence=1.0 if matched else 0.0,
                )
            )
        score = 100.0 * credits / max(1e-6, total_weight)
        return score, results, highlights, {"exact_only": True}
