from __future__ import annotations

import re

import numpy as np
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
import torch

from .common import (
    char_ngrams,
    normalize_text,
    repeat_ratio,
    split_sentences_with_spans,
    weighted_overall,
)
from .schemas import EvaluationRequest, EvaluationResponse, Highlight, KeywordResult


class ImprovedEvaluator:
    """意味表現と自然言語推論を導入した改善版。

    - 冗長性: n-gram + Sentence-BERT文類似度
    - 論理整合性: NLIによる contradiction 確率
    - 要求網羅度: 完全一致 + Sentence-BERT意味一致
    - 補助解析: GiNZA NER + TF-IDF参照差分
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        nli_model: str = "akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick",
        spacy_model: str = "ja_ginza",
        device: str | None = None,
    ) -> None:
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        if device:
            self.nli_model.to(device)
        self.nli_model.eval()
        self.device = next(self.nli_model.parameters()).device
        self.nlp = spacy.load(spacy_model)
        self.contradiction_index = self._resolve_contradiction_index()

    def _resolve_contradiction_index(self) -> int:
        id2label = getattr(self.nli_model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if "contrad" in str(label).lower() or "矛盾" in str(label):
                return int(idx)
        # 上記日本語NLIモデルの一般的な順序をフォールバックとして使用。
        return 2

    def evaluate(self, req: EvaluationRequest) -> EvaluationResponse:
        text = req.report.text
        rubric = req.assignment.rubric
        spans = split_sentences_with_spans(text)
        sentences = [x[2] for x in spans]

        redundancy, red_hl, red_diag = self._redundancy(
            text, sentences, spans, rubric.semantic_redundancy_threshold
        )
        consistency, con_hl, con_diag = self._consistency(
            sentences, spans, rubric.contradiction_threshold
        )
        coverage, kw_results, kw_hl, cov_diag = self._coverage(
            text,
            sentences,
            req.assignment.required_keywords,
            req.assignment.recommended_keywords,
            rubric.keyword_semantic_threshold,
            rubric.semantic_keyword_credit,
        )

        overall_100, overall_5 = weighted_overall(
            redundancy, consistency, coverage, rubric
        )
        doc = self.nlp(text)
        entities = [
            {"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
            for e in doc.ents
        ]
        missing = [r.keyword for r in kw_results if r.required and not r.matched]
        reference_terms = self._tfidf_reference_terms(
            text, req.assignment.prompt, req.assignment.reference_text
        )

        return EvaluationResponse(
            system="improved",
            overall_score_100=overall_100,
            overall_score_5=overall_5,
            redundancy_score=round(redundancy, 2),
            consistency_score=round(consistency, 2),
            coverage_score=round(coverage, 2),
            highlights=sorted(red_hl + con_hl + kw_hl, key=lambda h: (h.start, h.end, h.type)),
            keywords=kw_results,
            missing_required_keywords=missing,
            named_entities=entities,
            diagnostics={
                "redundancy": red_diag,
                "consistency": con_diag,
                "coverage": cov_diag,
                "tfidf_reference_gap_terms": reference_terms,
            },
        )

    def _redundancy(self, text, sentences, spans, threshold):
        bi = repeat_ratio(char_ngrams(text, 2))
        tri = repeat_ratio(char_ngrams(text, 3))
        lexical_penalty = min(1.0, 0.6 * bi + 0.4 * tri)

        highlights: list[Highlight] = []
        redundant_pairs = []
        semantic_penalties = []
        if len(sentences) >= 2:
            emb = np.asarray(self.embedder.encode(sentences, normalize_embeddings=True))
            sims = emb @ emb.T
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    sim = float(sims[i, j])
                    if sim >= threshold:
                        p = (sim - threshold) / max(1e-6, 1.0 - threshold)
                        semantic_penalties.append(p)
                        redundant_pairs.append({"i": i, "j": j, "semantic_similarity": round(sim, 4)})
                        start, end, sent = spans[j]
                        highlights.append(
                            Highlight(
                                start=start,
                                end=end,
                                type="redundancy",
                                reason=f"Sentence-BERTで文{i+1}と意味が類似 ({sim:.2f})",
                                confidence=min(1.0, sim),
                                sentence_index=j,
                                text=sent,
                            )
                        )
        semantic_penalty = min(1.0, sum(semantic_penalties) / max(1, len(sentences) - 1)) if semantic_penalties else 0.0
        penalty = min(1.0, 0.35 * lexical_penalty + 0.65 * semantic_penalty)
        return 100.0 * (1.0 - penalty), highlights, {
            "bigram_repeat_ratio": round(bi, 4),
            "trigram_repeat_ratio": round(tri, 4),
            "sentence_bert_pairs": redundant_pairs,
        }

    def _nli_probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        batch = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = self.nli_model(**batch).logits.detach().cpu().numpy()
        return softmax(logits, axis=1)

    def _consistency(self, sentences, spans, threshold):
        if len(sentences) < 2:
            return 100.0, [], {"pairs": []}
        pairs = []
        indexes = []
        for i in range(1, len(sentences)):
            premise = " ".join(sentences[max(0, i - 3) : i])
            pairs.append((premise, sentences[i]))
            indexes.append(i)
        probs = self._nli_probs(pairs)

        details = []
        penalties = []
        highlights: list[Highlight] = []
        for row, sentence_index in enumerate(indexes):
            p = float(probs[row, self.contradiction_index])
            details.append({
                "sentence_index": sentence_index,
                "contradiction_probability": round(p, 4),
            })
            penalty = max(0.0, (p - threshold) / max(1e-6, 1.0 - threshold))
            penalties.append(penalty)
            if p >= threshold:
                start, end, sent = spans[sentence_index]
                highlights.append(
                    Highlight(
                        start=start,
                        end=end,
                        type="contradiction",
                        reason=f"NLIで前段との矛盾確率が高い ({p:.2f})",
                        confidence=p,
                        sentence_index=sentence_index,
                        text=sent,
                    )
                )
        mean_p = float(np.mean(penalties)) if penalties else 0.0
        max_p = float(np.max(penalties)) if penalties else 0.0
        total = min(1.0, 0.5 * mean_p + 0.5 * max_p)
        return 100.0 * (1.0 - total), highlights, {
            "pairs": details,
            "contradiction_label_index": self.contradiction_index,
        }

    def _coverage(self, text, sentences, required, recommended, threshold, semantic_credit):
        all_keywords = [(k, True) for k in required] + [(k, False) for k in recommended]
        if not all_keywords:
            return 100.0, [], [], {}
        normalized_text = normalize_text(text)
        sentence_emb = (
            np.asarray(self.embedder.encode(sentences, normalize_embeddings=True))
            if sentences else None
        )

        results: list[KeywordResult] = []
        highlights: list[Highlight] = []
        weighted_credit = 0.0
        total_weight = 0.0
        for kw, is_required in all_keywords:
            weight = 2.0 if is_required else 1.0
            total_weight += weight
            exact = normalize_text(kw) in normalized_text
            matched_sentence = None
            confidence = 1.0 if exact else 0.0
            match_type = "exact" if exact else "missing"
            credit = 1.0 if exact else 0.0

            if exact:
                for m in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):
                    highlights.append(
                        Highlight(
                            start=m.start(), end=m.end(), type="keyword",
                            reason=f"完全一致キーワード: {kw}", confidence=1.0,
                            text=m.group(0),
                        )
                    )
            elif sentences and sentence_emb is not None:
                kw_emb = np.asarray(self.embedder.encode([kw], normalize_embeddings=True))[0]
                sims = sentence_emb @ kw_emb
                idx = int(np.argmax(sims))
                best = float(sims[idx])
                if best >= threshold:
                    match_type = "semantic"
                    matched_sentence = sentences[idx]
                    confidence = best
                    credit = semantic_credit

            weighted_credit += weight * credit
            results.append(
                KeywordResult(
                    keyword=kw,
                    required=is_required,
                    matched=match_type != "missing",
                    match_type=match_type,
                    confidence=round(confidence, 4),
                    matched_sentence=matched_sentence,
                )
            )
        return 100.0 * weighted_credit / max(1e-6, total_weight), results, highlights, {
            "semantic_threshold": threshold,
            "semantic_credit": semantic_credit,
        }

    @staticmethod
    def _tfidf_reference_terms(report: str, prompt: str, reference: str | None, top_k: int = 15):
        source = prompt + " " + (reference or "")
        if not source.strip() or not report.strip():
            return []
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        x = vectorizer.fit_transform([report, source]).toarray()
        terms = np.asarray(vectorizer.get_feature_names_out())
        gap = x[1] - x[0]
        idx = np.argsort(gap)[::-1][:top_k]
        return [
            {"term": str(terms[i]), "gap": round(float(gap[i]), 6)}
            for i in idx if gap[i] > 0
        ]
