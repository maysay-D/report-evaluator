from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ReportMeta(BaseModel):
    student_id: str | None = None
    submitted_at: str | None = None
    char_count: int | None = None
    attempt: int | None = None


class ReportSubmission(BaseModel):
    text: str
    meta: ReportMeta = Field(default_factory=ReportMeta)


class Rubric(BaseModel):
    redundancy_weight: float = 0.30
    consistency_weight: float = 0.40
    coverage_weight: float = 0.30

    baseline_sentence_similarity_threshold: float = 0.55
    baseline_contradiction_similarity_threshold: float = 0.35
    semantic_redundancy_threshold: float = 0.88
    contradiction_threshold: float = 0.65
    keyword_semantic_threshold: float = 0.72
    semantic_keyword_credit: float = 0.60


class AssignmentDefinition(BaseModel):
    prompt: str
    required_keywords: list[str] = Field(default_factory=list)
    recommended_keywords: list[str] = Field(default_factory=list)
    reference_text: str | None = None
    rubric: Rubric = Field(default_factory=Rubric)


class EvaluationRequest(BaseModel):
    report: ReportSubmission
    assignment: AssignmentDefinition


class Highlight(BaseModel):
    start: int
    end: int
    type: Literal["redundancy", "contradiction", "keyword"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    sentence_index: int | None = None
    text: str | None = None


class KeywordResult(BaseModel):
    keyword: str
    required: bool
    matched: bool
    match_type: Literal["exact", "semantic", "missing"]
    confidence: float = Field(ge=0.0, le=1.0)
    matched_sentence: str | None = None


class EvaluationResponse(BaseModel):
    system: Literal["baseline", "improved"]
    overall_score_100: float
    overall_score_5: float
    redundancy_score: float
    consistency_score: float
    coverage_score: float
    highlights: list[Highlight]
    keywords: list[KeywordResult]
    missing_required_keywords: list[str]
    named_entities: list[dict] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)
