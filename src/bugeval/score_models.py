"""Scoring models for evaluation results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CommentVerdict(StrEnum):
    tp = "TP"
    tp_novel = "TP-novel"
    fp = "FP"
    low_value = "low-value"


class CommentScore(BaseModel):
    comment_index: int
    verdict: CommentVerdict
    matched_buggy_line: int | None = None


class CaseScore(BaseModel):
    case_id: str
    tool: str
    caught: bool = False
    localization_distance: int | None = None
    detection_score: int = Field(default=0, ge=0, le=3)
    review_quality: int = Field(default=0, ge=0, le=4)
    comment_scores: list[CommentScore] = []
    reasoning: str = ""
    tp_count: int = 0
    fp_count: int = 0
    novel_count: int = 0
    false_alarm: bool = False
    potentially_contaminated: bool = False
    context_level: str = ""
    judge_failed: bool = False
