# src/bugeval/judge_models.py
"""Pydantic models for LLM judge results."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel


class CommentClassification(StrEnum):
    tp = "TP"
    fp = "FP"
    low_value = "low-value"


class CommentJudgment(BaseModel):
    """Judge's classification of a single tool comment."""

    id: int
    classification: CommentClassification
    relevance: str = ""  # "direct" | "adjacent" | "unrelated"


class NoiseStats(BaseModel):
    """Noise/SNR statistics derived from comment judgments."""

    total_comments: int = 0
    true_positives: int = 0
    snr: float = 0.0


class JudgeScore(BaseModel):
    """LLM judge output for one (case x tool) pair."""

    test_case_id: str
    tool: str
    score: int  # 0–3
    votes: list[int]
    reasoning: str
    comment_judgments: list[CommentJudgment] = []
    noise: NoiseStats = NoiseStats()
    vote_agreement: float = 0.0  # fraction of votes matching the majority


def majority_vote(votes: list[int]) -> int:
    """Return the most common vote. On tie: return the median value."""
    if not votes:
        return 0
    counter = Counter(votes)
    max_count = max(counter.values())
    candidates = sorted(v for v, c in counter.items() if c == max_count)
    return candidates[len(candidates) // 2]
