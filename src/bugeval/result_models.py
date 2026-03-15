"""Pydantic models for normalized tool evaluation results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CommentType(StrEnum):
    inline = "inline"
    pr_level = "pr-level"
    summary = "summary"


class Comment(BaseModel):
    """A single comment or finding from a tool review."""

    file: str = ""
    line: int = 0
    body: str
    type: CommentType = CommentType.inline
    confidence: float | None = None
    severity: str | None = None
    category: str | None = None
    suggested_fix: str | None = None
    reasoning: str | None = None


class ResultMetadata(BaseModel):
    """Execution metadata for a tool review."""

    tokens: int = 0
    cost_usd: float = 0.0
    time_seconds: float = 0.0


class DxAssessment(BaseModel):
    """Developer experience assessment on a 1–5 scale per dimension."""

    actionability: int = Field(ge=1, le=5, default=3)
    false_positive_burden: int = Field(ge=1, le=5, default=3)
    integration_friction: int = Field(ge=1, le=5, default=3)
    response_latency: int = Field(ge=1, le=5, default=3)
    notes: str = ""


class NormalizedResult(BaseModel):
    """Common output schema for all tool evaluation modes."""

    test_case_id: str
    tool: str
    context_level: str = ""
    comments: list[Comment] = []
    metadata: ResultMetadata = ResultMetadata()
    dx: DxAssessment | None = None
