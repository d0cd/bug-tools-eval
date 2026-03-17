"""Pydantic model for in-house agent evaluation results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Structured output from an in-house agent run."""

    findings: list[dict[str, Any]] = Field(default_factory=list)
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    token_count: int = 0
    cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    turns: int = 0
    model: str = ""
    context_level: str = ""
    response_text: str = ""
    error: str | None = None
