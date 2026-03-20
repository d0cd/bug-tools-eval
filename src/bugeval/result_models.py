"""Tool result models."""

from __future__ import annotations

from pydantic import BaseModel


class Comment(BaseModel):
    file: str = ""
    line: int = 0
    body: str = ""
    suggested_fix: str = ""


class ToolResult(BaseModel):
    case_id: str
    tool: str
    context_level: str = ""
    comments: list[Comment] = []
    time_seconds: float = 0.0
    cost_usd: float = 0.0
    error: str = ""
    transcript_path: str = ""
    potentially_contaminated: bool = False
