"""Pydantic schemas for test cases, candidates, and related types."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Category(StrEnum):
    logic = "logic"
    memory = "memory"
    concurrency = "concurrency"
    api_misuse = "api-misuse"
    type = "type"
    cryptographic = "cryptographic"
    constraint = "constraint"
    code_smell = "code-smell"
    security = "security"
    performance = "performance"
    style = "style"
    incomplete = "incomplete"


class Difficulty(StrEnum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class Severity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class PRSize(StrEnum):
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"
    xl = "xl"


class Visibility(StrEnum):
    public = "public"
    private = "private"


class ExpectedFinding(BaseModel):
    file: str
    line: int
    summary: str
    line_side: str = "pre_fix"  # "pre_fix" (- side) or "post_fix" (+ side)


class CaseStats(BaseModel):
    lines_added: int
    lines_deleted: int
    files_changed: int
    hunks: int


class TestCase(BaseModel):
    id: str
    repo: str
    base_commit: str
    head_commit: str
    fix_commit: str
    category: Category
    difficulty: Difficulty
    severity: Severity
    language: str
    pr_size: PRSize
    description: str
    expected_findings: list[ExpectedFinding]
    stats: CaseStats | None = None
    visibility: Visibility = Visibility.public
    needs_manual_review: bool = False
    verified: bool = False
    verified_by: str | None = None
    valid_for_code_review: bool = True
    introducing_commit: str | None = None
    """SHA of the commit that first introduced this issue. Analysis-only — never
    passed to evaluation agents. None when unknown (PR-scraped cases)."""
    pr_number: int | None = None
    reviewer_notes: list[str] = []
    reviewer_findings: list[ExpectedFinding] = []
    quality_flags: list[str] = []


class Candidate(BaseModel):
    repo: str
    pr_number: int
    fix_commit: str
    base_commit: str | None = None
    head_commit: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[str]
    title: str
    body: str
    labels: list[str]
    files_changed: list[str]
    diff_stats: CaseStats
    expected_findings: list[ExpectedFinding]
    language: str
    pr_size: PRSize
    reviewer_notes: list[str] = []  # reviewer comments that identified the bug
    reviewer_findings: list[ExpectedFinding] = []  # structured inline review positions


class ScrapeState(BaseModel):
    repo: str
    last_scraped_at: datetime
    processed_pr_numbers: list[int]
