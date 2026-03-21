"""Core data models for bug-finding evaluation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class CaseKind(StrEnum):
    bug = "bug"
    clean = "clean"


class BuggyLine(BaseModel):
    file: str
    line: int
    content: str = ""


class ReviewThread(BaseModel):
    path: str = ""
    line: int = 0
    is_resolved: bool = False
    comments: list[str] = []


class PRRelation(BaseModel):
    pr_number: int
    role: str  # "introducing" | "partial_fix" | "full_fix" | "revert" | "regression"
    commit: str
    title: str = ""
    merge_date: str = ""
    author: str = ""


class GroundTruth(BaseModel):
    introducing_commit: str | None = None
    blame_confidence: str = ""  # "A" | "B" | "C" | "D"
    buggy_lines: list[BuggyLine] = []
    fix_summary: str = ""
    fix_pr_numbers: list[int] = []


class Validation(BaseModel):
    claude_verdict: str = ""  # "confirmed" | "disputed" | "ambiguous"
    gemini_verdict: str = ""
    agreement: bool = False
    test_validated: bool = False


class CaseStats(BaseModel):
    lines_added: int = 0
    lines_deleted: int = 0
    files_changed: int = 0


class TestCase(BaseModel):
    # Identity
    id: str
    repo: str
    kind: CaseKind
    language: str = "rust"

    # Git coordinates
    base_commit: str
    fix_commit: str = ""
    fix_pr_number: int | None = None

    # Introducing PR data (what the tool sees)
    introducing_pr_number: int | None = None
    introducing_pr_title: str = ""
    introducing_pr_body: str = ""
    introducing_pr_commit_messages: list[str] = []
    introducing_pr_commit_shas: list[str] = []
    introducing_pr_author: str = ""
    introducing_pr_merge_date: str = ""
    introducing_pr_review_comments: list[str] = []
    introducing_pr_review_threads: list[ReviewThread] = []
    introducing_pr_ci_status: str = ""

    # Fix PR data (for ground truth construction)
    fix_pr_title: str = ""
    fix_pr_body: str = ""
    fix_pr_commit_messages: list[str] = []
    fix_pr_commit_shas: list[str] = []
    fix_pr_merge_date: str = ""
    fix_pr_review_comments: list[str] = []
    fix_pr_review_threads: list[ReviewThread] = []
    fix_pr_discussion_comments: list[str] = []
    fix_pr_merge_method: str = ""
    fix_pr_ci_status: str = ""

    # Issue data
    linked_issues: list[int] = []
    issue_bodies: dict[int, str] = {}
    issue_labels: list[str] = []
    referenced_issues: list[int] = []

    # PR relationship graph
    related_prs: list[PRRelation] = []

    # Ground truth (None for clean cases)
    truth: GroundTruth | None = None

    # Validation
    validation: Validation | None = None

    # Classification metadata
    category: str = ""
    difficulty: str = ""
    severity: str = ""
    pr_size: str = ""
    stats: CaseStats | None = None
    bug_description: str = ""
    bug_description_source: str = ""
    bug_latency_days: int | None = None
    same_author_fix: bool = False

    # Curation
    excluded: bool = False
    excluded_reason: str = ""
