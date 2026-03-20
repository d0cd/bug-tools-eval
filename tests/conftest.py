"""Shared test fixtures."""

from __future__ import annotations

import pytest

from bugeval.models import (
    BuggyLine,
    CaseKind,
    CaseStats,
    GroundTruth,
    PRRelation,
    TestCase,
    Validation,
)
from bugeval.result_models import Comment, ToolResult
from bugeval.score_models import CaseScore, CommentScore, CommentVerdict


@pytest.fixture
def sample_case() -> TestCase:
    return TestCase(
        id="snarkVM-001",
        repo="ProvableHQ/snarkVM",
        kind=CaseKind.bug,
        base_commit="abc123",
        fix_commit="def456",
        fix_pr_number=42,
        introducing_pr_number=30,
        introducing_pr_title="Add validator rotation",
        introducing_pr_body="Implements validator rotation logic.",
        introducing_pr_commit_messages=["feat: add rotation"],
        introducing_pr_author="alice",
        introducing_pr_merge_date="2024-06-15",
        introducing_pr_review_comments=["LGTM"],
        introducing_pr_ci_status="passing",
        fix_pr_title="Fix overflow in rotation counter",
        fix_pr_body="The counter overflowed when count > 128.",
        fix_pr_commit_messages=["fix: handle overflow"],
        fix_pr_merge_date="2024-07-10",
        fix_pr_review_comments=["Good catch"],
        fix_pr_discussion_comments=["Found this in prod"],
        fix_pr_merge_method="squash",
        linked_issues=[100],
        issue_bodies={100: "Counter overflows at 128 validators."},
        issue_labels=["bug", "critical"],
        referenced_issues=[99],
        related_prs=[
            PRRelation(
                pr_number=30,
                role="introducing",
                commit="abc123",
                title="Add validator rotation",
                merge_date="2024-06-15",
                author="alice",
            ),
            PRRelation(
                pr_number=42,
                role="full_fix",
                commit="def456",
                title="Fix overflow in rotation counter",
                merge_date="2024-07-10",
            ),
        ],
        truth=GroundTruth(
            introducing_commit="abc123",
            blame_confidence="A",
            buggy_lines=[
                BuggyLine(
                    file="consensus/src/worker.rs",
                    line=142,
                    content="let state = shared_state.clone();",
                ),
            ],
            fix_summary="Added lock before state mutation",
            fix_pr_numbers=[42],
        ),
        validation=Validation(
            claude_verdict="confirmed",
            gemini_verdict="confirmed",
            agreement=True,
        ),
        category="concurrency",
        difficulty="medium",
        severity="critical",
        pr_size="medium",
        stats=CaseStats(lines_added=15, lines_deleted=3, files_changed=2),
        bug_description="Missing lock acquisition before shared state mutation",
        bug_description_source="issue",
        bug_latency_days=25,
        same_author_fix=False,
    )


@pytest.fixture
def clean_case() -> TestCase:
    return TestCase(
        id="clean-001",
        repo="ProvableHQ/snarkVM",
        kind=CaseKind.clean,
        base_commit="aaa111",
    )


@pytest.fixture
def sample_result() -> ToolResult:
    return ToolResult(
        case_id="snarkVM-001",
        tool="copilot",
        comments=[
            Comment(
                file="consensus/src/worker.rs",
                line=143,
                body="Potential race condition on shared_state",
                suggested_fix="Use a mutex guard",
            ),
            Comment(
                file="consensus/src/worker.rs",
                line=200,
                body="Consider adding error handling here",
            ),
        ],
        time_seconds=45.2,
        cost_usd=0.0,
    )


@pytest.fixture
def sample_score() -> CaseScore:
    return CaseScore(
        case_id="snarkVM-001",
        tool="copilot",
        caught=True,
        localization_distance=1,
        detection_score=3,
        review_quality=3,
        comment_scores=[
            CommentScore(comment_index=0, verdict=CommentVerdict.tp, matched_buggy_line=0),
            CommentScore(comment_index=1, verdict=CommentVerdict.low_value),
        ],
        reasoning="Tool correctly identified the race condition.",
        tp_count=1,
        fp_count=0,
        novel_count=0,
    )
