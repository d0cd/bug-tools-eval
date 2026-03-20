"""Tests for core data models."""

from __future__ import annotations

from bugeval.models import (
    BuggyLine,
    CaseKind,
    CaseStats,
    GroundTruth,
    PRRelation,
    TestCase,
    Validation,
)


class TestCaseKind:
    def test_values(self) -> None:
        assert CaseKind.bug == "bug"
        assert CaseKind.clean == "clean"

    def test_str_enum(self) -> None:
        assert str(CaseKind.bug) == "bug"
        assert f"{CaseKind.clean}" == "clean"


class TestBuggyLine:
    def test_required_fields(self) -> None:
        bl = BuggyLine(file="foo.rs", line=42)
        assert bl.file == "foo.rs"
        assert bl.line == 42
        assert bl.content == ""

    def test_with_content(self) -> None:
        bl = BuggyLine(file="bar.rs", line=10, content="let x = 1;")
        assert bl.content == "let x = 1;"


class TestPRRelation:
    def test_minimal(self) -> None:
        pr = PRRelation(pr_number=100, role="introducing", commit="abc")
        assert pr.pr_number == 100
        assert pr.title == ""

    def test_full(self) -> None:
        pr = PRRelation(
            pr_number=100,
            role="full_fix",
            commit="abc",
            title="Fix bug",
            merge_date="2024-01-01",
            author="alice",
        )
        assert pr.author == "alice"


class TestGroundTruth:
    def test_defaults(self) -> None:
        gt = GroundTruth()
        assert gt.introducing_commit is None
        assert gt.blame_confidence == ""
        assert gt.buggy_lines == []
        assert gt.fix_pr_numbers == []

    def test_with_data(self) -> None:
        gt = GroundTruth(
            introducing_commit="abc",
            blame_confidence="A",
            buggy_lines=[BuggyLine(file="x.rs", line=1)],
            fix_pr_numbers=[42, 43],
        )
        assert len(gt.buggy_lines) == 1
        assert gt.fix_pr_numbers == [42, 43]


class TestValidation:
    def test_defaults(self) -> None:
        v = Validation()
        assert not v.agreement
        assert not v.test_validated


class TestCaseStats:
    def test_defaults(self) -> None:
        s = CaseStats()
        assert s.lines_added == 0


class TestTestCase:
    def test_minimal_bug(self) -> None:
        tc = TestCase(id="t-001", repo="Org/Repo", kind=CaseKind.bug, base_commit="abc")
        assert tc.id == "t-001"
        assert tc.kind == CaseKind.bug
        assert tc.truth is None
        assert tc.validation is None
        assert tc.linked_issues == []
        assert tc.issue_bodies == {}
        assert tc.related_prs == []

    def test_minimal_clean(self) -> None:
        tc = TestCase(id="c-001", repo="Org/Repo", kind=CaseKind.clean, base_commit="xyz")
        assert tc.truth is None

    def test_full_case(self, sample_case: TestCase) -> None:
        assert sample_case.id == "snarkVM-001"
        assert sample_case.kind == CaseKind.bug
        assert sample_case.truth is not None
        assert sample_case.truth.blame_confidence == "A"
        assert len(sample_case.truth.buggy_lines) == 1
        assert sample_case.validation is not None
        assert sample_case.validation.agreement is True
        assert len(sample_case.related_prs) == 2
        assert sample_case.issue_bodies[100] == "Counter overflows at 128 validators."
        assert sample_case.bug_latency_days == 25

    def test_clean_case(self, clean_case: TestCase) -> None:
        assert clean_case.kind == CaseKind.clean
        assert clean_case.truth is None
