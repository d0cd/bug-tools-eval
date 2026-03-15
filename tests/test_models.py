"""Tests for Pydantic models."""

import pytest
import yaml
from pydantic import ValidationError

from bugeval.models import (
    Candidate,
    CaseStats,
    Category,
    Difficulty,
    ExpectedFinding,
    PRSize,
    ScrapeState,
    Severity,
    TestCase,
    Visibility,
)


def make_test_case(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "aleo-lang-001",
        "repo": "provable-org/aleo-lang",
        "base_commit": "abc123def456abc123def456abc123def456abc1",
        "head_commit": "def456abc123def456abc123def456abc123def4",
        "fix_commit": "ghi789ghi789ghi789ghi789ghi789ghi789ghi7",
        "category": "logic",
        "difficulty": "medium",
        "severity": "high",
        "language": "rust",
        "pr_size": "small",
        "description": "A logic bug in the type checker",
        "expected_findings": [{"file": "src/main.rs", "line": 42, "summary": "Off-by-one error"}],
    }
    base.update(overrides)
    return base


def make_candidate(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "repo": "provable-org/aleo-lang",
        "pr_number": 123,
        "fix_commit": "abc123def456abc123def456abc123def456abc1",
        "confidence": 0.7,
        "signals": ["has_bug_label", "pr_references_issue"],
        "title": "Fix off-by-one error in type checker",
        "body": "Fixes #456",
        "labels": ["bug"],
        "files_changed": ["src/main.rs"],
        "diff_stats": {"lines_added": 5, "lines_deleted": 3, "files_changed": 1, "hunks": 2},
        "expected_findings": [],
        "language": "rust",
        "pr_size": "tiny",
    }
    base.update(overrides)
    return base


class TestTestCase:
    def test_valid_test_case(self) -> None:
        case = TestCase(**make_test_case())  # type: ignore[arg-type]
        assert case.id == "aleo-lang-001"
        assert case.category == Category.logic

    def test_yaml_round_trip(self) -> None:
        case = TestCase(**make_test_case())  # type: ignore[arg-type]
        # mode="json" converts enums to plain strings, safe for yaml.safe_dump
        data = case.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data)
        loaded = yaml.safe_load(yaml_str)
        case2 = TestCase(**loaded)
        assert case == case2

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TestCase(**make_test_case(category="invalid_category"))  # type: ignore[arg-type]

    def test_missing_required_field(self) -> None:
        data = make_test_case()
        del data["id"]
        with pytest.raises(ValidationError):
            TestCase(**data)  # type: ignore[arg-type]

    def test_optional_stats_none(self) -> None:
        case = TestCase(**make_test_case(stats=None))  # type: ignore[arg-type]
        assert case.stats is None

    def test_optional_stats_present(self) -> None:
        case = TestCase(
            **make_test_case(  # type: ignore[arg-type]
                stats={"lines_added": 10, "lines_deleted": 5, "files_changed": 2, "hunks": 3}
            )
        )
        assert case.stats is not None
        assert case.stats.lines_added == 10


class TestCandidate:
    def test_valid_candidate(self) -> None:
        candidate = Candidate(**make_candidate())  # type: ignore[arg-type]
        assert candidate.confidence == 0.7
        assert candidate.base_commit is None
        assert candidate.head_commit is None

    def test_confidence_too_high(self) -> None:
        with pytest.raises(ValidationError):
            Candidate(**make_candidate(confidence=1.5))  # type: ignore[arg-type]

    def test_confidence_too_low(self) -> None:
        with pytest.raises(ValidationError):
            Candidate(**make_candidate(confidence=-0.1))  # type: ignore[arg-type]

    def test_confidence_at_bounds(self) -> None:
        c1 = Candidate(**make_candidate(confidence=0.0))  # type: ignore[arg-type]
        assert c1.confidence == 0.0
        c2 = Candidate(**make_candidate(confidence=1.0))  # type: ignore[arg-type]
        assert c2.confidence == 1.0


class TestEnums:
    def test_pr_size_values(self) -> None:
        assert PRSize.tiny == "tiny"
        assert PRSize.xl == "xl"

    def test_severity_values(self) -> None:
        assert Severity.critical == "critical"

    def test_category_values(self) -> None:
        assert Category.logic == "logic"
        assert Category.memory == "memory"

    def test_difficulty_values(self) -> None:
        assert Difficulty.easy == "easy"
        assert Difficulty.hard == "hard"


class TestScrapeState:
    def test_valid_scrape_state(self) -> None:
        from datetime import datetime

        state = ScrapeState(
            repo="provable-org/aleo-lang",
            last_scraped_at=datetime.now(),
            processed_pr_numbers=[1, 2, 3],
        )
        assert state.repo == "provable-org/aleo-lang"
        assert 2 in state.processed_pr_numbers

    def test_empty_processed_list(self) -> None:
        from datetime import datetime

        state = ScrapeState(
            repo="foo/bar",
            last_scraped_at=datetime.now(),
            processed_pr_numbers=[],
        )
        assert state.processed_pr_numbers == []


class TestExpectedFinding:
    def test_valid_finding(self) -> None:
        finding = ExpectedFinding(file="src/main.rs", line=42, summary="Off-by-one")
        assert finding.file == "src/main.rs"
        assert finding.line == 42


class TestCaseStats:
    def test_valid_stats(self) -> None:
        stats = CaseStats(lines_added=10, lines_deleted=5, files_changed=2, hunks=3)
        assert stats.lines_added == 10
        assert stats.hunks == 3


class TestVerified:
    def test_verified_defaults_false(self) -> None:
        case = TestCase(**make_test_case())  # type: ignore[arg-type]
        assert case.verified is False

    def test_verified_by_defaults_none(self) -> None:
        case = TestCase(**make_test_case())  # type: ignore[arg-type]
        assert case.verified_by is None

    def test_verified_can_be_set(self) -> None:
        case = TestCase(**make_test_case(verified=True, verified_by="alice"))  # type: ignore[arg-type]
        assert case.verified is True
        assert case.verified_by == "alice"

    def test_yaml_round_trip_with_verified(self) -> None:
        case = TestCase(**make_test_case(verified=True, verified_by="bob"))  # type: ignore[arg-type]
        data = case.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data)
        loaded = yaml.safe_load(yaml_str)
        case2 = TestCase(**loaded)
        assert case2.verified is True
        assert case2.verified_by == "bob"

    def test_existing_yaml_without_verified_loads_with_defaults(self) -> None:
        """Backward compat: YAML without verified fields loads with verified=False."""
        data = make_test_case()
        # Simulate old YAML that has no verified/verified_by keys
        assert "verified" not in data
        case = TestCase(**data)  # type: ignore[arg-type]
        assert case.verified is False
        assert case.verified_by is None


class TestVisibility:
    def test_visibility_enum_values(self) -> None:
        assert Visibility.public == "public"
        assert Visibility.private == "private"

    def test_test_case_default_visibility(self) -> None:
        case = TestCase(**make_test_case())  # type: ignore[arg-type]
        assert case.visibility == Visibility.public

    def test_test_case_explicit_visibility(self) -> None:
        case = TestCase(**make_test_case(visibility="private"))  # type: ignore[arg-type]
        assert case.visibility == Visibility.private

    def test_test_case_invalid_visibility(self) -> None:
        with pytest.raises(ValidationError):
            TestCase(**make_test_case(visibility="internal"))  # type: ignore[arg-type]
