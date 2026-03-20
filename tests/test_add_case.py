"""Tests for the add_case module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from bugeval.add_case import (
    add_case_from_pr,
    find_duplicate,
    parse_pr_url,
)
from bugeval.models import CaseKind, TestCase


class TestParsePrUrl:
    def test_valid_url(self) -> None:
        owner, repo, number = parse_pr_url(
            "https://github.com/ProvableHQ/snarkVM/pull/2345"
        )
        assert owner == "ProvableHQ"
        assert repo == "snarkVM"
        assert number == 2345

    def test_valid_url_trailing_slash(self) -> None:
        owner, repo, number = parse_pr_url(
            "https://github.com/owner/repo/pull/99/"
        )
        assert owner == "owner"
        assert repo == "repo"
        assert number == 99

    def test_invalid_url_no_pull(self) -> None:
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            parse_pr_url("https://github.com/owner/repo/issues/123")

    def test_invalid_url_not_github(self) -> None:
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            parse_pr_url("https://gitlab.com/owner/repo/pull/123")

    def test_invalid_url_garbage(self) -> None:
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            parse_pr_url("not-a-url")


class TestFindDuplicate:
    def test_finds_existing(self, tmp_path: Path) -> None:
        case = TestCase(
            id="snarkVM-001",
            repo="ProvableHQ/snarkVM",
            kind=CaseKind.bug,
            base_commit="abc",
            fix_pr_number=42,
        )
        case_path = tmp_path / "snarkVM-001.yaml"
        case_path.write_text(
            yaml.safe_dump(case.model_dump(mode="json"), sort_keys=False)
        )

        result = find_duplicate(tmp_path, 42)
        assert result == "snarkVM-001"

    def test_returns_none_when_new(self, tmp_path: Path) -> None:
        case = TestCase(
            id="snarkVM-001",
            repo="ProvableHQ/snarkVM",
            kind=CaseKind.bug,
            base_commit="abc",
            fix_pr_number=42,
        )
        case_path = tmp_path / "snarkVM-001.yaml"
        case_path.write_text(
            yaml.safe_dump(case.model_dump(mode="json"), sort_keys=False)
        )

        result = find_duplicate(tmp_path, 99)
        assert result is None

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = find_duplicate(tmp_path, 42)
        assert result is None

    def test_case_without_fix_pr_number(self, tmp_path: Path) -> None:
        case = TestCase(
            id="snarkVM-001",
            repo="ProvableHQ/snarkVM",
            kind=CaseKind.bug,
            base_commit="abc",
            fix_pr_number=None,
        )
        case_path = tmp_path / "snarkVM-001.yaml"
        case_path.write_text(
            yaml.safe_dump(case.model_dump(mode="json"), sort_keys=False)
        )

        result = find_duplicate(tmp_path, 42)
        assert result is None


def _make_pr_json(number: int = 123) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Fix overflow in parser",
        "body": "Fixes #10",
        "labels": [{"name": "bug"}],
        "mergeCommit": {"oid": "abc123def456"},
        "baseRefName": "main",
        "headRefName": "fix/overflow",
        "files": [{"path": "src/parser.rs"}],
        "additions": 5,
        "deletions": 3,
        "changedFiles": 1,
        "mergedAt": "2025-01-15T12:00:00Z",
        "author": {"login": "alice"},
        "commits": [],
        "reviewDecision": "APPROVED",
        "statusCheckRollup": [],
    }


class TestAddCaseCreatesYaml:
    @patch("bugeval.add_case.run_gh")
    @patch("bugeval.add_case.fetch_pr_details_graphql")
    def test_creates_yaml(
        self,
        mock_graphql: Any,
        mock_gh: Any,
        tmp_path: Path,
    ) -> None:
        pr_data = _make_pr_json(123)
        mock_gh.return_value = json.dumps(pr_data)
        mock_graphql.return_value = {}

        cases_dir = tmp_path / "snarkVM"
        cases_dir.mkdir()

        result = add_case_from_pr(
            "https://github.com/ProvableHQ/snarkVM/pull/123",
            cases_dir,
            repo_dir=Path(""),
        )

        assert result is not None
        assert result.id == "snarkVM-001"
        assert result.fix_pr_number == 123
        assert result.repo == "ProvableHQ/snarkVM"

        written = list(cases_dir.glob("*.yaml"))
        assert len(written) == 1
        assert written[0].name == "snarkVM-001.yaml"

    @patch("bugeval.add_case.run_gh")
    @patch("bugeval.add_case.fetch_pr_details_graphql")
    def test_sequential_numbering(
        self,
        mock_graphql: Any,
        mock_gh: Any,
        tmp_path: Path,
    ) -> None:
        cases_dir = tmp_path / "snarkVM"
        cases_dir.mkdir()

        # Pre-existing case
        existing = TestCase(
            id="snarkVM-001",
            repo="ProvableHQ/snarkVM",
            kind=CaseKind.bug,
            base_commit="abc",
            fix_pr_number=42,
        )
        (cases_dir / "snarkVM-001.yaml").write_text(
            yaml.safe_dump(existing.model_dump(mode="json"), sort_keys=False)
        )

        pr_data = _make_pr_json(123)
        mock_gh.return_value = json.dumps(pr_data)
        mock_graphql.return_value = {}

        result = add_case_from_pr(
            "https://github.com/ProvableHQ/snarkVM/pull/123",
            cases_dir,
            repo_dir=Path(""),
        )

        assert result is not None
        assert result.id == "snarkVM-002"


class TestAddCaseDedupSkips:
    @patch("bugeval.add_case.run_gh")
    def test_dedup_skips(
        self,
        mock_gh: Any,
        tmp_path: Path,
    ) -> None:
        cases_dir = tmp_path / "snarkVM"
        cases_dir.mkdir()

        existing = TestCase(
            id="snarkVM-001",
            repo="ProvableHQ/snarkVM",
            kind=CaseKind.bug,
            base_commit="abc",
            fix_pr_number=123,
        )
        (cases_dir / "snarkVM-001.yaml").write_text(
            yaml.safe_dump(existing.model_dump(mode="json"), sort_keys=False)
        )

        result = add_case_from_pr(
            "https://github.com/ProvableHQ/snarkVM/pull/123",
            cases_dir,
            repo_dir=Path(""),
        )

        assert result is None
        mock_gh.assert_not_called()


class TestAddCaseDryRun:
    @patch("bugeval.add_case.run_gh")
    @patch("bugeval.add_case.fetch_pr_details_graphql")
    def test_dry_run_no_files(
        self,
        mock_graphql: Any,
        mock_gh: Any,
        tmp_path: Path,
    ) -> None:
        pr_data = _make_pr_json(123)
        mock_gh.return_value = json.dumps(pr_data)
        mock_graphql.return_value = {}

        cases_dir = tmp_path / "snarkVM"
        cases_dir.mkdir()

        result = add_case_from_pr(
            "https://github.com/ProvableHQ/snarkVM/pull/123",
            cases_dir,
            repo_dir=Path(""),
            dry_run=True,
        )

        assert result is not None
        assert result.fix_pr_number == 123

        written = list(cases_dir.glob("*.yaml"))
        assert len(written) == 0
