"""Tests for greptile_runner module (PR-based)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from bugeval.greptile_runner import (
    poll_for_greptile_review,
    run_greptile,
    scrape_greptile_comments,
)
from bugeval.models import CaseKind, TestCase
from bugeval.result_models import Comment


def _make_case(**overrides: object) -> TestCase:
    defaults: dict[str, object] = {
        "id": "leo-001",
        "repo": "AleoNet/leo",
        "kind": CaseKind.bug,
        "base_commit": "abc123",
        "introducing_pr_number": 42,
        "introducing_pr_title": "Add new feature",
        "introducing_pr_body": "This adds a feature",
    }
    defaults.update(overrides)
    return TestCase(**defaults)  # type: ignore[arg-type]


class TestPollForGreptileReview:
    @patch("bugeval.copilot_runner.time.sleep")
    @patch("bugeval.copilot_runner.run_gh")
    def test_found_review(
        self, mock_gh: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_gh.return_value = json.dumps({
            "reviews": [
                {"author": {"login": "greptile[bot]"}, "state": "COMMENTED"},
            ],
        })
        result = poll_for_greptile_review("testuser/leo", 99, timeout=60)
        assert result is True

    @patch("bugeval.copilot_runner.time.sleep")
    @patch("bugeval.copilot_runner.time.monotonic")
    @patch("bugeval.copilot_runner.run_gh")
    def test_timeout(
        self,
        mock_gh: MagicMock,
        mock_time: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        mock_gh.return_value = json.dumps({"reviews": []})
        mock_time.side_effect = [0.0, 100.0, 400.0]
        result = poll_for_greptile_review(
            "testuser/leo", 99, timeout=300, poll_interval=15,
        )
        assert result is False


class TestScrapeGreptileComments:
    @patch("bugeval.copilot_runner.run_gh")
    def test_parses_review_comments(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = json.dumps([
            {
                "path": "src/main.rs",
                "line": 42,
                "body": "Potential null deref here",
                "user": {"login": "greptile[bot]"},
            },
            {
                "path": "src/lib.rs",
                "line": 10,
                "body": "Consider error handling",
                "user": {"login": "greptile[bot]"},
            },
        ])
        comments = scrape_greptile_comments("testuser/leo", 99)
        assert len(comments) == 2
        assert comments[0].file == "src/main.rs"
        assert comments[0].line == 42
        assert comments[0].body == "Potential null deref here"

    @patch("bugeval.copilot_runner.run_gh")
    def test_filters_non_greptile(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = json.dumps([
            {
                "path": "src/main.rs",
                "line": 42,
                "body": "Greptile finding",
                "user": {"login": "greptile[bot]"},
            },
            {
                "path": "src/lib.rs",
                "line": 10,
                "body": "Human comment",
                "user": {"login": "somedev"},
            },
        ])
        comments = scrape_greptile_comments("testuser/leo", 99)
        assert len(comments) == 1
        assert comments[0].body == "Greptile finding"

    @patch("bugeval.copilot_runner.run_gh")
    def test_empty_comments(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = json.dumps([])
        comments = scrape_greptile_comments("testuser/leo", 99)
        assert comments == []


class TestRunGreptile:
    @patch("bugeval.greptile_runner.close_eval_pr")
    @patch("bugeval.greptile_runner._scrape_raw_greptile_comments")
    @patch("bugeval.greptile_runner.scrape_greptile_comments")
    @patch("bugeval.greptile_runner.poll_for_greptile_review")
    @patch("bugeval.greptile_runner.open_eval_pr")
    @patch("bugeval.greptile_runner.create_eval_branches")
    @patch("bugeval.greptile_runner.ensure_tool_repo")
    @patch("bugeval.greptile_runner._get_patch_diff")
    def test_success(
        self,
        mock_diff: MagicMock,
        mock_fork: MagicMock,
        mock_branch: MagicMock,
        mock_open: MagicMock,
        mock_poll: MagicMock,
        mock_scrape: MagicMock,
        mock_raw: MagicMock,
        mock_close: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_diff.return_value = "diff content"
        mock_fork.return_value = "testuser/leo"
        mock_branch.return_value = ("base-abc", "review-abc")
        mock_open.return_value = 99
        mock_poll.return_value = True
        mock_raw.return_value = [
            {"path": "src/main.rs", "line": 42, "body": "Bug found",
             "user": {"login": "greptile[bot]"}},
        ]
        mock_scrape.return_value = [
            Comment(file="src/main.rs", line=42, body="Bug found"),
        ]
        case = _make_case()
        result = run_greptile(case, tmp_path, org="testuser")
        assert result.case_id == "leo-001"
        assert result.tool == "greptile"
        assert len(result.comments) == 1
        assert result.error == ""
        mock_close.assert_called_once_with(
            "testuser/leo", 99, "review-abc", "base-abc",
        )

    @patch("bugeval.greptile_runner.close_eval_pr")
    @patch("bugeval.greptile_runner.poll_for_greptile_review")
    @patch("bugeval.greptile_runner.open_eval_pr")
    @patch("bugeval.greptile_runner.create_eval_branches")
    @patch("bugeval.greptile_runner.ensure_tool_repo")
    @patch("bugeval.greptile_runner._get_patch_diff")
    def test_timeout(
        self,
        mock_diff: MagicMock,
        mock_fork: MagicMock,
        mock_branch: MagicMock,
        mock_open: MagicMock,
        mock_poll: MagicMock,
        mock_close: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_diff.return_value = "diff content"
        mock_fork.return_value = "testuser/leo"
        mock_branch.return_value = ("base-abc", "review-abc")
        mock_open.return_value = 99
        mock_poll.return_value = False
        case = _make_case()
        result = run_greptile(case, tmp_path, timeout=60, org="testuser")
        assert result.case_id == "leo-001"
        assert result.tool == "greptile"
        assert "timeout" in result.error.lower()
        mock_close.assert_called_once_with(
            "testuser/leo", 99, "review-abc", "base-abc",
        )

    @patch("bugeval.greptile_runner.ensure_tool_repo")
    @patch("bugeval.greptile_runner._get_patch_diff")
    def test_error(
        self,
        mock_diff: MagicMock,
        mock_fork: MagicMock,
        tmp_path: Path,
    ) -> None:
        from bugeval.mine import GhError

        mock_diff.return_value = "diff content"
        mock_fork.side_effect = GhError(
            ["gh", "repo", "fork"], "network error",
        )
        case = _make_case()
        result = run_greptile(case, tmp_path)
        assert result.case_id == "leo-001"
        assert result.tool == "greptile"
        assert result.error != ""
        assert len(result.comments) == 0


class TestGreptileTranscript:
    @patch("bugeval.greptile_runner.close_eval_pr")
    @patch("bugeval.greptile_runner._scrape_raw_greptile_comments")
    @patch("bugeval.greptile_runner.scrape_greptile_comments")
    @patch("bugeval.greptile_runner.poll_for_greptile_review")
    @patch("bugeval.greptile_runner.open_eval_pr")
    @patch("bugeval.greptile_runner.create_eval_branches")
    @patch("bugeval.greptile_runner.ensure_tool_repo")
    @patch("bugeval.greptile_runner._get_patch_diff")
    def test_transcript_saved(
        self,
        mock_diff: MagicMock,
        mock_fork: MagicMock,
        mock_branch: MagicMock,
        mock_open: MagicMock,
        mock_poll: MagicMock,
        mock_scrape: MagicMock,
        mock_raw: MagicMock,
        mock_close: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_diff.return_value = "diff content"
        mock_fork.return_value = "testuser/leo"
        mock_branch.return_value = ("base-abc", "review-abc")
        mock_open.return_value = 99
        mock_poll.return_value = True
        mock_raw.return_value = [
            {"path": "f.rs", "line": 1, "body": "bug",
             "user": {"login": "greptile[bot]"}},
        ]
        mock_scrape.return_value = [
            Comment(file="f.rs", line=1, body="bug"),
        ]
        transcript_dir = tmp_path / "transcripts"
        case = _make_case()
        result = run_greptile(
            case, tmp_path, org="testuser",
            transcript_dir=transcript_dir,
        )
        assert result.transcript_path != ""
        path = Path(result.transcript_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "pr_metadata" in data
        assert "raw_comments" in data
        assert data["pr_metadata"]["pr_number"] == 99

    @patch("bugeval.greptile_runner.close_eval_pr")
    @patch("bugeval.greptile_runner._scrape_raw_greptile_comments")
    @patch("bugeval.greptile_runner.scrape_greptile_comments")
    @patch("bugeval.greptile_runner.poll_for_greptile_review")
    @patch("bugeval.greptile_runner.open_eval_pr")
    @patch("bugeval.greptile_runner.create_eval_branches")
    @patch("bugeval.greptile_runner.ensure_tool_repo")
    @patch("bugeval.greptile_runner._get_patch_diff")
    def test_no_transcript_without_dir(
        self,
        mock_diff: MagicMock,
        mock_fork: MagicMock,
        mock_branch: MagicMock,
        mock_open: MagicMock,
        mock_poll: MagicMock,
        mock_scrape: MagicMock,
        mock_raw: MagicMock,
        mock_close: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_diff.return_value = "diff content"
        mock_fork.return_value = "testuser/leo"
        mock_branch.return_value = ("base-abc", "review-abc")
        mock_open.return_value = 99
        mock_poll.return_value = True
        mock_raw.return_value = []
        mock_scrape.return_value = []
        case = _make_case()
        result = run_greptile(case, tmp_path, org="testuser")
        assert result.transcript_path == ""
