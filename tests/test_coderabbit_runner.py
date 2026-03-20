"""Tests for coderabbit_runner module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from bugeval.coderabbit_runner import (
    _save_coderabbit_transcript,
    poll_for_coderabbit_review,
    run_coderabbit,
    scrape_coderabbit_comments,
)
from bugeval.models import CaseKind, GroundTruth, TestCase
from bugeval.result_models import Comment


def _make_case(**overrides: object) -> TestCase:
    defaults: dict[str, object] = {
        "id": "leo-001",
        "repo": "AleoNet/leo",
        "kind": CaseKind.bug,
        "base_commit": "abc123",
        "fix_commit": "def456",
        "introducing_pr_title": "Add feature X",
        "introducing_pr_body": "Some description",
        "truth": GroundTruth(
            introducing_commit="intro999",
            fix_pr_numbers=[1],
        ),
    }
    defaults.update(overrides)
    return TestCase(**defaults)  # type: ignore[arg-type]


class TestPollForCodeRabbitReview:
    def test_found(self) -> None:
        reviews_data = json.dumps({
            "reviews": [
                {"author": {"login": "coderabbitai[bot]"}, "body": "review"},
            ],
        })

        with patch(
            "bugeval.copilot_runner.run_gh",
            return_value=reviews_data,
        ):
            result = poll_for_coderabbit_review(
                "org/repo", 42, timeout=10, poll_interval=1,
            )

        assert result is True

    def test_timeout(self) -> None:
        no_reviews = json.dumps({"reviews": []})

        with patch(
            "bugeval.copilot_runner.run_gh",
            return_value=no_reviews,
        ), patch("bugeval.copilot_runner.time") as mock_time:
            # First call: 0, second check: over timeout
            mock_time.monotonic.side_effect = [0.0, 0.0, 11.0]
            mock_time.sleep = lambda _: None
            result = poll_for_coderabbit_review(
                "org/repo", 42, timeout=10, poll_interval=1,
            )

        assert result is False

    def test_ignores_non_coderabbit_reviews(self) -> None:
        reviews_data = json.dumps({
            "reviews": [
                {"author": {"login": "some-user"}, "body": "lgtm"},
            ],
        })

        with patch(
            "bugeval.copilot_runner.run_gh",
            return_value=reviews_data,
        ), patch("bugeval.copilot_runner.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 0.0, 11.0]
            mock_time.sleep = lambda _: None
            result = poll_for_coderabbit_review(
                "org/repo", 42, timeout=10, poll_interval=1,
            )

        assert result is False


class TestScrapeCodeRabbitComments:
    def test_filters_to_coderabbit(self) -> None:
        raw = json.dumps([
            {
                "user": {"login": "coderabbitai[bot]"},
                "path": "src/lib.rs",
                "line": 42,
                "body": "Potential null deref",
            },
            {
                "user": {"login": "human-reviewer"},
                "path": "src/lib.rs",
                "line": 10,
                "body": "Looks good",
            },
            {
                "user": {"login": "coderabbitai[bot]"},
                "path": "src/main.rs",
                "line": 99,
                "body": "Off-by-one error",
            },
        ])

        with patch("bugeval.copilot_runner.run_gh", return_value=raw):
            comments = scrape_coderabbit_comments("org/repo", 1)

        assert len(comments) == 2
        assert comments[0].file == "src/lib.rs"
        assert comments[0].line == 42
        assert comments[1].file == "src/main.rs"

    def test_empty_when_no_coderabbit(self) -> None:
        raw = json.dumps([
            {
                "user": {"login": "human"},
                "path": "f.rs",
                "line": 1,
                "body": "ok",
            },
        ])

        with patch("bugeval.copilot_runner.run_gh", return_value=raw):
            comments = scrape_coderabbit_comments("org/repo", 1)

        assert comments == []


class TestSaveCodeRabbitTranscript:
    def test_saves_transcript(self, tmp_path: Path) -> None:
        result = _save_coderabbit_transcript(
            tmp_path, "leo-001",
            fork="org/leo",
            branch="eval/leo-001",
            pr_number=42,
            scrubbed_title="Add feature",
            scrubbed_body="Description",
            raw_comments=[{"body": "test"}],
            patch_diff="diff --git a/f.rs b/f.rs",
            time_seconds=12.5,
        )

        path = Path(result)
        assert path.exists()
        assert path.name == "leo-001-coderabbit.json"

        data = json.loads(path.read_text())
        assert data["pr_metadata"]["fork"] == "org/leo"
        assert data["pr_metadata"]["pr_number"] == 42
        assert data["scrubbed_title"] == "Add feature"
        assert data["time_seconds"] == 12.5


class TestRunCodeRabbit:
    def test_full_lifecycle(self, tmp_path: Path) -> None:
        case = _make_case()
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        transcript_dir = tmp_path / "transcripts"

        raw_comments = [
            {
                "user": {"login": "coderabbitai[bot]"},
                "path": "src/lib.rs",
                "line": 42,
                "body": "Bug here",
            },
        ]

        with (
            patch(
                "bugeval.coderabbit_runner._get_patch_diff",
                return_value="diff content",
            ),
            patch(
                "bugeval.coderabbit_runner.ensure_fork",
                return_value="org/leo",
            ),
            patch(
                "bugeval.coderabbit_runner.create_eval_branch",
                return_value="eval/leo-001",
            ),
            patch(
                "bugeval.coderabbit_runner._default_branch",
                return_value="main",
            ),
            patch("bugeval.coderabbit_runner._isolate_fork"),
            patch(
                "bugeval.coderabbit_runner.open_eval_pr",
                return_value=42,
            ),
            patch(
                "bugeval.coderabbit_runner.poll_for_coderabbit_review",
                return_value=True,
            ),
            patch(
                "bugeval.coderabbit_runner._scrape_raw_coderabbit_comments",
                return_value=raw_comments,
            ),
            patch(
                "bugeval.coderabbit_runner.scrape_coderabbit_comments",
                return_value=[
                    Comment(file="src/lib.rs", line=42, body="Bug here"),
                ],
            ),
            patch(
                "bugeval.coderabbit_runner.close_eval_pr",
            ) as mock_close,
        ):
            result = run_coderabbit(
                case, repo_dir, timeout=300,
                org="org", transcript_dir=transcript_dir,
            )

        assert result.tool == "coderabbit"
        assert result.context_level == "diff+repo"
        assert len(result.comments) == 1
        assert result.comments[0].file == "src/lib.rs"
        assert result.error == ""
        mock_close.assert_called_once_with("org/leo", 42, "eval/leo-001")

    def test_timeout_returns_error(self, tmp_path: Path) -> None:
        case = _make_case()
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        with (
            patch(
                "bugeval.coderabbit_runner._get_patch_diff",
                return_value="diff",
            ),
            patch(
                "bugeval.coderabbit_runner.ensure_fork",
                return_value="org/leo",
            ),
            patch(
                "bugeval.coderabbit_runner.create_eval_branch",
                return_value="eval/leo-001",
            ),
            patch(
                "bugeval.coderabbit_runner._default_branch",
                return_value="main",
            ),
            patch("bugeval.coderabbit_runner._isolate_fork"),
            patch(
                "bugeval.coderabbit_runner.open_eval_pr",
                return_value=42,
            ),
            patch(
                "bugeval.coderabbit_runner.poll_for_coderabbit_review",
                return_value=False,
            ),
            patch("bugeval.coderabbit_runner.close_eval_pr"),
        ):
            result = run_coderabbit(case, repo_dir, timeout=10)

        assert "Timeout" in result.error
        assert result.comments == []

    def test_exception_returns_error(self, tmp_path: Path) -> None:
        case = _make_case()
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        with patch(
            "bugeval.coderabbit_runner._get_patch_diff",
            side_effect=RuntimeError("network down"),
        ):
            result = run_coderabbit(case, repo_dir)

        assert "network down" in result.error
        assert result.tool == "coderabbit"
