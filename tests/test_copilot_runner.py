"""Tests for the copilot_runner module."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from bugeval.copilot_runner import (
    _get_patch_diff,
    _isolate_fork,
    close_eval_pr,
    create_eval_branch,
    ensure_fork,
    open_eval_pr,
    poll_for_review,
    run_copilot,
    scrape_pr_comments,
)
from bugeval.models import CaseKind, GroundTruth, TestCase
from bugeval.result_models import Comment


def _make_case(**overrides: object) -> TestCase:
    defaults = {
        "id": "snarkVM-001",
        "repo": "AleoNet/snarkVM",
        "kind": CaseKind.bug,
        "base_commit": "abc123",
        "introducing_pr_number": 42,
        "introducing_pr_title": "Add new feature",
        "introducing_pr_body": "This adds a feature",
    }
    defaults.update(overrides)
    return TestCase(**defaults)  # type: ignore[arg-type]


class TestEnsureFork:
    @patch("bugeval.copilot_runner.run_gh")
    def test_returns_fork_name(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = [
            # fork command
            "",
            # whoami
            "testuser\n",
        ]
        result = ensure_fork("AleoNet/snarkVM")
        assert result == "testuser/snarkVM"
        mock_gh.assert_any_call(
            "repo", "fork", "AleoNet/snarkVM", "--clone=false",
        )

    @patch("bugeval.copilot_runner.run_gh")
    def test_fork_already_exists(self, mock_gh: MagicMock) -> None:
        from bugeval.mine import GhError

        mock_gh.side_effect = [
            GhError(["gh", "repo", "fork"], "already exists"),
            "testuser\n",
        ]
        result = ensure_fork("AleoNet/snarkVM")
        assert result == "testuser/snarkVM"

    @patch("bugeval.copilot_runner.run_gh")
    def test_fork_with_org_returns_org_name(self, mock_gh: MagicMock) -> None:
        # When org is provided, only the fork command is called — no whoami
        mock_gh.return_value = ""
        result = ensure_fork("AleoNet/snarkVM", org="myorg")
        assert result == "myorg/snarkVM"
        call_args = mock_gh.call_args_list[0][0]
        assert "--org" in call_args
        assert "myorg" in call_args
        # Should NOT have called the user API
        assert mock_gh.call_count == 1


class TestCreateEvalBranch:
    @patch("bugeval.copilot_runner.subprocess.run")
    def test_creates_branch_and_pushes(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        case = _make_case()
        result = create_eval_branch(
            fork="testuser/snarkVM",
            case=case,
            patch_diff="diff --git a/f.rs b/f.rs\n",
            repo_dir=tmp_path,
        )
        assert result == "eval/snarkVM-001"
        assert mock_run.call_count >= 3  # checkout, apply, push
        # Verify checkout uses parent of base_commit (introducing_commit fallback)
        first_call_args = mock_run.call_args_list[0][0][0]
        assert "abc123~1" in first_call_args

    @patch("bugeval.copilot_runner.subprocess.run")
    def test_uses_introducing_commit_from_truth(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        case = _make_case(
            truth=GroundTruth(introducing_commit="intro999"),
        )
        create_eval_branch(
            fork="testuser/snarkVM",
            case=case,
            patch_diff="diff --git a/f.rs b/f.rs\n",
            repo_dir=tmp_path,
        )
        first_call_args = mock_run.call_args_list[0][0][0]
        assert "intro999~1" in first_call_args


class TestOpenEvalPr:
    @patch("bugeval.copilot_runner._default_branch", return_value="main")
    @patch("bugeval.copilot_runner.run_gh")
    def test_returns_pr_number(
        self, mock_gh: MagicMock, mock_default: MagicMock,
    ) -> None:
        mock_gh.return_value = json.dumps({"number": 99})
        case = _make_case()
        result = open_eval_pr(
            "testuser/snarkVM", "eval/snarkVM-001", case,
        )
        assert result == 99

    @patch("bugeval.copilot_runner._default_branch", return_value="develop")
    @patch("bugeval.copilot_runner.run_gh")
    def test_uses_detected_default_branch(
        self, mock_gh: MagicMock, mock_default: MagicMock,
    ) -> None:
        mock_gh.return_value = json.dumps({"number": 7})
        case = _make_case()
        open_eval_pr("testuser/snarkVM", "eval/snarkVM-001", case)
        call_args = mock_gh.call_args[0]
        assert "develop" in call_args

    @patch("bugeval.copilot_runner._default_branch", return_value="main")
    @patch("bugeval.copilot_runner.run_gh")
    def test_uses_pr_metadata(
        self, mock_gh: MagicMock, mock_default: MagicMock,
    ) -> None:
        mock_gh.return_value = json.dumps({"number": 7})
        case = _make_case(
            introducing_pr_title="Refactor validator rotation logic",
            introducing_pr_body="Detailed body about refactoring",
        )
        open_eval_pr("testuser/snarkVM", "eval/snarkVM-001", case)
        call_args = mock_gh.call_args[0]
        # Title and body are scrubbed for anti-contamination
        assert "Refactor validator rotation logic" in call_args
        assert "Detailed body about refactoring" in call_args

    @patch("bugeval.copilot_runner._default_branch", return_value="main")
    @patch("bugeval.copilot_runner.run_gh")
    def test_scrubs_fix_references_in_title(
        self, mock_gh: MagicMock, mock_default: MagicMock,
    ) -> None:
        mock_gh.return_value = json.dumps({"number": 7})
        case = _make_case(
            introducing_pr_title="Fix overflow",
            introducing_pr_body="",
        )
        open_eval_pr("testuser/snarkVM", "eval/snarkVM-001", case)
        call_args = mock_gh.call_args[0]
        # "Fix overflow" gets scrubbed, falls back to eval-{case.id}
        assert "eval-snarkVM-001" in call_args


class TestScrapePrComments:
    @patch("bugeval.copilot_runner.run_gh")
    def test_parses_review_comments(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = json.dumps([
            {
                "path": "src/main.rs",
                "line": 42,
                "body": "Potential null deref here",
                "user": {"login": "copilot[bot]"},
            },
            {
                "path": "src/lib.rs",
                "line": 10,
                "body": "Consider error handling",
                "user": {"login": "copilot[bot]"},
            },
        ])
        comments = scrape_pr_comments("testuser/snarkVM", 99)
        assert len(comments) == 2
        assert comments[0].file == "src/main.rs"
        assert comments[0].line == 42
        assert comments[0].body == "Potential null deref here"

    @patch("bugeval.copilot_runner.run_gh")
    def test_filters_non_copilot(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = json.dumps([
            {
                "path": "src/main.rs",
                "line": 42,
                "body": "Copilot finding",
                "user": {"login": "copilot[bot]"},
            },
            {
                "path": "src/lib.rs",
                "line": 10,
                "body": "Human comment",
                "user": {"login": "somedev"},
            },
        ])
        comments = scrape_pr_comments("testuser/snarkVM", 99)
        assert len(comments) == 1
        assert comments[0].body == "Copilot finding"

    @patch("bugeval.copilot_runner.run_gh")
    def test_empty_comments(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = json.dumps([])
        comments = scrape_pr_comments("testuser/snarkVM", 99)
        assert comments == []


class TestCloseEvalPr:
    @patch("bugeval.copilot_runner.run_gh")
    def test_closes_and_deletes_branch(
        self, mock_gh: MagicMock,
    ) -> None:
        mock_gh.return_value = ""
        close_eval_pr("testuser/snarkVM", 99, "eval/snarkVM-001")
        assert mock_gh.call_count == 2  # close PR + delete branch


class TestPollForReview:
    @patch("bugeval.copilot_runner.time.sleep")
    @patch("bugeval.copilot_runner.run_gh")
    def test_found_review(
        self, mock_gh: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_gh.return_value = json.dumps({
            "reviews": [
                {"author": {"login": "copilot[bot]"}, "state": "COMMENTED"},
            ],
        })
        result = poll_for_review("testuser/snarkVM", 99, timeout=60)
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
        # Simulate time passing beyond timeout
        mock_time.side_effect = [0.0, 100.0, 400.0]
        result = poll_for_review(
            "testuser/snarkVM", 99, timeout=300, poll_interval=15,
        )
        assert result is False


class TestRunCopilot:
    @patch("bugeval.copilot_runner.close_eval_pr")
    @patch("bugeval.copilot_runner._scrape_raw_comments")
    @patch("bugeval.copilot_runner.scrape_pr_comments")
    @patch("bugeval.copilot_runner.poll_for_review")
    @patch("bugeval.copilot_runner.open_eval_pr")
    @patch("bugeval.copilot_runner._isolate_fork")
    @patch("bugeval.copilot_runner._default_branch", return_value="main")
    @patch("bugeval.copilot_runner.create_eval_branch")
    @patch("bugeval.copilot_runner.ensure_fork")
    @patch("bugeval.copilot_runner._get_patch_diff")
    def test_success(
        self,
        mock_diff: MagicMock,
        mock_fork: MagicMock,
        mock_branch: MagicMock,
        mock_default_br: MagicMock,
        mock_isolate: MagicMock,
        mock_open: MagicMock,
        mock_poll: MagicMock,
        mock_scrape: MagicMock,
        mock_raw: MagicMock,
        mock_close: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_diff.return_value = "diff content"
        mock_fork.return_value = "testuser/snarkVM"
        mock_branch.return_value = "eval/snarkVM-001"
        mock_open.return_value = 99
        mock_poll.return_value = True
        mock_raw.return_value = [
            {"path": "src/main.rs", "line": 42, "body": "Bug found",
             "user": {"login": "copilot[bot]"}},
        ]
        mock_scrape.return_value = [
            Comment(
                file="src/main.rs", line=42,
                body="Bug found",
            ),
        ]
        case = _make_case()
        result = run_copilot(case, tmp_path)
        assert result.case_id == "snarkVM-001"
        assert result.tool == "copilot"
        assert len(result.comments) == 1
        assert result.error == ""
        mock_close.assert_called_once()
        mock_isolate.assert_called_once()

    @patch("bugeval.copilot_runner.close_eval_pr")
    @patch("bugeval.copilot_runner._scrape_raw_comments")
    @patch("bugeval.copilot_runner.scrape_pr_comments")
    @patch("bugeval.copilot_runner.poll_for_review")
    @patch("bugeval.copilot_runner.open_eval_pr")
    @patch("bugeval.copilot_runner._isolate_fork")
    @patch("bugeval.copilot_runner._default_branch", return_value="main")
    @patch("bugeval.copilot_runner.create_eval_branch")
    @patch("bugeval.copilot_runner.ensure_fork")
    @patch("bugeval.copilot_runner._get_patch_diff")
    def test_timeout(
        self,
        mock_diff: MagicMock,
        mock_fork: MagicMock,
        mock_branch: MagicMock,
        mock_default_br: MagicMock,
        mock_isolate: MagicMock,
        mock_open: MagicMock,
        mock_poll: MagicMock,
        mock_scrape: MagicMock,
        mock_raw: MagicMock,
        mock_close: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_diff.return_value = "diff content"
        mock_fork.return_value = "testuser/snarkVM"
        mock_branch.return_value = "eval/snarkVM-001"
        mock_open.return_value = 99
        mock_poll.return_value = False
        case = _make_case()
        result = run_copilot(case, tmp_path, timeout=60)
        assert result.case_id == "snarkVM-001"
        assert result.tool == "copilot"
        assert "timeout" in result.error.lower()
        mock_close.assert_called_once()

    @patch("bugeval.copilot_runner.ensure_fork")
    @patch("bugeval.copilot_runner._get_patch_diff")
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
        result = run_copilot(case, tmp_path)
        assert result.case_id == "snarkVM-001"
        assert result.tool == "copilot"
        assert result.error != ""
        assert len(result.comments) == 0


class TestGetPatchDiff:
    @patch("bugeval.copilot_runner.run_git")  # patched at import location
    def test_uses_introducing_commit(self, mock_git: MagicMock, tmp_path: Path) -> None:
        mock_git.return_value = "diff output"
        case = _make_case(
            truth=GroundTruth(introducing_commit="intro999"),
        )
        result = _get_patch_diff(case, tmp_path)
        assert result == "diff output"
        mock_git.assert_called_once_with(
            "diff", "intro999~1", "intro999", cwd=tmp_path,
        )

    @patch("bugeval.copilot_runner.run_git")  # patched at import location
    def test_falls_back_to_base_commit(
        self, mock_git: MagicMock, tmp_path: Path,
    ) -> None:
        mock_git.return_value = "diff output"
        case = _make_case()  # no truth, base_commit="abc123"
        result = _get_patch_diff(case, tmp_path)
        assert result == "diff output"
        mock_git.assert_called_once_with(
            "diff", "abc123~1", "abc123", cwd=tmp_path,
        )

    def test_no_commit_returns_empty(self, tmp_path: Path) -> None:
        case = _make_case(base_commit="")
        result = _get_patch_diff(case, tmp_path)
        assert result == ""


class TestIsolateFork:
    @patch("bugeval.copilot_runner.subprocess.run")
    def test_force_pushes_parent_to_default_branch(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        _isolate_fork("myorg/repo", "intro999", "main", tmp_path)
        call_args = mock_run.call_args[0][0]
        assert "--force" in call_args
        assert "intro999~1:refs/heads/main" in call_args
        assert "https://github.com/myorg/repo.git" in call_args

    @patch("bugeval.copilot_runner.subprocess.run")
    def test_raises_on_failure(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        from bugeval.mine import GhError

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="push failed",
        )
        try:
            _isolate_fork("myorg/repo", "intro999", "main", tmp_path)
            assert False, "Expected GhError"
        except GhError:
            pass


class TestCopilotTranscript:
    @patch("bugeval.copilot_runner.close_eval_pr")
    @patch("bugeval.copilot_runner._scrape_raw_comments")
    @patch("bugeval.copilot_runner.scrape_pr_comments")
    @patch("bugeval.copilot_runner.poll_for_review")
    @patch("bugeval.copilot_runner.open_eval_pr")
    @patch("bugeval.copilot_runner._isolate_fork")
    @patch("bugeval.copilot_runner._default_branch", return_value="main")
    @patch("bugeval.copilot_runner.create_eval_branch")
    @patch("bugeval.copilot_runner.ensure_fork")
    @patch("bugeval.copilot_runner._get_patch_diff")
    def test_transcript_saved(
        self,
        mock_diff: MagicMock,
        mock_fork: MagicMock,
        mock_branch: MagicMock,
        mock_default_br: MagicMock,
        mock_isolate: MagicMock,
        mock_open: MagicMock,
        mock_poll: MagicMock,
        mock_scrape: MagicMock,
        mock_raw: MagicMock,
        mock_close: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_diff.return_value = "diff content"
        mock_fork.return_value = "testuser/snarkVM"
        mock_branch.return_value = "eval/snarkVM-001"
        mock_open.return_value = 99
        mock_poll.return_value = True
        mock_raw.return_value = [
            {"path": "src/main.rs", "line": 42, "body": "Bug found",
             "user": {"login": "copilot[bot]"}},
        ]
        mock_scrape.return_value = [
            Comment(file="src/main.rs", line=42, body="Bug found"),
        ]
        transcript_dir = tmp_path / "transcripts"
        case = _make_case()
        result = run_copilot(
            case, tmp_path, transcript_dir=transcript_dir,
        )
        assert result.transcript_path != ""
        path = Path(result.transcript_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "pr_metadata" in data
        assert data["pr_metadata"]["fork"] == "testuser/snarkVM"
        assert data["pr_metadata"]["pr_number"] == 99
        assert "raw_comments" in data
        assert "patch_diff" in data
        assert "scrubbed_title" in data
        assert "scrubbed_body" in data
        assert "time_seconds" in data

    @patch("bugeval.copilot_runner.close_eval_pr")
    @patch("bugeval.copilot_runner._scrape_raw_comments")
    @patch("bugeval.copilot_runner.scrape_pr_comments")
    @patch("bugeval.copilot_runner.poll_for_review")
    @patch("bugeval.copilot_runner.open_eval_pr")
    @patch("bugeval.copilot_runner._isolate_fork")
    @patch("bugeval.copilot_runner._default_branch", return_value="main")
    @patch("bugeval.copilot_runner.create_eval_branch")
    @patch("bugeval.copilot_runner.ensure_fork")
    @patch("bugeval.copilot_runner._get_patch_diff")
    def test_no_transcript_without_dir(
        self,
        mock_diff: MagicMock,
        mock_fork: MagicMock,
        mock_branch: MagicMock,
        mock_default_br: MagicMock,
        mock_isolate: MagicMock,
        mock_open: MagicMock,
        mock_poll: MagicMock,
        mock_scrape: MagicMock,
        mock_raw: MagicMock,
        mock_close: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_diff.return_value = "diff content"
        mock_fork.return_value = "testuser/snarkVM"
        mock_branch.return_value = "eval/snarkVM-001"
        mock_open.return_value = 99
        mock_poll.return_value = True
        mock_raw.return_value = []
        mock_scrape.return_value = []
        case = _make_case()
        result = run_copilot(case, tmp_path)
        assert result.transcript_path == ""


class TestEnsureForkOrgReturn:
    @patch("bugeval.copilot_runner.run_gh")
    def test_org_returns_org_not_username(self, mock_gh: MagicMock) -> None:
        """When org is given, return org/name even if fork raises."""
        from bugeval.mine import GhError

        mock_gh.side_effect = GhError(["gh"], "already exists")
        result = ensure_fork("AleoNet/snarkVM", org="eval-org")
        assert result == "eval-org/snarkVM"
        # Should NOT have queried the user API
        assert mock_gh.call_count == 1
