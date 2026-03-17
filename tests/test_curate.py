"""Tests for the curate command: prompt construction, response parsing, CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from bugeval.curate import build_curation_prompt, curate, parse_llm_response
from bugeval.io import save_candidates
from bugeval.models import (
    Candidate,
    CaseStats,
    Category,
    Difficulty,
    ExpectedFinding,
    PRSize,
    Severity,
    TestCase,
)


def make_candidate(pr_number: int = 1) -> Candidate:
    return Candidate(
        repo="foo/bar",
        pr_number=pr_number,
        fix_commit="abc123def456abc123def456abc123def456abc1",
        confidence=0.8,
        signals=["has_bug_label", "pr_references_issue"],
        title=f"Fix off-by-one error in parser (PR #{pr_number})",
        body="Fixes #42\n\nThis PR fixes a critical off-by-one error.",
        labels=["bug"],
        files_changed=["src/parser.rs"],
        diff_stats=CaseStats(lines_added=5, lines_deleted=3, files_changed=1, hunks=1),
        expected_findings=[
            ExpectedFinding(file="src/parser.rs", line=42, summary="[auto] off-by-one")
        ],
        language="rust",
        pr_size=PRSize.tiny,
    )


def make_llm_response_data() -> dict[str, object]:
    return {
        "category": "logic",
        "difficulty": "easy",
        "severity": "low",
        "description": "An off-by-one error in the parser causes incorrect range calculations.",
        "expected_findings": [
            {"file": "src/parser.rs", "line": 42, "summary": "off-by-one in range calc"}
        ],
        "head_commit": None,
        "base_commit": None,
        "needs_manual_review": True,
    }


class TestBuildCurationPrompt:
    def test_contains_pr_title(self) -> None:
        candidate = make_candidate()
        prompt = build_curation_prompt(candidate, diff_context="diff here")
        assert candidate.title in prompt

    def test_contains_repo(self) -> None:
        candidate = make_candidate()
        prompt = build_curation_prompt(candidate, diff_context="")
        assert candidate.repo in prompt

    def test_contains_diff(self) -> None:
        candidate = make_candidate()
        prompt = build_curation_prompt(candidate, diff_context="--- a/parser.rs\n+++ b/parser.rs")
        assert "parser.rs" in prompt

    def test_contains_language(self) -> None:
        candidate = make_candidate()
        prompt = build_curation_prompt(candidate, diff_context="")
        assert "rust" in prompt.lower()

    def test_with_git_log(self) -> None:
        candidate = make_candidate()
        prompt = build_curation_prompt(
            candidate, diff_context="", git_log="abc123 introduce parser bug"
        )
        assert "abc123" in prompt

    def test_returns_nonempty_string(self) -> None:
        candidate = make_candidate()
        prompt = build_curation_prompt(candidate, diff_context="")
        assert len(prompt) > 100


class TestParseLlmResponse:
    def test_valid_response_creates_test_case(self) -> None:
        candidate = make_candidate()
        data = make_llm_response_data()
        case = parse_llm_response(data, case_id="bar-001", candidate=candidate)
        assert isinstance(case, TestCase)
        assert case.id == "bar-001"
        assert case.category == Category.logic
        assert case.difficulty == Difficulty.easy
        assert case.severity == Severity.low

    def test_uses_candidate_metadata(self) -> None:
        candidate = make_candidate()
        data = make_llm_response_data()
        case = parse_llm_response(data, case_id="bar-001", candidate=candidate)
        assert case.repo == candidate.repo
        assert case.language == candidate.language
        assert case.fix_commit == candidate.fix_commit

    def test_expected_findings_parsed(self) -> None:
        candidate = make_candidate()
        data = make_llm_response_data()
        case = parse_llm_response(data, case_id="bar-001", candidate=candidate)
        assert len(case.expected_findings) == 1
        assert case.expected_findings[0].file == "src/parser.rs"
        assert case.expected_findings[0].line == 42

    def test_head_base_commit_set(self) -> None:
        candidate = make_candidate()
        data = {
            **make_llm_response_data(),
            "head_commit": "deadbeef" * 5,
            "base_commit": "cafebabe" * 5,
        }
        case = parse_llm_response(data, case_id="bar-002", candidate=candidate)
        assert case.head_commit == "deadbeef" * 5

    def test_empty_findings(self) -> None:
        candidate = make_candidate()
        data = {**make_llm_response_data(), "expected_findings": []}
        case = parse_llm_response(data, case_id="bar-003", candidate=candidate)
        assert case.expected_findings == []

    def test_needs_manual_review_saved(self) -> None:
        candidate = make_candidate()
        data = {**make_llm_response_data(), "needs_manual_review": True}
        case = parse_llm_response(data, case_id="bar-005", candidate=candidate)
        assert case.needs_manual_review is True

    def test_needs_manual_review_defaults_false(self) -> None:
        candidate = make_candidate()
        data = make_llm_response_data()  # needs_manual_review: True in fixture
        data["needs_manual_review"] = False
        case = parse_llm_response(data, case_id="bar-006", candidate=candidate)
        assert case.needs_manual_review is False

    def test_null_commits_fallback(self) -> None:
        candidate = make_candidate()
        data = make_llm_response_data()  # head_commit: None, base_commit: None
        case = parse_llm_response(data, case_id="bar-004", candidate=candidate)
        assert case.head_commit == candidate.fix_commit
        assert case.base_commit == f"{candidate.fix_commit}^"


class TestCurateCliHelp:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(curate, ["--help"])
        assert result.exit_code == 0
        assert "--candidates" in result.output
        assert "--dry-run" in result.output
        assert "--output-dir" in result.output


def make_mock_subprocess(response_data: dict[str, object]) -> MagicMock:
    """Return a mock subprocess.run result that outputs stream-json (type=result event)."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    # Emit a stream-json result event so _parse_stream_json_output populates response_text
    result_event = {
        "type": "result",
        "result": json.dumps(response_data),
        "num_turns": 1,
        "total_cost_usd": 0.0,
        "usage": {},
    }
    mock_result.stdout = json.dumps(result_event) + "\n"
    mock_result.stderr = ""
    return mock_result


class TestCurateDryRun:
    def test_dry_run_does_not_call_api(self, tmp_path: Path) -> None:
        candidates = [make_candidate(1)]
        candidates_path = tmp_path / "candidates.yaml"
        save_candidates(candidates, candidates_path)

        runner = CliRunner()
        with patch("subprocess.run") as mock_run:
            result = runner.invoke(
                curate,
                [
                    "--candidates",
                    str(candidates_path),
                    "--dry-run",
                    "--output-dir",
                    str(tmp_path / "cases"),
                ],
            )
        assert result.exit_code == 0
        mock_run.assert_not_called()

    def test_dry_run_prints_candidate_info(self, tmp_path: Path) -> None:
        candidate = make_candidate(42)
        candidates_path = tmp_path / "cands.yaml"
        save_candidates([candidate], candidates_path)

        runner = CliRunner()
        result = runner.invoke(
            curate,
            [
                "--candidates",
                str(candidates_path),
                "--dry-run",
                "--output-dir",
                str(tmp_path / "cases"),
            ],
        )
        assert result.exit_code == 0
        assert "42" in result.output  # PR number


class TestCurateControls:
    """Tests for runaway-prevention controls: --limit, --fail-after, checkpoint."""

    def test_limit_caps_candidates(self, tmp_path: Path) -> None:
        candidates = [make_candidate(i) for i in range(1, 6)]
        candidates_path = tmp_path / "candidates.yaml"
        save_candidates(candidates, candidates_path)

        cases_dir = tmp_path / "cases"
        runner = CliRunner()
        mock_sub = make_mock_subprocess(make_llm_response_data())
        with patch("subprocess.run", return_value=mock_sub):
            result = runner.invoke(
                curate,
                [
                    "--candidates",
                    str(candidates_path),
                    "--output-dir",
                    str(cases_dir),
                    "--api-delay",
                    "0",
                    "--limit",
                    "2",
                ],
            )
        assert result.exit_code == 0
        assert len(list(cases_dir.glob("*.yaml"))) == 2

    def test_fail_after_aborts_on_consecutive_errors(self, tmp_path: Path) -> None:
        candidates = [make_candidate(i) for i in range(1, 6)]
        candidates_path = tmp_path / "candidates.yaml"
        save_candidates(candidates, candidates_path)

        cases_dir = tmp_path / "cases"
        runner = CliRunner()

        mock_sub = MagicMock()
        mock_sub.returncode = 1
        mock_sub.stderr = "API failure"
        mock_sub.stdout = ""

        with patch("subprocess.run", return_value=mock_sub):
            result = runner.invoke(
                curate,
                [
                    "--candidates",
                    str(candidates_path),
                    "--output-dir",
                    str(cases_dir),
                    "--api-delay",
                    "0",
                    "--fail-after",
                    "2",
                ],
            )
        assert result.exit_code == 0  # exits cleanly, not with exception
        assert "Aborting" in result.output

    def test_checkpoint_skips_already_done(self, tmp_path: Path) -> None:
        # Give each candidate a distinct fix_commit so checkpoint filtering works
        candidates = [
            make_candidate(i).model_copy(update={"fix_commit": f"{str(i) * 40}"[:40]})
            for i in range(1, 4)
        ]
        candidates_path = tmp_path / "candidates.yaml"
        save_candidates(candidates, candidates_path)

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()

        # Pre-populate checkpoint with first candidate's commit
        done_commit = candidates[0].fix_commit
        checkpoint = cases_dir / ".curate_checkpoint.json"
        checkpoint.write_text(json.dumps([done_commit]))

        mock_sub = make_mock_subprocess(make_llm_response_data())
        runner = CliRunner()
        with patch("subprocess.run", return_value=mock_sub) as mock_run:
            runner.invoke(
                curate,
                [
                    "--candidates",
                    str(candidates_path),
                    "--output-dir",
                    str(cases_dir),
                    "--api-delay",
                    "0",
                ],
            )
        # Only 2 of 3 candidates should have been processed (first was checkpointed)
        assert mock_run.call_count == 2

    def test_shard_splits_candidates(self, tmp_path: Path) -> None:
        # 6 candidates; shard 0/3 should process indices 0, 3 → 2 cases
        candidates = [
            make_candidate(i).model_copy(update={"fix_commit": f"{str(i) * 40}"[:40]})
            for i in range(1, 7)
        ]
        candidates_path = tmp_path / "candidates.yaml"
        save_candidates(candidates, candidates_path)

        cases_dir = tmp_path / "cases"
        runner = CliRunner()
        mock_sub = make_mock_subprocess(make_llm_response_data())
        with patch("subprocess.run", return_value=mock_sub):
            result = runner.invoke(
                curate,
                [
                    "--candidates",
                    str(candidates_path),
                    "--output-dir",
                    str(cases_dir),
                    "--api-delay",
                    "0",
                    "--shard",
                    "0/3",
                ],
            )
        assert result.exit_code == 0
        assert len(list(cases_dir.glob("*.yaml"))) == 2

    def test_shard_invalid_raises_error(self, tmp_path: Path) -> None:
        candidates = [make_candidate(1)]
        candidates_path = tmp_path / "candidates.yaml"
        save_candidates(candidates, candidates_path)

        runner = CliRunner()
        result = runner.invoke(
            curate,
            [
                "--candidates",
                str(candidates_path),
                "--output-dir",
                str(tmp_path / "cases"),
                "--shard",
                "bad",
            ],
        )
        assert result.exit_code != 0

    def test_no_checkpoint_flag_reprocesses_all(self, tmp_path: Path) -> None:
        candidates = [make_candidate(1)]
        candidates_path = tmp_path / "candidates.yaml"
        save_candidates(candidates, candidates_path)

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        # Mark the candidate as already done
        checkpoint = cases_dir / ".curate_checkpoint.json"
        checkpoint.write_text(json.dumps([candidates[0].fix_commit]))

        mock_sub = make_mock_subprocess(make_llm_response_data())
        runner = CliRunner()
        with patch("subprocess.run", return_value=mock_sub) as mock_run:
            runner.invoke(
                curate,
                [
                    "--candidates",
                    str(candidates_path),
                    "--output-dir",
                    str(cases_dir),
                    "--api-delay",
                    "0",
                    "--no-checkpoint",
                ],
            )
        assert mock_run.call_count == 1  # processed despite checkpoint


class TestCurateWithMockedApi:
    def test_curate_writes_case_file(self, tmp_path: Path) -> None:
        candidate = make_candidate(1)
        candidates_path = tmp_path / "candidates.yaml"
        save_candidates([candidate], candidates_path)

        cases_dir = tmp_path / "cases"
        runner = CliRunner()
        mock_sub = make_mock_subprocess(make_llm_response_data())
        with patch("subprocess.run", return_value=mock_sub):
            result = runner.invoke(
                curate,
                [
                    "--candidates",
                    str(candidates_path),
                    "--output-dir",
                    str(cases_dir),
                    "--api-delay",
                    "0",
                ],
            )
        assert result.exit_code == 0
        yaml_files = list(cases_dir.glob("*.yaml"))
        assert len(yaml_files) == 1

    def test_curate_skips_low_confidence(self, tmp_path: Path) -> None:
        candidate = make_candidate(1)
        candidate = candidate.model_copy(update={"confidence": 0.2})
        candidates_path = tmp_path / "candidates.yaml"
        save_candidates([candidate], candidates_path)

        cases_dir = tmp_path / "cases"
        runner = CliRunner()
        mock_sub = make_mock_subprocess(make_llm_response_data())
        with patch("subprocess.run", return_value=mock_sub):
            result = runner.invoke(
                curate,
                [
                    "--candidates",
                    str(candidates_path),
                    "--output-dir",
                    str(cases_dir),
                    "--min-confidence",
                    "0.5",
                    "--api-delay",
                    "0",
                ],
            )
        assert result.exit_code == 0
        yaml_files = list(cases_dir.glob("*.yaml"))
        assert len(yaml_files) == 0


class TestBuildCurationPromptReviewerContext:
    def test_includes_reviewer_notes_when_present(self) -> None:
        candidate = make_candidate()
        candidate = candidate.model_copy(
            update={"reviewer_notes": ["[inline] This will panic on empty input"]}
        )
        prompt = build_curation_prompt(candidate, diff_context="")
        assert "Human Reviewer Comments" in prompt
        assert "This will panic on empty input" in prompt

    def test_no_reviewer_notes_section_when_empty(self) -> None:
        candidate = make_candidate()
        prompt = build_curation_prompt(candidate, diff_context="")
        assert "Human Reviewer Comments" not in prompt

    def test_includes_reviewer_findings_when_present(self) -> None:
        from bugeval.models import ExpectedFinding

        candidate = make_candidate()
        candidate = candidate.model_copy(
            update={
                "reviewer_findings": [
                    ExpectedFinding(file="src/foo.rs", line=10, summary="bad logic")
                ]
            }
        )
        prompt = build_curation_prompt(candidate, diff_context="")
        assert "Reviewer-Anchored Findings" in prompt
        assert "src/foo.rs" in prompt

    def test_no_reviewer_findings_section_when_empty(self) -> None:
        candidate = make_candidate()
        prompt = build_curation_prompt(candidate, diff_context="")
        assert "Reviewer-Anchored Findings" not in prompt


class TestParseLlmResponsePreservesReviewerContext:
    def test_preserves_pr_number(self) -> None:
        candidate = make_candidate(pr_number=999)
        data = make_llm_response_data()
        case = parse_llm_response(data, case_id="bar-010", candidate=candidate)
        assert case.pr_number == 999

    def test_preserves_reviewer_notes(self) -> None:
        candidate = make_candidate()
        candidate = candidate.model_copy(
            update={"reviewer_notes": ["reviewer said X"]}
        )
        data = make_llm_response_data()
        case = parse_llm_response(data, case_id="bar-011", candidate=candidate)
        assert case.reviewer_notes == ["reviewer said X"]

    def test_preserves_reviewer_findings(self) -> None:
        from bugeval.models import ExpectedFinding

        candidate = make_candidate()
        candidate = candidate.model_copy(
            update={
                "reviewer_findings": [
                    ExpectedFinding(file="src/x.rs", line=5, summary="issue here")
                ]
            }
        )
        data = make_llm_response_data()
        case = parse_llm_response(data, case_id="bar-012", candidate=candidate)
        assert len(case.reviewer_findings) == 1
        assert case.reviewer_findings[0].file == "src/x.rs"


class TestCurateCandidateBackend:
    """curate_candidate must delegate subprocess calls to run_claude_cli."""

    def test_uses_run_claude_cli_backend(self) -> None:
        from bugeval.curate import curate_candidate

        candidate = make_candidate(1)
        mock_sub = make_mock_subprocess(make_llm_response_data())
        with patch("subprocess.run", return_value=mock_sub) as mock_run:
            result = curate_candidate(
                candidate=candidate,
                diff_context="diff here",
                git_log="",
                case_id="bar-001",
                system_prompt="classify this",
            )
        assert result is not None
        assert isinstance(result, __import__("bugeval.models", fromlist=["TestCase"]).TestCase)
        # subprocess.run must have been called (via run_claude_cli)
        assert mock_run.call_count == 1

    def test_default_max_turns_allows_tool_calls(self) -> None:
        """max_turns defaults to >1 so the agent can make WebSearch/WebFetch calls."""
        from bugeval.curate import curate_candidate

        candidate = make_candidate(1)
        mock_sub = make_mock_subprocess(make_llm_response_data())
        captured: list[list[str]] = []
        def _capture(cmd: list[str], **_kw: object) -> object:
            captured.append(cmd)
            return mock_sub

        with patch("subprocess.run", side_effect=_capture):
            curate_candidate(
                candidate=candidate,
                diff_context="",
                git_log="",
                case_id="bar-003",
                system_prompt="classify this",
            )
        assert captured, "subprocess.run not called"
        cmd = captured[0]
        turns_idx = cmd.index("--max-turns") if "--max-turns" in cmd else -1
        assert turns_idx != -1, "--max-turns not in command"
        assert int(cmd[turns_idx + 1]) > 1, "max_turns should be >1 to allow tool use"

    def test_timeout_is_at_least_300s(self) -> None:
        from bugeval.curate import curate_candidate

        candidate = make_candidate(1)
        mock_sub = make_mock_subprocess(make_llm_response_data())
        with patch("subprocess.run", return_value=mock_sub) as mock_run:
            curate_candidate(
                candidate=candidate,
                diff_context="",
                git_log="",
                case_id="bar-002",
                system_prompt="classify this",
            )
        _args, kwargs = mock_run.call_args
        assert kwargs.get("timeout", 0) >= 300
