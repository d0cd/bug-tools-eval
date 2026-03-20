"""Tests for evaluate module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from bugeval.evaluate import (
    evaluate_tool,
    get_diff_for_case,
    process_case,
    result_filename,
)
from bugeval.models import CaseKind, GroundTruth, TestCase
from bugeval.result_models import Comment, ToolResult


def _make_case(**overrides: object) -> TestCase:
    defaults: dict[str, object] = {
        "id": "leo-001",
        "repo": "AleoNet/leo",
        "kind": CaseKind.bug,
        "base_commit": "abc123",
        "fix_commit": "def456",
    }
    defaults.update(overrides)
    return TestCase(**defaults)  # type: ignore[arg-type]


class TestResultFilename:
    def test_with_context(self) -> None:
        assert result_filename("leo-001", "agent", "diff-only") == (
            "leo-001--agent--diff-only.yaml"
        )

    def test_without_context(self) -> None:
        assert result_filename("leo-001", "greptile", "") == (
            "leo-001--greptile.yaml"
        )

    def test_complex_context(self) -> None:
        assert result_filename("snarkVM-042", "agent", "diff+repo+domain") == (
            "snarkVM-042--agent--diff+repo+domain.yaml"
        )


class TestProcessCase:
    def test_dispatches_agent(self, tmp_path: Path) -> None:
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001",
            tool="agent",
            context_level="diff-only",
            comments=[Comment(file="f.rs", line=1, body="bug")],
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="some diff"),
            patch(
                "bugeval.agent_runner.run_anthropic_api",
                return_value=fake_result,
            ),
        ):
            result = process_case(
                case, "agent", "diff-only", repo_dir, run_dir, 300
            )

        assert result.tool == "agent"
        assert result.case_id == "leo-001"

    def test_dispatches_greptile(self, tmp_path: Path) -> None:
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="greptile",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.greptile_runner.run_greptile",
                return_value=fake_result,
            ) as mock_greptile,
        ):
            result = process_case(
                case, "greptile", "", repo_dir, run_dir, 300,
            )

        assert result.tool == "greptile"
        mock_greptile.assert_called_once()
        call_kwargs = mock_greptile.call_args
        assert call_kwargs.kwargs.get("transcript_dir") is not None

    def test_dispatches_copilot(self, tmp_path: Path) -> None:
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001",
            tool="copilot",
            comments=[Comment(file="f.rs", line=1, body="bug")],
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="some diff"),
            patch(
                "bugeval.copilot_runner.run_copilot",
                return_value=fake_result,
            ) as mock_copilot,
        ):
            result = process_case(
                case, "copilot", "", repo_dir, run_dir, 300,
            )

        assert result.tool == "copilot"
        assert result.case_id == "leo-001"
        mock_copilot.assert_called_once()
        call_kwargs = mock_copilot.call_args
        assert call_kwargs.kwargs.get("transcript_dir") is not None

    def test_dispatches_agent_cli(self, tmp_path: Path) -> None:
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent-cli-claude",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.run_agent_cli",
                return_value=fake_result,
            ) as mock_cli,
        ):
            result = process_case(
                case, "agent-cli", "diff-only", repo_dir, run_dir, 300,
            )

        assert result.tool == "agent-cli-claude"
        mock_cli.assert_called_once()

    def test_dispatches_agent_sdk(self, tmp_path: Path) -> None:
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent-sdk",
            error="agent-sdk runner not yet implemented",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.run_agent_sdk",
                return_value=fake_result,
            ) as mock_sdk,
        ):
            result = process_case(
                case, "agent-sdk", "diff-only", repo_dir, run_dir, 300,
            )

        assert result.tool == "agent-sdk"
        mock_sdk.assert_called_once()

    def test_dispatches_coderabbit(self, tmp_path: Path) -> None:
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="coderabbit",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.coderabbit_runner.run_coderabbit",
                return_value=fake_result,
            ) as mock_coderabbit,
        ):
            result = process_case(
                case, "coderabbit", "", repo_dir, run_dir, 300,
            )

        assert result.tool == "coderabbit"
        mock_coderabbit.assert_called_once()
        call_kwargs = mock_coderabbit.call_args
        assert call_kwargs.kwargs.get("transcript_dir") is not None

    def test_dispatches_agent_gemini(self, tmp_path: Path) -> None:
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent-gemini",
            context_level="diff-only",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.run_google_api",
                return_value=fake_result,
            ) as mock_google,
        ):
            result = process_case(
                case, "agent-gemini", "diff-only", repo_dir, run_dir, 300,
            )

        assert result.tool == "agent-gemini"
        mock_google.assert_called_once()

    def test_dispatches_agent_openai(self, tmp_path: Path) -> None:
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent-openai",
            context_level="diff-only",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.run_openai_api",
                return_value=fake_result,
            ) as mock_openai,
        ):
            result = process_case(
                case, "agent-openai", "diff-only", repo_dir, run_dir, 300,
            )

        assert result.tool == "agent-openai"
        mock_openai.assert_called_once()

    def test_unsupported_tool(self, tmp_path: Path) -> None:
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        with patch("bugeval.evaluate.get_diff_for_case", return_value="diff"):
            result = process_case(
                case, "unknown_tool", "", repo_dir, run_dir, 300,
            )

        assert "Unsupported tool" in result.error


class TestEvaluateTool:
    def test_checkpoint_skips_done(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Write a case file
        case = _make_case()
        import yaml
        with open(cases_dir / "leo-001.yaml", "w") as f:
            yaml.safe_dump(case.model_dump(mode="json"), f)

        # Pre-populate checkpoint with this case done
        ckpt = run_dir / "checkpoint.json"
        ckpt.write_text(json.dumps(["leo-001::agent::diff-only"]))

        with patch("bugeval.evaluate.process_case") as mock_process:
            evaluate_tool(
                "agent", cases_dir, run_dir, "diff-only",
                Path("."), 1, 300, False,
            )

        # process_case should NOT have been called
        mock_process.assert_not_called()

    def test_dry_run_skips_processing(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        run_dir = tmp_path / "run"

        case = _make_case()
        import yaml
        with open(cases_dir / "leo-001.yaml", "w") as f:
            yaml.safe_dump(case.model_dump(mode="json"), f)

        with patch("bugeval.evaluate.process_case") as mock_process:
            evaluate_tool(
                "agent", cases_dir, run_dir, "diff-only",
                Path("."), 1, 300, True,
            )

        mock_process.assert_not_called()
        # run_dir should have been created
        assert run_dir.exists()

    def test_no_cases_returns_early(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "empty_cases"
        cases_dir.mkdir()
        run_dir = tmp_path / "run"

        with patch("bugeval.evaluate.process_case") as mock_process:
            evaluate_tool(
                "agent", cases_dir, run_dir, "",
                Path("."), 1, 300, False,
            )

        mock_process.assert_not_called()


class TestGetDiffForCase:
    def test_uses_introducing_commit(self) -> None:
        """When truth.introducing_commit is set, diff that commit vs its parent."""
        case = _make_case(
            truth=GroundTruth(
                introducing_commit="intro999",
                fix_pr_numbers=[1],
            ),
        )

        with patch("bugeval.evaluate.get_diff", return_value="intro diff") as mock_diff:
            result = get_diff_for_case(case, Path("/repo"))

        assert result == "intro diff"
        mock_diff.assert_called_once_with(
            "intro999~1", "intro999", cwd=Path("/repo"),
        )

    def test_no_introducing_returns_empty(self) -> None:
        """When no introducing_commit, return empty string (no fallback to fix diff)."""
        case = _make_case()
        result = get_diff_for_case(case, Path("/repo"))
        assert result == ""

    def test_no_commits_returns_empty(self) -> None:
        """When no fix or base commit and no introducing, return empty string."""
        case = _make_case(fix_commit="", base_commit="")
        result = get_diff_for_case(case, Path("/repo"))
        assert result == ""

    def test_truth_none_returns_empty(self) -> None:
        """When truth is None, return empty (no fallback to fix diff)."""
        case = _make_case(truth=None)
        result = get_diff_for_case(case, Path("/repo"))
        assert result == ""


class TestEvaluatePassesTranscriptDir:
    def test_agent_receives_transcript_dir(self, tmp_path: Path) -> None:
        """Verify process_case passes transcript_dir to run_anthropic_api."""
        case = _make_case(
            truth=GroundTruth(introducing_commit="intro999"),
        )
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent", context_level="diff-only",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.run_anthropic_api",
                return_value=fake_result,
            ) as mock_api,
        ):
            process_case(
                case, "agent", "diff-only", repo_dir, run_dir, 300,
            )

        # run_anthropic_api should have been called with transcript_dir set
        call_kwargs = mock_api.call_args
        assert call_kwargs.kwargs.get("transcript_dir") is not None
        transcript_dir = call_kwargs.kwargs["transcript_dir"]
        assert transcript_dir == run_dir / "transcripts"
        assert transcript_dir.exists()


class TestEvaluateAgentWorkspaceSetup:
    def test_agent_calls_setup_workspace_for_repo_context(
        self, tmp_path: Path,
    ) -> None:
        """Verify that agent with diff+repo context sets up workspace."""
        case = _make_case(
            truth=GroundTruth(introducing_commit="intro999"),
        )
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        ws_path = tmp_path / "workspace"

        fake_result = ToolResult(
            case_id="leo-001", tool="agent", context_level="diff+repo",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.setup_workspace",
                return_value=ws_path,
            ) as mock_setup,
            patch(
                "bugeval.agent_runner.run_anthropic_api",
                return_value=fake_result,
            ) as mock_api,
        ):
            process_case(
                case, "agent", "diff+repo", repo_dir, run_dir, 300,
            )

        # setup_workspace should have been called
        mock_setup.assert_called_once()
        # run_anthropic_api should receive the workspace path
        call_args = mock_api.call_args
        assert call_args[0][2] == ws_path  # 3rd positional arg is repo_dir

    def test_agent_diff_only_skips_workspace(self, tmp_path: Path) -> None:
        """Verify diff-only context does NOT call setup_workspace."""
        case = _make_case(
            truth=GroundTruth(introducing_commit="intro999"),
        )
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent", context_level="diff-only",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.setup_workspace",
            ) as mock_setup,
            patch(
                "bugeval.agent_runner.run_anthropic_api",
                return_value=fake_result,
            ) as mock_api,
        ):
            process_case(
                case, "agent", "diff-only", repo_dir, run_dir, 300,
            )

        mock_setup.assert_not_called()
        # workspace should be None for diff-only
        call_args = mock_api.call_args
        assert call_args[0][2] is None


class TestEvaluateModelOverride:
    def test_model_passed_to_agent_api(self, tmp_path: Path) -> None:
        """Verify process_case passes model to run_anthropic_api."""
        case = _make_case(
            truth=GroundTruth(introducing_commit="intro999"),
        )
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent", context_level="diff-only",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.run_anthropic_api",
                return_value=fake_result,
            ) as mock_api,
        ):
            process_case(
                case, "agent", "diff-only", repo_dir, run_dir, 300,
                model="claude-opus-4-6",
            )

        call_kwargs = mock_api.call_args
        assert call_kwargs.kwargs.get("model") == "claude-opus-4-6"

    def test_model_passed_to_agent_cli(self, tmp_path: Path) -> None:
        """Verify process_case passes model to run_agent_cli."""
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent-cli-claude",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.run_agent_cli",
                return_value=fake_result,
            ) as mock_cli,
        ):
            process_case(
                case, "agent-cli-claude", "diff-only", repo_dir, run_dir,
                300, model="claude-opus-4-6",
            )

        call_kwargs = mock_cli.call_args
        assert call_kwargs.kwargs.get("model") == "claude-opus-4-6"

    def test_model_passed_to_agent_sdk(self, tmp_path: Path) -> None:
        """Verify process_case passes model to run_agent_sdk."""
        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent-sdk",
        )

        with (
            patch("bugeval.evaluate.get_diff_for_case", return_value="diff"),
            patch(
                "bugeval.agent_runner.run_agent_sdk",
                return_value=fake_result,
            ) as mock_sdk,
        ):
            process_case(
                case, "agent-sdk", "diff-only", repo_dir, run_dir, 300,
                model="claude-opus-4-6",
            )

        call_kwargs = mock_sdk.call_args
        assert call_kwargs.kwargs.get("model") == "claude-opus-4-6"
