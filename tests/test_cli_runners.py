"""Tests for CLI runner dispatch (claude, gemini, codex)."""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path
from unittest.mock import MagicMock, patch

from bugeval.agent_runner import (
    _estimate_claude_cli_cost,
    _run_claude_cli,
    _run_codex_cli,
    _run_gemini_cli,
    _save_cli_transcript,
    build_system_prompt,
    run_agent_cli,
)
from bugeval.models import CaseKind, TestCase
from bugeval.result_models import ToolResult


def _make_case(**overrides: object) -> TestCase:
    defaults: dict[str, object] = {
        "id": "leo-001",
        "repo": "AleoNet/leo",
        "kind": CaseKind.bug,
        "base_commit": "abc123",
        "fix_commit": "def456",
        "introducing_pr_title": "Add new parser",
        "introducing_pr_body": "Implements expression parsing.",
        "introducing_pr_commit_messages": ["feat: add parser"],
    }
    defaults.update(overrides)
    return TestCase(**defaults)  # type: ignore[arg-type]


SAMPLE_DIFF = "--- a/foo.rs\n+++ b/foo.rs\n@@ -1,3 +1,3 @@\n-old\n+new\n"


class TestEstimateClaudeCliCost:
    def test_basic_cost(self) -> None:
        cost_info = {"input_tokens": 1000, "output_tokens": 500}
        cost = _estimate_claude_cli_cost(cost_info)
        # 1000 * 3/1e6 + 500 * 15/1e6 = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 1e-6

    def test_with_cache_tokens(self) -> None:
        cost_info = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_input_tokens": 2000,
            "cache_creation_input_tokens": 500,
        }
        cost = _estimate_claude_cli_cost(cost_info)
        expected = (
            1000 * 3.0 / 1e6
            + 500 * 15.0 / 1e6
            + 2000 * 0.30 / 1e6
            + 500 * 3.75 / 1e6
        )
        assert abs(cost - round(expected, 6)) < 1e-6

    def test_empty_cost_info(self) -> None:
        assert _estimate_claude_cli_cost({}) == 0.0

    def test_none_values_treated_as_zero(self) -> None:
        cost_info = {
            "input_tokens": None,
            "output_tokens": None,
        }
        assert _estimate_claude_cli_cost(cost_info) == 0.0


class TestRunClaudeCliJsonOutput:
    @patch("bugeval.agent_runner.subprocess.run")
    def test_parses_json_output(self, mock_run: MagicMock) -> None:
        output = {
            "result": '[{"file":"f.rs","line":1,"description":"bug here"}]',
            "cost": {"input_tokens": 100, "output_tokens": 50},
            "session_id": "sess-1",
            "is_error": False,
            "duration_ms": 5000,
            "num_turns": 3,
        }
        mock_run.return_value = sp.CompletedProcess(
            args=["claude"], returncode=0,
            stdout=json.dumps(output), stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        result = _run_claude_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
        )
        assert result.case_id == "leo-001"
        assert result.tool == "agent-cli-claude"
        assert len(result.comments) == 1
        assert result.comments[0].file == "f.rs"
        assert result.cost_usd > 0
        assert result.error == ""

    @patch("bugeval.agent_runner.subprocess.run")
    def test_diff_only_disallows_tools(self, mock_run: MagicMock) -> None:
        output = {"result": "[]", "cost": {}}
        mock_run.return_value = sp.CompletedProcess(
            args=["claude"], returncode=0,
            stdout=json.dumps(output), stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        _run_claude_cli(case, SAMPLE_DIFF, None, "diff-only", 300, system)
        cmd = mock_run.call_args[0][0]
        assert "--disallowedTools" in cmd

    @patch("bugeval.agent_runner.subprocess.run")
    def test_repo_context_allows_tools(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        output = {"result": "[]", "cost": {}}
        mock_run.return_value = sp.CompletedProcess(
            args=["claude"], returncode=0,
            stdout=json.dumps(output), stderr="",
        )
        case = _make_case()
        repo = tmp_path / "repo"
        repo.mkdir()
        system = build_system_prompt("diff+repo")
        _run_claude_cli(
            case, SAMPLE_DIFF, repo, "diff+repo", 300, system,
        )
        cmd = mock_run.call_args[0][0]
        assert "--allowedTools" in cmd
        assert "--disallowedTools" not in cmd

    @patch("bugeval.agent_runner.subprocess.run")
    def test_json_decode_error_fallback(self, mock_run: MagicMock) -> None:
        mock_run.return_value = sp.CompletedProcess(
            args=["claude"], returncode=0,
            stdout='[{"file":"f.rs","line":1,"description":"plain text bug"}]',
            stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        result = _run_claude_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
        )
        # Should fall back to parsing stdout as plain text
        assert result.error == ""
        assert len(result.comments) == 1

    @patch("bugeval.agent_runner.subprocess.run")
    def test_timeout_returns_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = sp.TimeoutExpired(cmd="claude", timeout=300)
        case = _make_case()
        system = build_system_prompt("diff-only")
        result = _run_claude_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
        )
        assert "timed out" in result.error.lower()

    @patch("bugeval.agent_runner.subprocess.run")
    def test_not_found_returns_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("claude not found")
        case = _make_case()
        system = build_system_prompt("diff-only")
        result = _run_claude_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
        )
        assert "not found" in result.error.lower()


class TestRunClaudeCliTranscript:
    @patch("bugeval.agent_runner.subprocess.run")
    def test_transcript_saved(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        output = {"result": "[]", "cost": {"input_tokens": 10}}
        mock_run.return_value = sp.CompletedProcess(
            args=["claude"], returncode=0,
            stdout=json.dumps(output), stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        transcript_dir = tmp_path / "transcripts"
        result = _run_claude_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
            transcript_dir=transcript_dir,
        )
        assert result.transcript_path != ""
        path = Path(result.transcript_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["tool"] == "claude"
        assert "prompt" in data
        assert "output" in data


class TestRunGeminiCli:
    @patch("bugeval.agent_runner.subprocess.run")
    def test_correct_flags(self, mock_run: MagicMock) -> None:
        output = '[{"file":"g.rs","line":5,"description":"issue"}]'
        mock_run.return_value = sp.CompletedProcess(
            args=["gemini"], returncode=0,
            stdout=output, stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        result = _run_gemini_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
        )
        assert result.tool == "agent-cli-gemini"
        assert result.error == ""
        assert len(result.comments) == 1
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gemini"
        assert "-p" in cmd

    @patch("bugeval.agent_runner.subprocess.run")
    def test_uses_stdin(self, mock_run: MagicMock) -> None:
        """Verify Gemini CLI pipes prompt via stdin, not as CLI argument."""
        mock_run.return_value = sp.CompletedProcess(
            args=["gemini"], returncode=0, stdout="[]", stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        _run_gemini_cli(case, SAMPLE_DIFF, None, "diff-only", 300, system)
        call_kwargs = mock_run.call_args
        cmd = call_kwargs[0][0]
        # Prompt must NOT be in the command args
        for arg in cmd:
            assert "```diff" not in arg, "Prompt leaked into CLI args"
        # Prompt must be piped via input=
        assert call_kwargs.kwargs.get("input") is not None

    @patch("bugeval.agent_runner.subprocess.run")
    def test_repo_context_uses_yolo(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        mock_run.return_value = sp.CompletedProcess(
            args=["gemini"], returncode=0, stdout="[]", stderr="",
        )
        case = _make_case()
        repo = tmp_path / "repo"
        repo.mkdir()
        system = build_system_prompt("diff+repo")
        _run_gemini_cli(
            case, SAMPLE_DIFF, repo, "diff+repo", 300, system,
        )
        cmd = mock_run.call_args[0][0]
        assert "--yolo" in cmd

    @patch("bugeval.agent_runner.subprocess.run")
    def test_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = sp.TimeoutExpired(cmd="gemini", timeout=300)
        case = _make_case()
        system = build_system_prompt("diff-only")
        result = _run_gemini_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
        )
        assert "timed out" in result.error.lower()


class TestRunCodexCli:
    @patch("bugeval.agent_runner.subprocess.run")
    def test_correct_flags(self, mock_run: MagicMock) -> None:
        output = '[{"file":"c.rs","line":3,"description":"codex issue"}]'
        mock_run.return_value = sp.CompletedProcess(
            args=["codex"], returncode=0,
            stdout=output, stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        result = _run_codex_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
        )
        assert result.tool == "agent-cli-codex"
        assert result.error == ""
        assert len(result.comments) == 1
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "codex"
        assert "exec" in cmd
        assert "--sandbox" in cmd
        idx = cmd.index("--sandbox")
        assert cmd[idx + 1] == "read-only"

    @patch("bugeval.agent_runner.subprocess.run")
    def test_uses_stdin(self, mock_run: MagicMock) -> None:
        """Verify Codex CLI pipes prompt via stdin, not as CLI argument."""
        mock_run.return_value = sp.CompletedProcess(
            args=["codex"], returncode=0, stdout="[]", stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        _run_codex_cli(case, SAMPLE_DIFF, None, "diff-only", 300, system)
        call_kwargs = mock_run.call_args
        cmd = call_kwargs[0][0]
        # Prompt must NOT be in the command args
        for arg in cmd:
            assert "```diff" not in arg, "Prompt leaked into CLI args"
        # Prompt must be piped via input=
        assert call_kwargs.kwargs.get("input") is not None

    @patch("bugeval.agent_runner.subprocess.run")
    def test_repo_context_uses_workspace_write(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        mock_run.return_value = sp.CompletedProcess(
            args=["codex"], returncode=0, stdout="[]", stderr="",
        )
        case = _make_case()
        repo = tmp_path / "repo"
        repo.mkdir()
        system = build_system_prompt("diff+repo")
        _run_codex_cli(
            case, SAMPLE_DIFF, repo, "diff+repo", 300, system,
        )
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--sandbox")
        assert cmd[idx + 1] == "workspace-write"

    @patch("bugeval.agent_runner.subprocess.run")
    def test_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = sp.TimeoutExpired(cmd="codex", timeout=300)
        case = _make_case()
        system = build_system_prompt("diff-only")
        result = _run_codex_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
        )
        assert "timed out" in result.error.lower()


class TestCliDispatchAllTools:
    @patch("bugeval.agent_runner._run_claude_cli")
    def test_dispatch_claude(self, mock_claude: MagicMock) -> None:
        mock_claude.return_value = ToolResult(
            case_id="leo-001", tool="agent-cli-claude",
        )
        case = _make_case()
        result = run_agent_cli(
            case, SAMPLE_DIFF, None, "diff-only", cli_tool="claude",
        )
        assert result.tool == "agent-cli-claude"
        mock_claude.assert_called_once()

    @patch("bugeval.agent_runner._run_gemini_cli")
    def test_dispatch_gemini(self, mock_gemini: MagicMock) -> None:
        mock_gemini.return_value = ToolResult(
            case_id="leo-001", tool="agent-cli-gemini",
        )
        case = _make_case()
        result = run_agent_cli(
            case, SAMPLE_DIFF, None, "diff-only", cli_tool="gemini",
        )
        assert result.tool == "agent-cli-gemini"
        mock_gemini.assert_called_once()

    @patch("bugeval.agent_runner._run_codex_cli")
    def test_dispatch_codex(self, mock_codex: MagicMock) -> None:
        mock_codex.return_value = ToolResult(
            case_id="leo-001", tool="agent-cli-codex",
        )
        case = _make_case()
        result = run_agent_cli(
            case, SAMPLE_DIFF, None, "diff-only", cli_tool="codex",
        )
        assert result.tool == "agent-cli-codex"
        mock_codex.assert_called_once()

    def test_dispatch_unknown(self) -> None:
        case = _make_case()
        result = run_agent_cli(
            case, SAMPLE_DIFF, None, "diff-only", cli_tool="unknown",
        )
        assert "Unknown CLI tool" in result.error


class TestSaveCliTranscript:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        td = tmp_path / "transcripts"
        path = _save_cli_transcript(
            td, "leo-001", "claude", "my prompt", {"result": "ok"},
        )
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["tool"] == "claude"
        assert data["output"] == {"result": "ok"}

    def test_truncates_long_prompt(self, tmp_path: Path) -> None:
        td = tmp_path / "transcripts"
        long_prompt = "x" * 10000
        path = _save_cli_transcript(
            td, "leo-001", "claude", long_prompt, "out",
        )
        data = json.loads(Path(path).read_text())
        assert len(data["prompt"]) == 5000


class TestCliModelOverride:
    @patch("bugeval.agent_runner.subprocess.run")
    def test_claude_includes_model_flag(self, mock_run: MagicMock) -> None:
        output = {"result": "[]", "cost": {}}
        mock_run.return_value = sp.CompletedProcess(
            args=["claude"], returncode=0,
            stdout=json.dumps(output), stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        _run_claude_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
            model="claude-opus-4-6",
        )
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4-6"

    @patch("bugeval.agent_runner.subprocess.run")
    def test_claude_omits_model_flag_when_empty(
        self, mock_run: MagicMock,
    ) -> None:
        output = {"result": "[]", "cost": {}}
        mock_run.return_value = sp.CompletedProcess(
            args=["claude"], returncode=0,
            stdout=json.dumps(output), stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        _run_claude_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system, model="",
        )
        cmd = mock_run.call_args[0][0]
        assert "--model" not in cmd

    @patch("bugeval.agent_runner.subprocess.run")
    def test_gemini_includes_model_flag(self, mock_run: MagicMock) -> None:
        mock_run.return_value = sp.CompletedProcess(
            args=["gemini"], returncode=0, stdout="[]", stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        _run_gemini_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
            model="gemini-2.5-pro",
        )
        cmd = mock_run.call_args[0][0]
        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "gemini-2.5-pro"

    @patch("bugeval.agent_runner.subprocess.run")
    def test_codex_includes_model_flag(self, mock_run: MagicMock) -> None:
        mock_run.return_value = sp.CompletedProcess(
            args=["codex"], returncode=0, stdout="[]", stderr="",
        )
        case = _make_case()
        system = build_system_prompt("diff-only")
        _run_codex_cli(
            case, SAMPLE_DIFF, None, "diff-only", 300, system,
            model="o3",
        )
        cmd = mock_run.call_args[0][0]
        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "o3"

    @patch("bugeval.agent_runner.subprocess.run")
    def test_run_agent_cli_passes_model(self, mock_run: MagicMock) -> None:
        """Verify run_agent_cli threads model to the underlying CLI runner."""
        output = {"result": "[]", "cost": {}}
        mock_run.return_value = sp.CompletedProcess(
            args=["claude"], returncode=0,
            stdout=json.dumps(output), stderr="",
        )
        case = _make_case()
        run_agent_cli(
            case, SAMPLE_DIFF, None, "diff-only",
            cli_tool="claude", timeout=60, model="claude-opus-4-6",
        )
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4-6"


class TestEvaluateDispatchGeminiCodex:
    def test_dispatch_agent_cli_gemini(self, tmp_path: Path) -> None:
        from bugeval.evaluate import process_case

        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent-cli-gemini",
        )

        with (
            patch(
                "bugeval.evaluate.get_diff_for_case", return_value="diff",
            ),
            patch(
                "bugeval.agent_runner.run_agent_cli",
                return_value=fake_result,
            ) as mock_cli,
        ):
            result = process_case(
                case, "agent-cli-gemini", "diff-only",
                repo_dir, run_dir, 300,
            )

        assert result.tool == "agent-cli-gemini"
        call_kwargs = mock_cli.call_args
        assert call_kwargs.kwargs.get("cli_tool") == "gemini"

    def test_dispatch_agent_cli_codex(self, tmp_path: Path) -> None:
        from bugeval.evaluate import process_case

        case = _make_case()
        run_dir = tmp_path / "run"
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        fake_result = ToolResult(
            case_id="leo-001", tool="agent-cli-codex",
        )

        with (
            patch(
                "bugeval.evaluate.get_diff_for_case", return_value="diff",
            ),
            patch(
                "bugeval.agent_runner.run_agent_cli",
                return_value=fake_result,
            ) as mock_cli,
        ):
            result = process_case(
                case, "agent-cli-codex", "diff-only",
                repo_dir, run_dir, 300,
            )

        assert result.tool == "agent-cli-codex"
        call_kwargs = mock_cli.call_args
        assert call_kwargs.kwargs.get("cli_tool") == "codex"
