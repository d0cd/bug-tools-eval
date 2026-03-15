"""Tests for agent_cli_runner."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from bugeval.agent_cli_runner import (
    _parse_cli_findings,
    _parse_cli_token_count,
    run_claude_cli,
    run_claude_cli_docker,
    run_codex_cli,
    run_gemini_cli,
)


def test_run_claude_cli_success(tmp_path: Path) -> None:
    findings_json = '[{"file": "src/main.rs", "line": 10, "summary": "bug"}]'
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = f"Some output\n```json\n{findings_json}\n```\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_claude_cli(tmp_path, "review this patch")

    assert result.error is None
    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "src/main.rs"
    assert result.model == "claude-sonnet-4-6"
    assert result.wall_time_seconds >= 0


def test_run_claude_cli_timeout(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5)):
        result = run_claude_cli(tmp_path, "prompt", timeout_seconds=5)

    assert result.error == "timeout"
    assert result.findings == []


def test_run_claude_cli_nonzero_exit(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "claude: command not found"

    with patch("subprocess.run", return_value=mock_result):
        result = run_claude_cli(tmp_path, "prompt")

    assert result.error is not None
    assert "code 1" in result.error
    assert result.findings == []


def test_parse_cli_findings_with_json_array() -> None:
    stdout = 'Here are the findings:\n```json\n[{"file": "a.rs", "line": 5, "summary": "x"}]\n```'
    findings = _parse_cli_findings(stdout)
    assert len(findings) == 1
    assert findings[0]["file"] == "a.rs"


def test_parse_cli_findings_empty_output() -> None:
    findings = _parse_cli_findings("")
    assert findings == []


def test_run_claude_cli_passes_max_turns(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "[]"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        run_claude_cli(tmp_path, "prompt", max_turns=5)

    call_args = mock_run.call_args[0][0]
    assert "--max-turns" in call_args
    assert "5" in call_args


def test_run_claude_cli_docker_calls_docker(tmp_path: Path) -> None:
    """Verify docker run command is constructed correctly."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = '[{"file": "a.rs", "line": 1, "summary": "bug"}]'
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = run_claude_cli_docker(
            repo_dir=tmp_path,
            prompt="Review this code.",
            max_turns=5,
            image="bugeval-agent",
        )

    args = mock_run.call_args[0][0]
    assert args[0] == "docker"
    assert "run" in args
    assert "--rm" in args
    assert "-e" in args
    assert "ANTHROPIC_API_KEY" in args
    assert f"{tmp_path.resolve()}:/work" in args
    assert "bugeval-agent" in args
    assert "--max-turns" in args
    assert "5" in args
    assert result.findings == [{"file": "a.rs", "line": 1, "summary": "bug"}]


def test_run_claude_cli_docker_timeout(tmp_path: Path) -> None:
    """Timeout returns AgentResult with error='timeout'."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=5)):
        result = run_claude_cli_docker(tmp_path, "prompt", image="bugeval-agent")
    assert result.error == "timeout"


def test_run_claude_cli_docker_nonzero_exit(tmp_path: Path) -> None:
    """Non-zero exit code returns AgentResult with error set."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "container error"

    with patch("subprocess.run", return_value=mock_result):
        result = run_claude_cli_docker(tmp_path, "prompt", image="bugeval-agent")
    assert result.error is not None
    assert "code 1" in result.error


def test_run_gemini_cli_success(tmp_path: Path) -> None:
    findings_json = '[{"file": "src/main.rs", "line": 10, "summary": "bug"}]'
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = f"```json\n{findings_json}\n```\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_gemini_cli(tmp_path, "review this patch")

    assert result.error is None
    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "src/main.rs"
    assert result.model == "gemini-2.5-flash"
    assert result.wall_time_seconds >= 0


def test_run_gemini_cli_timeout(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="gemini", timeout=5)):
        result = run_gemini_cli(tmp_path, "prompt", timeout_seconds=5)

    assert result.error == "timeout"
    assert result.findings == []


def test_run_gemini_cli_nonzero_exit(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "gemini: command not found"

    with patch("subprocess.run", return_value=mock_result):
        result = run_gemini_cli(tmp_path, "prompt")

    assert result.error is not None
    assert "code 1" in result.error
    assert result.findings == []


def test_run_codex_cli_success(tmp_path: Path) -> None:
    findings_json = '[{"file": "src/lib.rs", "line": 5, "summary": "issue"}]'
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = f"```json\n{findings_json}\n```\n"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_codex_cli(tmp_path, "review this patch")

    assert result.error is None
    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "src/lib.rs"
    assert result.model == "o4-mini"
    assert result.wall_time_seconds >= 0


def test_run_codex_cli_timeout(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=5)):
        result = run_codex_cli(tmp_path, "prompt", timeout_seconds=5)

    assert result.error == "timeout"
    assert result.findings == []


def test_run_codex_cli_nonzero_exit(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "codex: command not found"

    with patch("subprocess.run", return_value=mock_result):
        result = run_codex_cli(tmp_path, "prompt")

    assert result.error is not None
    assert "code 1" in result.error
    assert result.findings == []


def test_run_gemini_cli_passes_model(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "[]"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        run_gemini_cli(tmp_path, "prompt", model="gemini-2.5-flash-lite")

    call_args = mock_run.call_args[0][0]
    assert "-m" in call_args
    assert "gemini-2.5-flash-lite" in call_args


# ---------------------------------------------------------------------------
# Token count parsing
# ---------------------------------------------------------------------------


def test_parse_cli_token_count_total_tokens_pattern() -> None:
    """Parses 'Total tokens: N' pattern."""
    assert _parse_cli_token_count("Total tokens: 1234") == 1234


def test_parse_cli_token_count_input_output_pattern() -> None:
    """Sums input + output tokens when both are present."""
    output = "Input tokens: 100\nOutput tokens: 50"
    assert _parse_cli_token_count(output) == 150


def test_parse_cli_token_count_case_insensitive() -> None:
    """Parsing is case-insensitive."""
    assert _parse_cli_token_count("TOTAL TOKENS: 999") == 999


def test_parse_cli_token_count_returns_zero_when_absent() -> None:
    """Returns 0 when no token info is found."""
    assert _parse_cli_token_count("Here is the review. No bugs found.") == 0


def test_parse_cli_token_count_returns_zero_on_empty() -> None:
    """Returns 0 for empty string."""
    assert _parse_cli_token_count("") == 0


def test_run_claude_cli_includes_token_count_from_output() -> None:
    """run_claude_cli populates token_count when the CLI outputs token info."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "[]"
    mock_result.stderr = "Total tokens: 500"

    with patch("subprocess.run", return_value=mock_result):
        result = run_claude_cli(Path("/tmp"), "review")

    assert result.token_count == 500


def test_run_codex_cli_passes_model(tmp_path: Path) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "[]"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        run_codex_cli(tmp_path, "prompt", model="gpt-4.1-mini")

    call_args = mock_run.call_args[0][0]
    assert "--model" in call_args
    assert "gpt-4.1-mini" in call_args
