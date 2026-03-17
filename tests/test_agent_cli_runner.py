"""Tests for agent_cli_runner."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from bugeval.agent_cli_runner import (
    _parse_cli_findings,
    _parse_cli_token_count,
    _parse_stream_json_output,
    run_claude_cli,
    run_claude_cli_docker,
    run_codex_cli,
    run_gemini_cli,
)


def _make_stream_jsonl(
    result_text: str,
    turns: int = 1,
    cost: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> str:
    """Build a minimal valid stream-json JSONL string for use in tests."""
    lines = [
        json.dumps(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": result_text}]}}
        ),
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "num_turns": turns,
                "result": result_text,
                "total_cost_usd": cost,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                },
            }
        ),
    ]
    return "\n".join(lines)


def test_run_claude_cli_success(tmp_path: Path) -> None:
    findings_json = '[{"file": "src/main.rs", "line": 10, "summary": "bug"}]'
    result_text = f"Some output\n```json\n{findings_json}\n```\n"
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _make_stream_jsonl(result_text)
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
    findings_json = '[{"file": "a.rs", "line": 1, "summary": "bug"}]'
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _make_stream_jsonl(f"```json\n{findings_json}\n```")
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
    """run_claude_cli populates token_count from the stream-json result event."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _make_stream_jsonl(
        "Some reasoning text", turns=3, cost=0.05, input_tokens=300, output_tokens=200
    )
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_claude_cli(Path("/tmp"), "review")

    assert result.token_count == 500
    assert result.turns == 3
    assert result.cost_usd == 0.05
    assert result.response_text == "Some reasoning text"


def test_run_claude_cli_extracts_envelope_metadata(tmp_path: Path) -> None:
    """run_claude_cli extracts turns, cost, tokens, response_text from stream-json result event."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _make_stream_jsonl(
        "Here is my analysis of the patch.",
        turns=2,
        cost=0.012,
        input_tokens=800,
        output_tokens=150,
    )
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_claude_cli(tmp_path, "prompt")

    assert result.turns == 2
    assert result.cost_usd == 0.012
    assert result.token_count == 950
    assert result.response_text == "Here is my analysis of the patch."


def test_run_claude_cli_missing_result_event_defaults_to_zero(tmp_path: Path) -> None:
    """When stdout has no type=result event, cost/turns/tokens all default to 0."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}}
    )
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_claude_cli(tmp_path, "prompt")

    assert result.turns == 0
    assert result.cost_usd == 0.0
    assert result.token_count == 0
    assert result.response_text == ""


def test_run_claude_cli_malformed_stdout_gives_empty_findings(tmp_path: Path) -> None:
    """When stdout is not valid stream-json, no findings or metadata are extracted."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "not json at all"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_claude_cli(tmp_path, "prompt")

    assert result.error is None
    assert result.findings == []
    assert result.response_text == ""
    assert result.token_count == 0


# ---------------------------------------------------------------------------
# _parse_stream_json_output
# ---------------------------------------------------------------------------


def test_parse_stream_json_output_extracts_result() -> None:
    """Parses result text, turns, cost, and tokens from type=result event."""
    stdout = _make_stream_jsonl(
        "hello world", turns=2, cost=0.05, input_tokens=100, output_tokens=50
    )
    conv, result_text, tokens, cost, turns = _parse_stream_json_output(stdout)
    assert result_text == "hello world"
    assert turns == 2
    assert cost == 0.05
    assert tokens == 150


def test_parse_stream_json_output_includes_cache_tokens() -> None:
    """Token count includes cache_creation + cache_read tokens."""
    stdout = _make_stream_jsonl(
        "x", input_tokens=10, output_tokens=5, cache_creation=200, cache_read=50
    )
    _, _, tokens, _, _ = _parse_stream_json_output(stdout)
    assert tokens == 265


def test_parse_stream_json_output_builds_conversation() -> None:
    """type=assistant events are included in conversation."""
    stdout = _make_stream_jsonl("my analysis")
    conv, _, _, _, _ = _parse_stream_json_output(stdout)
    assert len(conv) == 1
    assert conv[0]["role"] == "assistant"
    assert conv[0]["content"][0]["text"] == "my analysis"


def test_parse_stream_json_output_includes_tool_use_in_conversation() -> None:
    """tool_use content blocks in assistant messages appear in conversation."""
    lines = [
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read",
             "input": {"file_path": "src/main.rs"}}
        ]}}),
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "fn main() {}"}
        ]}}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "```json\n[]\n```"}
        ]}}),
        json.dumps({"type": "result", "num_turns": 2, "result": "```json\n[]\n```",
                    "total_cost_usd": 0.01, "usage": {"input_tokens": 0, "output_tokens": 0,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}),
    ]
    stdout = "\n".join(lines)
    conv, _, _, _, turns = _parse_stream_json_output(stdout)
    assert turns == 2
    assert len(conv) == 3
    assert conv[0]["role"] == "assistant"
    assert conv[0]["content"][0]["type"] == "tool_use"
    assert conv[1]["role"] == "user"
    assert conv[1]["content"][0]["type"] == "tool_result"


def test_parse_stream_json_output_ignores_noise_events() -> None:
    """system and stream_event types are ignored."""
    lines = [
        json.dumps({"type": "system", "subtype": "init", "data": "ignored"}),
        json.dumps({"type": "stream_event", "event": {"type": "content_block_delta"}}),
        json.dumps({"type": "rate_limit_event", "rate_limit_info": {}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}),
        json.dumps({"type": "result", "num_turns": 1, "result": "hi", "total_cost_usd": 0.0,
                    "usage": {"input_tokens": 0, "output_tokens": 0,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}),
    ]
    conv, result_text, _, _, _ = _parse_stream_json_output("\n".join(lines))
    assert len(conv) == 1
    assert result_text == "hi"


def test_run_claude_cli_populates_conversation(tmp_path: Path) -> None:
    """run_claude_cli populates AgentResult.conversation from stream-json assistant events."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _make_stream_jsonl("analysis text")
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = run_claude_cli(tmp_path, "prompt")

    assert len(result.conversation) == 1
    assert result.conversation[0]["role"] == "assistant"


def test_run_claude_cli_uses_stream_json_format(tmp_path: Path) -> None:
    """run_claude_cli passes --output-format stream-json --verbose to the subprocess."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _make_stream_jsonl("")
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        run_claude_cli(tmp_path, "prompt")

    call_args = mock_run.call_args[0][0]
    assert "stream-json" in call_args
    assert "--verbose" in call_args
    # --output-format value should be stream-json, not bare json
    fmt_idx = call_args.index("--output-format")
    assert call_args[fmt_idx + 1] == "stream-json"


def test_run_claude_cli_disables_user_settings(tmp_path: Path) -> None:
    """run_claude_cli passes --setting-sources project,local to skip user hooks."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _make_stream_jsonl("")
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        run_claude_cli(tmp_path, "prompt")

    call_args = mock_run.call_args[0][0]
    assert "--setting-sources" in call_args
    src_idx = call_args.index("--setting-sources")
    assert call_args[src_idx + 1] == "project,local"


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
