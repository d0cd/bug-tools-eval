"""Tests for agent_api_runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bugeval.agent_api_runner import (
    _parse_api_findings,
    execute_tool,
    run_agent_api,
)

# ---------------------------------------------------------------------------
# execute_tool tests
# ---------------------------------------------------------------------------


def test_execute_tool_read_file(tmp_path: Path) -> None:
    test_file = tmp_path / "src" / "main.rs"
    test_file.parent.mkdir()
    test_file.write_text("fn main() {}")
    result = execute_tool("read_file", {"path": "src/main.rs"}, tmp_path)
    assert "fn main()" in result


def test_execute_tool_list_directory(tmp_path: Path) -> None:
    (tmp_path / "alpha.rs").write_text("")
    (tmp_path / "beta.rs").write_text("")
    result = execute_tool("list_directory", {"path": "."}, tmp_path)
    assert "alpha.rs" in result
    assert "beta.rs" in result


def test_execute_tool_search_code(tmp_path: Path) -> None:
    test_file = tmp_path / "foo.rs"
    test_file.write_text("let x = panic!();\n")
    result = execute_tool("search_code", {"pattern": "panic", "path": "."}, tmp_path)
    assert "panic" in result


def test_execute_tool_path_traversal_blocked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Path traversal"):
        execute_tool("read_file", {"path": "../../etc/passwd"}, tmp_path)


def test_execute_tool_unknown_tool(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        execute_tool("rm_rf", {"path": "."}, tmp_path)


def test_execute_tool_search_code_invalid_regex(tmp_path: Path) -> None:
    """Invalid regex pattern returns an error string, not subprocess error."""
    result = execute_tool("search_code", {"pattern": "[unclosed", "path": "."}, tmp_path)
    assert "Invalid" in result


def test_execute_tool_search_code_timeout(tmp_path: Path) -> None:
    """Subprocess timeout from grep returns a readable error string."""
    import subprocess as sp

    with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="grep", timeout=30)):
        result = execute_tool("search_code", {"pattern": "x", "path": "."}, tmp_path)
    assert "timed out" in result.lower() or "timeout" in result.lower()


def test_execute_tool_list_directory_symlink_blocked(tmp_path: Path) -> None:
    """list_directory rejects symlinks even if they resolve within repo_dir."""
    target = tmp_path / "real_dir"
    target.mkdir()
    link = tmp_path / "link_dir"
    link.symlink_to(target)
    result = execute_tool("list_directory", {"path": "link_dir"}, tmp_path)
    assert "symlink" in result.lower() or "Error" in result


# ---------------------------------------------------------------------------
# run_agent_api tests
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    block.model_dump.return_value = {"type": "text", "text": text}
    return block


def _make_tool_use_block(name: str, input_data: dict, block_id: str = "tu_1") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data
    block.id = block_id
    block.model_dump.return_value = {"type": "tool_use", "name": name, "input": input_data}
    return block


def _make_response(
    content: list, stop_reason: str, input_tokens: int = 100, output_tokens: int = 50
) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.stop_reason = stop_reason
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


def test_run_agent_api_single_turn(tmp_path: Path) -> None:
    findings_text = '[{"file": "a.rs", "line": 1, "summary": "bug"}]'
    response = _make_response(
        content=[_make_text_block(findings_text)],
        stop_reason="end_turn",
    )

    mock_client = MagicMock()
    mock_client.messages.create.return_value = response

    with patch("bugeval.agent_api_runner.Anthropic", return_value=mock_client):
        result = run_agent_api(tmp_path, "system", "user prompt")

    assert result.turns == 1
    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "a.rs"
    assert result.token_count == 150
    assert result.error is None


def test_run_agent_api_multi_turn(tmp_path: Path) -> None:
    # First response: tool_use (list_directory)
    tool_block = _make_tool_use_block("list_directory", {"path": "."}, "tu_1")
    first_response = _make_response(content=[tool_block], stop_reason="tool_use")

    # Second response: end_turn with findings
    findings_text = '[{"file": "b.rs", "line": 5, "summary": "x"}]'
    second_response = _make_response(
        content=[_make_text_block(findings_text)],
        stop_reason="end_turn",
    )

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [first_response, second_response]

    with patch("bugeval.agent_api_runner.Anthropic", return_value=mock_client):
        result = run_agent_api(tmp_path, "system", "user prompt")

    assert result.turns == 2
    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "b.rs"


def test_run_agent_api_max_turns_cutoff(tmp_path: Path) -> None:
    # Always return tool_use → loop should stop at max_turns
    tool_block = _make_tool_use_block("list_directory", {"path": "."})
    tool_response = _make_response(content=[tool_block], stop_reason="tool_use")

    mock_client = MagicMock()
    mock_client.messages.create.return_value = tool_response

    with patch("bugeval.agent_api_runner.Anthropic", return_value=mock_client):
        result = run_agent_api(tmp_path, "system", "user prompt", max_turns=3)

    assert result.turns == 3
    assert mock_client.messages.create.call_count == 3


def test_parse_api_findings_with_json() -> None:
    text = 'Here are findings:\n```json\n[{"file": "x.rs", "line": 2, "summary": "y"}]\n```'
    findings = _parse_api_findings(text)
    assert len(findings) == 1
    assert findings[0]["file"] == "x.rs"


def test_parse_api_findings_empty() -> None:
    assert _parse_api_findings("No bugs found.") == []


def test_parse_api_findings_nested_arrays() -> None:
    """Findings with nested arrays inside objects are parsed correctly."""
    text = (
        "```json\n"
        '[{"file": "a.rs", "line": 1, "summary": "bug", "tags": ["error", "critical"]}]\n'
        "```"
    )
    result = _parse_api_findings(text)
    assert len(result) == 1
    assert result[0]["file"] == "a.rs"


def test_parse_api_findings_no_fence() -> None:
    """Raw JSON array without fence markers is parsed."""
    text = (
        "Here are the findings:\n"
        '[{"file": "b.rs", "line": 5, "summary": "issue"}]\n'
        "End of analysis."
    )
    result = _parse_api_findings(text)
    assert len(result) == 1
    assert result[0]["file"] == "b.rs"


def test_parse_api_findings_empty_array() -> None:
    """Empty JSON array returns empty list."""
    text = "No bugs found.\n```json\n[]\n```"
    result = _parse_api_findings(text)
    assert result == []


def test_run_agent_api_diff_only_no_file_tools(tmp_path: Path) -> None:
    """In diff-only mode, no file tools are provided to the model."""
    mock_response = MagicMock()
    mock_response.content = [_make_text_block("[]")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("bugeval.agent_api_runner.Anthropic", return_value=mock_client):
        run_agent_api(tmp_path, "system", "user", max_turns=1, context_level="diff-only")

    call_kwargs = mock_client.messages.create.call_args[1]
    tool_names = [t["name"] for t in call_kwargs.get("tools", [])]
    assert "read_file" not in tool_names
    assert "list_directory" not in tool_names
    assert "search_code" not in tool_names


def test_run_agent_api_diff_repo_has_file_tools(tmp_path: Path) -> None:
    """In diff+repo mode, file tools are provided to the model."""
    mock_response = MagicMock()
    mock_response.content = [_make_text_block("[]")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("bugeval.agent_api_runner.Anthropic", return_value=mock_client):
        run_agent_api(tmp_path, "system", "user", max_turns=1, context_level="diff+repo")

    call_kwargs = mock_client.messages.create.call_args[1]
    tool_names = [t["name"] for t in call_kwargs.get("tools", [])]
    assert "read_file" in tool_names


def test_run_agent_api_parses_findings_on_max_tokens(tmp_path: Path) -> None:
    """When stop_reason is max_tokens but output contains findings, they are still parsed."""
    findings_text = '[{"file": "c.rs", "line": 3, "summary": "truncated but parseable"}]'
    response = _make_response(
        content=[_make_text_block(findings_text)],
        stop_reason="max_tokens",
    )

    mock_client = MagicMock()
    mock_client.messages.create.return_value = response

    with patch("bugeval.agent_api_runner.Anthropic", return_value=mock_client):
        result = run_agent_api(tmp_path, "system", "user prompt")

    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "c.rs"


# ---------------------------------------------------------------------------
# New tools: git_blame + read_file_range
# ---------------------------------------------------------------------------


def test_agent_tools_includes_git_blame() -> None:
    """AGENT_TOOLS must include the git_blame tool definition."""
    from bugeval.agent_api_runner import AGENT_TOOLS

    names = [t["name"] for t in AGENT_TOOLS]
    assert "git_blame" in names


def test_agent_tools_includes_read_file_range() -> None:
    """AGENT_TOOLS must include the read_file_range tool definition."""
    from bugeval.agent_api_runner import AGENT_TOOLS

    names = [t["name"] for t in AGENT_TOOLS]
    assert "read_file_range" in names


def test_execute_tool_read_file_range_returns_lines(tmp_path: Path) -> None:
    """read_file_range returns only the requested line range (1-indexed, inclusive)."""
    f = tmp_path / "sample.rs"
    f.write_text("line1\nline2\nline3\nline4\nline5\n")
    result = execute_tool(
        "read_file_range", {"path": "sample.rs", "start_line": 2, "end_line": 4}, tmp_path
    )
    assert "line2" in result
    assert "line3" in result
    assert "line4" in result
    assert "line1" not in result
    assert "line5" not in result


def test_execute_tool_read_file_range_path_traversal_blocked(tmp_path: Path) -> None:
    """read_file_range rejects path traversal."""
    with pytest.raises(ValueError, match="Path traversal"):
        execute_tool(
            "read_file_range",
            {"path": "../../etc/passwd", "start_line": 1, "end_line": 5},
            tmp_path,
        )


def test_execute_tool_git_blame_path_traversal_blocked(tmp_path: Path) -> None:
    """git_blame rejects path traversal."""
    with pytest.raises(ValueError, match="Path traversal"):
        execute_tool(
            "git_blame",
            {"path": "../../etc/passwd", "start_line": 1, "end_line": 5},
            tmp_path,
        )


def test_execute_tool_git_blame_returns_output(tmp_path: Path) -> None:
    """git_blame runs git blame -L and returns stdout (mocked subprocess)."""
    import subprocess as sp

    blame_output = "abc1234 (Author 2024-01-01  1) fn foo() {}"
    with patch(
        "subprocess.run",
        return_value=sp.CompletedProcess(args=[], returncode=0, stdout=blame_output, stderr=""),
    ):
        result = execute_tool(
            "git_blame", {"path": "src/main.rs", "start_line": 1, "end_line": 1}, tmp_path
        )
    assert "abc1234" in result or blame_output[:10] in result


def test_execute_tool_git_blame_timeout(tmp_path: Path) -> None:
    """git_blame timeout returns a readable error string."""
    import subprocess as sp

    with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="git", timeout=30)):
        result = execute_tool(
            "git_blame", {"path": "src/main.rs", "start_line": 1, "end_line": 5}, tmp_path
        )
    assert "timed out" in result.lower() or "timeout" in result.lower()


# ---------------------------------------------------------------------------
# Retry logic: _retry_call unit tests
# ---------------------------------------------------------------------------


def test_retry_call_returns_on_first_success() -> None:
    """_retry_call returns immediately when fn succeeds on first try."""
    from bugeval.agent_api_runner import _retry_call

    result = _retry_call(lambda: 42, retryable=(ValueError,))
    assert result == 42


def test_retry_call_retries_on_retryable_error() -> None:
    """_retry_call retries when fn raises a retryable exception."""
    from bugeval.agent_api_runner import _retry_call

    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("transient")
        return "ok"

    with patch("bugeval.agent_api_runner.time") as mock_time:
        mock_time.monotonic.side_effect = [0.0, 1.0, 2.0, 3.0]
        result = _retry_call(fn, retryable=(ValueError,), max_retries=3, base_delay=0.0)

    assert result == "ok"
    assert len(calls) == 2


def test_retry_call_raises_after_max_retries() -> None:
    """_retry_call re-raises after max_retries attempts."""
    from bugeval.agent_api_runner import _retry_call

    def fn() -> None:
        raise ValueError("always fails")

    with patch("bugeval.agent_api_runner.time"):
        with pytest.raises(ValueError, match="always fails"):
            _retry_call(fn, retryable=(ValueError,), max_retries=2, base_delay=0.0)


def test_retry_call_does_not_retry_non_retryable() -> None:
    """Non-retryable exceptions propagate immediately without retry."""
    from bugeval.agent_api_runner import _retry_call

    calls: list[int] = []

    def fn() -> None:
        calls.append(1)
        raise TypeError("fatal")

    with pytest.raises(TypeError, match="fatal"):
        _retry_call(fn, retryable=(ValueError,), max_retries=3)

    assert len(calls) == 1


def test_run_agent_api_wraps_api_call_with_retry(tmp_path: Path) -> None:
    """run_agent_api delegates each create() call through _retry_call."""
    from bugeval import agent_api_runner

    success_response = _make_response(
        content=[_make_text_block("[]")],
        stop_reason="end_turn",
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = success_response

    retry_invocations: list[bool] = []
    real_retry = agent_api_runner._retry_call

    def spy_retry(fn, retryable, **kwargs):  # type: ignore[no-untyped-def]
        retry_invocations.append(True)
        return real_retry(fn, retryable, max_retries=0)

    with patch.object(agent_api_runner, "_retry_call", side_effect=spy_retry):
        with patch("bugeval.agent_api_runner.Anthropic", return_value=mock_client):
            run_agent_api(tmp_path, "system", "user")

    assert len(retry_invocations) >= 1


def test_run_agent_api_cost_usd(tmp_path: Path) -> None:
    """cost_usd should be computed from token usage when model is in default pricing."""
    response = _make_response(
        content=[_make_text_block("[]")],
        stop_reason="end_turn",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = response

    with patch("bugeval.agent_api_runner.Anthropic", return_value=mock_client):
        result = run_agent_api(tmp_path, "system", "user", model="claude-sonnet-4-6")

    # claude-sonnet-4-6: 3.0/1M input + 15.0/1M output = 18.0 per 1M each
    assert result.cost_usd == pytest.approx(18.0)
