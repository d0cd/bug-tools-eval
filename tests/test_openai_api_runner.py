"""Tests for openai_api_runner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bugeval.openai_api_runner import OPENAI_TOOLS, run_openai_api


def _make_stop_response(content: str, total_tokens: int = 100) -> MagicMock:
    """Build a mock OpenAI chat completion response with finish_reason=stop."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.total_tokens = total_tokens

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    return mock_response


def test_run_openai_api_success(tmp_path: Path) -> None:
    findings_json = '[{"file": "src/main.rs", "line": 10, "summary": "bug"}]'
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_stop_response(findings_json)

    with patch("bugeval.openai_api_runner.OpenAI", return_value=mock_client):
        result = run_openai_api(tmp_path, "system prompt", "user prompt")

    assert result.error is None
    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "src/main.rs"
    assert result.model == "o4-mini"
    assert result.turns == 1
    assert result.token_count == 100


def test_run_openai_api_custom_model(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_stop_response("[]")

    with patch("bugeval.openai_api_runner.OpenAI", return_value=mock_client):
        result = run_openai_api(tmp_path, "system", "user", model="gpt-4.1-mini")

    assert result.model == "gpt-4.1-mini"


def test_run_openai_api_diff_only_no_tools(tmp_path: Path) -> None:
    """In diff-only mode, tools kwarg should not be passed."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_stop_response("[]")

    with patch("bugeval.openai_api_runner.OpenAI", return_value=mock_client):
        run_openai_api(tmp_path, "system", "user", context_level="diff-only")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert "tools" not in call_kwargs


def test_run_openai_api_tool_call_and_follow_up(tmp_path: Path) -> None:
    """One round of tool use followed by a stop response."""
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "list_directory"
    tool_call.function.arguments = json.dumps({"path": "."})

    mock_message_tool = MagicMock()
    mock_message_tool.content = None
    mock_message_tool.tool_calls = [tool_call]

    mock_choice_tool = MagicMock()
    mock_choice_tool.message = mock_message_tool
    mock_choice_tool.finish_reason = "tool_calls"

    first_response = MagicMock()
    first_response.choices = [mock_choice_tool]
    first_response.usage = MagicMock()
    first_response.usage.total_tokens = 50

    findings_json = '[{"file": "a.rs", "line": 1, "summary": "x"}]'
    second_response = _make_stop_response(findings_json, 60)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first_response, second_response]

    with patch("bugeval.openai_api_runner.OpenAI", return_value=mock_client):
        result = run_openai_api(tmp_path, "system", "user", context_level="diff+repo")

    assert result.turns == 2
    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "a.rs"
    assert result.token_count == 110


def test_run_openai_api_wall_time(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_stop_response("[]")

    with patch("bugeval.openai_api_runner.OpenAI", return_value=mock_client):
        result = run_openai_api(tmp_path, "system", "user")

    assert result.wall_time_seconds >= 0


def test_openai_tools_schema() -> None:
    """OPENAI_TOOLS should contain all AGENT_TOOLS with correct structure."""
    from bugeval.agent_api_runner import AGENT_TOOLS

    assert len(OPENAI_TOOLS) == len(AGENT_TOOLS)
    for tool in OPENAI_TOOLS:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


def test_run_openai_api_wraps_api_call_with_retry(tmp_path: Path) -> None:
    """run_openai_api delegates create() calls through _retry_call."""
    from bugeval import openai_api_runner

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_stop_response("[]")

    retry_invocations: list[bool] = []

    def spy_retry(fn, retryable, **kwargs):  # type: ignore[no-untyped-def]
        retry_invocations.append(True)
        return fn()

    with patch.object(openai_api_runner, "_retry_call", side_effect=spy_retry):
        with patch("bugeval.openai_api_runner.OpenAI", return_value=mock_client):
            run_openai_api(tmp_path, "system", "user")

    assert len(retry_invocations) >= 1


def test_run_openai_api_cost_usd(tmp_path: Path) -> None:
    """cost_usd computed from prompt_tokens + completion_tokens."""
    mock_message = MagicMock()
    mock_message.content = "[]"
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.total_tokens = 1_000_000
    mock_usage.prompt_tokens = 600_000
    mock_usage.completion_tokens = 400_000

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("bugeval.openai_api_runner.OpenAI", return_value=mock_client):
        result = run_openai_api(tmp_path, "system", "user", model="gpt-4.1-mini")

    # gpt-4.1-mini: 0.40/1M input, 1.60/1M output
    expected = (600_000 * 0.40 + 400_000 * 1.60) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
