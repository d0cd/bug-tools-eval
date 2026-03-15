"""Tests for google_api_runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bugeval.google_api_runner import run_google_api


def _make_text_response(text: str, total_tokens: int = 100) -> MagicMock:
    """Build a mock genai response with a single text part and no function calls."""
    mock_part = MagicMock()
    mock_part.function_call = None
    mock_part.text = text

    mock_content = MagicMock()
    mock_content.parts = [mock_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.total_token_count = total_tokens

    return mock_response


def test_run_google_api_success(tmp_path: Path) -> None:
    findings_json = '[{"file": "src/main.rs", "line": 10, "summary": "bug"}]'
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_text_response(findings_json)

    with patch("bugeval.google_api_runner.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = run_google_api(tmp_path, "system prompt", "user prompt")

    assert result.error is None
    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "src/main.rs"
    assert result.model == "gemini-2.5-flash"
    assert result.turns == 1
    assert result.token_count == 100


def test_run_google_api_no_candidates(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.candidates = []
    mock_client.models.generate_content.return_value = mock_response

    with patch("bugeval.google_api_runner.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = run_google_api(tmp_path, "system", "user")

    assert result.findings == []
    assert result.turns == 1


def test_run_google_api_custom_model(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_text_response("[]")

    with patch("bugeval.google_api_runner.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = run_google_api(tmp_path, "system", "user", model="gemini-2.5-flash-lite")

    assert result.model == "gemini-2.5-flash-lite"


def test_run_google_api_diff_only_no_tools(tmp_path: Path) -> None:
    """In diff-only mode the config should have no tools."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_text_response("[]")

    with patch("bugeval.google_api_runner.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        run_google_api(tmp_path, "system", "user", context_level="diff-only")

    # GenerateContentConfig should have been called with empty tools
    cfg_call = mock_genai.types.GenerateContentConfig.call_args
    assert cfg_call is not None
    assert cfg_call.kwargs.get("tools") == []


def test_run_google_api_tool_call_and_follow_up(tmp_path: Path) -> None:
    """One round of function calling followed by a text response."""
    # First response: function call
    fn_part = MagicMock()
    fn_call = MagicMock()
    fn_call.name = "list_directory"
    fn_call.args = {"path": "."}
    fn_part.function_call = fn_call
    fn_part.text = None

    first_content = MagicMock()
    first_content.parts = [fn_part]
    first_candidate = MagicMock()
    first_candidate.content = first_content
    first_response = MagicMock()
    first_response.candidates = [first_candidate]
    first_response.usage_metadata = MagicMock()
    first_response.usage_metadata.total_token_count = 50

    # Second response: text with findings
    second_response = _make_text_response('[{"file": "a.rs", "line": 1, "summary": "x"}]', 60)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [first_response, second_response]

    with patch("bugeval.google_api_runner.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = run_google_api(tmp_path, "system", "user", context_level="diff+repo")

    assert result.turns == 2
    assert len(result.findings) == 1
    assert result.findings[0]["file"] == "a.rs"
    assert result.token_count == 110


def test_run_google_api_wall_time(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_text_response("[]")

    with patch("bugeval.google_api_runner.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = run_google_api(tmp_path, "system", "user")

    assert result.wall_time_seconds >= 0


def test_run_google_api_wraps_api_call_with_retry(tmp_path: Path) -> None:
    """run_google_api delegates generate_content calls through _retry_call."""
    from bugeval import google_api_runner

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_text_response("[]")

    retry_invocations: list[bool] = []

    def spy_retry(fn, retryable, **kwargs):  # type: ignore[no-untyped-def]
        retry_invocations.append(True)
        return fn()

    with patch.object(google_api_runner, "_retry_call", side_effect=spy_retry):
        with patch("bugeval.google_api_runner.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            run_google_api(tmp_path, "system", "user")

    assert len(retry_invocations) >= 1


def test_run_google_api_cost_usd(tmp_path: Path) -> None:
    """cost_usd computed from prompt_token_count + candidates_token_count."""
    mock_part = MagicMock()
    mock_part.function_call = None
    mock_part.text = "[]"

    mock_content = MagicMock()
    mock_content.parts = [mock_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content

    mock_usage = MagicMock()
    mock_usage.total_token_count = 1_000_000
    mock_usage.prompt_token_count = 600_000
    mock_usage.candidates_token_count = 400_000

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = mock_usage

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("bugeval.google_api_runner.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = run_google_api(tmp_path, "system", "user", model="gemini-2.5-flash")

    # gemini-2.5-flash: 0.15/1M input, 0.60/1M output
    expected = (600_000 * 0.15 + 400_000 * 0.60) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
