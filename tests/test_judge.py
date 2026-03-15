# tests/test_judge.py
"""Tests for the judge module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic
import pytest
import yaml

from bugeval.judge import (
    _build_judge_prompt,
    _extract_judge_json,
    judge_case,
    judge_normalized_results,
    load_judge_prompt,
)
from bugeval.result_models import Comment, NormalizedResult


def _make_result(case_id: str = "case-001", tool: str = "greptile") -> NormalizedResult:
    return NormalizedResult(
        test_case_id=case_id,
        tool=tool,
        comments=[Comment(body="off by one", file="a.rs", line=10)],
    )


# --- load_judge_prompt ---


def test_load_judge_prompt_from_file(tmp_path: Path) -> None:
    p = tmp_path / "judge_prompt.md"
    p.write_text("Custom judge prompt")
    assert load_judge_prompt(p) == "Custom judge prompt"


def test_load_judge_prompt_falls_back_to_default(tmp_path: Path) -> None:
    prompt = load_judge_prompt(tmp_path / "nonexistent.md")
    assert "rubric" in prompt.lower() or "score" in prompt.lower()


# --- _extract_judge_json ---


def test_extract_judge_json_valid() -> None:
    text = '```json\n{"score": 2, "reasoning": "ok", "comment_judgments": []}\n```'
    data = _extract_judge_json(text)
    assert data is not None
    assert data["score"] == 2


def test_extract_judge_json_bare_object() -> None:
    text = '{"score": 3, "reasoning": "great", "comment_judgments": []}'
    data = _extract_judge_json(text)
    assert data is not None
    assert data["score"] == 3


def test_extract_judge_json_invalid_returns_none() -> None:
    assert _extract_judge_json("not json at all") is None


def test_extract_judge_json_multiline_fence() -> None:
    text = '```json\n{"score": 2,\n "reasoning": "ok",\n "comment_judgments": []}\n```'
    data = _extract_judge_json(text)
    assert data is not None
    assert data["score"] == 2


# --- judge_case ---


def _make_mock_response(score: int) -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock()]
    response.content[0].type = "text"
    response.content[0].text = json.dumps(
        {
            "score": score,
            "reasoning": f"Score is {score}",
            "comment_judgments": [{"id": 0, "classification": "TP", "relevance": "direct"}],
        }
    )
    return response


def test_judge_case_majority_vote(tmp_path: Path) -> None:
    """3 votes [2, 2, 3] → score 2."""
    from tests.conftest import make_case

    case = make_case()
    result = _make_result()

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _make_mock_response(2),
        _make_mock_response(2),
        _make_mock_response(3),
    ]

    with patch("bugeval.judge.Anthropic", return_value=mock_client):
        score = judge_case(case, result, system_prompt="judge this")

    assert score.score == 2
    assert score.votes == [2, 2, 3]
    assert len(score.comment_judgments) > 0
    # New assertions:
    assert score.noise.total_comments == 1  # _make_result() has 1 comment
    assert score.noise.snr == pytest.approx(1.0)  # all judgments are TP


def test_judge_case_reasoning_captured(tmp_path: Path) -> None:
    """judge_case reasoning comes from the LLM response, not just vote counts."""
    from tests.conftest import make_case

    case = make_case()
    result = _make_result()

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _make_mock_response(2),
        _make_mock_response(2),
        _make_mock_response(2),
    ]

    with patch("bugeval.judge.Anthropic", return_value=mock_client):
        score = judge_case(case, result, system_prompt="judge this")

    assert score.score == 2
    # Reasoning should contain the LLM's text, not just "Votes: ..."
    assert "Score is 2" in score.reasoning


def test_judge_case_dry_run(tmp_path: Path) -> None:
    from tests.conftest import make_case

    case = make_case()
    result = _make_result()

    with patch("bugeval.judge.Anthropic") as mock_anthropic_cls:
        score = judge_case(case, result, system_prompt="p", dry_run=True)

    mock_anthropic_cls.assert_not_called()
    assert score.score == 0
    assert score.reasoning == "dry-run"


def test_judge_case_all_parse_failures() -> None:
    """When all votes fail to parse, score=0 with failure count in reasoning."""
    from tests.conftest import make_case

    case = make_case()
    result = _make_result()

    bad_response = MagicMock()
    bad_response.content = [MagicMock()]
    bad_response.content[0].type = "text"
    bad_response.content[0].text = "I cannot score this."

    mock_client = MagicMock()
    mock_client.messages.create.return_value = bad_response

    with patch("bugeval.judge.Anthropic", return_value=mock_client):
        score = judge_case(case, result, system_prompt="judge this")

    assert score.votes == [0, 0, 0]
    assert score.score == 0
    assert "failed to parse" in score.reasoning


# --- CLI smoke test ---


def test_judge_help() -> None:
    from click.testing import CliRunner

    from bugeval.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["judge", "--help"])
    assert result.exit_code == 0
    assert "--run-dir" in result.output
    assert "--dry-run" in result.output


def test_judge_no_results(tmp_path: Path) -> None:
    """Exits cleanly when no normalized results found."""
    from click.testing import CliRunner

    from bugeval.cli import cli

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "judge",
            "--run-dir",
            str(tmp_path),
            "--cases-dir",
            str(cases_dir),
        ],
    )
    assert result.exit_code == 0
    assert "No normalized results" in result.output


# --- judge_normalized_results ---


def _write_case_yaml(cases_dir: Path, case_id: str = "case-001") -> None:
    """Write a minimal TestCase YAML into cases_dir."""
    from bugeval.io import save_case
    from tests.conftest import make_case

    case = make_case(id=case_id)
    save_case(case, cases_dir / f"{case_id}.yaml")


def _write_normalized_yaml(
    run_dir: Path, case_id: str = "case-001", tool: str = "greptile"
) -> Path:
    """Write a minimal NormalizedResult YAML into run_dir."""
    result = NormalizedResult(
        test_case_id=case_id,
        tool=tool,
        comments=[Comment(body="suspicious line", file="src/main.rs", line=42)],
    )
    out = run_dir / f"{case_id}-{tool}.yaml"
    out.write_text(yaml.safe_dump(result.model_dump(mode="json"), sort_keys=False))
    return out


def test_judge_normalized_results_dry_run(tmp_path: Path) -> None:
    """dry_run=True: scores all results without making LLM calls; score YAMLs are still written."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()

    _write_case_yaml(cases_dir)
    _write_normalized_yaml(tmp_path)

    with patch("bugeval.judge.Anthropic") as mock_anthropic_cls:
        count = judge_normalized_results(tmp_path, cases_dir, dry_run=True)

    mock_anthropic_cls.assert_not_called()
    assert count == 1
    scores_dir = tmp_path / "scores"
    # scores/ dir is created and score file is written; dry_run only skips the LLM API call
    assert scores_dir.exists()
    assert len(list(scores_dir.glob("*.yaml"))) == 1


# --- _build_judge_prompt ---


def test_build_judge_prompt_includes_suggested_fix() -> None:
    """suggested_fix on enriched Comment should appear in the judge prompt."""
    from tests.conftest import make_case

    case = make_case()
    result = NormalizedResult(
        test_case_id="case-001",
        tool="anthropic-api",
        comments=[
            Comment(body="off by one", file="a.rs", line=10, suggested_fix="Add bounds check")
        ],
    )
    prompt = _build_judge_prompt(case, result)
    assert "Add bounds check" in prompt
    assert "Fix:" in prompt


def test_build_judge_prompt_no_fix_no_section() -> None:
    """When no suggested_fix, the Fix: line should not appear."""
    from tests.conftest import make_case

    case = make_case()
    result = NormalizedResult(
        test_case_id="case-001",
        tool="greptile",
        comments=[Comment(body="off by one", file="a.rs", line=10)],
    )
    prompt = _build_judge_prompt(case, result)
    assert "Fix:" not in prompt


def test_judge_case_uses_temperature_zero() -> None:
    """judge_case must pass temperature=0 to the LLM for reproducibility."""
    from tests.conftest import make_case

    case = make_case()
    result = _make_result()

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response(2)

    with patch("bugeval.judge.Anthropic", return_value=mock_client):
        judge_case(case, result, system_prompt="judge this")

    call_kwargs = mock_client.messages.create.call_args_list[0][1]
    assert call_kwargs.get("temperature") == 0


def test_judge_normalized_results_returns_count(tmp_path: Path) -> None:
    """Returns the count of results that were scored."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()

    _write_case_yaml(cases_dir)
    _write_normalized_yaml(tmp_path)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response(2)

    with patch("bugeval.judge.Anthropic", return_value=mock_client):
        count = judge_normalized_results(tmp_path, cases_dir, dry_run=False)

    assert count == 1


def test_judge_case_retries_on_rate_limit() -> None:
    """judge_case retries when the API raises RateLimitError, then succeeds."""
    from tests.conftest import make_case

    case = make_case()
    result = _make_result()

    mock_client = MagicMock()
    rate_limit_error = anthropic.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={}),
        body={},
    )
    mock_client.messages.create.side_effect = [
        rate_limit_error,
        _make_mock_response(2),
        _make_mock_response(2),
        _make_mock_response(2),
    ]

    with patch("bugeval.judge.Anthropic", return_value=mock_client):
        with patch("bugeval.agent_api_runner.time.sleep"):
            score = judge_case(case, result, system_prompt="judge this")

    assert score.score == 2
    assert mock_client.messages.create.call_count == 4  # 1 failure + 3 successes
