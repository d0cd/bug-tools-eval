# tests/test_judge.py
"""Tests for the judge module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from bugeval.judge import _extract_judge_json, judge_case, load_judge_prompt
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

    config_data = {"github": {"eval_org": "x"}, "tools": [], "repos": {}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
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
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert "No normalized results" in result.output
