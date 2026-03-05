# tests/test_normalize.py
"""Tests for normalize module."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from bugeval.normalize import (
    discover_raw_dirs,
    normalize_agent_result,
    normalize_api_result,
    normalize_pr_result,
)
from bugeval.result_models import CommentType

# --- normalize_pr_result ---


def test_normalize_pr_result_inline_comment(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "case-001-coderabbit"
    raw_dir.mkdir(parents=True)
    comments = [
        {"source": "inline_comment", "body": "Off by one", "path": "a.rs", "line": 42},
    ]
    (raw_dir / "comments.json").write_text(json.dumps(comments))
    result = normalize_pr_result("case-001", "coderabbit", raw_dir)
    assert result.test_case_id == "case-001"
    assert result.tool == "coderabbit"
    assert len(result.comments) == 1
    assert result.comments[0].file == "a.rs"
    assert result.comments[0].line == 42
    assert result.comments[0].type == CommentType.inline


def test_normalize_pr_result_review_comment(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "case-001-coderabbit"
    raw_dir.mkdir(parents=True)
    comments = [
        {"source": "review", "body": "Looks risky overall", "state": "COMMENT"},
    ]
    (raw_dir / "comments.json").write_text(json.dumps(comments))
    result = normalize_pr_result("case-001", "coderabbit", raw_dir)
    assert result.comments[0].type == CommentType.pr_level
    assert result.comments[0].file == ""
    assert result.comments[0].line == 0


def test_normalize_pr_result_empty(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "case-001-coderabbit"
    raw_dir.mkdir(parents=True)
    (raw_dir / "comments.json").write_text("[]")
    result = normalize_pr_result("case-001", "coderabbit", raw_dir)
    assert result.comments == []


# --- normalize_api_result ---


def test_normalize_api_result(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "case-001-greptile"
    raw_dir.mkdir(parents=True)
    findings = [
        {"source": "greptile", "body": "Buffer overread", "path": "src/main.rs", "line": 10},
    ]
    (raw_dir / "findings.json").write_text(json.dumps(findings))
    result = normalize_api_result("case-001", "greptile", "diff-only", raw_dir)
    assert result.comments[0].body == "Buffer overread"
    assert result.comments[0].file == "src/main.rs"
    assert result.context_level == "diff-only"


def test_normalize_api_result_empty(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "case-001-greptile"
    raw_dir.mkdir(parents=True)
    (raw_dir / "findings.json").write_text("[]")
    result = normalize_api_result("case-001", "greptile", "diff-only", raw_dir)
    assert result.comments == []


# --- normalize_agent_result ---


def test_normalize_agent_result(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "case-001-anthropic-api"
    raw_dir.mkdir(parents=True)
    findings = [{"file": "b.rs", "line": 5, "summary": "logic error"}]
    metadata = {
        "token_count": 500,
        "cost_usd": 0.02,
        "wall_time_seconds": 3.1,
        "context_level": "diff+repo",
        "model": "claude-sonnet-4-6",
        "turns": 2,
        "error": None,
    }
    (raw_dir / "findings.json").write_text(json.dumps(findings))
    (raw_dir / "metadata.json").write_text(json.dumps(metadata))
    result = normalize_agent_result("case-001", "anthropic-api", raw_dir)
    assert result.comments[0].file == "b.rs"
    assert result.comments[0].body == "logic error"
    assert result.metadata.tokens == 500
    assert result.metadata.time_seconds == 3.1
    assert result.context_level == "diff+repo"


def test_normalize_agent_result_missing_metadata(tmp_path: Path) -> None:
    """If metadata.json is absent, use defaults."""
    raw_dir = tmp_path / "raw" / "case-001-claude-code-cli"
    raw_dir.mkdir(parents=True)
    (raw_dir / "findings.json").write_text("[]")
    result = normalize_agent_result("case-001", "claude-code-cli", raw_dir)
    assert result.metadata.tokens == 0


# --- discover_raw_dirs ---


def test_discover_raw_dirs(tmp_path: Path) -> None:
    (tmp_path / "raw" / "case-001-greptile").mkdir(parents=True)
    (tmp_path / "raw" / "case-002-coderabbit").mkdir(parents=True)
    dirs = discover_raw_dirs(tmp_path)
    names = {d.name for d in dirs}
    assert names == {"case-001-greptile", "case-002-coderabbit"}


def test_discover_raw_dirs_missing(tmp_path: Path) -> None:
    """Returns empty list if raw/ doesn't exist."""
    dirs = discover_raw_dirs(tmp_path)
    assert dirs == []


def test_normalize_api_result_reads_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "case-001-greptile"
    raw_dir.mkdir(parents=True)
    (raw_dir / "findings.json").write_text("[]")
    (raw_dir / "metadata.json").write_text(json.dumps({"time_seconds": 1.5, "cost_usd": 0.03}))
    result = normalize_api_result("case-001", "greptile", "diff-only", raw_dir)
    assert result.metadata.time_seconds == 1.5
    assert result.metadata.cost_usd == 0.03


def test_normalize_api_result_missing_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "case-001-greptile"
    raw_dir.mkdir(parents=True)
    (raw_dir / "findings.json").write_text("[]")
    # No metadata.json
    result = normalize_api_result("case-001", "greptile", "diff-only", raw_dir)
    assert result.metadata.time_seconds == 0.0
    assert result.metadata.cost_usd == 0.0


# --- _parse_raw_dir_name ---


def test_parse_raw_dir_name() -> None:
    from bugeval.normalize import _parse_raw_dir_name

    # Regex path: case ID ends in -NNN
    assert _parse_raw_dir_name("case-001-coderabbit") == ("case-001", "coderabbit")
    assert _parse_raw_dir_name("aleo-lang-042-anthropic-api") == ("aleo-lang-042", "anthropic-api")
    # Fallback path: no three-digit suffix
    assert _parse_raw_dir_name("foo-bar") == ("foo", "bar")


def test_parse_raw_dir_name_two_digit_case_id() -> None:
    """Case ID with 2 digits + hyphenated tool name."""
    from bugeval.normalize import _parse_raw_dir_name

    case_id, tool = _parse_raw_dir_name("aleo-lang-42-anthropic-api")
    assert case_id == "aleo-lang-42"
    assert tool == "anthropic-api"


def test_parse_raw_dir_name_four_digit_case_id() -> None:
    """Case ID with 4 digits + hyphenated tool name."""
    from bugeval.normalize import _parse_raw_dir_name

    case_id, tool = _parse_raw_dir_name("sentry-1234-greptile")
    assert case_id == "sentry-1234"
    assert tool == "greptile"


def test_parse_raw_dir_name_single_digit_case_id() -> None:
    """Case ID with 1 digit + hyphenated tool name."""
    from bugeval.normalize import _parse_raw_dir_name

    case_id, tool = _parse_raw_dir_name("cal-1-claude-code-cli")
    assert case_id == "cal-1"
    assert tool == "claude-code-cli"


def _make_config_yaml(tmp_path: Path) -> Path:
    config_data = {
        "github": {"eval_org": "provable-eval"},
        "tools": [{"name": "greptile", "type": "api", "cooldown_seconds": 0}],
        "repos": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    return config_path


def test_normalize_dry_run_no_files(tmp_path: Path) -> None:
    from bugeval.cli import cli

    config_path = _make_config_yaml(tmp_path)
    raw_dir = tmp_path / "raw" / "case-001-greptile"
    raw_dir.mkdir(parents=True)
    (raw_dir / "findings.json").write_text("[]")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["normalize", "--run-dir", str(tmp_path), "--config", str(config_path), "--dry-run"],
    )
    assert result.exit_code == 0
    # No output YAML should have been created
    assert not (tmp_path / "case-001-greptile.yaml").exists()


def test_normalize_dry_run_prints_summary(tmp_path: Path) -> None:
    from bugeval.cli import cli

    config_path = _make_config_yaml(tmp_path)
    raw_dir = tmp_path / "raw" / "case-001-greptile"
    raw_dir.mkdir(parents=True)
    (raw_dir / "findings.json").write_text("[]")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["normalize", "--run-dir", str(tmp_path), "--config", str(config_path), "--dry-run"],
    )
    assert result.exit_code == 0
    assert "dry-run" in result.output or "Would normalize" in result.output
