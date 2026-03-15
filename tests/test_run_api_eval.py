"""Tests for run_api_eval orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from click.testing import CliRunner

from bugeval.cli import cli
from bugeval.pr_eval_models import CaseToolState, CaseToolStatus, RunState, ToolDef, ToolType
from bugeval.run_api_eval import process_case_tool_api
from tests.conftest import make_case


def _make_config_yaml(tmp_path: Path, tools: list[dict[str, Any]] | None = None) -> Path:
    tools_data = tools or [
        {
            "name": "greptile",
            "type": "api",
            "api_endpoint": "https://api.greptile.com/v2/review",
            "api_key_env": "GREPTILE_API_KEY",
            "cooldown_seconds": 0,
        }
    ]
    config_data = {
        "github": {"eval_org": "provable-eval"},
        "tools": tools_data,
        "repos": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    return config_path


def _make_case_yaml(cases_dir: Path, case_id: str = "case-001") -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    case_data = {
        "id": case_id,
        "repo": "provable-org/aleo-lang",
        "base_commit": "abc123",
        "head_commit": "def456",
        "fix_commit": "ghi789",
        "category": "logic",
        "difficulty": "medium",
        "severity": "high",
        "language": "rust",
        "pr_size": "small",
        "description": "A test case",
        "expected_findings": [{"file": "src/main.rs", "line": 42, "summary": "bug"}],
    }
    (cases_dir / f"{case_id}.yaml").write_text(yaml.dump(case_data))


# --- process_case_tool_api ---


def test_process_case_tool_api_dry_run(tmp_path: Path) -> None:
    case = make_case()
    tool = ToolDef(name="greptile", type=ToolType.api, cooldown_seconds=0)
    state = process_case_tool_api(
        case=case,
        tool=tool,
        patch_content="--- a\n+++ b",
        run_dir=tmp_path,
        context_level="diff-only",
        dry_run=True,
    )
    assert state.status == CaseToolStatus.done
    assert state.case_id == "case-001"
    assert state.tool == "greptile"


def test_process_case_tool_api_no_adapter_raises_failed(tmp_path: Path) -> None:
    case = make_case()
    # Give the tool valid api fields so validation passes and adapter lookup fires
    tool = ToolDef(
        name="unknown-tool",
        type=ToolType.api,
        api_endpoint="https://example.com",
        api_key_env="SOME_API_KEY",
        cooldown_seconds=0,
    )
    state = process_case_tool_api(
        case=case,
        tool=tool,
        patch_content="patch",
        run_dir=tmp_path,
        context_level="diff-only",
        dry_run=False,
    )
    assert state.status == CaseToolStatus.failed
    assert state.error is not None
    assert "No adapter" in state.error


def test_process_case_tool_api_saves_findings(tmp_path: Path) -> None:
    case = make_case()
    tool = ToolDef(
        name="greptile",
        type=ToolType.api,
        api_endpoint="https://api.greptile.com/v2/review",
        api_key_env="GREPTILE_API_KEY",
        cooldown_seconds=0,
    )

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value=[{"summary": "found", "file": "src/lib.rs", "line": 10}]
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        state = process_case_tool_api(
            case=case,
            tool=tool,
            patch_content="patch content",
            run_dir=tmp_path,
            context_level="diff-only",
            dry_run=False,
        )

    assert state.status == CaseToolStatus.done
    findings_file = tmp_path / "raw" / "case-001-greptile" / "findings.json"
    assert findings_file.exists()
    findings = json.loads(findings_file.read_text())
    assert len(findings) == 1
    assert findings[0]["body"] == "found"


# --- CLI via run-api-eval ---


def test_run_api_eval_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-api-eval", "--help"])
    assert result.exit_code == 0
    assert "--context-level" in result.output
    assert "--dry-run" in result.output
    assert "--cases-dir" in result.output


def test_run_api_eval_no_cases(tmp_path: Path) -> None:
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    # Don't create cases dir

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-api-eval",
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(tmp_path / "patches"),
            "--run-dir",
            str(tmp_path / "results"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "No cases found" in result.output


def test_run_api_eval_dry_run_with_case(tmp_path: Path) -> None:
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    _make_case_yaml(cases_dir)
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n@@ -1 +1 @@ foo")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-api-eval",
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(patches_dir),
            "--run-dir",
            str(tmp_path / "results"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "done" in result.output or "skip" in result.output


def test_run_api_eval_checkpoint_written(tmp_path: Path) -> None:
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    _make_case_yaml(cases_dir)
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b")
    run_dir = tmp_path / "results"

    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "run-api-eval",
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(patches_dir),
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ],
    )
    checkpoint = run_dir / "checkpoint.yaml"
    assert checkpoint.exists()
    rs = RunState.load(checkpoint)
    state = rs.get("case-001", "greptile")
    assert state.status == CaseToolStatus.done


def test_run_api_eval_missing_patch_marks_failed(tmp_path: Path) -> None:
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    _make_case_yaml(cases_dir)
    # No patch file created
    run_dir = tmp_path / "results"

    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "run-api-eval",
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(patches_dir),
            "--run-dir",
            str(run_dir),
        ],
    )
    checkpoint = run_dir / "checkpoint.yaml"
    assert checkpoint.exists()
    rs = RunState.load(checkpoint)
    state = rs.get("case-001", "greptile")
    assert state.status == CaseToolStatus.failed
    assert "patch not found" in (state.error or "")


def test_run_api_eval_checkpoint_resume_skips_done(tmp_path: Path) -> None:
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    _make_case_yaml(cases_dir)
    (patches_dir / "case-001.patch").write_text("patch")
    run_dir = tmp_path / "results"
    run_dir.mkdir(parents=True)

    # Pre-seed checkpoint with done state
    rs = RunState()
    rs.set(CaseToolState(case_id="case-001", tool="greptile", status=CaseToolStatus.done))
    rs.save(run_dir / "checkpoint.yaml")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-api-eval",
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(patches_dir),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0
    assert "skip" in result.output


def test_run_api_eval_context_level_passed(tmp_path: Path) -> None:
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    _make_case_yaml(cases_dir)
    (patches_dir / "case-001.patch").write_text("patch")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-api-eval",
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(patches_dir),
            "--run-dir",
            str(tmp_path / "results"),
            "--context-level",
            "diff+repo",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0


def test_process_case_api_writes_metadata_json(tmp_path: Path) -> None:
    case = make_case()
    tool = ToolDef(
        name="greptile",
        type=ToolType.api,
        api_endpoint="https://api.greptile.com/v2/review",
        api_key_env="GREPTILE_API_KEY",
        cooldown_seconds=0,
    )

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=[])
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        process_case_tool_api(
            case=case,
            tool=tool,
            patch_content="patch",
            run_dir=tmp_path,
            context_level="diff-only",
            dry_run=False,
        )

    metadata_file = tmp_path / "raw" / "case-001-greptile" / "metadata.json"
    assert metadata_file.exists()


def test_metadata_json_has_time_seconds(tmp_path: Path) -> None:
    case = make_case()
    tool = ToolDef(
        name="greptile",
        type=ToolType.api,
        api_endpoint="https://api.greptile.com/v2/review",
        api_key_env="GREPTILE_API_KEY",
        cooldown_seconds=0,
    )

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=[])
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        process_case_tool_api(
            case=case,
            tool=tool,
            patch_content="patch",
            run_dir=tmp_path,
            context_level="diff-only",
            dry_run=False,
        )

    metadata_file = tmp_path / "raw" / "case-001-greptile" / "metadata.json"
    meta = json.loads(metadata_file.read_text())
    assert "time_seconds" in meta
    assert isinstance(meta["time_seconds"], float)
    assert "cost_usd" not in meta


def test_run_api_eval_limit_slices_cases(tmp_path: Path) -> None:
    """--limit should process at most N cases per tool."""
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    for i in range(1, 4):
        _make_case_yaml(cases_dir, f"case-{i:03d}")
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    for i in range(1, 4):
        (patches_dir / f"case-{i:03d}.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-api-eval",
            "--config", str(config_path),
            "--cases-dir", str(cases_dir),
            "--patches-dir", str(patches_dir),
            "--run-dir", str(run_dir),
            "--limit", "2",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    checkpoint = run_dir / "checkpoint.yaml"
    rs = RunState.load(checkpoint)
    assert len(rs.states()) == 2


def test_run_api_eval_fail_after_aborts(tmp_path: Path) -> None:
    """--fail-after should abort the tool loop after N consecutive failures."""
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    for i in range(1, 5):
        _make_case_yaml(cases_dir, f"case-{i:03d}")
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    # No patches — all cases will fail with "patch not found"
    run_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-api-eval",
            "--config", str(config_path),
            "--cases-dir", str(cases_dir),
            "--patches-dir", str(patches_dir),
            "--run-dir", str(run_dir),
            "--fail-after", "2",
        ],
    )
    assert result.exit_code == 0
    checkpoint = run_dir / "checkpoint.yaml"
    rs = RunState.load(checkpoint)
    failed = [s for s in rs.states() if s.status == CaseToolStatus.failed]
    assert len(failed) == 2


def test_run_api_eval_unknown_tools_filter_exits(tmp_path: Path) -> None:
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run-api-eval",
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(patches_dir),
            "--run-dir",
            str(tmp_path / "results"),
            "--tools",
            "nonexistent",
        ],
    )
    assert result.exit_code != 0
