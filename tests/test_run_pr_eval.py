"""Tests for run_pr_eval CLI and helpers."""

from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from bugeval.models import Category, Difficulty, PRSize, Severity, TestCase
from bugeval.pr_eval_models import CaseToolStatus, RunState
from bugeval.run_pr_eval import load_cases, make_run_id, process_case_tool, run_pr_eval


def _make_config_file(tmp_path: Path, eval_org: str = "provable-eval") -> Path:
    data = {
        "github": {"eval_org": eval_org},
        "tools": [
            {
                "name": "coderabbit",
                "type": "pr",
                "github_app": "coderabbit-ai",
                "cooldown_seconds": 0,
            },
        ],
        "repos": {"aleo-lang": "provable-org/aleo-lang"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(data))
    return config_path


def _make_case_file(cases_dir: Path, case_id: str = "case-001") -> Path:
    cases_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": case_id,
        "repo": "provable-org/aleo-lang",
        "base_commit": "abc123",
        "head_commit": "def456",
        "fix_commit": "def456",
        "category": "logic",
        "difficulty": "medium",
        "severity": "high",
        "language": "rust",
        "pr_size": "small",
        "description": "A test bug case",
        "expected_findings": [],
    }
    path = cases_dir / f"{case_id}.yaml"
    path.write_text(yaml.dump(data))
    return path


def _make_case(case_id: str = "case-001") -> TestCase:
    return TestCase(
        id=case_id,
        repo="provable-org/aleo-lang",
        base_commit="abc123",
        head_commit="def456",
        fix_commit="def456",
        category=Category.logic,
        difficulty=Difficulty.medium,
        severity=Severity.high,
        language="rust",
        pr_size=PRSize.small,
        description="A test bug case",
        expected_findings=[],
    )


def test_make_run_id_format() -> None:
    run_id = make_run_id()
    assert run_id.startswith("run-")
    parts = run_id[4:].split("-")
    assert len(parts) == 3  # YYYY-MM-DD


def test_load_cases_missing_dir(tmp_path: Path) -> None:
    cases = load_cases(tmp_path / "nonexistent")
    assert cases == []


def test_load_cases_empty_dir(tmp_path: Path) -> None:
    empty_dir = tmp_path / "cases"
    empty_dir.mkdir()
    cases = load_cases(empty_dir)
    assert cases == []


def test_run_pr_eval_help() -> None:
    runner = CliRunner()
    result = runner.invoke(run_pr_eval, ["--help"])
    assert result.exit_code == 0
    assert "--cases-dir" in result.output
    assert "--run-dir" in result.output
    assert "--dry-run" in result.output


def test_run_pr_eval_no_cases(tmp_path: Path) -> None:
    config_path = _make_config_file(tmp_path)
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    run_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        run_pr_eval,
        [
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(tmp_path / "patches"),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0
    assert "No cases" in result.output


def test_checkpoint_created_on_dry_run(tmp_path: Path) -> None:
    config_path = _make_config_file(tmp_path)
    cases_dir = tmp_path / "cases"
    _make_case_file(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        run_pr_eval,
        [
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
    assert result.exit_code == 0
    checkpoint = run_dir / "checkpoint.yaml"
    assert checkpoint.exists()
    # Verify checkpoint records the case-tool pair as done
    state = RunState.load(checkpoint)
    pair = state.get("case-001", "coderabbit")
    assert pair.status == CaseToolStatus.done


def test_process_case_tool_dry_run_no_gh_calls(tmp_path: Path) -> None:
    from bugeval.pr_eval_models import load_eval_config

    config_path = _make_config_file(tmp_path)
    config = load_eval_config(config_path)
    tool = config.pr_tools[0]

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    case = _make_case()

    # In dry-run mode, open_pr returns 0 and no gh calls are made
    with patch("bugeval.pr_lifecycle.run_gh") as mock_gh:
        state = process_case_tool(case, tool, config, patches_dir, run_dir, None, dry_run=True)

    mock_gh.assert_not_called()
    assert state.status == CaseToolStatus.done


def test_process_case_tool_missing_patch(tmp_path: Path) -> None:
    from bugeval.pr_eval_models import load_eval_config

    config_path = _make_config_file(tmp_path)
    config = load_eval_config(config_path)
    tool = config.pr_tools[0]

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()  # no patch file
    run_dir = tmp_path / "results"

    case = _make_case()
    state = process_case_tool(case, tool, config, patches_dir, run_dir, None, dry_run=True)

    assert state.status == CaseToolStatus.failed
    assert "patch not found" in (state.error or "")


def test_process_case_tool_with_repo_dir(tmp_path: Path) -> None:
    """When repo_dir is provided, apply_patch_to_branch is called."""
    from bugeval.pr_eval_models import load_eval_config

    config_path = _make_config_file(tmp_path)
    config = load_eval_config(config_path)
    tool = config.pr_tools[0]

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    run_dir = tmp_path / "results"

    case = _make_case()

    with patch("bugeval.run_pr_eval.apply_patch_to_branch"):
        state = process_case_tool(case, tool, config, patches_dir, run_dir, repo_dir, dry_run=True)

    # dry_run=True skips the actual git call, but code path is exercised
    assert state.status == CaseToolStatus.done


def test_process_case_tool_with_repo_dir_applies_patch(tmp_path: Path) -> None:
    """With repo_dir and dry_run=False, apply_patch_to_branch is invoked."""
    from bugeval.pr_eval_models import load_eval_config

    config_path = _make_config_file(tmp_path)
    config = load_eval_config(config_path)
    tool = config.pr_tools[0]

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    run_dir = tmp_path / "results"

    case = _make_case()

    with (
        patch("bugeval.run_pr_eval.apply_patch_to_branch") as mock_apply,
        patch("bugeval.run_pr_eval.open_pr", return_value=7),
        patch("bugeval.run_pr_eval.poll_for_review", return_value=False),
        patch("bugeval.run_pr_eval.scrape_review_comments", return_value=[]),
        patch("bugeval.run_pr_eval.close_pr_delete_branch"),
    ):
        state = process_case_tool(case, tool, config, patches_dir, run_dir, repo_dir, dry_run=False)

    mock_apply.assert_called_once()
    assert state.status == CaseToolStatus.done


def test_scrape_always_called_even_when_poll_times_out(tmp_path: Path) -> None:
    """scrape_review_comments must be called even when poll_for_review returns False.

    Tools that post findings only to issue_comments (not formal reviews) would
    be silently missed if scraping is gated on poll_for_review returning True.
    """
    from bugeval.pr_eval_models import load_eval_config

    config_path = _make_config_file(tmp_path)
    config = load_eval_config(config_path)
    tool = config.pr_tools[0]

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    run_dir = tmp_path / "results"

    case = _make_case()

    with (
        patch("bugeval.run_pr_eval.apply_patch_to_branch"),
        patch("bugeval.run_pr_eval.open_pr", return_value=7),
        patch("bugeval.run_pr_eval.poll_for_review", return_value=False),  # timeout
        patch("bugeval.run_pr_eval.scrape_review_comments", return_value=[]) as mock_scrape,
        patch("bugeval.run_pr_eval.close_pr_delete_branch"),
    ):
        state = process_case_tool(case, tool, config, patches_dir, run_dir, repo_dir, dry_run=False)

    # Scrape MUST be called even though poll timed out (issue_comment tools)
    mock_scrape.assert_called_once()
    assert state.status == CaseToolStatus.done


def test_checkpoint_resume_skips_done(tmp_path: Path) -> None:
    config_path = _make_config_file(tmp_path)
    cases_dir = tmp_path / "cases"
    _make_case_file(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"
    run_dir.mkdir()

    # Pre-populate checkpoint with done state
    rs = RunState()
    from bugeval.pr_eval_models import CaseToolState

    rs.set(CaseToolState(case_id="case-001", tool="coderabbit", status=CaseToolStatus.done))
    rs.save(run_dir / "checkpoint.yaml")

    runner = CliRunner()
    with patch("bugeval.run_pr_eval.process_case_tool") as mock_process:
        result = runner.invoke(
            run_pr_eval,
            [
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

    # process_case_tool should NOT be called since status=done
    mock_process.assert_not_called()
    assert result.exit_code == 0


def test_run_pr_eval_limit_slices_cases(tmp_path: Path) -> None:
    """--limit should process at most N cases per tool."""
    config_path = _make_config_file(tmp_path)
    cases_dir = tmp_path / "cases"
    for i in range(1, 4):
        _make_case_file(cases_dir, f"case-{i:03d}")
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    for i in range(1, 4):
        (patches_dir / f"case-{i:03d}.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        run_pr_eval,
        [
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


def test_run_pr_eval_fail_after_aborts(tmp_path: Path) -> None:
    """--fail-after should abort the tool loop after N consecutive failures."""
    config_path = _make_config_file(tmp_path)
    cases_dir = tmp_path / "cases"
    for i in range(1, 5):
        _make_case_file(cases_dir, f"case-{i:03d}")
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    # No patches — patch not found → failed state → circuit breaker triggers
    run_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        run_pr_eval,
        [
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
