"""Tests for run_agent_eval orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import ANY, patch

import yaml
from click.testing import CliRunner

from bugeval.cli import cli
from bugeval.pr_eval_models import CaseToolState, CaseToolStatus, RunState, ToolDef, ToolType
from bugeval.run_agent_eval import process_case_agent
from tests.conftest import make_case


def _make_config_yaml(tmp_path: Path, tools: list[dict[str, Any]] | None = None) -> Path:
    tools_data = tools or [
        {
            "name": "claude-code-cli",
            "type": "agent",
            "cooldown_seconds": 0,
        },
        {
            "name": "anthropic-api",
            "type": "agent",
            "cooldown_seconds": 0,
            "api_key_env": "ANTHROPIC_API_KEY",
        },
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


# --- process_case_agent ---


def test_process_case_agent_dry_run(tmp_path: Path) -> None:
    case = make_case()
    tool = ToolDef(name="claude-code-cli", type=ToolType.agent, cooldown_seconds=0)
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    state = process_case_agent(
        case=case,
        tool=tool,
        patch_path=patch_path,
        run_dir=tmp_path,
        context_level="diff-only",
        dry_run=True,
        max_turns=5,
    )
    assert state.status == CaseToolStatus.done
    assert state.case_id == "case-001"
    assert state.tool == "claude-code-cli"


def test_process_case_agent_cli_mode(tmp_path: Path) -> None:
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(name="claude-code-cli", type=ToolType.agent, cooldown_seconds=0)
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    mock_result = AgentResult(findings=[{"file": "a.rs", "line": 1, "summary": "x"}])

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch("bugeval.run_agent_eval.run_claude_cli", return_value=mock_result):
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                state = process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff-only",
                    dry_run=False,
                    max_turns=10,
                )

    assert state.status == CaseToolStatus.done

    # Verify context_level is stamped into metadata.json for normalize
    import json

    meta = json.loads((tmp_path / "raw" / "case-001-claude-code-cli" / "metadata.json").read_text())
    assert meta.get("context_level") == "diff-only"


def test_process_case_agent_diff_only_cli_uses_isolated_dir(tmp_path: Path) -> None:
    """In diff-only mode, CLI runners receive an empty workspace dir, not the cloned repo."""
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(name="claude-code-cli", type=ToolType.agent, cooldown_seconds=0)
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "secret.rs").write_text("fn secret() {}")

    mock_result = AgentResult(findings=[])
    captured_dirs: list[Path] = []

    def capture_cli_dir(d: Path, *args: Any, **kwargs: Any) -> AgentResult:
        captured_dirs.append(d)
        return mock_result

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=repo_dir):
        with patch("bugeval.run_agent_eval.run_claude_cli", side_effect=capture_cli_dir):
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff-only",
                    dry_run=False,
                    max_turns=10,
                )

    assert len(captured_dirs) == 1
    cli_dir = captured_dirs[0]
    # CLI should NOT run in the cloned repo (which has repo files)
    assert not (cli_dir / "secret.rs").exists()


def test_process_case_agent_diff_repo_cli_uses_repo_dir(tmp_path: Path) -> None:
    """In diff+repo mode, CLI runners receive the full cloned repo dir."""
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(name="claude-code-cli", type=ToolType.agent, cooldown_seconds=0)
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "main.rs").write_text("fn main() {}")

    mock_result = AgentResult(findings=[])
    captured_dirs: list[Path] = []

    def capture_cli_dir(d: Path, *args: Any, **kwargs: Any) -> AgentResult:
        captured_dirs.append(d)
        return mock_result

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=repo_dir):
        with patch("bugeval.run_agent_eval.run_claude_cli", side_effect=capture_cli_dir):
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff+repo",
                    dry_run=False,
                    max_turns=10,
                )

    assert len(captured_dirs) == 1
    assert captured_dirs[0] == repo_dir


def test_process_case_agent_api_mode(tmp_path: Path) -> None:
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(name="anthropic-api", type=ToolType.agent, cooldown_seconds=0)
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    mock_result = AgentResult(findings=[])

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch("bugeval.run_agent_eval.run_agent_api", return_value=mock_result):
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                state = process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff-only",
                    dry_run=False,
                    max_turns=20,
                )

    assert state.status == CaseToolStatus.done


# --- CLI integration ---


def test_run_agent_eval_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-agent-eval", "--help"])
    assert result.exit_code == 0
    assert "--context-level" in result.output
    assert "--max-turns" in result.output
    assert "--dry-run" in result.output
    assert "--use-docker" in result.output
    assert "--docker-image" in result.output
    assert "--require-docker" in result.output


def test_run_agent_eval_no_cases(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "empty_cases"
    cases_dir.mkdir()
    result = runner.invoke(
        cli,
        [
            "run-agent-eval",
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(tmp_path),
            "--run-dir",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 0
    assert "No cases found" in result.output


def test_run_agent_eval_dry_run(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    result = runner.invoke(
        cli,
        [
            "run-agent-eval",
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
    assert "done" in result.output


def test_run_agent_eval_checkpoint_written(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    runner.invoke(
        cli,
        [
            "run-agent-eval",
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


def test_run_agent_eval_checkpoint_resume_skips_done(tmp_path: Path) -> None:
    """A done pair in the checkpoint should be skipped."""
    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    run_dir = tmp_path / "results"
    run_dir.mkdir()

    # Pre-populate checkpoint with done state
    rs = RunState()
    rs.set(CaseToolState(case_id="case-001", tool="claude-code-cli", status=CaseToolStatus.done))
    rs.save(run_dir / "checkpoint.yaml")

    result = runner.invoke(
        cli,
        [
            "run-agent-eval",
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


def test_run_agent_eval_missing_patch_fails(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    # No patch file created
    run_dir = tmp_path / "results"

    result = runner.invoke(
        cli,
        [
            "run-agent-eval",
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
    assert "failed" in result.output
    assert "patch not found" in result.output


def test_run_agent_eval_unknown_tools_filter_exits(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = _make_config_yaml(tmp_path)
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)

    result = runner.invoke(
        cli,
        [
            "run-agent-eval",
            "--config",
            str(config_path),
            "--cases-dir",
            str(cases_dir),
            "--patches-dir",
            str(tmp_path),
            "--run-dir",
            str(tmp_path / "results"),
            "--tools",
            "nonexistent-tool",
        ],
    )
    assert result.exit_code == 1


def test_is_docker_available_returns_bool() -> None:
    from bugeval.run_agent_eval import is_docker_available

    result = is_docker_available()
    assert isinstance(result, bool)


def test_run_agent_eval_warns_without_docker(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    with patch("bugeval.run_agent_eval.is_docker_available", return_value=False):
        result = runner.invoke(
            cli,
            [
                "run-agent-eval",
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
    assert "Warning" in result.stderr


def test_run_agent_eval_exits_with_require_docker(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    run_dir = tmp_path / "results"

    with patch("bugeval.run_agent_eval.is_docker_available", return_value=False):
        result = runner.invoke(
            cli,
            [
                "run-agent-eval",
                "--config",
                str(config_path),
                "--cases-dir",
                str(cases_dir),
                "--patches-dir",
                str(patches_dir),
                "--run-dir",
                str(run_dir),
                "--require-docker",
            ],
        )

    assert result.exit_code != 0
    assert "Error" in result.stderr


def test_run_agent_eval_max_turns_passed_through(tmp_path: Path) -> None:
    from bugeval.agent_models import AgentResult

    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    captured_max_turns: list[int] = []

    def fake_cli_runner(
        repo_dir: Any, prompt: Any, max_turns: int = 10, **kwargs: Any
    ) -> AgentResult:
        captured_max_turns.append(max_turns)
        return AgentResult()

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch("bugeval.run_agent_eval.run_claude_cli", side_effect=fake_cli_runner):
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                runner.invoke(
                    cli,
                    [
                        "run-agent-eval",
                        "--config",
                        str(config_path),
                        "--cases-dir",
                        str(cases_dir),
                        "--patches-dir",
                        str(patches_dir),
                        "--run-dir",
                        str(run_dir),
                        "--max-turns",
                        "7",
                    ],
                )

    assert 7 in captured_max_turns


def test_process_case_agent_uses_docker_when_flag_set(tmp_path: Path) -> None:
    """When use_docker=True, run_claude_cli_docker is called instead of run_claude_cli."""
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(name="claude-code-cli", type=ToolType.agent, cooldown_seconds=0)
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    mock_result = AgentResult(findings=[])

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch(
            "bugeval.run_agent_eval.run_claude_cli_docker", return_value=mock_result
        ) as mock_docker:
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff-only",
                    dry_run=False,
                    max_turns=5,
                    use_docker=True,
                    docker_image="bugeval-agent",
                )

    # In diff-only mode CLI tools receive an isolated workspace dir (not the repo).
    mock_docker.assert_called_once_with(
        ANY,
        ANY,
        max_turns=5,
        model="claude-sonnet-4-6",
        image="bugeval-agent",
    )
    called_dir = mock_docker.call_args[0][0]
    assert called_dir != tmp_path / "repo"


def test_process_case_agent_gemini_cli_mode(tmp_path: Path) -> None:
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(
        name="gemini-cli-flash", type=ToolType.agent, cooldown_seconds=0, model="gemini-2.5-flash"
    )
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    mock_result = AgentResult(findings=[])

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch("bugeval.run_agent_eval.run_gemini_cli", return_value=mock_result) as mock_fn:
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                state = process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff-only",
                    dry_run=False,
                    max_turns=10,
                )

    assert state.status == CaseToolStatus.done
    mock_fn.assert_called_once_with(ANY, ANY, model="gemini-2.5-flash")


def test_process_case_agent_codex_cli_mode(tmp_path: Path) -> None:
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(name="codex-cli-o4", type=ToolType.agent, cooldown_seconds=0, model="o4-mini")
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    mock_result = AgentResult(findings=[])

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch("bugeval.run_agent_eval.run_codex_cli", return_value=mock_result) as mock_fn:
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                state = process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff-only",
                    dry_run=False,
                    max_turns=10,
                )

    assert state.status == CaseToolStatus.done
    mock_fn.assert_called_once_with(ANY, ANY, model="o4-mini")


def test_process_case_agent_google_api_mode(tmp_path: Path) -> None:
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(
        name="google-api-flash", type=ToolType.agent, cooldown_seconds=0, model="gemini-2.5-flash"
    )
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    mock_result = AgentResult(findings=[])

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch("bugeval.run_agent_eval.run_google_api", return_value=mock_result) as mock_fn:
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                state = process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff+repo",
                    dry_run=False,
                    max_turns=10,
                )

    assert state.status == CaseToolStatus.done
    mock_fn.assert_called_once_with(
        ANY, ANY, ANY, max_turns=10, model="gemini-2.5-flash", context_level="diff+repo"
    )


def test_process_case_agent_openai_api_mode(tmp_path: Path) -> None:
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(name="openai-api-o4", type=ToolType.agent, cooldown_seconds=0, model="o4-mini")
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    mock_result = AgentResult(findings=[])

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch("bugeval.run_agent_eval.run_openai_api", return_value=mock_result) as mock_fn:
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                state = process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff+repo",
                    dry_run=False,
                    max_turns=10,
                )

    assert state.status == CaseToolStatus.done
    mock_fn.assert_called_once_with(
        ANY, ANY, ANY, max_turns=10, model="o4-mini", context_level="diff+repo"
    )


def test_process_case_agent_unknown_tool_raises(tmp_path: Path) -> None:
    case = make_case()
    tool = ToolDef(name="unknown-xyz", type=ToolType.agent, cooldown_seconds=0)
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch("bugeval.run_agent_eval.cleanup_repo"):
            state = process_case_agent(
                case=case,
                tool=tool,
                patch_path=patch_path,
                run_dir=tmp_path,
                context_level="diff-only",
                dry_run=False,
                max_turns=5,
            )

    assert state.status == CaseToolStatus.failed
    assert "unknown-xyz" in (state.error or "")


def test_run_agent_eval_writes_run_metadata_json(tmp_path: Path) -> None:
    """run-agent-eval should write run_metadata.json to the run directory."""
    import json

    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    _make_case_yaml(cases_dir)
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    (patches_dir / "case-001.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    runner.invoke(
        cli,
        [
            "run-agent-eval",
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

    meta_path = run_dir / "run_metadata.json"
    assert meta_path.exists(), "run_metadata.json should be written"
    meta = json.loads(meta_path.read_text())
    assert "created_at" in meta
    assert "git_sha" in meta
    assert "config_hash" in meta
    assert "context_level" in meta
    assert "tools" in meta
    assert "python_version" in meta


def test_run_agent_eval_limit_slices_cases(tmp_path: Path) -> None:
    """--limit should process at most N cases per tool."""
    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    for i in range(1, 4):
        _make_case_yaml(cases_dir, f"case-{i:03d}")
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    for i in range(1, 4):
        (patches_dir / f"case-{i:03d}.patch").write_text("--- a\n+++ b\n")
    run_dir = tmp_path / "results"

    result = runner.invoke(
        cli,
        [
            "run-agent-eval",
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


def test_run_agent_eval_fail_after_aborts(tmp_path: Path) -> None:
    """--fail-after should abort the tool loop after N consecutive failures."""
    runner = CliRunner()
    config_path = _make_config_yaml(
        tmp_path,
        tools=[{"name": "claude-code-cli", "type": "agent", "cooldown_seconds": 0}],
    )
    cases_dir = tmp_path / "cases"
    for i in range(1, 5):
        _make_case_yaml(cases_dir, f"case-{i:03d}")
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    # No patches — all cases will fail with "patch not found"
    run_dir = tmp_path / "results"

    result = runner.invoke(
        cli,
        [
            "run-agent-eval",
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
    # Circuit breaker fires after 2 failures — only 2 states written
    assert len(failed) == 2


def test_process_case_agent_no_docker_by_default(tmp_path: Path) -> None:
    """When use_docker=False (default), run_claude_cli is called (not docker variant)."""
    from bugeval.agent_models import AgentResult

    case = make_case()
    tool = ToolDef(name="claude-code-cli", type=ToolType.agent, cooldown_seconds=0)
    patch_path = tmp_path / "case-001.patch"
    patch_path.write_text("--- a\n+++ b\n")

    mock_result = AgentResult(findings=[])

    with patch("bugeval.run_agent_eval.setup_repo_for_case", return_value=tmp_path / "repo"):
        with patch("bugeval.run_agent_eval.run_claude_cli", return_value=mock_result) as mock_local:
            with patch("bugeval.run_agent_eval.cleanup_repo"):
                process_case_agent(
                    case=case,
                    tool=tool,
                    patch_path=patch_path,
                    run_dir=tmp_path,
                    context_level="diff-only",
                    dry_run=False,
                    max_turns=5,
                    use_docker=False,
                )

    mock_local.assert_called_once()
