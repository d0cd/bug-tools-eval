"""run-agent-eval CLI: async orchestrator for in-house agent evaluation."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import click

from bugeval.agent_api_runner import run_agent_api
from bugeval.agent_cli_runner import run_claude_cli, run_claude_cli_docker
from bugeval.agent_models import AgentResult
from bugeval.agent_prompts import build_user_prompt, load_agent_prompt
from bugeval.models import TestCase
from bugeval.pr_eval_models import (
    CaseToolState,
    CaseToolStatus,
    EvalConfig,
    RunState,
    ToolDef,
    load_eval_config,
)
from bugeval.repo_setup import cleanup_repo, setup_repo_for_case
from bugeval.run_pr_eval import load_cases, make_run_id


def is_docker_available() -> bool:
    """Return True if Docker daemon is reachable via `docker info`."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def process_case_agent(
    case: TestCase,
    tool: ToolDef,
    patch_path: Path,
    run_dir: Path,
    context_level: str,
    dry_run: bool,
    max_turns: int,
    use_docker: bool = False,
    docker_image: str = "bugeval-agent",
) -> CaseToolState:
    """Run the agent state machine for one (case, tool) pair. Returns final state."""
    now = datetime.now(tz=UTC).isoformat()
    state = CaseToolState(case_id=case.id, tool=tool.name, started_at=now)

    if dry_run:
        state.status = CaseToolStatus.done
        state.completed_at = datetime.now(tz=UTC).isoformat()
        return state

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        patch_content = patch_path.read_text()
        system_prompt = load_agent_prompt()
        user_prompt = build_user_prompt(case, patch_content, context_level)

        state.status = CaseToolStatus.cloning
        repo_dir = setup_repo_for_case(case, patch_path, tmp_dir)

        state.status = CaseToolStatus.running
        if tool.name == "claude-code-cli":
            if use_docker:
                result: AgentResult = run_claude_cli_docker(
                    repo_dir, user_prompt, max_turns=max_turns, image=docker_image
                )
            else:
                result = run_claude_cli(repo_dir, user_prompt, max_turns=max_turns)
        elif tool.name == "anthropic-api":
            result = run_agent_api(
                repo_dir,
                system_prompt,
                user_prompt,
                max_turns=max_turns,
                context_level=context_level,
            )
        else:
            raise ValueError(f"Unknown agent tool: {tool.name!r}")

        # Stamp context_level so normalize can populate the context-level slice.
        result.context_level = context_level

        state.status = CaseToolStatus.collecting
        out_dir = run_dir / "raw" / f"{case.id}-{tool.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "findings.json").write_text(json.dumps(result.findings, indent=2))
        (out_dir / "conversation.json").write_text(json.dumps(result.conversation, indent=2))
        exclude_keys = {"findings", "conversation", "stdout"}
        (out_dir / "metadata.json").write_text(
            json.dumps(result.model_dump(mode="json", exclude=exclude_keys), indent=2)
        )
        if result.stdout:
            (out_dir / "stdout.txt").write_text(result.stdout)

        if result.error:
            state.status = CaseToolStatus.failed
            state.error = result.error
            state.completed_at = datetime.now(tz=UTC).isoformat()
            return state

    except Exception as exc:
        state.status = CaseToolStatus.failed
        state.error = str(exc)
        state.completed_at = datetime.now(tz=UTC).isoformat()
        return state
    finally:
        cleanup_repo(tmp_dir)

    state.status = CaseToolStatus.done
    state.completed_at = datetime.now(tz=UTC).isoformat()
    return state


async def _eval_agent_tool(
    tool: ToolDef,
    cases: list[TestCase],
    patches_dir: Path,
    run_dir: Path,
    context_level: str,
    run_state: RunState,
    checkpoint_path: Path,
    dry_run: bool,
    max_turns: int,
    use_docker: bool = False,
    docker_image: str = "bugeval-agent",
) -> None:
    """Evaluate all cases against one agent tool, sequentially."""
    for case in cases:
        existing = run_state.get(case.id, tool.name)
        if existing.status == CaseToolStatus.done:
            click.echo(f"[skip] {case.id} x {tool.name} (already done)")
            continue

        patch_path = patches_dir / f"{case.id}.patch"
        if not patch_path.exists():
            state = CaseToolState(
                case_id=case.id,
                tool=tool.name,
                status=CaseToolStatus.failed,
                error=f"patch not found: {patch_path}",
                started_at=datetime.now(tz=UTC).isoformat(),
                completed_at=datetime.now(tz=UTC).isoformat(),
            )
            run_state.set(state)
            run_state.save(checkpoint_path)
            click.echo(f"[failed] {case.id} x {tool.name}: patch not found")
            continue

        click.echo(f"[start] {case.id} x {tool.name}")
        final_state = await asyncio.to_thread(
            process_case_agent,
            case,
            tool,
            patch_path,
            run_dir,
            context_level,
            dry_run,
            max_turns,
            use_docker,
            docker_image,
        )
        run_state.set(final_state)
        run_state.save(checkpoint_path)
        click.echo(f"[{final_state.status}] {case.id} x {tool.name}")

        if tool.cooldown_seconds > 0 and not dry_run:
            await asyncio.sleep(tool.cooldown_seconds)


@click.command("run-agent-eval")
@click.option(
    "--config",
    "config_path",
    default="config/config.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml",
)
@click.option(
    "--cases-dir",
    default="cases/",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory containing case YAML files",
)
@click.option(
    "--patches-dir",
    default="patches/",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory containing patch files",
)
@click.option(
    "--run-dir",
    default=None,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Output directory for run results (default: results/{run-id})",
)
@click.option("--tools", "tools_filter", default=None, help="Comma-separated tool names to include")
@click.option(
    "--context-level",
    default="diff-only",
    show_default=True,
    type=click.Choice(["diff-only", "diff+repo", "diff+repo+domain"]),
    help="How much context to give the agent",
)
@click.option(
    "--max-turns",
    default=20,
    show_default=True,
    type=int,
    help="Maximum number of agentic turns",
)
@click.option("--dry-run", is_flag=True, default=False, help="Simulate run without calling APIs")
@click.option(
    "--require-docker",
    is_flag=True,
    default=False,
    help="Exit with error if Docker daemon is not reachable.",
)
@click.option(
    "--use-docker",
    is_flag=True,
    default=False,
    help="Run claude-code-cli inside a Docker container for isolation.",
)
@click.option(
    "--docker-image",
    default="bugeval-agent",
    show_default=True,
    help="Docker image name to use with --use-docker.",
)
def run_agent_eval(
    config_path: str,
    cases_dir: str,
    patches_dir: str,
    run_dir: str | None,
    tools_filter: str | None,
    context_level: str,
    max_turns: int,
    dry_run: bool,
    require_docker: bool,
    use_docker: bool,
    docker_image: str,
) -> None:
    """Async orchestrator: run in-house agent evaluation across all (case × tool) pairs."""
    if not is_docker_available():
        msg = (
            "Docker daemon not reachable. "
            "Agent runs clone to a local temp dir (no container isolation)."
        )
        if require_docker:
            click.echo(f"Error: {msg}", err=True)
            raise SystemExit(1)
        click.echo(f"Warning: {msg}", err=True)

    config: EvalConfig = load_eval_config(Path(config_path))

    run_id = make_run_id()
    resolved_run_dir = Path(run_dir) if run_dir else Path("results") / run_id
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolved_run_dir / "checkpoint.yaml"
    run_state = RunState.load(checkpoint_path)

    cases = load_cases(Path(cases_dir))
    if not cases:
        click.echo(f"No cases found in {cases_dir}")
        return

    agent_tools = config.agent_tools
    if tools_filter:
        names = {n.strip() for n in tools_filter.split(",")}
        agent_tools = [t for t in agent_tools if t.name in names]
        if not agent_tools:
            click.echo(f"No agent tools matched: {tools_filter}", err=True)
            sys.exit(1)

    resolved_patches_dir = Path(patches_dir)

    async def _run() -> None:
        await asyncio.gather(
            *[
                _eval_agent_tool(
                    tool,
                    cases,
                    resolved_patches_dir,
                    resolved_run_dir,
                    context_level,
                    run_state,
                    checkpoint_path,
                    dry_run,
                    max_turns,
                    use_docker,
                    docker_image,
                )
                for tool in agent_tools
            ]
        )

    asyncio.run(_run())
    click.echo(f"Run complete. Results in: {resolved_run_dir}")
