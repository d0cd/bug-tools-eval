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
from bugeval.agent_cli_runner import (
    run_claude_cli,
    run_claude_cli_docker,
    run_codex_cli,
    run_gemini_cli,
)
from bugeval.agent_models import AgentResult
from bugeval.agent_prompts import build_user_prompt, load_agent_prompt
from bugeval.agent_sdk_runner import run_agent_sdk
from bugeval.google_api_runner import run_google_api
from bugeval.io import write_run_metadata
from bugeval.models import TestCase
from bugeval.openai_api_runner import run_openai_api
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


def _is_cli_tool(name: str) -> bool:
    """Return True for tools that run via subprocess (not API loops)."""
    return name in ("claude-code-cli",) or any(
        name.startswith(prefix) for prefix in ("claude-cli", "gemini-cli", "codex-cli")
    )


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


_BASE_TOOLS = "Read,Glob,Grep,WebSearch,WebFetch"
_DOCKER_TOOLS = f"Bash,{_BASE_TOOLS}"


def _resolve_allowed_tools(
    context_level: str, use_docker: bool, override: str | None
) -> str:
    """Return the tool allowlist string for the given context and runner.

    Resolution:
      - If override is provided, use it verbatim.
      - diff-only: no tools (empty string regardless of docker).
      - diff+repo*: Read,Glob,Grep,WebSearch,WebFetch; adds Bash when in Docker.
    """
    if override is not None:
        return override
    if context_level == "diff-only":
        return ""
    return _DOCKER_TOOLS if use_docker else _BASE_TOOLS


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
    repo_cache_dir: Path | None = None,
    allowed_tools: str | None = None,
) -> CaseToolState:
    """Run the agent state machine for one (case, tool) pair. Returns final state."""
    now = datetime.now(tz=UTC).isoformat()
    state = CaseToolState(case_id=case.id, tool=tool.name, started_at=now)

    if dry_run:
        state.status = CaseToolStatus.done
        state.completed_at = datetime.now(tz=UTC).isoformat()
        return state

    tmp_dir = Path(tempfile.mkdtemp())
    repo_dir: Path | None = None
    try:
        patch_content = patch_path.read_text()
        system_prompt = load_agent_prompt(language=case.language, context_level=context_level)
        user_prompt = build_user_prompt(case, patch_content, context_level)

        # Combine system prompt with user prompt for CLI runners.
        # API runners receive system_prompt as a separate argument.
        cli_prompt = f"{system_prompt}\n\n{user_prompt}" if _is_cli_tool(tool.name) else user_prompt

        # diff-only + CLI: no repo needed — create empty workspace only.
        if context_level == "diff-only" and _is_cli_tool(tool.name):
            cli_dir = tmp_dir / "workspace"
            cli_dir.mkdir(parents=True, exist_ok=True)
        else:
            state.status = CaseToolStatus.cloning
            repo_dir = setup_repo_for_case(case, patch_path, tmp_dir, cache_dir=repo_cache_dir)
            cli_dir = repo_dir

        cli_allowed_tools = _resolve_allowed_tools(context_level, use_docker, allowed_tools)

        state.status = CaseToolStatus.running
        if tool.name == "claude-code-cli" or tool.name.startswith("claude-cli"):
            agent_model = tool.model or "claude-sonnet-4-6"
            if use_docker:
                result: AgentResult = run_claude_cli_docker(
                    cli_dir,
                    cli_prompt,
                    max_turns=max_turns,
                    model=agent_model,
                    timeout_seconds=tool.timeout_seconds,
                    image=docker_image,
                    allowed_tools=cli_allowed_tools,
                    dangerously_skip_permissions=True,
                )
            else:
                result = run_claude_cli(
                    cli_dir,
                    cli_prompt,
                    max_turns=max_turns,
                    model=agent_model,
                    timeout_seconds=tool.timeout_seconds,
                    allowed_tools=cli_allowed_tools,
                )
        elif tool.name.startswith("gemini-cli"):
            agent_model = tool.model or "gemini-2.5-flash"
            result = run_gemini_cli(
                cli_dir, cli_prompt, model=agent_model, timeout_seconds=tool.timeout_seconds
            )
        elif tool.name.startswith("codex-cli"):
            agent_model = tool.model or "o4-mini"
            result = run_codex_cli(
                cli_dir, cli_prompt, model=agent_model, timeout_seconds=tool.timeout_seconds
            )
        elif tool.name.startswith("anthropic-api"):
            assert repo_dir is not None
            result = run_agent_api(
                repo_dir,
                system_prompt,
                user_prompt,
                max_turns=max_turns,
                context_level=context_level,
            )
        elif tool.name.startswith("google-api"):
            assert repo_dir is not None
            agent_model = tool.model or "gemini-2.5-flash"
            result = run_google_api(
                repo_dir,
                system_prompt,
                user_prompt,
                max_turns=max_turns,
                model=agent_model,
                context_level=context_level,
            )
        elif tool.name.startswith("openai-api"):
            assert repo_dir is not None
            agent_model = tool.model or "o4-mini"
            result = run_openai_api(
                repo_dir,
                system_prompt,
                user_prompt,
                max_turns=max_turns,
                model=agent_model,
                context_level=context_level,
            )
        elif tool.name.startswith("claude-agent-sdk"):
            assert repo_dir is not None
            agent_model = tool.model or "claude-sonnet-4-6"
            result = asyncio.run(
                run_agent_sdk(
                    repo_dir,
                    system_prompt,
                    user_prompt,
                    max_turns=max_turns,
                    model=agent_model,
                )
            )
        else:
            raise ValueError(f"Unknown agent tool: {tool.name!r}")

        # Stamp context_level so normalize can populate the context-level slice.
        result.context_level = context_level

        state.status = CaseToolStatus.collecting
        out_dir = run_dir / "raw" / f"{case.id}-{tool.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt.txt").write_text(cli_prompt)
        (out_dir / "findings.json").write_text(json.dumps(result.findings, indent=2))
        (out_dir / "conversation.json").write_text(json.dumps(result.conversation, indent=2))
        exclude_keys = {"findings", "conversation", "stdout", "response_text"}
        (out_dir / "metadata.json").write_text(
            json.dumps(result.model_dump(mode="json", exclude=exclude_keys), indent=2)
        )
        if result.stdout:
            (out_dir / "stdout.txt").write_text(result.stdout)
        if result.response_text:
            (out_dir / "response_text.txt").write_text(result.response_text)

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
    semaphore: asyncio.Semaphore,
    use_docker: bool = False,
    docker_image: str = "bugeval-agent",
    fail_after: int = 5,
    repo_cache_dir: Path | None = None,
    allowed_tools: str | None = None,
) -> None:
    """Evaluate all cases against one agent tool, concurrently (bounded by semaphore)."""
    failure_count = 0
    abort = False
    lock = asyncio.Lock()

    async def run_one(case: TestCase) -> None:
        nonlocal failure_count, abort

        async with lock:
            if abort:
                return
            existing = run_state.get(case.id, tool.name)
            if existing.status == CaseToolStatus.done:
                click.echo(f"[skip] {case.id} x {tool.name} (already done)")
                return

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
            async with lock:
                run_state.set(state)
                run_state.save(checkpoint_path)
                failure_count += 1
                click.echo(f"[failed] {case.id} x {tool.name}: patch not found")
                if fail_after > 0 and failure_count >= fail_after:
                    abort = True
                    click.echo(f"[abort] {tool.name}: {fail_after} failures, aborting")
            return

        click.echo(f"[start] {case.id} x {tool.name}")
        async with semaphore:
            async with lock:
                if abort:
                    return
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
                repo_cache_dir,
                allowed_tools,
            )

        async with lock:
            run_state.set(final_state)
            run_state.save(checkpoint_path)
            click.echo(f"[{final_state.status}] {case.id} x {tool.name}")
            if final_state.status == CaseToolStatus.failed:
                failure_count += 1
                if fail_after > 0 and failure_count >= fail_after:
                    abort = True
                    click.echo(f"[abort] {tool.name}: {fail_after} failures, aborting")

        if tool.cooldown_seconds > 0 and not dry_run:
            await asyncio.sleep(tool.cooldown_seconds)

    await asyncio.gather(*[run_one(case) for case in cases])


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
    default=30,
    show_default=True,
    type=int,
    help="Maximum number of agentic turns",
)
@click.option("--dry-run", is_flag=True, default=False, help="Simulate run without calling APIs")
@click.option(
    "--limit",
    default=0,
    show_default=True,
    type=int,
    help="Max cases to process per tool (0 = no limit)",
)
@click.option(
    "--fail-after",
    default=5,
    show_default=True,
    type=int,
    help="Abort tool after N total failures (0 = no limit)",
)
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
@click.option(
    "--max-concurrent",
    default=None,
    type=int,
    help="Max simultaneous agent calls (overrides config max_concurrent; default: 1).",
)
@click.option(
    "--repo-cache-dir",
    default=None,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Cache dir for repo clones. First use fetches from GitHub; later uses hardlink clone.",
)
@click.option(
    "--allowed-tools",
    default=None,
    help=(
        "Comma-separated Claude tools to allow (e.g. 'Read,Glob,Grep,Bash,WebSearch,WebFetch'). "
        "Default for diff+repo: Read,Glob,Grep,WebSearch,WebFetch; "
        "adds Bash automatically with --use-docker. "
        "diff-only always uses no tools."
    ),
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
    limit: int,
    fail_after: int,
    require_docker: bool,
    use_docker: bool,
    docker_image: str,
    max_concurrent: int | None,
    repo_cache_dir: str | None,
    allowed_tools: str | None,
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

    if limit > 0:
        cases = cases[:limit]

    agent_tools = config.agent_tools
    if tools_filter:
        prefixes = {n.strip() for n in tools_filter.split(",")}
        agent_tools = [t for t in agent_tools if any(t.name.startswith(p) for p in prefixes)]
        if not agent_tools:
            click.echo(f"No agent tools matched: {tools_filter}", err=True)
            sys.exit(1)

    # Resolve the effective allowed-tools string for metadata recording.
    effective_tools = _resolve_allowed_tools(context_level, use_docker, allowed_tools)

    # Write run_metadata.json for reproducibility tracing.
    write_run_metadata(
        resolved_run_dir,
        [t.name for t in agent_tools],
        context_level,
        Path(cases_dir),
        limit=limit,
        patches_dir=Path(patches_dir),
        config_path=config_path,
        allowed_tools=effective_tools,
    )

    resolved_patches_dir = Path(patches_dir)
    resolved_cache_dir = Path(repo_cache_dir) if repo_cache_dir else None
    if resolved_cache_dir:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    concurrency = max_concurrent if max_concurrent is not None else config.max_concurrent

    async def _run() -> None:
        semaphore = asyncio.Semaphore(concurrency)
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
                    semaphore,
                    use_docker,
                    docker_image,
                    fail_after,
                    resolved_cache_dir,
                    allowed_tools,
                )
                for tool in agent_tools
            ]
        )

    asyncio.run(_run())
    click.echo(f"Run complete. Results in: {resolved_run_dir}")
