"""run-api-eval CLI: async orchestrator for API-mode evaluation."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from bugeval.adapters import get_adapter
from bugeval.models import TestCase
from bugeval.pr_eval_models import (
    CaseToolState,
    CaseToolStatus,
    EvalConfig,
    RunState,
    ToolDef,
    load_eval_config,
)
from bugeval.run_pr_eval import load_cases, make_run_id


def process_case_tool_api(
    case: TestCase,
    tool: ToolDef,
    patch_content: str,
    run_dir: Path,
    context_level: str,
    dry_run: bool,
) -> CaseToolState:
    """Run the API state machine for one (case, tool) pair. Returns final state."""
    now = datetime.now(tz=UTC).isoformat()
    state = CaseToolState(case_id=case.id, tool=tool.name, started_at=now)

    if dry_run:
        state.status = CaseToolStatus.done
        state.completed_at = datetime.now(tz=UTC).isoformat()
        return state

    try:
        if not tool.api_key_env:
            raise ValueError(f"Tool {tool.name!r} is missing api_key_env in config")
        if not tool.api_endpoint:
            raise ValueError(f"Tool {tool.name!r} is missing api_endpoint in config")
        api_key = os.environ.get(tool.api_key_env, "")
        adapter_cls: Any = get_adapter(tool.name)
        adapter = adapter_cls(api_endpoint=tool.api_endpoint, api_key=api_key)

        state.status = CaseToolStatus.submitting
        t0 = time.monotonic()
        findings: list[dict[str, Any]] = asyncio.run(
            adapter.submit(case, patch_content, context_level)
        )
        elapsed = time.monotonic() - t0

        state.status = CaseToolStatus.collecting
        out_dir = run_dir / "raw" / f"{case.id}-{tool.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "findings.json").write_text(json.dumps(findings, indent=2))
        (out_dir / "metadata.json").write_text(json.dumps({"time_seconds": round(elapsed, 2)}))

    except Exception as exc:
        state.status = CaseToolStatus.failed
        state.error = str(exc)
        state.completed_at = datetime.now(tz=UTC).isoformat()
        return state

    state.status = CaseToolStatus.done
    state.completed_at = datetime.now(tz=UTC).isoformat()
    return state


async def _eval_api_tool(
    tool: ToolDef,
    cases: list[TestCase],
    patches_dir: Path,
    run_dir: Path,
    context_level: str,
    run_state: RunState,
    checkpoint_path: Path,
    dry_run: bool,
    fail_after: int = 5,
) -> None:
    """Evaluate all cases against one API tool, sequentially."""
    consecutive_failures = 0
    for case in cases:
        existing = run_state.get(case.id, tool.name)
        if existing.status == CaseToolStatus.done:
            click.echo(f"[skip] {case.id} x {tool.name} (already done)")
            continue

        patch_path = patches_dir / f"{case.id}.patch"
        try:
            patch_content = patch_path.read_text()
        except FileNotFoundError:
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
            consecutive_failures += 1
            if fail_after > 0 and consecutive_failures >= fail_after:
                click.echo(f"[abort] {tool.name}: {fail_after} consecutive failures, aborting")
                break
            continue
        click.echo(f"[start] {case.id} x {tool.name}")
        final_state = await asyncio.to_thread(
            process_case_tool_api,
            case,
            tool,
            patch_content,
            run_dir,
            context_level,
            dry_run,
        )
        run_state.set(final_state)
        run_state.save(checkpoint_path)
        click.echo(f"[{final_state.status}] {case.id} x {tool.name}")

        if final_state.status == CaseToolStatus.failed:
            consecutive_failures += 1
            if fail_after > 0 and consecutive_failures >= fail_after:
                click.echo(f"[abort] {tool.name}: {fail_after} consecutive failures, aborting")
                break
        else:
            consecutive_failures = 0

        if tool.cooldown_seconds > 0 and not dry_run:
            await asyncio.sleep(tool.cooldown_seconds)


@click.command("run-api-eval")
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
    help="How much context to send to the API",
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
    help="Abort tool after N consecutive failures (0 = no limit)",
)
def run_api_eval(
    config_path: str,
    cases_dir: str,
    patches_dir: str,
    run_dir: str | None,
    tools_filter: str | None,
    context_level: str,
    dry_run: bool,
    limit: int,
    fail_after: int,
) -> None:
    """Async orchestrator: run API-mode evaluation across all (case × tool) pairs."""
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

    api_tools = config.api_tools
    if tools_filter:
        names = {n.strip() for n in tools_filter.split(",")}
        api_tools = [t for t in api_tools if t.name in names]
        if not api_tools:
            click.echo(f"No API tools matched: {tools_filter}", err=True)
            sys.exit(1)

    resolved_patches_dir = Path(patches_dir)

    async def _run() -> None:
        await asyncio.gather(
            *[
                _eval_api_tool(
                    tool,
                    cases,
                    resolved_patches_dir,
                    resolved_run_dir,
                    context_level,
                    run_state,
                    checkpoint_path,
                    dry_run,
                    fail_after,
                )
                for tool in api_tools
            ]
        )

    asyncio.run(_run())
    click.echo(f"Run complete. Results in: {resolved_run_dir}")
