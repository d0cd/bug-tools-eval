"""run-pr-eval CLI: async orchestrator for PR-mode evaluation."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from bugeval.io import load_all_cases
from bugeval.manage_forks import fork_name
from bugeval.models import TestCase
from bugeval.pr_eval_models import (
    CaseToolState,
    CaseToolStatus,
    EvalConfig,
    RunState,
    ToolDef,
    load_eval_config,
)
from bugeval.pr_lifecycle import (
    apply_patch_to_branch,
    close_pr_delete_branch,
    make_branch_name,
    open_pr,
    poll_for_review,
    scrape_review_comments,
)


def make_run_id() -> str:
    """Generate a run ID based on the current UTC date."""
    return f"run-{datetime.now(tz=UTC).strftime('%Y-%m-%d')}"


def load_cases(cases_dir: Path) -> list[TestCase]:
    """Load all test cases from a directory. Returns empty list if dir missing."""
    if not cases_dir.exists():
        return []
    return load_all_cases(cases_dir)


def process_case_tool(
    case: TestCase,
    tool: ToolDef,
    config: EvalConfig,
    patches_dir: Path,
    run_dir: Path,
    repo_dir: Path | None,
    dry_run: bool,
) -> CaseToolState:
    """Run the state machine for one (case, tool) pair. Returns final state."""
    now = datetime.now(tz=UTC).isoformat()
    state = CaseToolState(case_id=case.id, tool=tool.name, started_at=now)

    # Locate patch file
    patch_path = patches_dir / f"{case.id}.patch"
    if not patch_path.exists():
        state.status = CaseToolStatus.failed
        state.error = f"patch not found: {patch_path}"
        state.completed_at = datetime.now(tz=UTC).isoformat()
        return state

    fork_repo = f"{config.eval_org}/{fork_name(case.repo, tool.name)}"
    upstream_repo = config.repos.get(case.repo.split("/")[-1]) or case.repo
    branch = make_branch_name(case.id, tool.name)

    try:
        # --- branching + applying ---
        if repo_dir is not None:
            state.status = CaseToolStatus.branching
            state.status = CaseToolStatus.applying
            if not dry_run:
                fork_url = f"git@github.com:{fork_repo}.git"
                apply_patch_to_branch(branch, case.base_commit, patch_path, fork_url, repo_dir)

        # --- open PR ---
        state.status = CaseToolStatus.pr_open
        pr_number = open_pr(fork_repo, upstream_repo, branch, case, dry_run=dry_run)
        state.pr_number = pr_number
        state.branch_name = branch

        if dry_run:
            state.status = CaseToolStatus.done
            state.completed_at = datetime.now(tz=UTC).isoformat()
            return state

        # --- polling ---
        state.status = CaseToolStatus.polling
        poll_for_review(fork_repo, pr_number)  # waits for tool to respond; scrape regardless

        # --- scraping ---
        # Always scrape all three endpoints (reviews, inline, issue_comments).
        # Some tools post findings only to issue_comments without a formal review,
        # so conditioning on poll result would silently miss their output.
        state.status = CaseToolStatus.scraping
        comments: list[dict[str, Any]] = scrape_review_comments(fork_repo, pr_number)

        # Save comments
        out_dir = run_dir / "raw" / f"{case.id}-{tool.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "comments.json").write_text(json.dumps(comments, indent=2))

        # --- closing ---
        state.status = CaseToolStatus.closing
        close_pr_delete_branch(fork_repo, pr_number, branch, dry_run=False)

    except Exception as exc:
        state.status = CaseToolStatus.failed
        state.error = str(exc)
        state.completed_at = datetime.now(tz=UTC).isoformat()
        return state

    state.status = CaseToolStatus.done
    state.completed_at = datetime.now(tz=UTC).isoformat()
    return state


async def _eval_tool(
    tool: ToolDef,
    cases: list[TestCase],
    config: EvalConfig,
    patches_dir: Path,
    run_dir: Path,
    repo_dir: Path | None,
    run_state: RunState,
    checkpoint_path: Path,
    dry_run: bool,
    fail_after: int = 5,
) -> None:
    """Evaluate all cases against one tool, sequentially."""
    consecutive_failures = 0
    for case in cases:
        existing = run_state.get(case.id, tool.name)
        if existing.status == CaseToolStatus.done:
            click.echo(f"[skip] {case.id} x {tool.name} (already done)")
            continue

        click.echo(f"[start] {case.id} x {tool.name}")
        final_state = await asyncio.to_thread(
            process_case_tool,
            case,
            tool,
            config,
            patches_dir,
            run_dir,
            repo_dir,
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


@click.command("run-pr-eval")
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
    "--repo-dir",
    default=None,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Local repo checkout for branch/patch operations (optional)",
)
@click.option(
    "--run-dir",
    default=None,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Output directory for run results (default: results/{run-id})",
)
@click.option("--tools", "tools_filter", default=None, help="Comma-separated tool names to include")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Simulate run without calling gh or git"
)
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
def run_pr_eval(
    config_path: str,
    cases_dir: str,
    patches_dir: str,
    repo_dir: str | None,
    run_dir: str | None,
    tools_filter: str | None,
    dry_run: bool,
    limit: int,
    fail_after: int,
) -> None:
    """Async orchestrator: run PR-mode evaluation across all (case × tool) pairs."""
    config: EvalConfig = load_eval_config(Path(config_path))

    # Resolve run dir
    run_id = make_run_id()
    resolved_run_dir = Path(run_dir) if run_dir else Path("results") / run_id
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolved_run_dir / "checkpoint.yaml"
    run_state = RunState.load(checkpoint_path)

    # Load cases
    cases = load_cases(Path(cases_dir))
    if not cases:
        click.echo(f"No cases found in {cases_dir}")
        return

    if limit > 0:
        cases = cases[:limit]

    # Select tools
    pr_tools = config.pr_tools
    if tools_filter:
        names = {n.strip() for n in tools_filter.split(",")}
        pr_tools = [t for t in pr_tools if t.name in names]
        if not pr_tools:
            click.echo(f"No PR tools matched: {tools_filter}", err=True)
            sys.exit(1)

    resolved_repo_dir = Path(repo_dir) if repo_dir else None
    resolved_patches_dir = Path(patches_dir)

    async def _run() -> None:
        await asyncio.gather(
            *[
                _eval_tool(
                    tool,
                    cases,
                    config,
                    resolved_patches_dir,
                    resolved_run_dir,
                    resolved_repo_dir,
                    run_state,
                    checkpoint_path,
                    dry_run,
                    fail_after,
                )
                for tool in pr_tools
            ]
        )

    asyncio.run(_run())
    click.echo(f"Run complete. Results in: {resolved_run_dir}")
