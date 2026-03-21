"""Evaluation orchestrator: dispatch to runners, manage checkpoints."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bugeval.git_utils import get_diff
from bugeval.io import (
    load_cases,
    load_checkpoint,
    save_checkpoint,
    save_result,
    write_run_metadata,
)
from bugeval.models import TestCase
from bugeval.result_models import ToolResult

logger = logging.getLogger(__name__)

_checkpoint_lock = threading.Lock()


def get_diff_for_case(case: TestCase, repo_dir: Path) -> str:
    """Get the introducing commit's diff (what the tool reviews)."""
    introducing = None
    if case.truth and case.truth.introducing_commit:
        introducing = case.truth.introducing_commit
    if not introducing:
        logger.warning(
            "No introducing commit for case %s, skipping", case.id,
        )
        return ""
    # Diff introducing commit against its parent
    return get_diff(f"{introducing}~1", introducing, cwd=repo_dir)


def result_filename(case_id: str, tool: str, context: str) -> str:
    """Build the result filename for a case/tool/context combo."""
    if context:
        return f"{case_id}--{tool}--{context}.yaml"
    return f"{case_id}--{tool}.yaml"


def _checkpoint_key(case_id: str, tool: str, context: str) -> str:
    if context:
        return f"{case_id}::{tool}::{context}"
    return f"{case_id}::{tool}"


def process_case(
    case: TestCase,
    tool: str,
    context_level: str,
    repo_dir: Path,
    run_dir: Path,
    timeout: int,
    thinking_budget: int = 0,
    model: str = "",
    org: str = "",
    docker: bool = False,
    docker_image: str = "bugeval-agent",
) -> ToolResult:
    """Dispatch to the appropriate runner and save the result."""
    diff = get_diff_for_case(case, repo_dir)

    workspace: Path | None = None
    if context_level != "diff-only":
        workspace = repo_dir

    # Docker dispatch: run agent-cli-* or agent-sdk inside a container
    if docker and (tool.startswith("agent-cli") or tool == "agent-sdk"):
        from bugeval.agent_runner import run_docker, setup_workspace

        if context_level != "diff-only":
            ws_dir = run_dir / "workspaces"
            ws_dir.mkdir(parents=True, exist_ok=True)
            workspace = setup_workspace(
                case, f"https://github.com/{case.repo}.git",
                context_level, ws_dir,
            )

        parts = tool.split("-", 2)
        cli_name = parts[2] if len(parts) > 2 else "claude"

        transcript_dir = run_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        result = run_docker(
            case, diff, workspace, context_level,
            cli_tool=cli_name, timeout=timeout,
            transcript_dir=transcript_dir,
            model=model, image=docker_image,
        )
        results_dir = run_dir / "results"
        fname = result_filename(case.id, tool, context_level)
        save_result(result, results_dir / fname)
        return result

    if tool == "greptile":
        from bugeval.greptile_runner import run_greptile

        transcript_dir = run_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        result = run_greptile(
            case, repo_dir, timeout=timeout,
            org=org, transcript_dir=transcript_dir,
        )
    elif tool == "coderabbit":
        from bugeval.coderabbit_runner import run_coderabbit

        transcript_dir = run_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        result = run_coderabbit(
            case, repo_dir, timeout=timeout,
            org=org, transcript_dir=transcript_dir,
        )
    elif tool == "copilot":
        from bugeval.copilot_runner import run_copilot

        transcript_dir = run_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        result = run_copilot(
            case, repo_dir, timeout=timeout,
            org=org, transcript_dir=transcript_dir,
        )
    elif tool == "agent":
        from bugeval.agent_runner import run_anthropic_api, setup_workspace

        # Set up isolated workspace at the correct commit
        if context_level != "diff-only":
            ws_dir = run_dir / "workspaces"
            ws_dir.mkdir(parents=True, exist_ok=True)
            workspace = setup_workspace(
                case, f"https://github.com/{case.repo}.git",
                context_level, ws_dir,
            )

        transcript_dir = run_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        result = run_anthropic_api(
            case, diff, workspace, context_level,
            timeout=timeout,
            transcript_dir=transcript_dir,
            thinking_budget=thinking_budget,
            model=model,
        )
    elif tool == "agent-gemini":
        from bugeval.agent_runner import run_google_api, setup_workspace

        if context_level != "diff-only":
            ws_dir = run_dir / "workspaces"
            ws_dir.mkdir(parents=True, exist_ok=True)
            workspace = setup_workspace(
                case, f"https://github.com/{case.repo}.git",
                context_level, ws_dir,
            )

        transcript_dir = run_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        result = run_google_api(
            case, diff, workspace, context_level,
            timeout=timeout,
            transcript_dir=transcript_dir,
            thinking_budget=thinking_budget,
            model=model,
        )
    elif tool == "agent-openai":
        from bugeval.agent_runner import run_openai_api, setup_workspace

        if context_level != "diff-only":
            ws_dir = run_dir / "workspaces"
            ws_dir.mkdir(parents=True, exist_ok=True)
            workspace = setup_workspace(
                case, f"https://github.com/{case.repo}.git",
                context_level, ws_dir,
            )

        transcript_dir = run_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        result = run_openai_api(
            case, diff, workspace, context_level,
            timeout=timeout,
            transcript_dir=transcript_dir,
            thinking_budget=thinking_budget,
            model=model,
        )
    elif tool.startswith("agent-cli"):
        from bugeval.agent_runner import run_agent_cli, setup_workspace

        # Set up isolated workspace at the correct commit
        if context_level != "diff-only":
            ws_dir = run_dir / "workspaces"
            ws_dir.mkdir(parents=True, exist_ok=True)
            workspace = setup_workspace(
                case, f"https://github.com/{case.repo}.git",
                context_level, ws_dir,
            )

        parts = tool.split("-", 2)
        cli_name = parts[2] if len(parts) > 2 else "claude"

        transcript_dir = run_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        result = run_agent_cli(
            case, diff, workspace, context_level,
            cli_tool=cli_name, timeout=timeout,
            transcript_dir=transcript_dir,
            model=model,
        )
    elif tool == "agent-sdk":
        from bugeval.agent_runner import run_agent_sdk, setup_workspace

        # Set up isolated workspace at the correct commit
        if context_level != "diff-only":
            ws_dir = run_dir / "workspaces"
            ws_dir.mkdir(parents=True, exist_ok=True)
            workspace = setup_workspace(
                case, f"https://github.com/{case.repo}.git",
                context_level, ws_dir,
            )

        transcript_dir = run_dir / "transcripts"
        result = run_agent_sdk(
            case, diff, workspace, context_level,
            timeout=timeout, transcript_dir=transcript_dir,
            model=model,
        )
    else:
        # Unsupported tool — return error result
        result = ToolResult(
            case_id=case.id,
            tool=tool,
            context_level=context_level,
            error=f"Unsupported tool: {tool}",
        )

    # Save result
    results_dir = run_dir / "results"
    fname = result_filename(case.id, tool, context_level)
    save_result(result, results_dir / fname)
    return result


def evaluate_tool(
    tool: str,
    cases_dir: Path,
    run_dir: Path,
    context_level: str,
    repo_dir: Path,
    concurrency: int,
    timeout: int,
    dry_run: bool,
    thinking_budget: int = 0,
    model: str = "",
    org: str = "",
    docker: bool = False,
    docker_image: str = "bugeval-agent",
) -> None:
    """Main orchestrator: load cases, process each, checkpoint progress."""
    cases = load_cases(cases_dir)
    if not cases:
        logger.warning("No cases found in %s", cases_dir)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    done = load_checkpoint(checkpoint_path)

    # Write run metadata
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        write_run_metadata(
            run_dir, tool, context_level, cases_dir,
            model=model, thinking_budget=thinking_budget,
            timeout=timeout,
        )

    # Filter to pending cases
    pending: list[TestCase] = []
    for c in cases:
        key = _checkpoint_key(c.id, tool, context_level)
        if key not in done:
            pending.append(c)

    logger.info(
        "Evaluating %s: %d pending, %d done, %d total",
        tool, len(pending), len(done), len(cases),
    )

    if dry_run:
        for c in pending:
            logger.info("[dry-run] Would process %s with %s", c.id, tool)
        return

    checkpoint_batch_size = 5

    if concurrency <= 1:
        pending_keys: list[str] = []
        for c in pending:
            try:
                process_case(
                    c, tool, context_level, repo_dir, run_dir, timeout,
                    thinking_budget=thinking_budget,
                    model=model, org=org,
                    docker=docker, docker_image=docker_image,
                )
            except Exception:
                logger.exception("Error processing %s", c.id)
            key = _checkpoint_key(c.id, tool, context_level)
            pending_keys.append(key)
            if len(pending_keys) >= checkpoint_batch_size:
                with _checkpoint_lock:
                    done.update(pending_keys)
                    save_checkpoint(done, checkpoint_path)
                pending_keys = []
        if pending_keys:
            with _checkpoint_lock:
                done.update(pending_keys)
                save_checkpoint(done, checkpoint_path)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    process_case, c, tool, context_level,
                    repo_dir, run_dir, timeout,
                    thinking_budget=thinking_budget,
                    model=model, org=org,
                    docker=docker, docker_image=docker_image,
                ): c
                for c in pending
            }
            pending_keys_concurrent: list[str] = []
            for future in as_completed(futures):
                c = futures[future]
                try:
                    future.result()
                    key = _checkpoint_key(c.id, tool, context_level)
                    pending_keys_concurrent.append(key)
                    if len(pending_keys_concurrent) >= checkpoint_batch_size:
                        with _checkpoint_lock:
                            done.update(pending_keys_concurrent)
                            save_checkpoint(done, checkpoint_path)
                        pending_keys_concurrent = []
                except Exception:
                    logger.warning(
                        "Error processing %s, will retry next run", c.id,
                    )
            if pending_keys_concurrent:
                with _checkpoint_lock:
                    done.update(pending_keys_concurrent)
                    save_checkpoint(done, checkpoint_path)
