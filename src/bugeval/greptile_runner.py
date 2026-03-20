"""Greptile evaluation: PR-based lifecycle via GitHub App."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from bugeval.agent_runner import _scrub_fix_references
from bugeval.copilot_runner import (
    _get_patch_diff,
    _isolate_fork,
    close_eval_pr,
    create_eval_branch,
    ensure_fork,
    open_eval_pr,
    poll_for_review,
    scrape_pr_comments,
)
from bugeval.mine import run_gh
from bugeval.models import TestCase
from bugeval.result_models import Comment, ToolResult

log = logging.getLogger(__name__)


def poll_for_greptile_review(
    fork: str,
    pr_number: int,
    timeout: int = 300,
    poll_interval: int = 15,
) -> bool:
    """Poll until a Greptile review appears or timeout."""
    return poll_for_review(fork, pr_number, "greptile", timeout, poll_interval)


def scrape_greptile_comments(fork: str, pr_number: int) -> list[Comment]:
    """Scrape review comments from a PR, filtering to Greptile only."""
    return scrape_pr_comments(fork, pr_number, "greptile")


def _scrape_raw_greptile_comments(
    fork: str, pr_number: int,
) -> list[dict[str, Any]]:
    """Scrape all raw review comments from a PR (unfiltered)."""
    output = run_gh(
        "api",
        f"repos/{fork}/pulls/{pr_number}/comments",
    )
    raw: list[dict[str, Any]] = json.loads(output)
    return raw


def _save_greptile_transcript(
    transcript_dir: Path,
    case_id: str,
    *,
    fork: str,
    branch: str,
    pr_number: int,
    scrubbed_title: str,
    scrubbed_body: str,
    raw_comments: list[dict[str, Any]],
    patch_diff: str,
    time_seconds: float,
) -> str:
    """Save the Greptile interaction transcript for audit."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"{case_id}-greptile.json"
    data = {
        "pr_metadata": {
            "fork": fork,
            "branch": branch,
            "pr_number": pr_number,
        },
        "scrubbed_title": scrubbed_title,
        "scrubbed_body": scrubbed_body,
        "raw_comments": raw_comments,
        "patch_diff": patch_diff,
        "time_seconds": time_seconds,
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path)


def _default_branch(fork: str) -> str:
    output = run_gh(
        "repo", "view", fork,
        "--json", "defaultBranchRef",
        "-q", ".defaultBranchRef.name",
    )
    return output.strip() or "main"


def run_greptile(
    case: TestCase,
    repo_dir: Path,
    timeout: int = 300,
    org: str = "",
    transcript_dir: Path | None = None,
) -> ToolResult:
    """Run the full Greptile evaluation lifecycle for a test case."""
    start = time.monotonic()
    fork = ""
    branch = ""
    pr_number = 0
    patch_diff = ""
    try:
        patch_diff = _get_patch_diff(case, repo_dir)
        fork = ensure_fork(case.repo, org=org)
        branch = create_eval_branch(
            fork=fork,
            case=case,
            patch_diff=patch_diff,
            repo_dir=repo_dir,
        )

        # Isolate the fork
        introducing = (
            (case.truth.introducing_commit if case.truth else None)
            or case.base_commit
        )
        if introducing:
            default_br = _default_branch(fork)
            _isolate_fork(fork, introducing, default_br, repo_dir)

        pr_number = open_eval_pr(fork, branch, case)

        scrubbed_title = (
            _scrub_fix_references(case.introducing_pr_title)
            if case.introducing_pr_title else ""
        )
        scrubbed_body = (
            _scrub_fix_references(case.introducing_pr_body)
            if case.introducing_pr_body else ""
        )

        found = poll_for_greptile_review(fork, pr_number, timeout=timeout)

        if not found:
            elapsed = time.monotonic() - start
            if transcript_dir:
                _save_greptile_transcript(
                    transcript_dir, case.id,
                    fork=fork, branch=branch, pr_number=pr_number,
                    scrubbed_title=scrubbed_title,
                    scrubbed_body=scrubbed_body,
                    raw_comments=[], patch_diff=patch_diff,
                    time_seconds=elapsed,
                )
            return ToolResult(
                case_id=case.id,
                tool="greptile",
                context_level="diff+repo",
                comments=[],
                time_seconds=elapsed,
                error=f"Timeout waiting for Greptile review ({timeout}s)",
            )

        raw_comments = _scrape_raw_greptile_comments(fork, pr_number)
        comments = scrape_greptile_comments(fork, pr_number)
        elapsed = time.monotonic() - start

        transcript_path = ""
        if transcript_dir:
            transcript_path = _save_greptile_transcript(
                transcript_dir, case.id,
                fork=fork, branch=branch, pr_number=pr_number,
                scrubbed_title=scrubbed_title,
                scrubbed_body=scrubbed_body,
                raw_comments=raw_comments, patch_diff=patch_diff,
                time_seconds=elapsed,
            )
        return ToolResult(
            case_id=case.id,
            tool="greptile",
            context_level="diff+repo",
            comments=comments,
            time_seconds=elapsed,
            transcript_path=transcript_path,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return ToolResult(
            case_id=case.id,
            tool="greptile",
            context_level="diff+repo",
            comments=[],
            time_seconds=elapsed,
            error=str(exc),
        )
    finally:
        if pr_number and fork and branch:
            try:
                close_eval_pr(fork, pr_number, branch)
            except Exception:
                log.warning(
                    "Failed to clean up PR #%d on %s",
                    pr_number, fork,
                )
