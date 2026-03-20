"""Copilot evaluation: create PR on fork, wait for review, scrape comments."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from bugeval.agent_runner import _scrub_fix_references
from bugeval.git_utils import run_git
from bugeval.mine import GhError, run_gh
from bugeval.models import TestCase
from bugeval.result_models import Comment, ToolResult

log = logging.getLogger(__name__)


def ensure_fork(repo: str, org: str = "") -> str:
    """Ensure a GitHub fork exists for the repo, return fork name."""
    args = ["repo", "fork", repo, "--clone=false"]
    if org:
        args.extend(["--org", org])
    try:
        run_gh(*args)
    except GhError:
        # Fork likely already exists — continue
        pass
    _, name = repo.split("/", 1)
    if org:
        return f"{org}/{name}"
    username = run_gh(
        "api", "user", "--jq", ".login",
    ).strip()
    return f"{username}/{name}"


def create_eval_branch(
    fork: str,
    case: TestCase,
    patch_diff: str,
    repo_dir: Path,
) -> str:
    """Create an eval branch at the introducing commit's parent with the patch applied."""
    branch = f"eval/{case.id}"
    introducing = (
        (case.truth.introducing_commit if case.truth else None) or case.base_commit
    )

    def _git(*args: str) -> None:
        cmd = ["git", "-C", str(repo_dir), *args]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise GhError(cmd, result.stderr)

    _git("checkout", "-B", branch, f"{introducing}~1")

    # Apply the introducing commit's changes via stdin
    cmd = [
        "git", "-C", str(repo_dir), "apply", "--allow-empty",
    ]
    proc = subprocess.run(
        cmd, input=patch_diff, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise GhError(cmd, proc.stderr)

    _git("add", "-A")
    _git("commit", "-m", f"eval: {case.id}", "--allow-empty")
    _git(
        "push", "--force", f"https://github.com/{fork}.git",
        f"{branch}:{branch}",
    )
    return branch


def _default_branch(fork: str) -> str:
    output = run_gh(
        "repo", "view", fork,
        "--json", "defaultBranchRef",
        "-q", ".defaultBranchRef.name",
    )
    return output.strip() or "main"


def open_eval_pr(fork: str, branch: str, case: TestCase) -> int:
    """Open a PR on the fork with introducing PR metadata."""
    scrubbed_title = (
        _scrub_fix_references(case.introducing_pr_title)
        if case.introducing_pr_title else ""
    )
    title = scrubbed_title or f"eval-{case.id}"
    body = (
        _scrub_fix_references(case.introducing_pr_body)
        if case.introducing_pr_body else ""
    )
    base = _default_branch(fork)
    output = run_gh(
        "pr", "create",
        "--repo", fork,
        "--head", branch,
        "--base", base,
        "--title", title,
        "--body", body,
        "--json", "number",
    )
    data = json.loads(output)
    pr_number: int = data["number"]
    log.info("Opened PR #%d on %s", pr_number, fork)
    return pr_number


def poll_for_review(
    fork: str,
    pr_number: int,
    bot_name: str = "copilot",
    timeout: int = 300,
    poll_interval: int = 15,
) -> bool:
    """Poll until a review from the named bot appears or timeout."""
    start = time.monotonic()
    while True:
        output = run_gh(
            "pr", "view", str(pr_number),
            "--repo", fork,
            "--json", "reviews",
        )
        data = json.loads(output)
        reviews = data.get("reviews") or []
        for review in reviews:
            author = (review.get("author") or {}).get("login", "")
            if bot_name.lower() in author.lower():
                log.info(
                    "%s review found on PR #%d", bot_name, pr_number,
                )
                return True
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            log.warning(
                "Timed out waiting for %s review on PR #%d",
                bot_name, pr_number,
            )
            return False
        time.sleep(poll_interval)


def scrape_pr_comments(
    fork: str, pr_number: int, bot_name: str = "copilot",
) -> list[Comment]:
    """Scrape review comments from a PR, filtering to the named bot."""
    output = run_gh(
        "api",
        f"repos/{fork}/pulls/{pr_number}/comments",
    )
    raw_comments: list[dict[str, object]] = json.loads(output)

    comments: list[Comment] = []
    for rc in raw_comments:
        user = rc.get("user") or {}
        login = str(
            user.get("login", "") if isinstance(user, dict) else ""
        )
        if bot_name.lower() not in login.lower():
            continue
        comments.append(Comment(
            file=str(rc.get("path", "")),
            line=int(rc.get("line") or 0),  # type: ignore[arg-type]
            body=str(rc.get("body", "")),
        ))
    return comments


def close_eval_pr(fork: str, pr_number: int, branch: str) -> None:
    """Close the eval PR and delete the remote branch."""
    run_gh(
        "pr", "close", str(pr_number), "--repo", fork,
    )
    run_gh(
        "api", "--method", "DELETE",
        f"repos/{fork}/git/refs/heads/{branch}",
    )
    log.info("Closed PR #%d and deleted branch %s", pr_number, branch)


def _get_patch_diff(case: TestCase, repo_dir: Path) -> str:
    """Get the diff for the introducing commit."""
    introducing = (
        (case.truth.introducing_commit if case.truth else None) or case.base_commit
    )
    if not introducing:
        return ""
    return run_git("diff", f"{introducing}~1", introducing, cwd=repo_dir)


def _isolate_fork(
    fork: str, introducing: str, default_branch: str, repo_dir: Path,
) -> None:
    """Force-push the fork's default branch to introducing~1 for isolation."""
    cmd = [
        "git", "-C", str(repo_dir), "push", "--force",
        f"https://github.com/{fork}.git",
        f"{introducing}~1:refs/heads/{default_branch}",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise GhError(cmd, result.stderr)


def _save_copilot_transcript(
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
    """Save the Copilot interaction transcript for audit."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"{case_id}-copilot.json"
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


def _scrape_raw_comments(fork: str, pr_number: int) -> list[dict[str, Any]]:
    """Scrape all raw review comments from a PR (unfiltered)."""
    output = run_gh(
        "api",
        f"repos/{fork}/pulls/{pr_number}/comments",
    )
    raw: list[dict[str, Any]] = json.loads(output)
    return raw


def run_copilot(
    case: TestCase,
    repo_dir: Path,
    timeout: int = 300,
    org: str = "",
    transcript_dir: Path | None = None,
) -> ToolResult:
    """Run the full Copilot evaluation lifecycle for a test case."""
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

        # Isolate the fork: reset default branch to the introducing commit's
        # parent so the PR diff only contains the introducing changes.
        introducing = (
            (case.truth.introducing_commit if case.truth else None)
            or case.base_commit
        )
        if introducing:
            default_br = _default_branch(fork)
            _isolate_fork(fork, introducing, default_br, repo_dir)

        pr_number = open_eval_pr(fork, branch, case)
        found = poll_for_review(fork, pr_number, timeout=timeout)

        if not found:
            elapsed = time.monotonic() - start
            if transcript_dir:
                _save_copilot_transcript(
                    transcript_dir, case.id,
                    fork=fork, branch=branch, pr_number=pr_number,
                    scrubbed_title=_scrub_fix_references(
                        case.introducing_pr_title or "",
                    ),
                    scrubbed_body=_scrub_fix_references(
                        case.introducing_pr_body or "",
                    ),
                    raw_comments=[], patch_diff=patch_diff,
                    time_seconds=elapsed,
                )
            return ToolResult(
                case_id=case.id,
                tool="copilot",
                context_level="diff+repo",
                comments=[],
                time_seconds=elapsed,
                error=f"Timeout waiting for Copilot review ({timeout}s)",
            )

        raw_comments = _scrape_raw_comments(fork, pr_number)
        comments = scrape_pr_comments(fork, pr_number)
        elapsed = time.monotonic() - start

        scrubbed_title = _scrub_fix_references(
            case.introducing_pr_title or "",
        )
        scrubbed_body = _scrub_fix_references(
            case.introducing_pr_body or "",
        )
        transcript_path = ""
        if transcript_dir:
            transcript_path = _save_copilot_transcript(
                transcript_dir, case.id,
                fork=fork, branch=branch, pr_number=pr_number,
                scrubbed_title=scrubbed_title,
                scrubbed_body=scrubbed_body,
                raw_comments=raw_comments, patch_diff=patch_diff,
                time_seconds=elapsed,
            )
        return ToolResult(
            case_id=case.id,
            tool="copilot",
            context_level="diff+repo",
            comments=comments,
            time_seconds=elapsed,
            transcript_path=transcript_path,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return ToolResult(
            case_id=case.id,
            tool="copilot",
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
