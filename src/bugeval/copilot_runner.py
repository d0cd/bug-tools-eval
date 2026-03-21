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


def _tool_repo_name(repo: str, tool: str) -> str:
    """Build the per-tool repo name: {repo_slug}-{tool}."""
    _, name = repo.split("/", 1)
    return f"{name}-{tool}"


def ensure_tool_repo(repo: str, tool: str, org: str) -> str:
    """Ensure a dedicated per-tool repo exists in the org, return full name.

    Creates `{org}/{repo_slug}-{tool}` (e.g. `bug-tools-eval/leo-copilot`)
    as a private repo if it doesn't already exist. Each tool gets its own
    isolated repo — no cross-tool contamination.
    """
    repo_name = _tool_repo_name(repo, tool)
    full_name = f"{org}/{repo_name}"

    # Check if it already exists
    try:
        run_gh("repo", "view", full_name, "--json", "name")
        return full_name
    except GhError:
        pass

    # Create it
    try:
        run_gh(
            "repo", "create", full_name,
            "--private",
            "--description", f"bugeval: {tool} evaluation repo for {repo}",
        )
    except GhError:
        # May already exist (race condition) — verify
        run_gh("repo", "view", full_name, "--json", "name")

    return full_name


def ensure_fork(repo: str, org: str = "") -> str:
    """Legacy: create a standard GitHub fork. Prefer ensure_tool_repo()."""
    args = ["repo", "fork", repo, "--clone=false"]
    if org:
        args.extend(["--org", org])
    try:
        run_gh(*args)
    except GhError:
        pass
    _, name = repo.split("/", 1)
    if org:
        return f"{org}/{name}"
    username = run_gh(
        "api", "user", "--jq", ".login",
    ).strip()
    return f"{username}/{name}"


def _opaque_id() -> str:
    """Generate a short opaque ID for branch names (no case info leakage)."""
    import hashlib

    return hashlib.sha256(str(time.monotonic()).encode()).hexdigest()[:10]


def create_eval_branches(
    fork: str,
    case: TestCase,
    patch_diff: str,
    repo_dir: Path,
) -> tuple[str, str]:
    """Push base and review branches to the tool repo. Returns (base_branch, head_branch).

    - base_branch: repo state at introducing~1 (what was there before the bug)
    - head_branch: repo state after applying the introducing changes (the buggy PR)
    - Branch names are opaque (no case ID, no commit SHA) to prevent info leakage
    - Commit messages are generic
    """
    opaque = _opaque_id()
    base_branch = f"base-{opaque}"
    head_branch = f"review-{opaque}"
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

    # Push base branch (introducing~1)
    _git("checkout", "-B", base_branch, f"{introducing}~1")
    _git(
        "push", "--force", f"https://github.com/{fork}.git",
        f"{base_branch}:{base_branch}",
    )

    # Push head branch (introducing changes applied on top of base)
    _git("checkout", "-B", head_branch, f"{introducing}~1")

    cmd = [
        "git", "-C", str(repo_dir), "apply", "--allow-empty",
    ]
    proc = subprocess.run(
        cmd, input=patch_diff, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise GhError(cmd, proc.stderr)

    _git("add", "-A")
    _git("commit", "-m", "code changes", "--allow-empty")
    _git(
        "push", "--force", f"https://github.com/{fork}.git",
        f"{head_branch}:{head_branch}",
    )
    return base_branch, head_branch


def _default_branch(fork: str) -> str:
    output = run_gh(
        "repo", "view", fork,
        "--json", "defaultBranchRef",
        "-q", ".defaultBranchRef.name",
    )
    return output.strip() or "main"


def open_eval_pr(
    fork: str, head_branch: str, base_branch: str, case: TestCase,
) -> int:
    """Open a PR on the tool repo: head_branch → base_branch."""
    scrubbed_title = (
        _scrub_fix_references(case.introducing_pr_title)
        if case.introducing_pr_title else ""
    )
    title = scrubbed_title or "code changes"
    body = (
        _scrub_fix_references(case.introducing_pr_body)
        if case.introducing_pr_body else ""
    )
    # Write body to temp file to avoid shell escaping issues with special chars
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False,
    ) as f:
        f.write(body)
        body_file = f.name

    try:
        output = run_gh(
            "pr", "create",
            "--repo", fork,
            "--head", head_branch,
            "--base", base_branch,
            "--title", title,
            "--body-file", body_file,
        )
    finally:
        Path(body_file).unlink(missing_ok=True)

    url = output.strip()
    pr_number = int(url.rstrip("/").split("/")[-1])
    log.info("Opened PR #%d on %s", pr_number, fork)
    return pr_number


def poll_for_review(
    fork: str,
    pr_number: int,
    bot_name: str = "copilot",
    timeout: int = 300,
    poll_interval: int = 15,
) -> bool:
    """Poll until a review or inline comments from the named bot appear, or timeout."""
    start = time.monotonic()
    while True:
        # Check reviews (e.g., copilot-pull-request-reviewer[bot])
        try:
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
        except (GhError, json.JSONDecodeError):
            pass  # gh pr view may fail on fresh repos; fall through to comment check

        # Also check inline PR comments (some bots post as comments, not reviews)
        try:
            output = run_gh(
                "api",
                f"repos/{fork}/pulls/{pr_number}/comments",
            )
            comments = json.loads(output)
            if isinstance(comments, list):
                for comment in comments:
                    if not isinstance(comment, dict):
                        continue
                    user = comment.get("user") or {}
                    login = str(user.get("login", ""))
                    if bot_name.lower() in login.lower():
                        log.info(
                            "%s comment found on PR #%d", bot_name, pr_number,
                        )
                        return True
        except (GhError, json.JSONDecodeError):
            pass

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


def close_eval_pr(
    fork: str, pr_number: int,
    head_branch: str, base_branch: str = "",
) -> None:
    """Close the eval PR and delete remote branches."""
    run_gh(
        "pr", "close", str(pr_number), "--repo", fork,
    )
    # Delete head branch
    try:
        run_gh(
            "api", "--method", "DELETE",
            f"repos/{fork}/git/refs/heads/{head_branch}",
        )
    except GhError:
        pass  # Branch may already be gone
    # Delete base branch if provided
    if base_branch:
        try:
            run_gh(
                "api", "--method", "DELETE",
                f"repos/{fork}/git/refs/heads/{base_branch}",
            )
        except GhError:
            pass
    log.info("Closed PR #%d, cleaned branches", pr_number)


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
    base_branch = ""
    head_branch = ""
    pr_number = 0
    patch_diff = ""
    try:
        patch_diff = _get_patch_diff(case, repo_dir)
        fork = ensure_tool_repo(case.repo, "copilot", org) if org else ensure_fork(case.repo)

        # Push opaque base + head branches (no case ID in names)
        base_branch, head_branch = create_eval_branches(
            fork=fork,
            case=case,
            patch_diff=patch_diff,
            repo_dir=repo_dir,
        )

        pr_number = open_eval_pr(fork, head_branch, base_branch, case)
        found = poll_for_review(fork, pr_number, timeout=timeout)

        if not found:
            elapsed = time.monotonic() - start
            if transcript_dir:
                _save_copilot_transcript(
                    transcript_dir, case.id,
                    fork=fork, branch=head_branch, pr_number=pr_number,
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
                fork=fork, branch=head_branch, pr_number=pr_number,
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
        if pr_number and fork and head_branch:
            try:
                close_eval_pr(fork, pr_number, head_branch, base_branch)
            except Exception:
                log.warning(
                    "Failed to clean up PR #%d on %s",
                    pr_number, fork,
                )
