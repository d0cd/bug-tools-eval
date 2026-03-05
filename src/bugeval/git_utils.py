"""Thin subprocess wrapper for git operations."""

from __future__ import annotations

import subprocess
from pathlib import Path

from bugeval.models import CaseStats


class GitError(Exception):
    """Raised when a git command fails."""

    def __init__(self, command: list[str], stderr: str) -> None:
        self.command = command
        self.stderr = stderr
        super().__init__(f"Git command failed: {' '.join(command)}\n{stderr}")


def run_git(*args: str, cwd: Path, timeout: int = 60) -> str:
    """Run a git command and return stdout. Raises GitError on failure."""
    cmd = ["git", *args]
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise GitError(cmd, f"Command timed out after {timeout}s")
    if result.returncode != 0:
        raise GitError(cmd, result.stderr)
    return result.stdout


def commit_exists(sha: str, cwd: Path) -> bool:
    """Check if a commit SHA exists in the repo."""
    try:
        run_git("cat-file", "-e", sha, cwd=cwd)
        return True
    except GitError:
        return False


def get_diff_stats(base: str, head: str, cwd: Path) -> CaseStats:
    """Get diff statistics between two commits."""
    stat_output = run_git("diff", "--numstat", base, head, cwd=cwd)
    lines_added = 0
    lines_deleted = 0
    files_changed = 0

    for line in stat_output.strip().splitlines():
        if line:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] != "-" and parts[1] != "-":
                try:
                    lines_added += int(parts[0])
                    lines_deleted += int(parts[1])
                except ValueError:
                    pass
                files_changed += 1

    diff_output = run_git("diff", base, head, cwd=cwd)
    hunks = diff_output.count("\n@@")

    return CaseStats(
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        files_changed=files_changed,
        hunks=hunks,
    )


def format_patch(base: str, head: str, cwd: Path) -> str:
    """Generate patch content between two commits."""
    return run_git("diff", base, head, cwd=cwd)


def apply_patch_check(patch_path: Path, cwd: Path) -> bool:
    """Check if a patch applies cleanly (dry run). Returns True if it applies."""
    try:
        run_git("apply", "--check", str(patch_path), cwd=cwd)
        return True
    except GitError:
        return False


def clone_repo(url: str, dest: Path, branch: str | None = None, timeout: int = 600) -> Path:
    """Clone a git repository. Returns the destination path."""
    args = ["git", "clone", url, str(dest)]
    if branch:
        args.extend(["-b", branch])
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise GitError(args, f"Clone timed out after {timeout}s")
    if result.returncode != 0:
        raise GitError(args, result.stderr)
    return dest


def is_repo(path: Path) -> bool:
    """Check if a path is a git repository."""
    try:
        run_git("rev-parse", "--git-dir", cwd=path)
        return True
    except (GitError, FileNotFoundError):
        return False
