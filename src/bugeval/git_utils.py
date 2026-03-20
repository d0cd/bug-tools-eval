"""Thin subprocess wrapper for git operations."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(Exception):
    def __init__(self, command: list[str], stderr: str) -> None:
        self.command = command
        self.stderr = stderr
        super().__init__(f"Git command failed: {' '.join(command)}\n{stderr}")


def run_git(*args: str, cwd: Path, timeout: int = 60) -> str:
    """Run a git command and return stdout. Raises GitError on failure."""
    cmd = ["git", *args]
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise GitError(cmd, f"Command timed out after {timeout}s")
    if result.returncode != 0:
        raise GitError(cmd, result.stderr)
    return result.stdout


def commit_exists(sha: str, cwd: Path) -> bool:
    """Check whether a commit SHA exists in the repo."""
    try:
        run_git("cat-file", "-e", sha, cwd=cwd)
        return True
    except GitError:
        return False


def get_diff(base: str, head: str, cwd: Path) -> str:
    """Return the diff between two commits."""
    return run_git("diff", base, head, cwd=cwd)


def get_changed_files(base: str, head: str, cwd: Path) -> list[str]:
    """Return list of files changed between two commits."""
    output = run_git("diff", "--name-only", base, head, cwd=cwd)
    return [f for f in output.strip().splitlines() if f]


def clone_at_sha(
    url: str, dest: Path, sha: str, timeout: int = 600
) -> Path:
    """Clone repo and checkout at specific SHA (truncated history)."""
    if dest.exists():
        # Already cloned (e.g., previous aborted run) — just checkout
        run_git("checkout", sha, cwd=dest)
        return dest
    cmd = ["git", "clone", "--single-branch", url, str(dest)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise GitError(cmd, f"Clone timed out after {timeout}s")
    if result.returncode != 0:
        raise GitError(cmd, result.stderr)
    run_git("checkout", sha, cwd=dest)
    return dest
