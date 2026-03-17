"""Shared repo setup and cleanup for agent evaluation modes."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

from bugeval.git_utils import clone_repo, clone_repo_local, is_repo, run_git
from bugeval.models import TestCase

# Per-repo locks prevent concurrent threads from racing to create the same cache
# entry. asyncio.to_thread runs setup_repo_for_case in a thread pool, so we need
# threading.Lock (not asyncio.Lock) here.
_cache_locks: dict[str, threading.Lock] = {}
_cache_locks_mutex = threading.Lock()


def _get_cache_lock(repo: str) -> threading.Lock:
    with _cache_locks_mutex:
        if repo not in _cache_locks:
            _cache_locks[repo] = threading.Lock()
        return _cache_locks[repo]


def get_or_create_cached_repo(repo: str, cache_dir: Path) -> Path:
    """Return path to a cached clone of repo, creating it if absent.

    Thread-safe: uses a per-repo lock so concurrent callers wait rather than
    racing to clone. Detects incomplete caches (directory exists but is not a
    valid git repo) and removes them before re-cloning.
    """
    name = repo.replace("/", "-")
    cache_path = cache_dir / name
    with _get_cache_lock(repo):
        if not is_repo(cache_path):
            # Remove any partial directory left by a previous interrupted clone.
            if cache_path.exists():
                shutil.rmtree(cache_path)
            clone_repo(f"https://github.com/{repo}.git", cache_path)
    return cache_path


def setup_repo_for_case(
    case: TestCase,
    patch_path: Path,
    work_dir: Path,
    cache_dir: Path | None = None,
    apply_patch: bool = False,
) -> Path:
    """Clone repo and checkout base commit. Returns repo directory.

    By default the repo is left at base_commit (the pre-fix state with the bug
    present), which matches what a reviewer would see when a PR is filed. Pass
    apply_patch=True to apply the diff on top, leaving the repo in the post-fix
    state (the old behaviour, now only used in tests).
    """
    repo_dir = work_dir / case.id
    if cache_dir is not None:
        cache_path = get_or_create_cached_repo(case.repo, cache_dir)
        clone_repo_local(cache_path, repo_dir)
    else:
        clone_repo(f"https://github.com/{case.repo}.git", repo_dir)
    run_git("checkout", case.base_commit, cwd=repo_dir)
    if apply_patch:
        run_git("apply", str(patch_path.resolve()), cwd=repo_dir)
    return repo_dir


def cleanup_repo(repo_dir: Path) -> None:
    """Remove the cloned repo directory."""
    shutil.rmtree(repo_dir, ignore_errors=True)
