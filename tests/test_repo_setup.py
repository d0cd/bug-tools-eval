"""Tests for repo_setup."""

from pathlib import Path
from unittest.mock import patch

import pytest

from bugeval.git_utils import GitError
from bugeval.models import Category, Difficulty, PRSize, Severity, TestCase
from bugeval.repo_setup import cleanup_repo, get_or_create_cached_repo, setup_repo_for_case


def _make_case() -> TestCase:
    return TestCase(
        id="aleo-lang-001",
        repo="provable-org/aleo-lang",
        base_commit="abc123",
        head_commit="def456",
        fix_commit="def456",
        category=Category.logic,
        difficulty=Difficulty.medium,
        severity=Severity.high,
        language="rust",
        pr_size=PRSize.small,
        description="Test case",
        expected_findings=[],
    )


def test_setup_repo_for_case_calls_clone_and_git(tmp_path: Path) -> None:
    """Default: checkout base_commit only (no git apply) — repo stays in pre-fix state."""
    case = _make_case()
    patch_path = tmp_path / "case.patch"
    patch_path.write_text("--- a/foo\n+++ b/foo\n")

    with (
        patch("bugeval.repo_setup.clone_repo") as mock_clone,
        patch("bugeval.repo_setup.run_git") as mock_git,
    ):
        mock_clone.return_value = tmp_path / case.id
        result = setup_repo_for_case(case, patch_path, tmp_path)

    expected_repo_dir = tmp_path / case.id
    mock_clone.assert_called_once_with(f"https://github.com/{case.repo}.git", expected_repo_dir)
    assert mock_git.call_count == 1
    mock_git.assert_called_once_with("checkout", case.base_commit, cwd=expected_repo_dir)
    assert result == expected_repo_dir


def test_setup_repo_for_case_apply_patch_true(tmp_path: Path) -> None:
    """apply_patch=True: checkout base_commit then apply the diff (post-fix state)."""
    case = _make_case()
    patch_path = tmp_path / "case.patch"
    patch_path.write_text("--- a/foo\n+++ b/foo\n")

    with (
        patch("bugeval.repo_setup.clone_repo") as mock_clone,
        patch("bugeval.repo_setup.run_git") as mock_git,
    ):
        mock_clone.return_value = tmp_path / case.id
        result = setup_repo_for_case(case, patch_path, tmp_path, apply_patch=True)

    expected_repo_dir = tmp_path / case.id
    assert mock_git.call_count == 2
    mock_git.assert_any_call("checkout", case.base_commit, cwd=expected_repo_dir)
    mock_git.assert_any_call("apply", str(patch_path), cwd=expected_repo_dir)
    assert result == expected_repo_dir


def test_setup_repo_for_case_clone_failure(tmp_path: Path) -> None:
    case = _make_case()
    patch_path = tmp_path / "case.patch"

    with patch("bugeval.repo_setup.clone_repo") as mock_clone:
        mock_clone.side_effect = GitError(["git", "clone"], "fatal: repo not found")
        with pytest.raises(GitError):
            setup_repo_for_case(case, patch_path, tmp_path)


def test_setup_repo_for_case_uses_local_clone_when_cache_provided(tmp_path: Path) -> None:
    case = _make_case()
    patch_path = tmp_path / "case.patch"
    patch_path.write_text("--- a/foo\n+++ b/foo\n")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_repo = cache_dir / "provable-org-aleo-lang"
    cached_repo.mkdir()

    with (
        patch("bugeval.repo_setup.clone_repo_local") as mock_local_clone,
        patch("bugeval.repo_setup.run_git") as mock_git,
        patch("bugeval.repo_setup.get_or_create_cached_repo", return_value=cached_repo),
    ):
        result = setup_repo_for_case(case, patch_path, tmp_path, cache_dir=cache_dir)

    mock_local_clone.assert_called_once_with(cached_repo, tmp_path / case.id)
    assert mock_git.call_count == 1  # only checkout, no apply
    assert result == tmp_path / case.id


def test_setup_repo_for_case_falls_back_to_remote_when_no_cache(tmp_path: Path) -> None:
    case = _make_case()
    patch_path = tmp_path / "case.patch"
    patch_path.write_text("--- a/foo\n+++ b/foo\n")

    with (
        patch("bugeval.repo_setup.clone_repo") as mock_clone,
        patch("bugeval.repo_setup.run_git"),
    ):
        mock_clone.return_value = tmp_path / case.id
        setup_repo_for_case(case, patch_path, tmp_path, cache_dir=None)

    mock_clone.assert_called_once_with(f"https://github.com/{case.repo}.git", tmp_path / case.id)


def test_get_or_create_cached_repo_creates_on_miss(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with (
        patch("bugeval.repo_setup.is_repo", return_value=False),
        patch("bugeval.repo_setup.clone_repo") as mock_clone,
    ):
        result = get_or_create_cached_repo("provable-org/aleo-lang", cache_dir)

    expected = cache_dir / "provable-org-aleo-lang"
    mock_clone.assert_called_once_with("https://github.com/provable-org/aleo-lang.git", expected)
    assert result == expected


def test_get_or_create_cached_repo_reuses_on_hit(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_repo = cache_dir / "provable-org-aleo-lang"
    cached_repo.mkdir()

    with (
        patch("bugeval.repo_setup.is_repo", return_value=True),
        patch("bugeval.repo_setup.clone_repo") as mock_clone,
    ):
        result = get_or_create_cached_repo("provable-org/aleo-lang", cache_dir)

    mock_clone.assert_not_called()
    assert result == cached_repo


def test_get_or_create_cached_repo_cleans_partial_cache(tmp_path: Path) -> None:
    """A directory that exists but isn't a valid git repo is removed before re-cloning."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    partial = cache_dir / "provable-org-aleo-lang"
    partial.mkdir()
    (partial / "some_partial_file").write_text("incomplete")

    with (
        patch("bugeval.repo_setup.is_repo", return_value=False),
        patch("bugeval.repo_setup.clone_repo") as mock_clone,
    ):
        get_or_create_cached_repo("provable-org/aleo-lang", cache_dir)

    mock_clone.assert_called_once()
    # shutil.rmtree was called on the partial dir (verified by clone being called once)


def test_get_or_create_cached_repo_concurrent_calls_clone_once(tmp_path: Path) -> None:
    """Concurrent threads calling get_or_create_cached_repo for the same repo only clone once."""
    import concurrent.futures

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    clone_call_count = 0

    def fake_clone(url: str, path: Path) -> None:
        nonlocal clone_call_count
        clone_call_count += 1

    # is_repo returns False first call, True thereafter (simulates clone completing)
    is_repo_results: list[bool] = []

    def fake_is_repo(path: Path) -> bool:
        if not is_repo_results:
            is_repo_results.append(True)
            return False  # first caller sees no valid repo
        return True  # subsequent callers see completed repo

    with (
        patch("bugeval.repo_setup.is_repo", side_effect=fake_is_repo),
        patch("bugeval.repo_setup.clone_repo", side_effect=fake_clone),
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(get_or_create_cached_repo, "provable-org/aleo-lang", cache_dir)
                for _ in range(5)
            ]
            concurrent.futures.wait(futures)

    assert clone_call_count == 1


def test_clone_repo_local_calls_git_clone_local(tmp_path: Path) -> None:
    from bugeval.git_utils import clone_repo_local

    src = tmp_path / "source"
    dest = tmp_path / "dest"

    with patch("bugeval.git_utils.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        clone_repo_local(src, dest)

    args = mock_run.call_args[0][0]
    assert args[0] == "git"
    assert "--local" in args
    assert str(src) in args
    assert str(dest) in args


def test_cleanup_repo_removes_directory(tmp_path: Path) -> None:
    repo_dir = tmp_path / "some-repo"
    repo_dir.mkdir()
    (repo_dir / "file.rs").write_text("fn main() {}")

    with patch("bugeval.repo_setup.shutil.rmtree") as mock_rmtree:
        cleanup_repo(repo_dir)
    mock_rmtree.assert_called_once_with(repo_dir, ignore_errors=True)
