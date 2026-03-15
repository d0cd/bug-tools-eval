"""Tests for git utilities using real git repos in tmp_path. No mocking."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from bugeval.git_utils import (
    GitError,
    apply_patch_check,
    clone_repo,
    commit_exists,
    format_patch,
    get_changed_files,
    get_diff_stats,
    is_repo,
    run_git,
)


def make_repo(path: Path) -> Path:
    """Initialize a git repo with an initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True
    )
    (path / "file.txt").write_text("initial content\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"], cwd=path, check=True, capture_output=True
    )
    return path


def add_commit(path: Path, filename: str, content: str, message: str) -> str:
    """Add a file and commit it. Returns the commit SHA."""
    (path / filename).write_text(content)
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=path, check=True, capture_output=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


class TestRunGit:
    def test_run_git_success(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        output = run_git("rev-parse", "HEAD", cwd=repo)
        assert len(output.strip()) == 40

    def test_run_git_failure(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        with pytest.raises(GitError):
            run_git("cat-file", "-e", "nonexistentsha1234567890123456789012345678", cwd=repo)

    def test_git_error_contains_command(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        with pytest.raises(GitError) as exc_info:
            run_git("cat-file", "-e", "nonexistentsha1234567890123456789012345678", cwd=repo)
        assert exc_info.value.command[0] == "git"


class TestCommitExists:
    def test_existing_commit(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        sha = run_git("rev-parse", "HEAD", cwd=repo).strip()
        assert commit_exists(sha, repo) is True

    def test_nonexistent_commit(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        assert commit_exists("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", repo) is False


class TestGetChangedFiles:
    def test_returns_changed_file(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = run_git("rev-parse", "HEAD", cwd=repo).strip()
        add_commit(repo, "new.txt", "content\n", "add new file")
        head = run_git("rev-parse", "HEAD", cwd=repo).strip()

        files = get_changed_files(base, head, repo)
        assert "new.txt" in files

    def test_empty_when_same_commit(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        sha = run_git("rev-parse", "HEAD", cwd=repo).strip()

        files = get_changed_files(sha, sha, repo)
        assert files == []

    def test_multiple_files(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = run_git("rev-parse", "HEAD", cwd=repo).strip()
        # Add two files in one commit
        (repo / "a.txt").write_text("a\n")
        (repo / "b.txt").write_text("b\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add two files"], cwd=repo, check=True, capture_output=True
        )
        head = run_git("rev-parse", "HEAD", cwd=repo).strip()

        files = get_changed_files(base, head, repo)
        assert "a.txt" in files
        assert "b.txt" in files

    def test_returns_list_of_strings(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = run_git("rev-parse", "HEAD", cwd=repo).strip()
        add_commit(repo, "x.txt", "x\n", "add x")
        head = run_git("rev-parse", "HEAD", cwd=repo).strip()

        files = get_changed_files(base, head, repo)
        assert isinstance(files, list)
        assert all(isinstance(f, str) for f in files)


class TestGetDiffStats:
    def test_added_lines(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = run_git("rev-parse", "HEAD", cwd=repo).strip()
        add_commit(repo, "new.txt", "line1\nline2\nline3\n", "add new file")
        head = run_git("rev-parse", "HEAD", cwd=repo).strip()

        stats = get_diff_stats(base, head, repo)
        assert stats.lines_added == 3
        assert stats.lines_deleted == 0
        assert stats.files_changed == 1
        assert stats.hunks >= 1

    def test_deleted_lines(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        # Modify initial file
        (repo / "file.txt").write_text("initial content\nmore content\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add more"], cwd=repo, check=True, capture_output=True
        )
        base = run_git("rev-parse", "HEAD", cwd=repo).strip()

        (repo / "file.txt").write_text("initial content\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "remove line"], cwd=repo, check=True, capture_output=True
        )
        head = run_git("rev-parse", "HEAD", cwd=repo).strip()

        stats = get_diff_stats(base, head, repo)
        assert stats.lines_deleted == 1
        assert stats.files_changed == 1


class TestFormatPatch:
    def test_patch_contains_filename(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = run_git("rev-parse", "HEAD", cwd=repo).strip()
        add_commit(repo, "patch_file.txt", "patched content\n", "add patch file")
        head = run_git("rev-parse", "HEAD", cwd=repo).strip()

        patch = format_patch(base, head, repo)
        assert "patch_file.txt" in patch
        assert "patched content" in patch

    def test_patch_is_nonempty(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = run_git("rev-parse", "HEAD", cwd=repo).strip()
        add_commit(repo, "f.txt", "x\n", "add f")
        head = run_git("rev-parse", "HEAD", cwd=repo).strip()

        patch = format_patch(base, head, repo)
        assert len(patch) > 0


class TestApplyPatchCheck:
    def test_valid_patch_applies(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = run_git("rev-parse", "HEAD", cwd=repo).strip()
        add_commit(repo, "apply_file.txt", "new content\n", "add apply file")
        head = run_git("rev-parse", "HEAD", cwd=repo).strip()

        patch_content = format_patch(base, head, repo)
        patch_file = tmp_path / "test.patch"
        patch_file.write_text(patch_content)

        # Reset back to base so patch can be applied
        subprocess.run(["git", "reset", "--hard", base], cwd=repo, capture_output=True, check=True)
        assert apply_patch_check(patch_file, repo) is True

    def test_invalid_patch_fails(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        bad_patch = tmp_path / "bad.patch"
        bad_patch.write_text("this is not a valid patch\n")
        assert apply_patch_check(bad_patch, repo) is False


class TestIsRepo:
    def test_is_repo_true(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        assert is_repo(repo) is True

    def test_is_repo_false(self, tmp_path: Path) -> None:
        not_repo = tmp_path / "not_a_repo"
        not_repo.mkdir()
        assert is_repo(not_repo) is False


class TestTimeouts:
    def test_run_git_timeout(self, tmp_path: Path) -> None:
        """run_git raises GitError on subprocess timeout."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=60),
        ):
            with pytest.raises(GitError, match="timed out"):
                run_git("log", cwd=tmp_path)

    def test_clone_repo_timeout(self, tmp_path: Path) -> None:
        """clone_repo raises GitError on subprocess timeout."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=600),
        ):
            with pytest.raises(GitError, match="timed out"):
                clone_repo("https://example.com/repo.git", tmp_path / "repo")


class TestCloneRepo:
    def test_clone_creates_repo(self, tmp_path: Path) -> None:
        source = make_repo(tmp_path / "source")
        dest = tmp_path / "dest"
        result_path = clone_repo(str(source), dest)
        assert result_path == dest
        assert is_repo(dest)

    def test_clone_copies_commits(self, tmp_path: Path) -> None:
        source = make_repo(tmp_path / "source")
        add_commit(source, "new.txt", "content\n", "add file")
        source_sha = run_git("rev-parse", "HEAD", cwd=source).strip()

        dest = tmp_path / "dest"
        clone_repo(str(source), dest)
        dest_sha = run_git("rev-parse", "HEAD", cwd=dest).strip()
        assert dest_sha == source_sha

    def test_clone_invalid_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(GitError):
            clone_repo("/nonexistent/path/to/repo", tmp_path / "dest")
