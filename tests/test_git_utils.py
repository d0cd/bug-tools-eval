"""Tests for git subprocess wrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from bugeval.git_utils import (
    GitError,
    clone_at_sha,
    commit_exists,
    get_changed_files,
    get_diff,
    run_git,
)


def _make_completed(
    stdout: str = "", stderr: str = "", returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


class TestRunGit:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _make_completed(stdout="ok\n"),
        )
        assert run_git("status", cwd=Path("/tmp")) == "ok\n"

    def test_failure_raises_git_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _make_completed(
                returncode=1, stderr="fatal: bad"
            ),
        )
        with pytest.raises(GitError, match="fatal: bad"):
            run_git("status", cwd=Path("/tmp"))

    def test_timeout_raises_git_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def raise_timeout(*a: object, **kw: object) -> None:
            raise subprocess.TimeoutExpired(cmd="git", timeout=60)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        with pytest.raises(GitError, match="timed out"):
            run_git("log", cwd=Path("/tmp"))


class TestCommitExists:
    def test_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _make_completed(),
        )
        assert commit_exists("abc123", cwd=Path("/tmp")) is True

    def test_not_exists(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _make_completed(
                returncode=1, stderr="not found"
            ),
        )
        assert commit_exists("abc123", cwd=Path("/tmp")) is False


class TestGetChangedFiles:
    def test_parses_output(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _make_completed(
                stdout="src/main.rs\nsrc/lib.rs\n"
            ),
        )
        files = get_changed_files("a", "b", cwd=Path("/tmp"))
        assert files == ["src/main.rs", "src/lib.rs"]

    def test_empty_output(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _make_completed(stdout=""),
        )
        assert get_changed_files("a", "b", cwd=Path("/tmp")) == []


class TestGetDiff:
    def test_returns_stdout(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _make_completed(stdout="diff output"),
        )
        assert get_diff("a", "b", cwd=Path("/tmp")) == "diff output"


class TestCloneAtSha:
    def test_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        calls: list[list[str]] = []

        def mock_run(
            cmd: list[str], **kw: object,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            return _make_completed()

        monkeypatch.setattr(subprocess, "run", mock_run)
        dest = tmp_path / "repo"
        result = clone_at_sha("https://example.com/r", dest, "abc")
        assert result == dest
        assert calls[0][0] == "git"
        assert "clone" in calls[0]

    def test_clone_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        def raise_timeout(*a: object, **kw: object) -> None:
            raise subprocess.TimeoutExpired(cmd="git", timeout=600)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        with pytest.raises(GitError, match="timed out"):
            clone_at_sha(
                "https://example.com/r", tmp_path / "repo", "abc",
            )
