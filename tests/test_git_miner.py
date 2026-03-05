"""Tests for git_miner module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from bugeval.git_miner import (
    detect_fix_keywords,
    find_introducing_commit,
    parse_fix_commits,
    score_git_candidate,
)

# --- parse_fix_commits ---

MOCK_GIT_LOG_TWO_COMMITS = (
    "COMMIT_START\n"
    "abc123\n"
    "fix: off by one in parser\n"
    "\n"
    "5\t2\tsrc/parser.rs\n"
    "COMMIT_START\n"
    "def456\n"
    "fix: null pointer in handler\n"
    "\n"
    "3\t1\tsrc/handler.rs\n"
)

MOCK_GIT_LOG_MERGE_NO_NUMSTAT = (
    "COMMIT_START\n"
    "aaa111\n"
    "Merge branch 'feature'\n"
    "parent1 parent2\n"
    "COMMIT_START\n"
    "bbb222\n"
    "fix: race condition\n"
    "\n"
    "10\t5\tsrc/lock.rs\n"
)

MOCK_GIT_LOG_BINARY_FILE = (
    "COMMIT_START\nccc333\nfix: update icon asset\n\n-\t-\tassets/icon.png\n2\t1\tsrc/config.rs\n"
)


def test_parse_fix_commits_two_commits(tmp_path: Path) -> None:
    """Two normal fix commits are both parsed correctly."""
    with patch("bugeval.git_miner.run_git", return_value=MOCK_GIT_LOG_TWO_COMMITS):
        commits = parse_fix_commits(tmp_path, "main", limit=500)
    assert len(commits) == 2
    assert commits[0]["sha"] == "abc123"
    assert commits[1]["sha"] == "def456"
    assert commits[0]["files"] == ["src/parser.rs"]
    assert commits[1]["files"] == ["src/handler.rs"]


def test_parse_fix_commits_merge_no_numstat(tmp_path: Path) -> None:
    """Merge commit with no numstat doesn't swallow the next commit."""
    with patch("bugeval.git_miner.run_git", return_value=MOCK_GIT_LOG_MERGE_NO_NUMSTAT):
        commits = parse_fix_commits(tmp_path, "main", limit=500)
    shas = [c["sha"] for c in commits]
    assert "bbb222" in shas


def test_parse_fix_commits_binary_file_included(tmp_path: Path) -> None:
    """Binary files (- numstat) are included in the files list."""
    with patch("bugeval.git_miner.run_git", return_value=MOCK_GIT_LOG_BINARY_FILE):
        commits = parse_fix_commits(tmp_path, "main", limit=500)
    assert len(commits) == 1
    assert "assets/icon.png" in commits[0]["files"]
    assert "src/config.rs" in commits[0]["files"]


def test_parse_fix_commits_empty_output(tmp_path: Path) -> None:
    """Empty git log returns empty list."""
    with patch("bugeval.git_miner.run_git", return_value=""):
        commits = parse_fix_commits(tmp_path, "main", limit=500)
    assert commits == []


# --- detect_fix_keywords ---


def test_detect_fix_keywords_fix() -> None:
    signals = detect_fix_keywords("fix: off by one in parser")
    assert "keyword:fix" in signals


def test_detect_fix_keywords_issue_ref() -> None:
    signals = detect_fix_keywords("closes #42: memory leak in allocator")
    assert "keyword:issue_ref" in signals


def test_detect_fix_keywords_no_match() -> None:
    signals = detect_fix_keywords("add feature: support new syntax")
    assert signals == []


def test_detect_fix_keywords_both() -> None:
    signals = detect_fix_keywords("fix: closes #99 buffer overflow")
    assert "keyword:fix" in signals
    assert "keyword:issue_ref" in signals


# --- score_git_candidate ---


def test_score_high_confidence() -> None:
    commit = {
        "message": "fix: closes #1 off-by-one",
        "lines_added": 10,
        "lines_deleted": 5,
        "files": ["src/main.rs"],
        "is_merge": True,
    }
    confidence, signals = score_git_candidate(commit, has_introducing=True)
    assert confidence > 0.5


def test_score_low_confidence() -> None:
    commit = {
        "message": "refactor everything",
        "lines_added": 500,
        "lines_deleted": 500,
        "files": ["a.rs", "b.rs", "c.rs", "d.rs"],
        "is_merge": False,
    }
    confidence, signals = score_git_candidate(commit, has_introducing=False)
    assert confidence < 0.3
    assert signals == []


def test_score_capped_at_1() -> None:
    # All six signals fire → sum = 1.0 exactly (the cap value)
    commit = {
        "message": "fix: closes #1 resolve bug",
        "lines_added": 5,
        "lines_deleted": 3,
        "files": ["src/main.rs"],
        "is_merge": True,
    }
    confidence, signals = score_git_candidate(commit, has_introducing=True)
    assert confidence == 1.0


def test_score_small_diff_signal() -> None:
    commit = {
        "message": "no keywords",
        "lines_added": 50,
        "lines_deleted": 30,
        "files": [],
        "is_merge": False,
    }
    import pytest

    confidence, signals = score_git_candidate(commit, has_introducing=False)
    assert "signal:small_diff" in signals
    assert confidence == pytest.approx(0.20)


def test_score_signals_absent_for_large_diff() -> None:
    commit = {
        "message": "no keywords",
        "lines_added": 300,
        "lines_deleted": 100,
        "files": ["a.rs", "b.rs", "c.rs", "d.rs"],
        "is_merge": False,
    }
    confidence, signals = score_git_candidate(commit, has_introducing=False)
    assert "signal:small_diff" not in signals
    assert "signal:few_files" not in signals
    assert confidence == 0.0


def test_score_introducing_signal() -> None:
    commit = {
        "message": "no keywords",
        "lines_added": 300,
        "lines_deleted": 300,
        "files": [],
        "is_merge": False,
    }
    confidence, signals = score_git_candidate(commit, has_introducing=True)
    assert "signal:has_introducing" in signals
    assert "signal:merge_commit" not in signals


# --- find_introducing_commit ---


def _make_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "T"], cwd=path, check=True, capture_output=True)


def _commit(repo: Path, filename: str, content: str, message: str) -> str:
    (repo / filename).write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_find_introducing_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    introducing_sha = _commit(repo, "main.rs", "fn foo() { bug }\n", "add bug")
    fix_sha = _commit(repo, "main.rs", "fn foo() { ok }\n", "fix: fix bug")

    result = find_introducing_commit(fix_sha, ["main.rs"], repo)
    assert result == introducing_sha


def test_find_introducing_commit_no_overlap(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    _commit(repo, "other.rs", "fn bar() {}\n", "add other")
    fix_sha = _commit(repo, "main.rs", "fn foo() { ok }\n", "fix: new file")

    result = find_introducing_commit(fix_sha, ["main.rs"], repo)
    # main.rs was never modified before fix, so no introducing commit found
    assert result is None
