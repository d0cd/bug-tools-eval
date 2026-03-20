"""Tests for clean_cases module."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from bugeval.io import load_case, save_case, save_checkpoint
from bugeval.models import CaseKind


def _make_pr(
    number: int = 1,
    title: str = "Add feature",
    body: str = "New stuff",
    labels: list[dict[str, str]] | None = None,
    additions: int = 50,
    deletions: int = 10,
    files: list[dict[str, str]] | None = None,
    merged_at: str = "2024-03-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "body": body,
        "labels": labels or [],
        "additions": additions,
        "deletions": deletions,
        "changedFiles": len(files) if files else 1,
        "files": files or [{"path": "src/main.rs"}],
        "mergeCommit": {"oid": f"sha{number}"},
        "mergedAt": merged_at,
        "author": {"login": "dev"},
    }


class TestFetchCleanPrs:
    def test_filters_fix_signals(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from bugeval.clean_cases import fetch_clean_prs

        prs = [
            _make_pr(1, title="Fix overflow bug"),
            _make_pr(2, title="Add feature X"),
            _make_pr(3, title="Bug in parser"),
            _make_pr(4, title="Refactor auth module"),
            _make_pr(5, labels=[{"name": "bug"}]),
        ]

        def mock_run(
            cmd: list[str], **kw: Any,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout=json.dumps(prs), stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = fetch_clean_prs("org/repo", count=10, since="2023-01-01")
        numbers = [pr["number"] for pr in result]
        assert 1 not in numbers  # "Fix" keyword
        assert 3 not in numbers  # "Bug" keyword
        assert 5 not in numbers  # bug label
        assert 2 in numbers
        assert 4 in numbers

    def test_filters_size(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from bugeval.clean_cases import fetch_clean_prs

        prs = [
            _make_pr(1, additions=1, deletions=0),  # too small (1 < 3)
            _make_pr(2, additions=50, deletions=10),  # ok (60)
            _make_pr(3, additions=800, deletions=300),  # too large (1100)
        ]

        def mock_run(
            cmd: list[str], **kw: Any,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout=json.dumps(prs), stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = fetch_clean_prs("org/repo", count=10, since="2023-01-01")
        numbers = [pr["number"] for pr in result]
        assert 1 not in numbers
        assert 3 not in numbers
        assert 2 in numbers

    def test_filters_non_code(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from bugeval.clean_cases import fetch_clean_prs

        prs = [
            _make_pr(1, files=[{"path": "README.md"}]),
            _make_pr(
                2, files=[{"path": "src/lib.rs"}, {"path": "README.md"}],
            ),
        ]

        def mock_run(
            cmd: list[str], **kw: Any,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout=json.dumps(prs), stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = fetch_clean_prs("org/repo", count=10, since="2023-01-01")
        numbers = [pr["number"] for pr in result]
        assert 1 not in numbers
        assert 2 in numbers


class TestCheckNotSubsequentlyFixed:
    def test_clean(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from bugeval.clean_cases import check_not_subsequently_fixed

        pr = _make_pr(10)

        def mock_run(
            cmd: list[str], **kw: Any,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout=json.dumps([]), stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        assert check_not_subsequently_fixed("org/repo", pr) is True

    def test_dirty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from bugeval.clean_cases import check_not_subsequently_fixed

        pr = _make_pr(10)
        fix_prs = [
            {
                "number": 15,
                "title": "Fix regression from #10",
                "body": "",
                "labels": [],
            },
        ]

        def mock_run(
            cmd: list[str], **kw: Any,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout=json.dumps(fix_prs), stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        assert check_not_subsequently_fixed("org/repo", pr) is False


class TestBuildCleanCase:
    def test_correct_fields(self) -> None:
        from bugeval.clean_cases import build_clean_case

        pr = _make_pr(
            42,
            title="Refactor auth",
            body="Clean up auth module",
            additions=30,
            deletions=10,
            files=[{"path": "src/auth.rs"}, {"path": "src/util.rs"}],
        )
        case = build_clean_case("org/repo", pr, "repo-clean-001")
        assert case.id == "repo-clean-001"
        assert case.repo == "org/repo"
        assert case.kind == CaseKind.clean
        assert case.truth is None
        assert case.introducing_pr_number == 42
        assert case.introducing_pr_title == "Refactor auth"
        assert case.introducing_pr_body == "Clean up auth module"
        assert case.base_commit != ""
        assert case.stats is not None
        assert case.stats.lines_added == 30
        assert case.stats.lines_deleted == 10
        assert case.stats.files_changed == 2
        assert case.pr_size == "small"
        assert case.language == "rust"


class TestMineCleanCasesCheckpoint:
    def test_checkpoint_resume(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        from bugeval.clean_cases import mine_clean_cases

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir(parents=True)
        save_checkpoint({"10"}, repo_dir / ".clean_checkpoint.json")

        prs = [
            _make_pr(10, title="Already done"),
            _make_pr(20, title="New feature"),
        ]

        call_log: list[str] = []

        def mock_run(
            cmd: list[str], **kw: Any,
        ) -> subprocess.CompletedProcess[str]:
            cmd_str = " ".join(cmd)
            call_log.append(cmd_str)
            if "pr" in cmd and "list" in cmd and "--search" in cmd:
                search_idx = cmd.index("--search")
                search_val = cmd[search_idx + 1]
                if "#" in search_val:
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0,
                        stdout=json.dumps([]), stderr="",
                    )
            if "pr" in cmd and "list" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout=json.dumps(prs), stderr="",
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0,
                stdout=json.dumps([]), stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        cases = mine_clean_cases(
            "org/repo", count=5, output_dir=tmp_path, since="2023-01-01",
        )
        case_numbers = [c.introducing_pr_number for c in cases]
        assert 10 not in case_numbers
        assert 20 in case_numbers


class TestCleanCaseRoundTrip:
    def test_save_load(self, tmp_path: Path) -> None:
        from bugeval.clean_cases import build_clean_case

        pr = _make_pr(7, title="Add caching", additions=20, deletions=5)
        case = build_clean_case("org/repo", pr, "repo-clean-001")
        path = tmp_path / "repo-clean-001.yaml"
        save_case(case, path)
        loaded = load_case(path)
        assert loaded.kind == CaseKind.clean
        assert loaded.truth is None
        assert loaded.id == "repo-clean-001"
        assert loaded.introducing_pr_number == 7
