"""Tests for the mine module."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from bugeval.io import load_case
from bugeval.mine import (
    GhError,
    _compute_pr_size,
    _detect_language,
    _is_non_code_only,
    build_case_from_pr,
    build_pr_relations,
    detect_cross_references,
    detect_reverts,
    extract_closing_issues,
    extract_referenced_issues,
    has_fix_signal,
    mine_repo,
    run_gh,
)
from bugeval.models import CaseKind, PRRelation


class TestHasFixSignal:
    def test_fix_in_title(self) -> None:
        assert has_fix_signal("Fix overflow bug", "", []) is True

    def test_bug_label(self) -> None:
        assert has_fix_signal("Add feature", "", ["bug"]) is True

    def test_hotfix_label(self) -> None:
        assert has_fix_signal("Update", "", ["hotfix"]) is True

    def test_no_signal(self) -> None:
        assert has_fix_signal("Add new feature", "Nice stuff", []) is False

    def test_fix_in_body(self) -> None:
        assert has_fix_signal("Update", "This fixes the issue", []) is True

    def test_revert_keyword(self) -> None:
        assert has_fix_signal("Revert bad change", "", []) is True

    def test_case_insensitive(self) -> None:
        assert has_fix_signal("BUG in parser", "", []) is True

    def test_regression_label(self) -> None:
        assert has_fix_signal("", "", ["regression"]) is True


class TestExtractClosingIssues:
    def test_fixes_hash(self) -> None:
        assert extract_closing_issues("Fixes #42") == [42]

    def test_closes_hash(self) -> None:
        assert extract_closing_issues("closes #100") == [100]

    def test_resolves_hash(self) -> None:
        assert extract_closing_issues("Resolves #7") == [7]

    def test_multiple(self) -> None:
        text = "Fixes #1 and closes #2"
        assert extract_closing_issues(text) == [1, 2]

    def test_no_match(self) -> None:
        assert extract_closing_issues("No issues here") == []

    def test_fixed_variant(self) -> None:
        assert extract_closing_issues("Fixed #99") == [99]


class TestExtractReferencedIssues:
    def test_see_hash(self) -> None:
        assert extract_referenced_issues("see #42") == [42]

    def test_related_to(self) -> None:
        assert extract_referenced_issues("related to #10") == [10]

    def test_followup_to(self) -> None:
        assert extract_referenced_issues("followup to #5") == [5]

    def test_no_match(self) -> None:
        assert extract_referenced_issues("Nothing here") == []


class TestIsNonCodeOnly:
    def test_all_docs(self) -> None:
        assert _is_non_code_only(["README.md", "CHANGELOG.md"]) is True

    def test_ci_files(self) -> None:
        assert _is_non_code_only([".github/workflows/ci.yml"]) is True

    def test_code_files(self) -> None:
        assert _is_non_code_only(["src/main.rs"]) is False

    def test_mixed(self) -> None:
        assert _is_non_code_only(["README.md", "src/lib.rs"]) is False

    def test_empty(self) -> None:
        assert _is_non_code_only([]) is True

    def test_toml_only(self) -> None:
        assert _is_non_code_only(["Cargo.toml"]) is True

    def test_lock_only(self) -> None:
        assert _is_non_code_only(["Cargo.lock"]) is True


class TestComputePrSize:
    def test_tiny(self) -> None:
        assert _compute_pr_size(3, 2) == "tiny"

    def test_small(self) -> None:
        assert _compute_pr_size(20, 10) == "small"

    def test_medium(self) -> None:
        assert _compute_pr_size(100, 50) == "medium"

    def test_large(self) -> None:
        assert _compute_pr_size(200, 100) == "large"

    def test_xl(self) -> None:
        assert _compute_pr_size(400, 200) == "xl"

    def test_boundary_tiny_small(self) -> None:
        assert _compute_pr_size(5, 4) == "tiny"
        assert _compute_pr_size(5, 5) == "small"

    def test_boundary_small_medium(self) -> None:
        assert _compute_pr_size(25, 24) == "small"
        assert _compute_pr_size(25, 25) == "medium"


class TestDetectLanguage:
    def test_rust(self) -> None:
        assert _detect_language(["src/main.rs", "src/lib.rs"]) == "rust"

    def test_python(self) -> None:
        assert _detect_language(["app.py", "tests/test.py"]) == "python"

    def test_mixed_majority_wins(self) -> None:
        files = ["a.rs", "b.rs", "c.py"]
        assert _detect_language(files) == "rust"

    def test_unknown_for_no_code(self) -> None:
        assert _detect_language(["README.md"]) == "unknown"

    def test_empty(self) -> None:
        assert _detect_language([]) == "unknown"

    def test_typescript(self) -> None:
        assert _detect_language(["app.ts", "component.tsx"]) == "typescript"

    def test_leo(self) -> None:
        assert _detect_language(["main.leo"]) == "leo"


class TestDetectCrossReferences:
    def test_finds_references(self) -> None:
        prs = [
            {"number": 1, "title": "Fix for #2", "body": ""},
            {"number": 2, "title": "Original", "body": ""},
        ]
        refs = detect_cross_references(prs)
        assert refs == {1: [2]}

    def test_ignores_self_reference(self) -> None:
        prs = [
            {"number": 1, "title": "See #1", "body": ""},
        ]
        assert detect_cross_references(prs) == {}

    def test_ignores_external_numbers(self) -> None:
        prs = [
            {"number": 1, "title": "See #999", "body": ""},
        ]
        assert detect_cross_references(prs) == {}

    def test_body_references(self) -> None:
        prs = [
            {"number": 10, "title": "Fix", "body": "Related to #20"},
            {"number": 20, "title": "Original", "body": ""},
        ]
        refs = detect_cross_references(prs)
        assert refs == {10: [20]}


class TestDetectReverts:
    def test_finds_revert(self) -> None:
        prs = [
            {"number": 5, "title": "Revert #3"},
            {"number": 3, "title": "Bad change"},
        ]
        assert detect_reverts(prs) == {5: 3}

    def test_no_reverts(self) -> None:
        prs = [{"number": 1, "title": "Normal PR"}]
        assert detect_reverts(prs) == {}

    def test_case_insensitive(self) -> None:
        prs = [{"number": 10, "title": "REVERT PR #7"}]
        assert detect_reverts(prs) == {10: 7}


class TestBuildPrRelations:
    def test_builds_related(self) -> None:
        prs_by_num: dict[int, dict[str, Any]] = {
            1: {
                "number": 1, "title": "Fix",
                "mergeCommit": {"oid": "aaa"},
                "mergedAt": "2024-01-01",
                "author": {"login": "alice"},
            },
            2: {
                "number": 2, "title": "Original",
                "mergeCommit": {"oid": "bbb"},
                "mergedAt": "2024-01-02",
                "author": {"login": "bob"},
            },
        }
        cross = {1: [2]}
        reverts: dict[int, int] = {}
        rels = build_pr_relations(1, prs_by_num, cross, reverts)
        assert len(rels) == 1
        assert rels[0].pr_number == 2
        assert rels[0].role == "related"
        assert rels[0].commit == "bbb"

    def test_revert_role(self) -> None:
        prs_by_num: dict[int, dict[str, Any]] = {
            5: {
                "number": 5, "title": "Revert #3",
                "mergeCommit": {"oid": "eee"},
                "mergedAt": "", "author": {"login": "x"},
            },
            3: {
                "number": 3, "title": "Bad",
                "mergeCommit": {"oid": "ccc"},
                "mergedAt": "", "author": {"login": "y"},
            },
        }
        cross = {5: [3]}
        reverts = {5: 3}
        rels = build_pr_relations(5, prs_by_num, cross, reverts)
        assert rels[0].role == "revert"

    def test_missing_pr_returns_empty(self) -> None:
        rels = build_pr_relations(999, {}, {}, {})
        assert rels == []


class TestFixPrInRelatedPrs:
    def test_fix_pr_in_related_prs(self) -> None:
        """Verify fix PR is added to related_prs with role='full_fix'."""
        pr: dict[str, Any] = {
            "number": 42,
            "title": "Fix overflow",
            "body": "",
            "mergeCommit": {"oid": "abc123"},
            "additions": 15,
            "deletions": 3,
            "changedFiles": 1,
            "files": [{"path": "src/main.rs"}],
            "labels": [],
            "mergedAt": "2024-07-10",
            "author": {"login": "alice"},
        }
        case = build_case_from_pr(
            repo="org/repo", pr=pr, case_id="r-001",
        )
        fix_rels = [
            r for r in case.related_prs if r.role == "full_fix"
        ]
        assert len(fix_rels) == 1
        assert fix_rels[0].pr_number == 42
        assert fix_rels[0].commit == "abc123"
        assert fix_rels[0].author == "alice"

    def test_fix_pr_is_first_relation(self) -> None:
        """Fix PR relation should be first in the list."""
        pr: dict[str, Any] = {
            "number": 10,
            "title": "Fix",
            "body": "",
            "mergeCommit": {"oid": "sha1"},
            "additions": 10,
            "deletions": 5,
            "changedFiles": 1,
            "files": [{"path": "x.rs"}],
            "labels": [],
            "mergedAt": "",
            "author": {"login": "dev"},
        }
        existing_rel = PRRelation(
            pr_number=5, role="related", commit="other",
        )
        case = build_case_from_pr(
            repo="org/repo", pr=pr, case_id="r-001",
            relations=[existing_rel],
        )
        assert case.related_prs[0].role == "full_fix"
        assert case.related_prs[0].pr_number == 10
        assert case.related_prs[1].pr_number == 5


class TestBuildCaseFromPr:
    def test_basic_construction(self) -> None:
        pr: dict[str, Any] = {
            "number": 42,
            "title": "Fix overflow",
            "body": "Fixes #10",
            "mergeCommit": {"oid": "abc123"},
            "additions": 15,
            "deletions": 3,
            "changedFiles": 2,
            "files": [
                {"path": "src/main.rs"},
                {"path": "src/lib.rs"},
            ],
            "labels": [{"name": "bug"}],
            "mergedAt": "2024-07-10",
            "author": {"login": "alice"},
        }
        case = build_case_from_pr(
            repo="ProvableHQ/snarkVM",
            pr=pr,
            case_id="snarkVM-001",
        )
        assert case.id == "snarkVM-001"
        assert case.repo == "ProvableHQ/snarkVM"
        assert case.kind == CaseKind.bug
        assert case.language == "rust"
        assert case.fix_commit == "abc123"
        assert case.fix_pr_number == 42
        assert case.fix_pr_title == "Fix overflow"
        assert case.linked_issues == [10]
        assert case.issue_labels == ["bug"]
        assert case.pr_size == "small"
        assert case.stats is not None
        assert case.stats.lines_added == 15

    def test_with_graphql_data(self) -> None:
        pr: dict[str, Any] = {
            "number": 1,
            "title": "Fix",
            "body": "",
            "mergeCommit": {"oid": "sha1"},
            "additions": 10,
            "deletions": 5,
            "changedFiles": 1,
            "files": [{"path": "a.py"}],
            "labels": [],
            "mergedAt": "2024-01-01",
            "author": {"login": "dev"},
        }
        gql: dict[str, Any] = {
            "commits": {
                "nodes": [
                    {"commit": {"oid": "c1", "message": "fix: thing"}}
                ],
            },
            "reviews": {
                "nodes": [{"body": "LGTM", "state": "APPROVED"}],
            },
            "reviewThreads": {
                "nodes": [
                    {
                        "comments": {
                            "nodes": [{"body": "nit: spacing"}],
                        },
                    },
                ],
            },
            "comments": {
                "nodes": [{"body": "Thanks!"}],
            },
            "closingIssuesReferences": {
                "nodes": [
                    {
                        "number": 99,
                        "body": "Bug report",
                        "labels": {"nodes": [{"name": "critical"}]},
                    },
                ],
            },
        }
        case = build_case_from_pr(
            repo="org/repo", pr=pr, case_id="repo-001",
            graphql_data=gql,
        )
        assert case.fix_pr_commit_messages == ["fix: thing"]
        assert any("LGTM" in c for c in case.fix_pr_review_comments)
        assert any("APPROVED" in c for c in case.fix_pr_review_comments)
        assert "nit: spacing" in case.fix_pr_review_comments
        assert case.fix_pr_discussion_comments == ["Thanks!"]
        assert 99 in case.linked_issues
        assert case.issue_bodies[99] == "Bug report"
        assert "critical" in case.issue_labels

    def test_with_issue_data(self) -> None:
        pr: dict[str, Any] = {
            "number": 1,
            "title": "Fix",
            "body": "fixes #5",
            "mergeCommit": {"oid": "sha"},
            "additions": 10,
            "deletions": 5,
            "changedFiles": 1,
            "files": [{"path": "x.rs"}],
            "labels": [],
            "mergedAt": "",
            "author": {"login": "dev"},
        }
        issue_data = {
            5: {"body": "Something broke", "labels": [{"name": "p0"}]},
        }
        case = build_case_from_pr(
            repo="org/repo", pr=pr, case_id="r-001",
            issue_data=issue_data,
        )
        assert case.issue_bodies[5] == "Something broke"
        assert "p0" in case.issue_labels

    def test_with_relations(self) -> None:
        pr: dict[str, Any] = {
            "number": 1,
            "title": "Fix",
            "body": "",
            "mergeCommit": {"oid": "sha"},
            "additions": 10,
            "deletions": 5,
            "changedFiles": 1,
            "files": [{"path": "x.rs"}],
            "labels": [],
            "mergedAt": "",
            "author": {"login": "dev"},
        }
        rels = [
            PRRelation(pr_number=2, role="related", commit="abc"),
        ]
        case = build_case_from_pr(
            repo="org/repo", pr=pr, case_id="r-001",
            relations=rels,
        )
        # Fix PR (full_fix) is auto-prepended + the explicit relation
        assert len(case.related_prs) == 2
        assert case.related_prs[0].role == "full_fix"
        assert case.related_prs[0].pr_number == 1
        assert case.related_prs[1].pr_number == 2


class TestBuildCaseRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        from bugeval.io import load_case, save_case

        pr: dict[str, Any] = {
            "number": 7,
            "title": "Fix bug",
            "body": "Fixes #3",
            "mergeCommit": {"oid": "deadbeef"},
            "additions": 20,
            "deletions": 5,
            "changedFiles": 2,
            "files": [
                {"path": "src/main.rs"},
                {"path": "src/util.rs"},
            ],
            "labels": [{"name": "bug"}],
            "mergedAt": "2024-08-01",
            "author": {"login": "dev"},
        }
        case = build_case_from_pr(
            repo="org/repo", pr=pr, case_id="repo-001",
        )
        path = tmp_path / "repo-001.yaml"
        save_case(case, path)
        loaded = load_case(path)
        assert loaded.id == "repo-001"
        assert loaded.fix_commit == "deadbeef"
        assert loaded.linked_issues == [3]
        assert loaded.pr_size == "small"
        assert loaded.stats is not None
        assert loaded.stats.lines_added == 20


class TestRunGh:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr="",
            ),
        )
        assert run_gh("pr", "list") == "ok"

    def test_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="err",
            ),
        )
        with pytest.raises(GhError, match="err"):
            run_gh("pr", "list")

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_timeout(*a: object, **kw: object) -> None:
            raise subprocess.TimeoutExpired(cmd="gh", timeout=60)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        with pytest.raises(GhError, match="timed out"):
            run_gh("pr", "list")


class TestMineRepo:
    def test_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Test mine_repo with mocked gh calls."""
        pr_list_data = [
            {
                "number": 42,
                "title": "Fix overflow bug",
                "body": "Fixes #10",
                "mergeCommit": {"oid": "abc123"},
                "additions": 20,
                "deletions": 5,
                "changedFiles": 2,
                "files": [
                    {"path": "src/main.rs"},
                    {"path": "src/lib.rs"},
                ],
                "labels": [{"name": "bug"}],
                "mergedAt": "2024-07-10",
                "author": {"login": "alice"},
                "commits": [],
                "reviewDecision": "APPROVED",
                "statusCheckRollup": [],
                "baseRefName": "main",
                "headRefName": "fix-overflow",
            },
        ]
        graphql_response = {
            "data": {
                "repository": {
                    "pr_42": {
                        "number": 42,
                        "title": "Fix overflow bug",
                        "body": "Fixes #10",
                        "mergedAt": "2024-07-10",
                        "mergeCommit": {"oid": "abc123"},
                        "author": {"login": "alice"},
                        "commits": {
                            "nodes": [
                                {
                                    "commit": {
                                        "oid": "c1",
                                        "message": "fix: overflow",
                                    },
                                },
                            ],
                        },
                        "reviews": {"nodes": []},
                        "reviewThreads": {"nodes": []},
                        "comments": {"nodes": []},
                        "closingIssuesReferences": {"nodes": []},
                    },
                },
            },
        }
        issue_data = {
            "number": 10,
            "title": "Overflow",
            "body": "Counter overflows",
            "labels": [{"name": "bug"}],
        }

        call_count = {"n": 0}

        def mock_run(
            cmd: list[str], **kw: Any,
        ) -> subprocess.CompletedProcess[str]:
            call_count["n"] += 1
            if "pr" in cmd and "list" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout=json.dumps(pr_list_data), stderr="",
                )
            if "graphql" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout=json.dumps(graphql_response), stderr="",
                )
            if "issue" in cmd and "view" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout=json.dumps(issue_data), stderr="",
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="unknown",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)

        cases = mine_repo(
            repo="ProvableHQ/snarkVM",
            limit=200,
            since="2023-01-01",
            output_dir=tmp_path,
        )
        assert len(cases) == 1
        assert cases[0].id == "snarkVM-001"
        assert cases[0].fix_pr_number == 42

        # Verify file was written
        case_file = tmp_path / "snarkVM" / "snarkVM-001.yaml"
        assert case_file.exists()
        loaded = load_case(case_file)
        assert loaded.id == "snarkVM-001"

        # Verify checkpoint
        ckpt = tmp_path / "snarkVM" / ".mine_checkpoint.json"
        assert ckpt.exists()

    def test_checkpoint_skips_done(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Already-checkpointed PRs are skipped."""
        from bugeval.io import save_checkpoint

        repo_dir = tmp_path / "snarkVM"
        repo_dir.mkdir(parents=True)
        save_checkpoint({"42"}, repo_dir / ".mine_checkpoint.json")

        pr_list_data = [
            {
                "number": 42,
                "title": "Fix overflow bug",
                "body": "",
                "mergeCommit": {"oid": "abc"},
                "additions": 20,
                "deletions": 5,
                "changedFiles": 1,
                "files": [{"path": "src/main.rs"}],
                "labels": [{"name": "bug"}],
                "mergedAt": "2024-01-01",
                "author": {"login": "x"},
                "commits": [],
                "reviewDecision": "",
                "statusCheckRollup": [],
                "baseRefName": "main",
                "headRefName": "fix",
            },
        ]

        def mock_run(
            cmd: list[str], **kw: Any,
        ) -> subprocess.CompletedProcess[str]:
            if "pr" in cmd and "list" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout=json.dumps(pr_list_data), stderr="",
                )
            if "graphql" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout=json.dumps({"data": {"repository": {}}}),
                    stderr="",
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="[]", stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)

        cases = mine_repo(
            repo="ProvableHQ/snarkVM",
            limit=200,
            since="2023-01-01",
            output_dir=tmp_path,
        )
        assert len(cases) == 0
