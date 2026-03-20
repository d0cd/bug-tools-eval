"""Tests for blame module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from bugeval.blame import (
    blame_cases,
    file_level_fallback,
    find_introducing_commit,
    parse_diff_added_lines,
    parse_diff_deleted_lines,
    populate_blame,
    resolve_introducing_pr,
    run_blame,
    walk_merge_commit,
)
from bugeval.models import CaseKind, GroundTruth, TestCase

# --- parse_diff_deleted_lines ---


class TestParseDiffDeletedLines:
    def test_single_file_deletions(self) -> None:
        diff = (
            "diff --git a/src/foo.rs b/src/foo.rs\n"
            "index abc..def 100644\n"
            "--- a/src/foo.rs\n"
            "+++ b/src/foo.rs\n"
            "@@ -10,4 +10,3 @@ fn main() {\n"
            "     let x = 1;\n"
            "-    let y = 2;\n"
            "-    let z = 3;\n"
            "+    let y = 3;\n"
            "     let w = 4;\n"
        )
        result = parse_diff_deleted_lines(diff)
        assert result == {"src/foo.rs": [11, 12]}

    def test_multiple_files(self) -> None:
        diff = (
            "diff --git a/a.rs b/a.rs\n"
            "--- a/a.rs\n"
            "+++ b/a.rs\n"
            "@@ -5,3 +5,2 @@\n"
            "     ok;\n"
            "-    removed;\n"
            "     end;\n"
            "diff --git a/b.rs b/b.rs\n"
            "--- a/b.rs\n"
            "+++ b/b.rs\n"
            "@@ -20,3 +20,2 @@\n"
            "     start;\n"
            "-    gone;\n"
            "     finish;\n"
        )
        result = parse_diff_deleted_lines(diff)
        assert result == {"a.rs": [6], "b.rs": [21]}

    def test_no_deletions(self) -> None:
        diff = (
            "diff --git a/a.rs b/a.rs\n"
            "--- a/a.rs\n"
            "+++ b/a.rs\n"
            "@@ -5,2 +5,3 @@\n"
            "     ok;\n"
            "+    added;\n"
            "     end;\n"
        )
        result = parse_diff_deleted_lines(diff)
        assert result == {}

    def test_empty_diff(self) -> None:
        assert parse_diff_deleted_lines("") == {}

    def test_multiple_hunks(self) -> None:
        diff = (
            "diff --git a/x.rs b/x.rs\n"
            "--- a/x.rs\n"
            "+++ b/x.rs\n"
            "@@ -3,3 +3,2 @@\n"
            "     a;\n"
            "-    b;\n"
            "     c;\n"
            "@@ -20,3 +19,2 @@\n"
            "     d;\n"
            "-    e;\n"
            "     f;\n"
        )
        result = parse_diff_deleted_lines(diff)
        assert result == {"x.rs": [4, 21]}


# --- run_blame ---


PORCELAIN_OUTPUT = """\
abc123abc123abc123abc123abc123abc123abc1 10 10 1
author Alice
author-mail <alice@example.com>
author-time 1234567890
author-tz +0000
committer Bob
committer-mail <bob@example.com>
committer-time 1234567890
committer-tz +0000
summary Some commit message
filename src/foo.rs
\tlet y = 2;
"""


class TestRunBlame:
    @patch("bugeval.blame.run_git")
    def test_basic(self, mock_run_git: object) -> None:
        from unittest.mock import MagicMock

        mock_fn = MagicMock(return_value=PORCELAIN_OUTPUT)
        with patch("bugeval.blame.run_git", mock_fn):
            result = run_blame("src/foo.rs", [10], cwd=Path("/repo"))
        assert result == {10: "abc123abc123abc123abc123abc123abc123abc1"}
        mock_fn.assert_called_once()

    @patch("bugeval.blame.run_git")
    def test_multiple_lines(self, mock_run_git: object) -> None:
        from unittest.mock import MagicMock

        output1 = PORCELAIN_OUTPUT
        output2 = PORCELAIN_OUTPUT.replace(
            "abc123abc123abc123abc123abc123abc123abc1",
            "def456def456def456def456def456def456def4",
        )
        mock_fn = MagicMock(side_effect=[output1, output2])
        with patch("bugeval.blame.run_git", mock_fn):
            result = run_blame("src/foo.rs", [10, 11], cwd=Path("/repo"))
        assert result == {
            10: "abc123abc123abc123abc123abc123abc123abc1",
            11: "def456def456def456def456def456def456def4",
        }

    def test_git_error_skips_line(self) -> None:
        from unittest.mock import MagicMock

        from bugeval.git_utils import GitError

        mock_fn = MagicMock(side_effect=GitError(["git"], "fatal"))
        with patch("bugeval.blame.run_git", mock_fn):
            result = run_blame("src/foo.rs", [10], cwd=Path("/repo"))
        assert result == {}


# --- walk_merge_commit ---


class TestWalkMergeCommit:
    def test_merge_commit_resolved(self) -> None:
        from unittest.mock import MagicMock

        # First call: list parents (2 = merge), second: resolve second parent
        def side_effect(*args: str, cwd: Path, timeout: int = 60) -> str:
            if "--format=%P" in args:
                return "parent1 parent2\n"
            if "^2" in " ".join(args):
                return "featureabc\n"
            return ""

        mock_fn = MagicMock(side_effect=side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            result = walk_merge_commit("mergesha", cwd=Path("/repo"))
        assert result == "featureabc"

    def test_non_merge_returns_same(self) -> None:
        from unittest.mock import MagicMock

        def side_effect(*args: str, cwd: Path, timeout: int = 60) -> str:
            if "--format=%P" in args:
                return "singleparent\n"
            return ""

        mock_fn = MagicMock(side_effect=side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            result = walk_merge_commit("abc123", cwd=Path("/repo"))
        assert result == "abc123"


# --- find_introducing_commit ---


class TestFindIntroducingCommit:
    def test_tier_a_single_sha(self) -> None:
        """Single clear SHA from blame → tier A."""
        from unittest.mock import MagicMock

        case = TestCase(
            id="test-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fixsha123",
            truth=GroundTruth(fix_pr_numbers=[1]),
        )

        diff_output = (
            "diff --git a/src/foo.rs b/src/foo.rs\n"
            "--- a/src/foo.rs\n"
            "+++ b/src/foo.rs\n"
            "@@ -10,3 +10,2 @@\n"
            "     ok;\n"
            "-    buggy_line;\n"
            "-    another_buggy;\n"
            "+    fixed;\n"
        )

        porcelain = (
            "intro111intro111intro111intro111intro111 10 10 1\n"
            "author A\n"
            "summary msg\n"
            "filename src/foo.rs\n"
            "\tline\n"
        )

        def git_side_effect(
            *args: str, cwd: Path, timeout: int = 60
        ) -> str:
            joined = " ".join(args)
            if "diff" in joined and "fixsha123" in joined:
                return diff_output
            if "blame" in joined:
                return porcelain
            if "--format=%P" in joined:
                return "singleparent\n"
            if "rev-list" in joined:
                return ""
            return ""

        mock_fn = MagicMock(side_effect=git_side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            sha, confidence = find_introducing_commit(
                case, Path("/repo")
            )
        assert sha == "intro111intro111intro111intro111intro111"
        assert confidence == "A"

    def test_tier_b_multiple_shas(self) -> None:
        """Multiple SHAs but one dominant (40-60%) → tier B."""
        from unittest.mock import MagicMock

        case = TestCase(
            id="test-002",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fixsha",
            truth=GroundTruth(fix_pr_numbers=[1]),
        )

        diff_output = (
            "diff --git a/src/foo.rs b/src/foo.rs\n"
            "--- a/src/foo.rs\n"
            "+++ b/src/foo.rs\n"
            "@@ -10,5 +10,2 @@\n"
            "     ok;\n"
            "-    line1;\n"
            "-    line2;\n"
            "-    line3;\n"
            "-    line4;\n"
            "+    fixed;\n"
        )

        blame_shas = [
            "aaa1" * 10,  # 2 lines → 50%
            "aaa1" * 10,
            "bbb2" * 10,  # 2 lines → 50%
            "bbb2" * 10,
        ]
        call_idx = [0]

        def git_side_effect(
            *args: str, cwd: Path, timeout: int = 60
        ) -> str:
            joined = " ".join(args)
            if "diff" in joined and "fixsha" in joined:
                return diff_output
            if "blame" in joined:
                sha = blame_shas[call_idx[0]]
                call_idx[0] += 1
                return (
                    f"{sha} 10 10 1\n"
                    "author A\n"
                    "summary msg\n"
                    "filename src/foo.rs\n"
                    "\tline\n"
                )
            if "--format=%P" in joined:
                return "singleparent\n"
            if "rev-list" in joined:
                return ""
            return ""

        mock_fn = MagicMock(side_effect=git_side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            sha, confidence = find_introducing_commit(
                case, Path("/repo")
            )
        assert sha == "aaa1" * 10
        assert confidence == "B"

    def test_tier_c_blame_fails_file_fallback(self) -> None:
        """Blame fails → file-level fallback → tier C."""
        from unittest.mock import MagicMock

        from bugeval.git_utils import GitError

        case = TestCase(
            id="test-003",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fixsha",
            truth=GroundTruth(fix_pr_numbers=[1]),
        )

        diff_output = (
            "diff --git a/src/foo.rs b/src/foo.rs\n"
            "--- a/src/foo.rs\n"
            "+++ b/src/foo.rs\n"
            "@@ -10,3 +10,2 @@\n"
            "     ok;\n"
            "-    buggy;\n"
            "+    fixed;\n"
        )

        def git_side_effect(
            *args: str, cwd: Path, timeout: int = 60
        ) -> str:
            joined = " ".join(args)
            if "diff" in joined and "fixsha" in joined:
                return diff_output
            if "blame" in joined:
                raise GitError(["git", "blame"], "fatal: no such path")
            if "log" in joined and "--format=%H" in joined:
                return "fallbacksha123\n"
            if "--format=%P" in joined:
                return "singleparent\n"
            if "rev-list" in joined:
                return ""
            return ""

        mock_fn = MagicMock(side_effect=git_side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            sha, confidence = find_introducing_commit(
                case, Path("/repo")
            )
        assert sha == "fallbacksha123"
        assert confidence == "C"

    def test_omission_calls_enclosing_function(self) -> None:
        """Omission bug (no deleted lines) tries blame_enclosing_function before fallback."""
        from unittest.mock import MagicMock

        case = TestCase(
            id="test-omit",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fixsha",
            truth=GroundTruth(fix_pr_numbers=[1]),
        )

        # Pure addition diff — no deletions
        diff_output = (
            "diff --git a/src/foo.rs b/src/foo.rs\n"
            "--- a/src/foo.rs\n"
            "+++ b/src/foo.rs\n"
            "@@ -10,2 +10,3 @@\n"
            "     ok;\n"
            "+    added_line;\n"
            "     end;\n"
        )

        def git_side_effect(
            *args: str, cwd: Path, timeout: int = 60
        ) -> str:
            joined = " ".join(args)
            if "diff" in joined and "fixsha" in joined:
                return diff_output
            # blame_enclosing_function uses git log -L
            if "log" in joined and "-L" in joined:
                return "enclosing123enclosing123enclosing12345\n"
            if "--format=%P" in joined:
                return "singleparent\n"
            if "rev-list" in joined:
                return ""
            return ""

        mock_fn = MagicMock(side_effect=git_side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            sha, confidence = find_introducing_commit(
                case, Path("/repo")
            )
        assert sha == "enclosing123enclosing123enclosing12345"
        assert confidence == "D"

    def test_omission_falls_to_file_fallback(self) -> None:
        """Omission bug where enclosing function blame fails falls to file fallback."""
        from unittest.mock import MagicMock

        from bugeval.git_utils import GitError

        case = TestCase(
            id="test-omit-fb",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fixsha",
            truth=GroundTruth(fix_pr_numbers=[1]),
        )

        diff_output = (
            "diff --git a/src/foo.rs b/src/foo.rs\n"
            "--- a/src/foo.rs\n"
            "+++ b/src/foo.rs\n"
            "@@ -10,2 +10,3 @@\n"
            "     ok;\n"
            "+    added_line;\n"
            "     end;\n"
        )

        call_count = {"log_l": 0, "log_h": 0}

        def git_side_effect(
            *args: str, cwd: Path, timeout: int = 60
        ) -> str:
            joined = " ".join(args)
            if "diff" in joined and "fixsha" in joined:
                return diff_output
            # blame_enclosing_function fails
            if "log" in joined and "-L" in joined:
                call_count["log_l"] += 1
                raise GitError(["git"], "fatal")
            # file_level_fallback
            if "log" in joined and "--format=%H" in joined:
                call_count["log_h"] += 1
                return "filefallback123\n"
            if "--format=%P" in joined:
                return "singleparent\n"
            if "rev-list" in joined:
                return ""
            return ""

        mock_fn = MagicMock(side_effect=git_side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            sha, confidence = find_introducing_commit(
                case, Path("/repo")
            )
        # Enclosing was tried
        assert call_count["log_l"] >= 1
        assert sha == "filefallback123"
        assert confidence == "C"


# --- parse_diff_added_lines ---


class TestParseDiffAddedLines:
    def test_single_file_additions(self) -> None:
        diff = (
            "diff --git a/src/foo.rs b/src/foo.rs\n"
            "--- a/src/foo.rs\n"
            "+++ b/src/foo.rs\n"
            "@@ -10,2 +10,4 @@ fn main() {\n"
            "     let x = 1;\n"
            "+    let y = 2;\n"
            "+    let z = 3;\n"
            "     let w = 4;\n"
        )
        result = parse_diff_added_lines(diff)
        assert result == {"src/foo.rs": [11, 12]}

    def test_no_additions(self) -> None:
        diff = (
            "diff --git a/a.rs b/a.rs\n"
            "--- a/a.rs\n"
            "+++ b/a.rs\n"
            "@@ -5,3 +5,2 @@\n"
            "     ok;\n"
            "-    removed;\n"
            "     end;\n"
        )
        result = parse_diff_added_lines(diff)
        assert result == {}

    def test_empty_diff(self) -> None:
        assert parse_diff_added_lines("") == {}


# --- file_level_fallback ---


class TestFileLevelFallback:
    def test_finds_most_recent_commit(self) -> None:
        from unittest.mock import MagicMock

        mock_fn = MagicMock(return_value="recentsha\n")
        with patch("bugeval.blame.run_git", mock_fn):
            result = file_level_fallback(
                ["src/a.rs", "src/b.rs"], "fixsha~1", Path("/repo")
            )
        assert result == "recentsha"

    def test_no_prior_commit(self) -> None:
        from unittest.mock import MagicMock

        mock_fn = MagicMock(return_value="\n")
        with patch("bugeval.blame.run_git", mock_fn):
            result = file_level_fallback(
                ["src/a.rs"], "fixsha~1", Path("/repo")
            )
        assert result is None

    def test_git_error_returns_none(self) -> None:
        from unittest.mock import MagicMock

        from bugeval.git_utils import GitError

        mock_fn = MagicMock(
            side_effect=GitError(["git"], "fatal")
        )
        with patch("bugeval.blame.run_git", mock_fn):
            result = file_level_fallback(
                ["src/a.rs"], "fixsha~1", Path("/repo")
            )
        assert result is None


# --- populate_blame ---


class TestPopulateBlame:
    def test_full_integration(self) -> None:
        from unittest.mock import MagicMock

        case = TestCase(
            id="test-int",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fixsha",
            truth=GroundTruth(fix_pr_numbers=[1]),
        )

        diff_output = (
            "diff --git a/src/foo.rs b/src/foo.rs\n"
            "--- a/src/foo.rs\n"
            "+++ b/src/foo.rs\n"
            "@@ -10,3 +10,2 @@\n"
            "     ok;\n"
            "-    buggy;\n"
            "+    fixed;\n"
        )

        porcelain = (
            "introsha1introsha1introsha1introsha1intr 10 10 1\n"
            "author A\n"
            "summary msg\n"
            "filename src/foo.rs\n"
            "\tline\n"
        )

        def git_side_effect(
            *args: str, cwd: Path, timeout: int = 60
        ) -> str:
            joined = " ".join(args)
            if "diff" in joined and "fixsha" in joined:
                return diff_output
            if "blame" in joined:
                return porcelain
            if "--format=%P" in joined:
                return "singleparent\n"
            if "rev-list" in joined:
                return ""
            if "rev-parse" in joined:
                return "parentofintro\n"
            return ""

        mock_fn = MagicMock(side_effect=git_side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            updated = populate_blame(case, Path("/repo"))

        assert updated.truth is not None
        assert (
            updated.truth.introducing_commit
            == "introsha1introsha1introsha1introsha1intr"
        )
        assert updated.truth.blame_confidence == "A"
        assert updated.base_commit == "parentofintro"

    def test_excluded_initial_commit(self) -> None:
        from unittest.mock import MagicMock

        case = TestCase(
            id="test-excl",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fixsha",
            truth=GroundTruth(fix_pr_numbers=[1]),
        )

        diff_output = (
            "diff --git a/src/foo.rs b/src/foo.rs\n"
            "--- a/src/foo.rs\n"
            "+++ b/src/foo.rs\n"
            "@@ -10,3 +10,2 @@\n"
            "     ok;\n"
            "-    buggy;\n"
            "+    fixed;\n"
        )

        porcelain = (
            "introsha1introsha1introsha1introsha1intr 10 10 1\n"
            "author A\n"
            "summary msg\n"
            "filename src/foo.rs\n"
            "\tline\n"
        )

        def git_side_effect(
            *args: str, cwd: Path, timeout: int = 60
        ) -> str:
            joined = " ".join(args)
            if "diff" in joined and "fixsha" in joined:
                return diff_output
            if "blame" in joined:
                return porcelain
            if "--format=%P" in joined:
                return "singleparent\n"
            # rev-list returns empty → it IS the initial commit
            if "rev-list" in joined and "--count" in joined:
                return "1\n"
            if "rev-list" in joined:
                return ""
            return ""

        mock_fn = MagicMock(side_effect=git_side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            updated = populate_blame(case, Path("/repo"))

        assert updated.truth is not None
        assert updated.truth.introducing_commit is None
        assert updated.truth.blame_confidence == "excluded"


# --- blame_cases with checkpoint ---


class TestBlameCases:
    def test_checkpoint_resume(self, tmp_path: Path) -> None:
        import json
        from unittest.mock import patch as mpatch

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()

        # Write two case YAMLs
        import yaml

        case1 = TestCase(
            id="repo-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fix1",
            truth=GroundTruth(fix_pr_numbers=[1]),
        )
        case2 = TestCase(
            id="repo-002",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fix2",
            truth=GroundTruth(fix_pr_numbers=[2]),
        )

        for c in [case1, case2]:
            p = cases_dir / f"{c.id}.yaml"
            with open(p, "w") as f:
                yaml.safe_dump(
                    c.model_dump(mode="json"), f, sort_keys=False
                )

        # Pre-populate checkpoint with case1 done
        ckpt = cases_dir / ".blame_checkpoint.json"
        ckpt.write_text(json.dumps(["repo-001"]))

        # Mock populate_blame to track calls
        calls: list[str] = []

        def mock_populate(
            case: TestCase, repo_dir: Path
        ) -> TestCase:
            calls.append(case.id)
            if case.truth:
                case.truth.introducing_commit = "found"
                case.truth.blame_confidence = "A"
            return case

        with mpatch(
            "bugeval.blame.populate_blame", side_effect=mock_populate
        ):
            blame_cases(cases_dir, Path("/repo"), concurrency=1)

        # Only case2 should have been processed
        assert calls == ["repo-002"]

        # Checkpoint should now contain both
        updated_ckpt = json.loads(ckpt.read_text())
        assert "repo-001" in updated_ckpt
        assert "repo-002" in updated_ckpt


# --- resolve_introducing_pr ---


class TestResolveIntroducingPr:
    def test_populates_all_fields(self) -> None:
        """Mock gh API + GraphQL, verify all introducing_pr_* fields."""
        import json

        case = TestCase(
            id="test-pr",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="base",
            fix_commit="fixsha",
            truth=GroundTruth(
                introducing_commit="introsha123",
                blame_confidence="A",
            ),
        )

        rest_response = json.dumps([{
            "number": 42,
            "title": "Add feature X",
            "body": "This adds feature X",
            "merged_at": "2024-01-15T10:00:00Z",
            "user": {"login": "alice"},
        }])

        graphql_data = {
            42: {
                "author": {"login": "alice"},
                "mergedAt": "2024-01-15T10:00:00Z",
                "statusCheckRollup": {"state": "SUCCESS"},
                "commits": {
                    "nodes": [
                        {"commit": {"oid": "sha1", "message": "init"}},
                        {"commit": {"oid": "sha2", "message": "fix lint"}},
                    ]
                },
                "reviews": {
                    "nodes": [
                        {"body": "LGTM"},
                    ]
                },
                "reviewThreads": {
                    "nodes": [
                        {
                            "comments": {
                                "nodes": [
                                    {"body": "nit: rename this"},
                                ]
                            }
                        }
                    ]
                },
            }
        }

        with (
            patch("bugeval.blame.run_gh", return_value=rest_response),
            patch(
                "bugeval.blame.fetch_pr_details_graphql",
                return_value=graphql_data,
            ),
        ):
            updated = resolve_introducing_pr(case, "org/repo")

        assert updated.introducing_pr_number == 42
        assert updated.introducing_pr_title == "Add feature X"
        assert updated.introducing_pr_body == "This adds feature X"
        assert updated.introducing_pr_commit_messages == ["init", "fix lint"]
        assert updated.introducing_pr_commit_shas == ["sha1", "sha2"]
        assert updated.introducing_pr_author == "alice"
        assert updated.introducing_pr_merge_date == "2024-01-15T10:00:00Z"
        assert updated.introducing_pr_review_comments == [
            "LGTM", "nit: rename this"
        ]
        assert updated.introducing_pr_ci_status == "SUCCESS"

    def test_no_pr_found_returns_unchanged(self) -> None:
        """When no PR maps to the commit, case is unchanged."""
        import json

        case = TestCase(
            id="test-nopr",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="base",
            fix_commit="fixsha",
            truth=GroundTruth(
                introducing_commit="introsha123",
                blame_confidence="A",
            ),
        )

        with patch("bugeval.blame.run_gh", return_value=json.dumps([])):
            updated = resolve_introducing_pr(case, "org/repo")

        assert updated.introducing_pr_number is None
        assert updated.introducing_pr_title == ""

    def test_no_truth_returns_unchanged(self) -> None:
        """Case without truth is returned as-is."""
        case = TestCase(
            id="test-notruth",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="base",
            truth=None,
        )
        updated = resolve_introducing_pr(case, "org/repo")
        assert updated.introducing_pr_number is None

    def test_no_introducing_commit_returns_unchanged(self) -> None:
        """Case with truth but no introducing_commit is returned as-is."""
        case = TestCase(
            id="test-nointro",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="base",
            truth=GroundTruth(introducing_commit=None),
        )
        updated = resolve_introducing_pr(case, "org/repo")
        assert updated.introducing_pr_number is None

    def test_gh_error_returns_unchanged(self) -> None:
        """gh CLI failure is caught and case returned unchanged."""
        from bugeval.mine import GhError

        case = TestCase(
            id="test-err",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="base",
            truth=GroundTruth(
                introducing_commit="introsha",
                blame_confidence="A",
            ),
        )

        with patch(
            "bugeval.blame.run_gh",
            side_effect=GhError(["gh"], "not found"),
        ):
            updated = resolve_introducing_pr(case, "org/repo")

        assert updated.introducing_pr_number is None


# --- run_blame at revision ---


class TestRunBlameAtRevision:
    def test_revision_argument_passed(self) -> None:
        """Verify at_rev is passed to git blame."""
        from unittest.mock import MagicMock

        mock_fn = MagicMock(return_value=PORCELAIN_OUTPUT)
        with patch("bugeval.blame.run_git", mock_fn):
            result = run_blame(
                "src/foo.rs", [10], cwd=Path("/repo"),
                at_rev="abc123~1",
            )
        assert result == {10: "abc123abc123abc123abc123abc123abc123abc1"}
        call_args = mock_fn.call_args
        assert "abc123~1" in call_args[0]

    def test_default_revision_is_head(self) -> None:
        """Without at_rev, blame runs at HEAD."""
        from unittest.mock import MagicMock

        mock_fn = MagicMock(return_value=PORCELAIN_OUTPUT)
        with patch("bugeval.blame.run_git", mock_fn):
            run_blame("src/foo.rs", [10], cwd=Path("/repo"))
        call_args = mock_fn.call_args
        assert "HEAD" in call_args[0]


# --- fix_pr_numbers populated ---


class TestFixPrNumbersPopulated:
    def test_fix_pr_numbers_set_from_fix_pr_number(self) -> None:
        """populate_blame sets truth.fix_pr_numbers from case.fix_pr_number."""
        from unittest.mock import MagicMock

        case = TestCase(
            id="test-fpn",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fixsha",
            fix_pr_number=99,
        )

        # Return excluded so we skip all the blame logic
        def git_side_effect(
            *args: str, cwd: Path, timeout: int = 60
        ) -> str:
            joined = " ".join(args)
            if "diff" in joined:
                return ""
            return ""

        mock_fn = MagicMock(side_effect=git_side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            updated = populate_blame(case, Path("/repo"))

        assert updated.truth is not None
        assert updated.truth.fix_pr_numbers == [99]

    def test_fix_pr_numbers_empty_when_no_fix_pr(self) -> None:
        """Without fix_pr_number, fix_pr_numbers stays empty."""
        from unittest.mock import MagicMock

        case = TestCase(
            id="test-fpn2",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="",
            fix_commit="fixsha",
            fix_pr_number=None,
        )

        def git_side_effect(
            *args: str, cwd: Path, timeout: int = 60
        ) -> str:
            return ""

        mock_fn = MagicMock(side_effect=git_side_effect)
        with patch("bugeval.blame.run_git", mock_fn):
            updated = populate_blame(case, Path("/repo"))

        assert updated.truth is not None
        assert updated.truth.fix_pr_numbers == []
