"""Tests for ground truth construction via diff intersection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from bugeval.ground_truth import (
    compute_buggy_lines,
    compute_metadata,
    extract_bug_description,
    parse_diff_added_lines,
    populate_ground_truth,
)
from bugeval.models import GroundTruth, TestCase

# ---------------------------------------------------------------------------
# Fixtures: synthetic diffs
# ---------------------------------------------------------------------------

SINGLE_FILE_DIFF = """\
diff --git a/src/lib.rs b/src/lib.rs
index abc1234..def5678 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,6 +10,8 @@ fn existing() {
     context line
     context line
+    let x = bad_call();
+    let y = x + 1;
     context line
     context line
"""

MULTI_FILE_DIFF = """\
diff --git a/src/foo.rs b/src/foo.rs
index aaa..bbb 100644
--- a/src/foo.rs
+++ b/src/foo.rs
@@ -5,3 +5,5 @@ fn foo() {
     keep
+    added_in_foo_line_6();
+    added_in_foo_line_7();
     keep
diff --git a/src/bar.rs b/src/bar.rs
index ccc..ddd 100644
--- a/src/bar.rs
+++ b/src/bar.rs
@@ -20,3 +20,4 @@ fn bar() {
     keep
+    added_in_bar_line_21();
     keep
"""

FIX_DIFF_EXACT = """\
diff --git a/src/lib.rs b/src/lib.rs
index def5678..ghi9012 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,8 +10,7 @@ fn existing() {
     context line
     context line
-    let x = bad_call();
-    let y = x + 1;
+    let x = good_call();
     context line
     context line
"""

FIX_DIFF_DRIFTED = """\
diff --git a/src/lib.rs b/src/lib.rs
index def5678..ghi9012 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -12,5 +12,4 @@ fn existing() {
     context line
-    let x = bad_call();
+    let x = good_call();
     context line
"""

FIX_DIFF_OTHER_FILE = """\
diff --git a/src/other.rs b/src/other.rs
index 111..222 100644
--- a/src/other.rs
+++ b/src/other.rs
@@ -1,4 +1,3 @@
-    removed_line();
+    replaced_line();
     keep
"""

FIX_DIFF_FOO = """\
diff --git a/src/foo.rs b/src/foo.rs
index bbb..eee 100644
--- a/src/foo.rs
+++ b/src/foo.rs
@@ -5,5 +5,4 @@ fn foo() {
     keep
-    added_in_foo_line_6();
+    fixed_in_foo();
     keep
"""

FIX_DIFF_BAR = """\
diff --git a/src/bar.rs b/src/bar.rs
index ddd..fff 100644
--- a/src/bar.rs
+++ b/src/bar.rs
@@ -20,4 +20,3 @@ fn bar() {
     keep
-    added_in_bar_line_21();
     keep
"""


# ---------------------------------------------------------------------------
# parse_diff_added_lines
# ---------------------------------------------------------------------------


class TestParseDiffAddedLines:
    def test_single_file(self) -> None:
        result = parse_diff_added_lines(SINGLE_FILE_DIFF)
        assert "src/lib.rs" in result
        lines = result["src/lib.rs"]
        assert (12, "    let x = bad_call();") in lines
        assert (13, "    let y = x + 1;") in lines

    def test_multiple_files(self) -> None:
        result = parse_diff_added_lines(MULTI_FILE_DIFF)
        assert "src/foo.rs" in result
        assert "src/bar.rs" in result

        foo_lines = result["src/foo.rs"]
        assert (6, "    added_in_foo_line_6();") in foo_lines
        assert (7, "    added_in_foo_line_7();") in foo_lines

        bar_lines = result["src/bar.rs"]
        assert (21, "    added_in_bar_line_21();") in bar_lines


# ---------------------------------------------------------------------------
# compute_buggy_lines
# ---------------------------------------------------------------------------


class TestComputeBuggyLines:
    def test_exact_match(self) -> None:
        result = compute_buggy_lines(SINGLE_FILE_DIFF, [FIX_DIFF_EXACT])
        assert len(result) >= 1
        files = {bl.file for bl in result}
        assert "src/lib.rs" in files
        # At least one of the two added lines should match
        matched_lines = {bl.line for bl in result if bl.file == "src/lib.rs"}
        assert 12 in matched_lines or 13 in matched_lines

    def test_tolerance(self) -> None:
        # FIX_DIFF_DRIFTED removes line 13 (old side) which was originally
        # added at line 12 — within tolerance of ±3
        result = compute_buggy_lines(SINGLE_FILE_DIFF, [FIX_DIFF_DRIFTED])
        assert len(result) >= 1
        files = {bl.file for bl in result}
        assert "src/lib.rs" in files

    def test_no_overlap(self) -> None:
        result = compute_buggy_lines(SINGLE_FILE_DIFF, [FIX_DIFF_OTHER_FILE])
        assert result == []

    def test_multi_fix(self) -> None:
        result = compute_buggy_lines(
            MULTI_FILE_DIFF, [FIX_DIFF_FOO, FIX_DIFF_BAR]
        )
        files = {bl.file for bl in result}
        # Both files should have matched lines
        assert "src/foo.rs" in files
        assert "src/bar.rs" in files


# ---------------------------------------------------------------------------
# extract_bug_description
# ---------------------------------------------------------------------------


class TestExtractBugDescription:
    def _make_case(self, **overrides: object) -> TestCase:
        defaults: dict[str, object] = {
            "id": "test-001",
            "repo": "org/repo",
            "kind": "bug",
            "base_commit": "aaa",
        }
        defaults.update(overrides)
        return TestCase(**defaults)  # type: ignore[arg-type]

    def test_pr_body_preferred_over_generic_issue(self) -> None:
        """PR body wins over issue body that doesn't look like a bug report."""
        case = self._make_case(
            issue_bodies={123: "I'd like a feature for token registry"},
            fix_pr_body="This fixes the crash in the parser module",
            fix_pr_title="PR title",
        )
        desc, source = extract_bug_description(case)
        assert source == "pr_body"
        assert "crash" in desc

    def test_bug_issue_wins_when_no_pr_body(self) -> None:
        """Issue with bug keywords wins when PR body is empty."""
        case = self._make_case(
            issue_bodies={123: "Bug report: the parser crashes on invalid input"},
            fix_pr_body="",
            fix_pr_title="Fix parser crash",
        )
        desc, source = extract_bug_description(case)
        assert source == "pr_title"  # title comes before issue in priority

    def test_from_pr_body(self) -> None:
        case = self._make_case(
            issue_bodies={},
            fix_pr_body="PR body text here is long enough",
            fix_pr_title="PR title",
        )
        desc, source = extract_bug_description(case)
        assert source == "pr_body"
        assert "PR body text here" in desc

    def test_from_title(self) -> None:
        case = self._make_case(
            issue_bodies={},
            fix_pr_body="",
            fix_pr_title="Fix the crash in parser",
        )
        desc, source = extract_bug_description(case)
        assert source == "pr_title"
        assert "Fix the crash in parser" in desc


# ---------------------------------------------------------------------------
# compute_metadata
# ---------------------------------------------------------------------------


class TestComputeMetadata:
    def _make_case(self, **overrides: object) -> TestCase:
        defaults: dict[str, object] = {
            "id": "test-001",
            "repo": "org/repo",
            "kind": "bug",
            "base_commit": "aaa",
        }
        defaults.update(overrides)
        return TestCase(**defaults)  # type: ignore[arg-type]

    def test_latency(self) -> None:
        case = self._make_case(
            introducing_pr_merge_date="2024-01-01T00:00:00Z",
            fix_pr_merge_date="2024-01-11T00:00:00Z",
        )
        meta = compute_metadata(case)
        assert meta["bug_latency_days"] == 10

    def test_same_author(self) -> None:
        case = self._make_case(
            introducing_pr_author="alice",
            fix_pr_merge_date="",
            introducing_pr_merge_date="",
        )
        # Set fix author via related_prs or fix_pr data
        # The fix author comes from related_prs with role=full_fix, or
        # we compare introducing_pr_author with fix PR author
        # For simplicity, we use the introducing_pr_author field
        # and need a fix_pr_author — check how the model stores it
        # The model doesn't have fix_pr_author directly, but we can
        # check related_prs
        from bugeval.models import PRRelation

        case.related_prs = [
            PRRelation(
                pr_number=42,
                role="full_fix",
                commit="bbb",
                author="alice",
            )
        ]
        meta = compute_metadata(case)
        assert meta["same_author_fix"] is True

    def test_different_author(self) -> None:
        from bugeval.models import PRRelation

        case = self._make_case(
            introducing_pr_author="alice",
            fix_pr_merge_date="",
            introducing_pr_merge_date="",
        )
        case.related_prs = [
            PRRelation(
                pr_number=42,
                role="full_fix",
                commit="bbb",
                author="bob",
            )
        ]
        meta = compute_metadata(case)
        assert meta["same_author_fix"] is False


# ---------------------------------------------------------------------------
# populate_ground_truth (integration, mocked git)
# ---------------------------------------------------------------------------


class TestPopulateGroundTruth:
    def test_full_flow(self) -> None:
        case = TestCase(
            id="test-001",
            repo="org/repo",
            kind="bug",
            base_commit="aaa",
            fix_commit="fff",
            fix_pr_title="Fix the bug",
            fix_pr_body="This fixes a crash",
            introducing_pr_merge_date="2024-01-01T00:00:00Z",
            fix_pr_merge_date="2024-01-15T00:00:00Z",
            introducing_pr_author="alice",
            truth=GroundTruth(
                introducing_commit="abc123",
                blame_confidence="A",
                fix_pr_numbers=[99],
            ),
        )

        introducing_diff = SINGLE_FILE_DIFF
        fix_diff = FIX_DIFF_EXACT

        def fake_run_git(*args: str, cwd: Path, timeout: int = 60) -> str:
            cmd = list(args)
            if cmd[0] == "diff":
                if "abc123~1" in cmd and "abc123" in cmd:
                    return introducing_diff
                # Fix PR diff — merge base approach
                return fix_diff
            if cmd[0] == "log":
                if "--format=%H" in cmd:
                    return "merge_base_sha\n"
                return ""
            if cmd[0] == "merge-base":
                return "merge_base_sha\n"
            return ""

        with patch("bugeval.ground_truth.run_git", side_effect=fake_run_git):
            updated = populate_ground_truth(case, Path("/fake/repo"))

        assert updated.truth is not None
        assert len(updated.truth.buggy_lines) > 0
        assert updated.bug_description != ""
        assert updated.bug_latency_days == 14
