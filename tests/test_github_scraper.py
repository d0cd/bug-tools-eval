"""Tests for GitHub scraper: scoring, linking, and utility functions. No gh calls."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from bugeval.github_scraper import (
    GhError,
    _batch_fetch_pr_reviews_graphql,
    _extract_issue_numbers,
    build_candidates,
    build_labeled_pr_candidates,
    build_pr_only_candidates,
    compute_pr_size,
    detect_language,
    enrich_git_candidates_with_github,
    enrich_with_reviews,
    extract_expected_findings,
    extract_reviewer_bug_signals,
    fetch_prs_by_label,
    filter_already_processed,
    link_issues_to_prs,
    load_scrape_state,
    save_scrape_state,
    score_candidate,
)
from bugeval.models import Candidate, CaseStats, PRSize, ScrapeState


def make_issue(
    number: int = 1,
    labels: list[str] | None = None,
    title: str = "",
    body: str = "",
) -> dict[str, object]:
    if labels is None:
        labels = ["bug"]
    return {
        "number": number,
        "title": title or f"Bug in feature #{number}",
        "body": body or f"Description of bug {number}",
        "labels": [{"name": lbl} for lbl in labels],
        "closedAt": "2024-01-01T00:00:00Z",
    }


def make_pr(
    number: int = 100,
    title: str = "Fix bug in feature",
    body: str = "",
    closed_issues: list[int] | None = None,
    additions: int = 10,
    deletions: int = 5,
    labels: list[str] | None = None,
) -> dict[str, object]:
    ci = closed_issues or []
    return {
        "number": number,
        "title": title,
        "body": body or (f"Fixes #{ci[0]}" if ci else ""),
        "labels": [{"name": lbl} for lbl in (labels or [])],
        "mergeCommit": {"oid": "abc123def456abc123def456abc123def456abc1"},
        "baseRefName": "main",
        "headRefName": "fix/bug",
        "files": [{"path": "src/main.rs"}],
        "additions": additions,
        "deletions": deletions,
        "changedFiles": 1,
        "closedIssues": [{"number": n} for n in ci],
    }


class TestScoreCandidate:
    def test_high_confidence_with_bug_label_and_reference(self) -> None:
        issue = make_issue(1, labels=["bug"])
        pr = make_pr(100, title="Fix bug #1", closed_issues=[1], additions=20, deletions=10)
        confidence, signals = score_candidate(issue, pr)
        assert confidence > 0.5
        assert "has_bug_label" in signals
        assert "pr_references_issue" in signals

    def test_low_confidence_no_bug_label_no_reference_large_diff(self) -> None:
        issue = make_issue(1, labels=["enhancement"])
        pr = make_pr(
            100,
            title="Add feature",
            body="Unrelated PR",
            closed_issues=[],
            additions=300,
            deletions=200,
        )
        confidence, signals = score_candidate(issue, pr)
        assert confidence < 0.5
        assert "has_bug_label" not in signals

    def test_small_diff_adds_signal(self) -> None:
        issue = make_issue(1, labels=[])
        pr = make_pr(100, body="", closed_issues=[], additions=5, deletions=3)
        _, signals = score_candidate(issue, pr)
        assert "small_diff" in signals

    def test_confidence_capped_at_1(self) -> None:
        issue = make_issue(1, labels=["bug"])
        pr = make_pr(100, title="Fix bug", closed_issues=[1], additions=5, deletions=3)
        confidence, _ = score_candidate(issue, pr)
        assert confidence <= 1.0

    def test_fix_keywords_in_title(self) -> None:
        issue = make_issue(1, labels=[])
        pr = make_pr(100, title="fix: resolve the regression", body="", closed_issues=[])
        _, signals = score_candidate(issue, pr)
        assert "fix_keywords_in_title" in signals

    def test_regression_label_counts_as_bug(self) -> None:
        issue = make_issue(1, labels=["regression"])
        pr = make_pr(100, body="", closed_issues=[])
        _, signals = score_candidate(issue, pr)
        assert "has_bug_label" in signals

    def test_has_linked_issue_not_double_counted(self) -> None:
        # When pr_references_issue fires for this issue, has_linked_issue must NOT also fire
        issue = make_issue(1, labels=["bug"])
        pr = make_pr(100, title="Fix bug", closed_issues=[1])
        _, signals = score_candidate(issue, pr)
        assert "pr_references_issue" in signals
        assert "has_linked_issue" not in signals

    def test_has_linked_issue_fires_for_unrelated_link(self) -> None:
        # PR links to issue #99 but is being scored against issue #1 (no direct link)
        issue = make_issue(1, labels=["bug"])
        pr = make_pr(100, title="Add feature", body="Fixes #99", closed_issues=[99])
        _, signals = score_candidate(issue, pr)
        assert "pr_references_issue" not in signals
        assert "has_linked_issue" in signals


class TestLinkIssuesToPrs:
    def test_link_via_closed_issues(self) -> None:
        issue = make_issue(1)
        pr = make_pr(100, closed_issues=[1])
        pairs = link_issues_to_prs([issue], [pr])
        assert len(pairs) == 1
        assert pairs[0][0]["number"] == 1
        assert pairs[0][1]["number"] == 100

    def test_link_via_body_reference(self) -> None:
        issue = make_issue(42)
        pr = make_pr(200, body="This closes #42 by fixing the race condition", closed_issues=[])
        pairs = link_issues_to_prs([issue], [pr])
        assert len(pairs) == 1
        assert pairs[0][0]["number"] == 42

    def test_link_via_title_reference(self) -> None:
        issue = make_issue(7)
        pr = make_pr(300, title="Fixes #7: off-by-one error", body="", closed_issues=[])
        pairs = link_issues_to_prs([issue], [pr])
        assert len(pairs) == 1

    def test_no_match(self) -> None:
        issue = make_issue(99)
        pr = make_pr(100, body="Unrelated PR", closed_issues=[])
        pairs = link_issues_to_prs([issue], [pr])
        assert len(pairs) == 0

    def test_multiple_issues_prs(self) -> None:
        issues = [make_issue(1), make_issue(2), make_issue(3)]
        prs = [make_pr(10, closed_issues=[1]), make_pr(20, closed_issues=[2])]
        pairs = link_issues_to_prs(issues, prs)
        assert len(pairs) == 2

    def test_same_issue_not_matched_twice(self) -> None:
        issue = make_issue(1)
        pr1 = make_pr(10, closed_issues=[1])
        pr2 = make_pr(20, body="Also fixes #1", closed_issues=[])
        pairs = link_issues_to_prs([issue], [pr1, pr2])
        # Issue 1 should only appear once
        linked_issues = [p[0]["number"] for p in pairs]
        assert linked_issues.count(1) == 1


class TestExtractIssueNumbers:
    def test_fixes_pattern(self) -> None:
        assert _extract_issue_numbers("fixes #123") == {123}

    def test_closes_pattern(self) -> None:
        assert _extract_issue_numbers("closes #456") == {456}

    def test_fix_pattern(self) -> None:
        assert _extract_issue_numbers("fix #7") == {7}

    def test_multiple_references(self) -> None:
        assert _extract_issue_numbers("fixes #1, closes #2") == {1, 2}

    def test_no_references(self) -> None:
        assert _extract_issue_numbers("unrelated text #not-a-number") == set()

    def test_case_insensitive(self) -> None:
        assert _extract_issue_numbers("Fixes #10") == {10}


class TestDetectLanguage:
    def test_rust_files(self) -> None:
        assert detect_language(["src/main.rs", "src/lib.rs", "src/parser.rs"]) == "rust"

    def test_python_files(self) -> None:
        assert detect_language(["main.py", "utils.py"]) == "python"

    def test_typescript_files(self) -> None:
        assert detect_language(["app.ts", "component.tsx"]) == "typescript"

    def test_mixed_files_dominant(self) -> None:
        result = detect_language(["a.rs", "b.rs", "c.py"])
        assert result == "rust"

    def test_unknown_extensions(self) -> None:
        assert detect_language(["readme.md", "config.toml"]) == "unknown"

    def test_empty_list(self) -> None:
        assert detect_language([]) == "unknown"


class TestComputePRSize:
    def test_tiny(self) -> None:
        assert compute_pr_size(3, 5) == PRSize.tiny

    def test_small(self) -> None:
        assert compute_pr_size(30, 15) == PRSize.small

    def test_medium(self) -> None:
        assert compute_pr_size(100, 80) == PRSize.medium

    def test_large(self) -> None:
        assert compute_pr_size(300, 150) == PRSize.large

    def test_xl(self) -> None:
        assert compute_pr_size(400, 200) == PRSize.xl

    def test_boundary_tiny_small(self) -> None:
        assert compute_pr_size(5, 4) == PRSize.tiny  # 9 = tiny
        assert compute_pr_size(5, 5) == PRSize.small  # 10 = small


class TestFilterAlreadyProcessed:
    def test_no_state_returns_all(self) -> None:
        prs = [make_pr(1), make_pr(2)]
        result = filter_already_processed(prs, None)
        assert len(result) == 2

    def test_filter_removes_processed(self) -> None:
        state = ScrapeState(
            repo="foo/bar",
            last_scraped_at=datetime.now(),
            processed_pr_numbers=[1, 2],
        )
        prs = [make_pr(1), make_pr(2), make_pr(3)]
        result = filter_already_processed(prs, state)
        assert len(result) == 1
        assert result[0]["number"] == 3

    def test_empty_state_returns_all(self) -> None:
        state = ScrapeState(
            repo="foo/bar",
            last_scraped_at=datetime.now(),
            processed_pr_numbers=[],
        )
        prs = [make_pr(1), make_pr(2)]
        result = filter_already_processed(prs, state)
        assert len(result) == 2


class TestExtractExpectedFindings:
    def test_extracts_from_patch(self) -> None:
        pr_diff_files = [
            {
                "filename": "src/main.rs",
                "patch": "@@ -1,3 +1,4 @@\n context\n+added line\n context\n",
            }
        ]
        findings = extract_expected_findings(pr_diff_files)
        assert len(findings) >= 1
        assert findings[0].file == "src/main.rs"
        assert findings[0].line == 1

    def test_empty_patch_skipped(self) -> None:
        findings = extract_expected_findings([{"filename": "x.rs", "patch": ""}])
        assert findings == []

    def test_missing_patch_skipped(self) -> None:
        findings = extract_expected_findings([{"filename": "x.rs"}])
        assert findings == []

    def test_summary_starts_with_auto(self) -> None:
        pr_diff_files = [
            {
                "filename": "lib.rs",
                "patch": "@@ -5,3 +5,4 @@\n ctx\n+new line\n ctx\n",
            }
        ]
        findings = extract_expected_findings(pr_diff_files)
        assert findings[0].summary.startswith("[auto]")


class TestExtractReviewerBugSignals:
    def test_reviewer_explicit_bug_word(self) -> None:
        reviews = [{"body": "This is a bug — the counter overflows", "_source": "review"}]
        signals, notes = extract_reviewer_bug_signals(reviews)
        assert "reviewer_bug_feedback" in signals
        assert len(notes) == 1
        assert "[review]" in notes[0]

    def test_changes_requested_adds_signal(self) -> None:
        reviews = [{"body": "Please fix this", "state": "CHANGES_REQUESTED", "_source": "review"}]
        signals, _ = extract_reviewer_bug_signals(reviews)
        assert "reviewer_changes_requested" in signals

    def test_empty_body_skipped(self) -> None:
        reviews = [{"body": "", "_source": "inline"}]
        signals, notes = extract_reviewer_bug_signals(reviews)
        assert signals == []
        assert notes == []

    def test_no_bug_language_no_signal(self) -> None:
        reviews = [{"body": "Looks good to me, nice work!", "_source": "thread"}]
        signals, notes = extract_reviewer_bug_signals(reviews)
        assert "reviewer_bug_feedback" not in signals
        assert notes == []

    def test_deduplication_of_signals(self) -> None:
        reviews = [
            {"body": "This will panic here", "_source": "inline"},
            {"body": "Also this crashes", "_source": "inline"},
        ]
        signals, notes = extract_reviewer_bug_signals(reviews)
        assert signals.count("reviewer_bug_feedback") == 1
        assert len(notes) == 2

    def test_note_capped_at_300_chars(self) -> None:
        long_body = "This is wrong. " * 30  # > 300 chars
        reviews = [{"body": long_body, "_source": "review"}]
        _, notes = extract_reviewer_bug_signals(reviews)
        assert len(notes[0]) <= 310  # "[review] " prefix + 300


class TestBuildPrOnlyCandidates:
    def test_bug_labeled_pr_included(self) -> None:
        pr = make_pr(200, title="Fix edge case", labels=["bug"])
        result = build_pr_only_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert len(result) == 1
        assert result[0].pr_number == 200
        assert "pr_only" in result[0].signals
        assert "has_bug_label" in result[0].signals

    def test_fix_keyword_in_title_included(self) -> None:
        pr = make_pr(201, title="fix: resolve integer overflow", labels=[])
        result = build_pr_only_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert len(result) == 1
        assert "fix_keywords_in_title" in result[0].signals

    def test_already_matched_pr_skipped(self) -> None:
        pr = make_pr(202, title="fix bug", labels=["bug"])
        result = build_pr_only_candidates("owner/repo", [pr], existing_pr_numbers={202})
        assert len(result) == 0

    def test_unrelated_pr_excluded(self) -> None:
        pr = make_pr(203, title="Add new feature", labels=["enhancement"])
        result = build_pr_only_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert len(result) == 0

    def test_small_diff_boosts_confidence(self) -> None:
        pr = make_pr(204, title="fix: off-by-one", labels=["bug"], additions=5, deletions=3)
        result = build_pr_only_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert result[0].confidence > 0.3


class TestEnrichWithReviews:
    def test_reviewer_feedback_boosts_confidence(self) -> None:
        issue = make_issue(1, labels=["bug"])
        pr = make_pr(100, title="Fix bug", body="Fixes #1")
        candidates = build_candidates("owner/repo", [issue], [pr])
        graphql_data = {
            100: [{"body": "This will panic on empty input", "_source": "inline", "state": ""}]
        }
        mock_fn = "bugeval.github_scraper._batch_fetch_pr_reviews_graphql"
        with patch(mock_fn, return_value=graphql_data):
            enriched = enrich_with_reviews("owner/repo", candidates)
        c = next(c for c in enriched if c.pr_number == 100)
        assert "reviewer_bug_feedback" in c.signals
        assert len(c.reviewer_notes) == 1
        assert c.confidence > 0.5

    def test_pr_only_candidate_gets_reviewer_boost(self) -> None:
        pr = make_pr(201, title="fix: crash on null", labels=["bug"])
        candidates = build_candidates("owner/repo", [], [pr])
        graphql_data = {
            201: [{"body": "This crashes when the field is None", "_source": "thread", "state": ""}]
        }
        mock_fn = "bugeval.github_scraper._batch_fetch_pr_reviews_graphql"
        with patch(mock_fn, return_value=graphql_data):
            enriched = enrich_with_reviews("owner/repo", candidates)
        c = next(c for c in enriched if c.pr_number == 201)
        assert "reviewer_bug_feedback" in c.signals
        assert c.reviewer_notes

    def test_no_reviews_means_no_change(self) -> None:
        issue = make_issue(1, labels=["bug"])
        pr = make_pr(100, body="Fixes #1")
        candidates = build_candidates("owner/repo", [issue], [pr])
        original_conf = candidates[0].confidence
        with patch("bugeval.github_scraper._batch_fetch_pr_reviews_graphql", return_value={}):
            enriched = enrich_with_reviews("owner/repo", candidates)
        assert enriched[0].confidence == original_conf

    def test_batches_in_chunks_of_25(self) -> None:
        """More than 25 candidates should trigger multiple GraphQL calls."""
        prs = [make_pr(i, title=f"fix bug {i}", labels=["bug"]) for i in range(1, 35)]
        candidates = build_candidates("owner/repo", [], prs)
        mock_fn = "bugeval.github_scraper._batch_fetch_pr_reviews_graphql"
        with patch(mock_fn, return_value={}) as mock_gql:
            enrich_with_reviews("owner/repo", candidates, top_n=30)
        assert mock_gql.call_count == 2  # 25 + 5


class TestBatchFetchPrReviewsGraphql:
    def test_returns_empty_on_graphql_error(self) -> None:
        with patch("bugeval.github_scraper.run_gh", side_effect=GhError(["gh"], "error")):
            result = _batch_fetch_pr_reviews_graphql("owner", "repo", [1, 2])
        assert result == {}

    def test_empty_pr_list_returns_empty(self) -> None:
        result = _batch_fetch_pr_reviews_graphql("owner", "repo", [])
        assert result == {}

    def test_parses_review_bodies(self) -> None:
        graphql_response = json.dumps({
            "data": {
                "repository": {
                    "pr_42": {
                        "reviews": {"nodes": [
                            {"body": "This is wrong", "state": "CHANGES_REQUESTED"}
                        ]},
                        "reviewThreads": {"nodes": []},
                        "comments": {"nodes": []},
                    }
                }
            }
        })
        with patch("bugeval.github_scraper.run_gh", return_value=graphql_response):
            result = _batch_fetch_pr_reviews_graphql("owner", "repo", [42])
        assert 42 in result
        assert result[42][0]["body"] == "This is wrong"
        assert result[42][0]["state"] == "CHANGES_REQUESTED"
        assert result[42][0]["_source"] == "review"


class TestFetchPrsByLabel:
    def test_deduplicates_across_labels(self) -> None:
        """The same PR returned under two labels should appear only once."""
        pr = make_pr(500, title="fix: crash on null", labels=["bug"])
        pr_json = json.dumps([pr])

        with patch("bugeval.github_scraper.run_gh", return_value=pr_json):
            result = fetch_prs_by_label("owner/repo", labels=["bug", "bugfix"])

        assert len(result) == 1
        assert result[0]["number"] == 500

    def test_skips_label_on_gh_error(self) -> None:
        """A GhError for one label should be silently skipped."""
        pr = make_pr(501, title="fix: overflow", labels=["bugfix"])
        pr_json = json.dumps([pr])

        def side_effect(*args: str) -> str:
            if "--label" in args and args[list(args).index("--label") + 1] == "bug":
                raise GhError(list(args), "label not found")
            return pr_json

        with patch("bugeval.github_scraper.run_gh", side_effect=side_effect):
            result = fetch_prs_by_label("owner/repo", labels=["bug", "bugfix"])

        assert len(result) == 1
        assert result[0]["number"] == 501

    def test_returns_empty_on_all_label_errors(self) -> None:
        """All labels failing should return empty list, not raise."""
        with patch("bugeval.github_scraper.run_gh", side_effect=GhError(["gh"], "error")):
            result = fetch_prs_by_label("owner/repo", labels=["bug"])
        assert result == []

    def test_default_labels_used_when_none(self) -> None:
        """When labels=None the function uses its built-in default list."""
        with patch("bugeval.github_scraper.run_gh", return_value="[]") as mock_gh:
            fetch_prs_by_label("owner/repo", labels=None)
        # Should have been called once per default label
        assert mock_gh.call_count > 1

    def test_since_forwarded_to_search(self) -> None:
        """The since parameter should be included in the --search argument."""
        with patch("bugeval.github_scraper.run_gh", return_value="[]") as mock_gh:
            fetch_prs_by_label("owner/repo", labels=["bug"], since="2024-06-01")
        call_args = mock_gh.call_args[0]
        assert "created:>2024-06-01" in " ".join(call_args)


class TestBuildLabeledPrCandidates:
    def _make_labeled_pr(
        self,
        number: int = 300,
        title: str = "fix: crash on null",
        additions: int = 10,
        deletions: int = 5,
        labels: list[str] | None = None,
        body: str = "",
    ) -> dict[str, object]:
        return {
            "number": number,
            "title": title,
            "body": body,
            "labels": [{"name": lbl} for lbl in (labels or ["bug"])],
            "mergeCommit": {"oid": "aabbccdd" * 5},
            "additions": additions,
            "deletions": deletions,
            "changedFiles": 1,
            "files": [{"path": "src/lib.rs"}],
        }

    def test_base_confidence_is_0_4(self) -> None:
        # Large diff (no small_diff bonus), generic title, no issue ref → only base 0.4
        pr = self._make_labeled_pr(
            title="chore: cleanup", additions=300, deletions=300, body=""
        )
        result = build_labeled_pr_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert len(result) == 1
        assert result[0].confidence == 0.4

    def test_labeled_bug_signal_always_present(self) -> None:
        pr = self._make_labeled_pr()
        result = build_labeled_pr_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert "labeled_bug" in result[0].signals

    def test_small_diff_boosts_confidence(self) -> None:
        pr = self._make_labeled_pr(additions=10, deletions=5)
        result = build_labeled_pr_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert result[0].confidence > 0.4
        assert "small_diff" in result[0].signals

    def test_fix_keyword_in_title_boosts_confidence(self) -> None:
        pr = self._make_labeled_pr(title="fix: null pointer", additions=300, deletions=300)
        result = build_labeled_pr_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert "fix_keywords_in_title" in result[0].signals
        assert result[0].confidence > 0.4

    def test_issue_ref_in_body_boosts_confidence(self) -> None:
        pr = self._make_labeled_pr(
            title="chore: cleanup", additions=300, deletions=300, body="Fixes #42"
        )
        result = build_labeled_pr_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert "has_issue_ref" in result[0].signals
        assert result[0].confidence > 0.4

    def test_existing_pr_skipped(self) -> None:
        pr = self._make_labeled_pr(number=300)
        result = build_labeled_pr_candidates("owner/repo", [pr], existing_pr_numbers={300})
        assert result == []

    def test_confidence_capped_at_1(self) -> None:
        # All bonuses: small diff + fix kw + issue ref = 0.4 + 0.1 + 0.1 + 0.1 = 0.7
        pr = self._make_labeled_pr(title="fix bug", additions=5, deletions=3, body="Fixes #1")
        result = build_labeled_pr_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert result[0].confidence <= 1.0

    def test_language_detected_from_files(self) -> None:
        pr = self._make_labeled_pr()
        result = build_labeled_pr_candidates("owner/repo", [pr], existing_pr_numbers=set())
        assert result[0].language == "rust"

    def test_empty_prs_list_returns_empty(self) -> None:
        result = build_labeled_pr_candidates("owner/repo", [], existing_pr_numbers=set())
        assert result == []


def make_git_candidate(
    fix_commit: str = "a" * 40,
    confidence: float = 0.6,
    repo: str = "ProvableHQ/snarkVM",
) -> Candidate:
    return Candidate(
        repo=repo,
        pr_number=0,
        fix_commit=fix_commit,
        base_commit=None,
        head_commit=fix_commit,
        confidence=confidence,
        signals=["keyword:fix", "signal:has_introducing"],
        title="fix: off by one in field arithmetic",
        body="",
        labels=[],
        files_changed=["src/lib.rs"],
        diff_stats=CaseStats(lines_added=5, lines_deleted=3, files_changed=1, hunks=1),
        expected_findings=[],
        language="rust",
        pr_size=PRSize.small,
    )


class TestEnrichGitCandidatesWithGithub:
    def test_enriches_with_pr_metadata(self) -> None:
        candidate = make_git_candidate()
        api_response = json.dumps([{"number": 42}])
        pr_detail = json.dumps({
            "number": 42,
            "title": "Fix: integer overflow in BLS12 field",
            "body": "Fixes #101 — overflow in field arithmetic",
            "labels": [{"name": "bug"}],
            "additions": 10,
            "deletions": 4,
            "files": [{"path": "src/field.rs"}],
        })
        side_effects = [api_response, pr_detail, GhError([], "")]
        with patch("bugeval.github_scraper.run_gh", side_effect=side_effects):
            result = enrich_git_candidates_with_github("ProvableHQ/snarkVM", [candidate])

        assert len(result) == 1
        enriched = result[0]
        assert enriched.pr_number == 42
        assert enriched.title == "Fix: integer overflow in BLS12 field"
        assert "bug" in enriched.labels
        assert "has_bug_label" in enriched.signals
        assert "github_pr" in enriched.signals
        assert enriched.confidence > candidate.confidence

    def test_no_pr_found_leaves_candidate_unchanged(self) -> None:
        candidate = make_git_candidate()
        with patch("bugeval.github_scraper.run_gh", return_value="[]"):
            result = enrich_git_candidates_with_github("ProvableHQ/snarkVM", [candidate])
        assert result[0].pr_number == candidate.pr_number
        assert result[0].confidence == candidate.confidence

    def test_gh_error_leaves_candidate_unchanged(self) -> None:
        candidate = make_git_candidate()
        with patch("bugeval.github_scraper.run_gh", side_effect=GhError([], "network error")):
            result = enrich_git_candidates_with_github("ProvableHQ/snarkVM", [candidate])
        assert result[0].pr_number == candidate.pr_number

    def test_respects_top_n_limit(self) -> None:
        candidates = [make_git_candidate(fix_commit=c * 40, confidence=0.5 + i * 0.1)
                      for i, c in enumerate("abcde")]
        # top_n=2: only first 2 should be enriched; rest passed through
        call_count = 0
        def mock_gh(*args: str) -> str:
            nonlocal call_count
            call_count += 1
            return "[]"  # no PR found → passthrough

        with patch("bugeval.github_scraper.run_gh", side_effect=mock_gh):
            result = enrich_git_candidates_with_github("ProvableHQ/snarkVM", candidates, top_n=2)

        assert len(result) == 5
        assert call_count == 2  # only 2 gh api calls made

    def test_no_bug_label_does_not_boost_confidence(self) -> None:
        candidate = make_git_candidate()
        api_response = json.dumps([{"number": 7}])
        pr_detail = json.dumps({
            "number": 7,
            "title": "refactor: clean up imports",
            "body": "",
            "labels": [{"name": "cleanup"}],
            "additions": 5,
            "deletions": 3,
            "files": [{"path": "src/lib.rs"}],
        })
        side_effects = [api_response, pr_detail, GhError([], "")]
        with patch("bugeval.github_scraper.run_gh", side_effect=side_effects):
            result = enrich_git_candidates_with_github("ProvableHQ/snarkVM", [candidate])
        assert result[0].confidence == candidate.confidence
        assert "has_bug_label" not in result[0].signals


class TestScrapeStateRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        state = ScrapeState(
            repo="foo/bar",
            last_scraped_at=datetime(2024, 1, 15, 12, 0, 0),
            processed_pr_numbers=[1, 2, 3],
        )
        path = tmp_path / "foo-bar.state.yaml"
        save_scrape_state(state, path)

        loaded = load_scrape_state(path)
        assert loaded is not None
        assert loaded.repo == "foo/bar"
        assert loaded.processed_pr_numbers == [1, 2, 3]

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = load_scrape_state(tmp_path / "nonexistent.yaml")
        assert result is None


def test_run_gh_timeout() -> None:
    """run_gh raises GhError when the subprocess times out."""
    from bugeval.github_scraper import run_gh

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["gh"], timeout=60)):
        try:
            run_gh("repo", "list")
            assert False, "expected GhError"
        except GhError as e:
            assert "timed out" in str(e)
