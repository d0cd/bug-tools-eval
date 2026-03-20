"""Tests for scoring logic (mechanical + LLM judge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bugeval.io import save_case, save_result
from bugeval.models import BuggyLine, CaseKind, GroundTruth, TestCase
from bugeval.result_models import Comment, ToolResult
from bugeval.score import (
    _files_match,
    build_judge_prompt,
    classify_comments,
    detect_contamination,
    mechanical_catch,
    score_case,
    score_run,
)
from bugeval.score_models import CaseScore, CommentVerdict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def truth_one_line() -> GroundTruth:
    return GroundTruth(
        introducing_commit="abc123",
        blame_confidence="A",
        buggy_lines=[
            BuggyLine(file="src/main.rs", line=100, content="bad line"),
        ],
        fix_summary="Fixed the bad line",
        fix_pr_numbers=[1],
    )


@pytest.fixture
def result_exact(truth_one_line: GroundTruth) -> ToolResult:
    return ToolResult(
        case_id="test-001",
        tool="copilot",
        comments=[
            Comment(file="src/main.rs", line=100, body="Bug here"),
        ],
    )


@pytest.fixture
def result_near(truth_one_line: GroundTruth) -> ToolResult:
    return ToolResult(
        case_id="test-001",
        tool="copilot",
        comments=[
            Comment(file="src/main.rs", line=105, body="Something wrong nearby"),
        ],
    )


@pytest.fixture
def result_miss() -> ToolResult:
    return ToolResult(
        case_id="test-001",
        tool="copilot",
        comments=[
            Comment(file="other/file.rs", line=50, body="Looks fishy"),
        ],
    )


# ---------------------------------------------------------------------------
# mechanical_catch
# ---------------------------------------------------------------------------


class TestMechanicalCatch:
    def test_exact_match(
        self, result_exact: ToolResult, truth_one_line: GroundTruth
    ) -> None:
        caught, dist = mechanical_catch(result_exact, truth_one_line)
        assert caught is True
        assert dist == 0

    def test_within_tolerance(
        self, result_near: ToolResult, truth_one_line: GroundTruth
    ) -> None:
        caught, dist = mechanical_catch(result_near, truth_one_line)
        assert caught is True
        assert dist == 5

    def test_miss(
        self, result_miss: ToolResult, truth_one_line: GroundTruth
    ) -> None:
        caught, dist = mechanical_catch(result_miss, truth_one_line)
        assert caught is False
        assert dist is None

    def test_no_truth(self, result_exact: ToolResult) -> None:
        caught, dist = mechanical_catch(result_exact, None)
        assert caught is False
        assert dist is None

    def test_outside_tolerance(
        self, truth_one_line: GroundTruth
    ) -> None:
        result = ToolResult(
            case_id="test-001",
            tool="copilot",
            comments=[
                Comment(file="src/main.rs", line=150, body="Bug?"),
            ],
        )
        caught, dist = mechanical_catch(result, truth_one_line, tolerance=10)
        assert caught is False
        assert dist is None


# ---------------------------------------------------------------------------
# classify_comments
# ---------------------------------------------------------------------------


class TestClassifyComments:
    def test_tp(self, truth_one_line: GroundTruth) -> None:
        result = ToolResult(
            case_id="test-001",
            tool="t",
            comments=[
                Comment(
                    file="src/main.rs",
                    line=102,
                    body="Race condition on this line",
                ),
            ],
        )
        scores = classify_comments(result, truth_one_line)
        assert len(scores) == 1
        assert scores[0].verdict == CommentVerdict.tp
        assert scores[0].matched_buggy_line == 0

    def test_fp(self, truth_one_line: GroundTruth) -> None:
        result = ToolResult(
            case_id="test-001",
            tool="t",
            comments=[
                Comment(
                    file="other/file.rs",
                    line=50,
                    body="This seems like an issue to me",
                ),
            ],
        )
        scores = classify_comments(result, truth_one_line)
        assert len(scores) == 1
        assert scores[0].verdict == CommentVerdict.fp

    def test_low_value(self, truth_one_line: GroundTruth) -> None:
        result = ToolResult(
            case_id="test-001",
            tool="t",
            comments=[
                Comment(file="src/main.rs", line=200, body="LGTM"),
            ],
        )
        scores = classify_comments(result, truth_one_line)
        assert len(scores) == 1
        assert scores[0].verdict == CommentVerdict.low_value

    def test_no_truth(self) -> None:
        result = ToolResult(
            case_id="test-001",
            tool="t",
            comments=[
                Comment(file="src/main.rs", line=10, body="This variable is used incorrectly"),
            ],
        )
        scores = classify_comments(result, None)
        assert len(scores) == 1
        assert scores[0].verdict == CommentVerdict.fp


# ---------------------------------------------------------------------------
# score_case
# ---------------------------------------------------------------------------


class TestScoreCase:
    def test_mechanical(
        self, sample_case: TestCase, sample_result: ToolResult
    ) -> None:
        cs = score_case(sample_case, sample_result, use_llm=False)
        assert cs.case_id == "snarkVM-001"
        assert cs.tool == "copilot"
        assert cs.caught is True
        assert cs.localization_distance == 1
        assert len(cs.comment_scores) == 2

    def test_clean_case_no_comments(self, clean_case: TestCase) -> None:
        result = ToolResult(case_id="clean-001", tool="copilot", comments=[])
        cs = score_case(clean_case, result, use_llm=False)
        assert cs.false_alarm is False
        assert cs.caught is False

    def test_clean_case_with_bug_comment(self, clean_case: TestCase) -> None:
        result = ToolResult(
            case_id="clean-001",
            tool="copilot",
            comments=[
                Comment(file="lib.rs", line=10, body="Bug found here!"),
            ],
        )
        cs = score_case(clean_case, result, use_llm=False)
        assert cs.false_alarm is True

    def test_clean_case_fp_count_excludes_low_value(
        self, clean_case: TestCase,
    ) -> None:
        """Clean case with some line=0 comments: fp_count < total comments."""
        result = ToolResult(
            case_id="clean-001",
            tool="copilot",
            comments=[
                Comment(file="lib.rs", line=0, body="General comment"),
                Comment(
                    file="lib.rs", line=10,
                    body="This looks like a bug to me",
                ),
                Comment(file="lib.rs", line=0, body="Another file-level note"),
            ],
        )
        cs = score_case(clean_case, result, use_llm=False)
        # 2 line=0 comments -> low_value, 1 line=10 -> fp
        assert cs.fp_count == 1
        assert cs.fp_count < len(result.comments)


# ---------------------------------------------------------------------------
# detect_contamination
# ---------------------------------------------------------------------------


class TestDetectContamination:
    def test_positive(self, sample_case: TestCase) -> None:
        result = ToolResult(
            case_id="snarkVM-001",
            tool="copilot",
            comments=[
                Comment(
                    file="consensus/src/worker.rs",
                    line=142,
                    body="Fix overflow in rotation counter",
                ),
            ],
        )
        assert detect_contamination(result, sample_case) is True

    def test_negative(self, sample_case: TestCase) -> None:
        result = ToolResult(
            case_id="snarkVM-001",
            tool="copilot",
            comments=[
                Comment(
                    file="consensus/src/worker.rs",
                    line=142,
                    body="Potential race condition on shared state access",
                ),
            ],
        )
        assert detect_contamination(result, sample_case) is False

    def test_includes_commit_messages(self) -> None:
        """Verify fix_pr_commit_messages are included in contamination pool."""
        case = TestCase(
            id="cm-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            fix_pr_title="",
            fix_pr_body="",
            fix_pr_commit_messages=["refactor validator rotation overflow"],
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="cm-001",
            tool="copilot",
            comments=[
                Comment(
                    file="f.rs",
                    line=10,
                    body="refactor validator rotation overflow here",
                ),
            ],
        )
        assert detect_contamination(result, case) is True


# ---------------------------------------------------------------------------
# build_judge_prompt
# ---------------------------------------------------------------------------


class TestBuildJudgePrompt:
    def test_includes_key_parts(
        self, sample_case: TestCase, sample_result: ToolResult
    ) -> None:
        prompt = build_judge_prompt(sample_case, sample_result)
        assert "detection_score" in prompt
        assert "review_quality" in prompt
        assert sample_case.bug_description in prompt
        assert "consensus/src/worker.rs" in prompt
        assert "Potential race condition" in prompt

    def test_does_not_contain_tool_name(
        self, sample_case: TestCase, sample_result: ToolResult
    ) -> None:
        """Judge prompt must not reveal which tool produced the output."""
        prompt = build_judge_prompt(sample_case, sample_result)
        assert sample_result.tool not in prompt
        assert "## Review Comments" in prompt


# ---------------------------------------------------------------------------
# score_run (checkpoint resume)
# ---------------------------------------------------------------------------


class TestScoreRun:
    def test_checkpoint_resume(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases" / "repo"
        cases_dir.mkdir(parents=True)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        results_dir = run_dir / "results"
        results_dir.mkdir()

        case = TestCase(
            id="r-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        save_case(case, cases_dir / "r-001.yaml")

        result = ToolResult(
            case_id="r-001",
            tool="copilot",
            comments=[
                Comment(file="f.rs", line=10, body="Bug on this line"),
            ],
        )
        save_result(result, results_dir / "r-001__copilot.yaml")

        # First run
        score_run(run_dir, tmp_path / "cases", dry_run=True)

        scores_dir = run_dir / "scores"
        assert (scores_dir / "r-001__copilot.yaml").exists()

        # Second run should skip (checkpoint)
        score_run(run_dir, tmp_path / "cases", dry_run=True)
        assert (scores_dir / "r-001__copilot.yaml").exists()


# ---------------------------------------------------------------------------
# Contamination wired into score_run
# ---------------------------------------------------------------------------


class TestContaminationWiredInScoreRun:
    def test_contamination_flag_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify score_run calls detect_contamination before scoring."""
        cases_dir = tmp_path / "cases" / "repo"
        cases_dir.mkdir(parents=True)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        results_dir = run_dir / "results"
        results_dir.mkdir()

        case = TestCase(
            id="c-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            fix_pr_title="Fix overflow in rotation counter",
            fix_pr_body="The counter overflowed when count > 128.",
            fix_pr_review_comments=["Good catch on the overflow"],
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed overflow",
            ),
        )
        save_case(case, cases_dir / "c-001.yaml")

        result = ToolResult(
            case_id="c-001",
            tool="copilot",
            comments=[
                Comment(
                    file="f.rs",
                    line=10,
                    body="Fix overflow in rotation counter overflowed",
                ),
            ],
        )
        save_result(result, results_dir / "c-001__copilot.yaml")

        # Track whether detect_contamination was called
        contamination_calls: list[bool] = []
        import bugeval.score as score_mod
        orig_detect = score_mod.detect_contamination

        def tracking_detect(
            r: ToolResult, c: TestCase
        ) -> bool:
            val = orig_detect(r, c)
            contamination_calls.append(val)
            return val

        monkeypatch.setattr(score_mod, "detect_contamination", tracking_detect)

        score_run(run_dir, tmp_path / "cases", dry_run=True)

        assert len(contamination_calls) == 1
        assert contamination_calls[0] is True


# ---------------------------------------------------------------------------
# Field validation on CaseScore
# ---------------------------------------------------------------------------


class TestCaseScoreValidation:
    def test_detection_score_too_high(self) -> None:
        with pytest.raises(Exception):
            CaseScore(
                case_id="x",
                tool="t",
                detection_score=5,
            )

    def test_detection_score_negative(self) -> None:
        with pytest.raises(Exception):
            CaseScore(
                case_id="x",
                tool="t",
                detection_score=-1,
            )

    def test_review_quality_too_high(self) -> None:
        with pytest.raises(Exception):
            CaseScore(
                case_id="x",
                tool="t",
                review_quality=5,
            )

    def test_review_quality_negative(self) -> None:
        with pytest.raises(Exception):
            CaseScore(
                case_id="x",
                tool="t",
                review_quality=-1,
            )

    def test_valid_scores_accepted(self) -> None:
        cs = CaseScore(
            case_id="x",
            tool="t",
            detection_score=3,
            review_quality=4,
        )
        assert cs.detection_score == 3
        assert cs.review_quality == 4


# ---------------------------------------------------------------------------
# _files_match path boundary
# ---------------------------------------------------------------------------


class TestFilesMatchBoundary:
    def test_exact_match(self) -> None:
        assert _files_match("src/main.rs", "src/main.rs") is True

    def test_suffix_match(self) -> None:
        assert _files_match("repo/src/main.rs", "src/main.rs") is True

    def test_no_partial_match(self) -> None:
        # "other/main.rs" should NOT match "src/main.rs"
        assert _files_match("other/main.rs", "src/main.rs") is False

    def test_basename_only(self) -> None:
        assert _files_match("main.rs", "main.rs") is True

    def test_different_basenames(self) -> None:
        assert _files_match("main.rs", "lib.rs") is False

    def test_suffix_without_separator(self) -> None:
        # "xsrc/main.rs" should NOT match "src/main.rs"
        assert _files_match("xsrc/main.rs", "src/main.rs") is False


# ---------------------------------------------------------------------------
# mechanical_catch skips line=0
# ---------------------------------------------------------------------------


class TestMechanicalCatchSkipsLineZero:
    def test_line_zero_ignored(self) -> None:
        truth = GroundTruth(
            buggy_lines=[
                BuggyLine(file="src/main.rs", line=5, content="bad"),
            ],
        )
        result = ToolResult(
            case_id="t-001",
            tool="t",
            comments=[
                Comment(file="src/main.rs", line=0, body="General comment"),
            ],
        )
        caught, dist = mechanical_catch(result, truth)
        assert caught is False
        assert dist is None

    def test_line_nonzero_still_works(self) -> None:
        truth = GroundTruth(
            buggy_lines=[
                BuggyLine(file="src/main.rs", line=5, content="bad"),
            ],
        )
        result = ToolResult(
            case_id="t-001",
            tool="t",
            comments=[
                Comment(file="src/main.rs", line=0, body="General comment"),
                Comment(file="src/main.rs", line=5, body="Bug on this line"),
            ],
        )
        caught, dist = mechanical_catch(result, truth)
        assert caught is True
        assert dist == 0


# ---------------------------------------------------------------------------
# classify_comments skips line=0
# ---------------------------------------------------------------------------


class TestClassifyCommentsSkipsLineZero:
    def test_line_zero_is_low_value_not_tp(self) -> None:
        """A comment with line=0 on the same file as a buggy line must NOT be TP."""
        truth = GroundTruth(
            buggy_lines=[
                BuggyLine(file="src/main.rs", line=0, content="bad"),
            ],
        )
        result = ToolResult(
            case_id="t-001",
            tool="t",
            comments=[
                Comment(
                    file="src/main.rs",
                    line=0,
                    body="This entire file has a race condition problem",
                ),
            ],
        )
        scores = classify_comments(result, truth)
        assert len(scores) == 1
        assert scores[0].verdict == CommentVerdict.low_value

    def test_line_zero_among_others(self) -> None:
        """Line=0 comment is low_value while a real-line comment still matches."""
        truth = GroundTruth(
            buggy_lines=[
                BuggyLine(file="src/main.rs", line=10, content="bad"),
            ],
        )
        result = ToolResult(
            case_id="t-001",
            tool="t",
            comments=[
                Comment(
                    file="src/main.rs",
                    line=0,
                    body="General file-level observation about this module",
                ),
                Comment(
                    file="src/main.rs",
                    line=10,
                    body="Bug on this specific line needs fixing",
                ),
            ],
        )
        scores = classify_comments(result, truth)
        assert len(scores) == 2
        assert scores[0].verdict == CommentVerdict.low_value
        assert scores[1].verdict == CommentVerdict.tp


# ---------------------------------------------------------------------------
# CaseScore has contaminated flag propagated
# ---------------------------------------------------------------------------


class TestCaseScoreContaminatedFlag:
    def test_field_exists(self) -> None:
        cs = CaseScore(case_id="x", tool="t")
        assert cs.potentially_contaminated is False

    def test_propagated_from_result(self) -> None:
        case = TestCase(
            id="c-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="c-001",
            tool="copilot",
            potentially_contaminated=True,
            comments=[
                Comment(file="f.rs", line=10, body="Bug on this line here"),
            ],
        )
        cs = score_case(case, result, use_llm=False)
        assert cs.potentially_contaminated is True


# ---------------------------------------------------------------------------
# CaseScore has context_level propagated
# ---------------------------------------------------------------------------


class TestCaseScoreContextLevel:
    def test_field_populated_from_result(self) -> None:
        case = TestCase(
            id="c-002",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="c-002",
            tool="agent",
            context_level="diff+repo",
            comments=[
                Comment(file="f.rs", line=10, body="Bug on this line here"),
            ],
        )
        cs = score_case(case, result, use_llm=False)
        assert cs.context_level == "diff+repo"

    def test_defaults_to_empty(self) -> None:
        cs = CaseScore(case_id="x", tool="t")
        assert cs.context_level == ""


# ---------------------------------------------------------------------------
# judge_failed flag
# ---------------------------------------------------------------------------


class TestJudgeFailedFlag:
    def test_default_false(self) -> None:
        cs = CaseScore(case_id="x", tool="t")
        assert cs.judge_failed is False

    def test_call_judge_sets_flag_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When call_judge raises internally, the returned dict has judge_failed."""
        import bugeval.score as score_mod

        # Simulate an API failure by monkeypatching anthropic import
        def _failing_judge(
            prompt: str, model: str = "", case_id: str = ""
        ) -> dict:
            # Simulate the except branch directly
            return {
                "detection_score": 0,
                "review_quality": 0,
                "comment_verdicts": [],
                "reasoning": "LLM judge failed",
                "judge_failed": True,
            }

        monkeypatch.setattr(score_mod, "call_judge", _failing_judge)

        case = TestCase(
            id="jf-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="jf-001",
            tool="copilot",
            comments=[
                Comment(file="f.rs", line=10, body="Bug on this line here"),
            ],
        )
        cs = score_case(case, result, use_llm=True)
        assert cs.judge_failed is True
        assert cs.detection_score == 0
        assert cs.review_quality == 0

    def test_successful_judge_not_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When call_judge succeeds, judge_failed is False."""
        import bugeval.score as score_mod

        def _ok_judge(
            prompt: str, model: str = "", case_id: str = ""
        ) -> dict:
            return {
                "detection_score": 2,
                "review_quality": 3,
                "comment_verdicts": ["TP"],
                "reasoning": "Good find",
            }

        monkeypatch.setattr(score_mod, "call_judge", _ok_judge)

        case = TestCase(
            id="jf-002",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="jf-002",
            tool="copilot",
            comments=[
                Comment(file="f.rs", line=10, body="Bug on this line here"),
            ],
        )
        cs = score_case(case, result, use_llm=True)
        assert cs.judge_failed is False


# ---------------------------------------------------------------------------
# LLM override changes counts
# ---------------------------------------------------------------------------


class TestLlmOverrideChangesCounts:
    def test_llm_override_changes_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LLM overrides mechanical verdicts, tp/fp/novel counts change."""
        import bugeval.score as score_mod

        case = TestCase(
            id="ov-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                    BuggyLine(file="f.rs", line=20, content="y"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="ov-001",
            tool="agent",
            comments=[
                # Mechanically TP (matches buggy line 10)
                Comment(file="f.rs", line=10, body="Bug on this line definitely"),
                # Mechanically FP (wrong file)
                Comment(file="other.rs", line=50, body="This looks wrong to me here"),
                # Mechanically low_value (line=0)
                Comment(file="f.rs", line=0, body="General comment about the file"),
            ],
        )

        # Verify mechanical scores first
        mech_scores = classify_comments(result, case.truth)
        assert mech_scores[0].verdict == CommentVerdict.tp
        assert mech_scores[1].verdict == CommentVerdict.fp
        assert mech_scores[2].verdict == CommentVerdict.low_value

        # Mock LLM to disagree: [FP, TP-novel, FP]
        def _override_judge(
            prompt: str, model: str = "", case_id: str = ""
        ) -> dict:
            return {
                "detection_score": 1,
                "review_quality": 2,
                "comment_verdicts": ["FP", "TP-novel", "FP"],
                "reasoning": "Overridden",
            }

        monkeypatch.setattr(score_mod, "call_judge", _override_judge)

        cs = score_case(case, result, use_llm=True)
        assert cs.tp_count == 0  # was 1 mechanically
        assert cs.fp_count == 2  # was 1 mechanically
        assert cs.novel_count == 1  # was 0 mechanically


# ---------------------------------------------------------------------------
# Bug case with empty comments
# ---------------------------------------------------------------------------


class TestScoreBugCaseEmptyComments:
    def test_score_bug_case_empty_comments(self) -> None:
        """Bug case with no comments: caught=False, all counts=0."""
        case = TestCase(
            id="empty-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(case_id="empty-001", tool="agent", comments=[])
        score = score_case(case, result, use_llm=False)
        assert score.caught is False
        assert score.detection_score == 0
        assert score.tp_count == 0
        assert score.fp_count == 0
        assert score.comment_scores == []


# ---------------------------------------------------------------------------
# Detection score heuristic values
# ---------------------------------------------------------------------------


class TestDetectionScoreHeuristic:
    def test_detection_score_heuristic_with_fix(self) -> None:
        """Caught + suggested_fix -> detection_score=3."""
        case = TestCase(
            id="ds-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="ds-001",
            tool="agent",
            comments=[
                Comment(
                    file="f.rs",
                    line=10,
                    body="Bug on this line needs fixing now",
                    suggested_fix="Use a lock guard here",
                ),
            ],
        )
        score = score_case(case, result, use_llm=False)
        assert score.caught is True
        assert score.detection_score == 3

    def test_detection_score_heuristic_without_fix(self) -> None:
        """Caught but no suggested_fix -> detection_score=2."""
        case = TestCase(
            id="ds-002",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="ds-002",
            tool="agent",
            comments=[
                Comment(
                    file="f.rs",
                    line=10,
                    body="Bug on this line needs fixing now",
                ),
            ],
        )
        score = score_case(case, result, use_llm=False)
        assert score.caught is True
        assert score.detection_score == 2

    def test_detection_score_heuristic_missed(self) -> None:
        """No catch -> detection_score=0."""
        case = TestCase(
            id="ds-003",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="ds-003",
            tool="agent",
            comments=[
                Comment(
                    file="other.rs",
                    line=999,
                    body="Something unrelated to the actual bug",
                ),
            ],
        )
        score = score_case(case, result, use_llm=False)
        assert score.caught is False
        assert score.detection_score == 0


# ---------------------------------------------------------------------------
# Result filename roundtrip
# ---------------------------------------------------------------------------


class TestResultFilenameRoundtrip:
    def test_result_filename_roundtrip(self) -> None:
        """Verify result_filename produces parseable names."""
        from bugeval.evaluate import result_filename

        name = result_filename("snarkVM-001", "agent", "diff+repo")
        assert "snarkVM-001" in name
        assert "agent" in name
        assert name.endswith(".yaml")

    def test_result_filename_no_context(self) -> None:
        """Filename without context level."""
        from bugeval.evaluate import result_filename

        name = result_filename("snarkVM-001", "agent", "")
        assert "snarkVM-001" in name
        assert "agent" in name
        assert name.endswith(".yaml")


# ---------------------------------------------------------------------------
# Contamination short-circuit for short fix text
# ---------------------------------------------------------------------------


class TestJudgeModelThreading:
    def test_score_case_passes_judge_model(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify score_case passes judge_model to call_judge."""
        import bugeval.score as score_mod

        captured_models: list[str] = []

        def _tracking_judge(
            prompt: str, model: str = "", case_id: str = "",
        ) -> dict:
            captured_models.append(model)
            return {
                "detection_score": 2,
                "review_quality": 3,
                "comment_verdicts": ["TP"],
                "reasoning": "Good",
            }

        monkeypatch.setattr(score_mod, "call_judge", _tracking_judge)

        case = TestCase(
            id="jm-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        result = ToolResult(
            case_id="jm-001",
            tool="agent",
            comments=[
                Comment(file="f.rs", line=10, body="Bug on this line here"),
            ],
        )
        score_case(
            case, result, use_llm=True,
            judge_model="claude-sonnet-4-20250514",
        )
        assert len(captured_models) == 1
        assert captured_models[0] == "claude-sonnet-4-20250514"

    def test_score_run_passes_judge_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify score_run passes judge_model through to score_case."""
        import bugeval.score as score_mod

        captured_models: list[str] = []

        orig_score_case = score_mod.score_case

        def _tracking_score_case(
            case: TestCase,
            result: ToolResult,
            use_llm: bool = True,
            judge_model: str = "claude-haiku-4-5-20251001",
        ) -> CaseScore:
            captured_models.append(judge_model)
            return orig_score_case(case, result, use_llm=False)

        monkeypatch.setattr(score_mod, "score_case", _tracking_score_case)

        cases_dir = tmp_path / "cases" / "repo"
        cases_dir.mkdir(parents=True)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        results_dir = run_dir / "results"
        results_dir.mkdir()

        case = TestCase(
            id="jm-002",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            bug_description="A bug",
            truth=GroundTruth(
                buggy_lines=[
                    BuggyLine(file="f.rs", line=10, content="x"),
                ],
                fix_summary="Fixed",
            ),
        )
        save_case(case, cases_dir / "jm-002.yaml")

        result = ToolResult(
            case_id="jm-002",
            tool="agent",
            comments=[
                Comment(file="f.rs", line=10, body="Bug on this line"),
            ],
        )
        save_result(result, results_dir / "jm-002__agent.yaml")

        score_run(
            run_dir, tmp_path / "cases", dry_run=True,
            judge_model="claude-sonnet-4-20250514",
        )
        assert len(captured_models) == 1
        assert captured_models[0] == "claude-sonnet-4-20250514"


class TestContaminationShortFixText:
    def test_contamination_short_fix_text(self) -> None:
        """Cases with very short fix text bypass contamination detection."""
        case = TestCase(
            id="short-001",
            repo="org/repo",
            kind=CaseKind.bug,
            base_commit="aaa",
            fix_pr_title="Fix",
            fix_pr_body="",
        )
        result = ToolResult(
            case_id="short-001",
            tool="agent",
            comments=[Comment(body="Fix the bug")],
        )
        assert detect_contamination(result, case) is False
