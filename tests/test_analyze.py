"""Tests for analyze module."""

from __future__ import annotations

from pathlib import Path

from bugeval.analyze import (
    benjamini_hochberg,
    bootstrap_ci,
    build_comparison_table,
    compute_catch_rate,
    cost_per_bug,
    export_csv,
    false_alarm_rate,
    load_scores,
    median_localization_distance,
    permutation_test,
    run_analysis,
    severity_weighted_catch_rate,
    signal_to_noise,
    slice_scores,
)
from bugeval.io import save_score
from bugeval.models import CaseKind, GroundTruth, TestCase
from bugeval.result_models import ToolResult
from bugeval.score_models import CaseScore, CommentScore, CommentVerdict


def _bug_case(case_id: str, repo: str = "R/a", severity: str = "medium") -> TestCase:
    return TestCase(
        id=case_id,
        repo=repo,
        kind=CaseKind.bug,
        base_commit="abc",
        severity=severity,
        truth=GroundTruth(buggy_lines=[]),
    )


def _clean_case(case_id: str, repo: str = "R/a") -> TestCase:
    return TestCase(id=case_id, repo=repo, kind=CaseKind.clean, base_commit="abc")


def _score(
    case_id: str,
    tool: str = "copilot",
    caught: bool = False,
    loc_dist: int | None = None,
    det: int = 0,
    quality: int = 0,
    tp: int = 0,
    fp: int = 0,
    novel: int = 0,
    false_al: bool = False,
) -> CaseScore:
    scores: list[CommentScore] = []
    for i in range(tp):
        scores.append(CommentScore(comment_index=i, verdict=CommentVerdict.tp))
    for i in range(novel):
        scores.append(
            CommentScore(comment_index=tp + i, verdict=CommentVerdict.tp_novel)
        )
    for i in range(fp):
        scores.append(
            CommentScore(comment_index=tp + novel + i, verdict=CommentVerdict.fp)
        )
    return CaseScore(
        case_id=case_id,
        tool=tool,
        caught=caught,
        localization_distance=loc_dist,
        detection_score=det,
        review_quality=quality,
        tp_count=tp,
        fp_count=fp,
        novel_count=novel,
        false_alarm=false_al,
        comment_scores=scores,
    )


class TestComputeCatchRate:
    def test_basic(self) -> None:
        scores = [
            _score("a", caught=True),
            _score("b", caught=True),
            _score("c", caught=True),
            _score("d"),
            _score("e"),
        ]
        assert compute_catch_rate(scores) == 0.6

    def test_empty(self) -> None:
        assert compute_catch_rate([]) == 0.0


class TestBootstrapCI:
    def test_returns_interval(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = bootstrap_ci(values, n_bootstrap=5000)
        mean = sum(values) / len(values)
        assert lo <= mean <= hi

    def test_all_same(self) -> None:
        lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_bootstrap=1000)
        assert lo == 1.0
        assert hi == 1.0


class TestPermutationTest:
    def test_identical_groups(self) -> None:
        g = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = permutation_test(g, g, n_permutations=5000)
        assert p > 0.05

    def test_different_groups(self) -> None:
        a = [100.0, 101.0, 102.0, 103.0, 104.0]
        b = [0.0, 1.0, 2.0, 3.0, 4.0]
        p = permutation_test(a, b, n_permutations=5000)
        assert p < 0.05


class TestBenjaminiHochberg:
    def test_known(self) -> None:
        pvals = [0.001, 0.01, 0.04, 0.06, 0.5]
        sig = benjamini_hochberg(pvals, alpha=0.05)
        # Rank 1: 0.001 <= 0.01, Rank 2: 0.01 <= 0.02, Rank 3: 0.04 > 0.03
        assert sig[0] is True
        assert sig[1] is True
        assert sig[2] is False
        assert sig[3] is False
        assert sig[4] is False


class TestSeverityWeightedCatchRate:
    def test_weighting(self) -> None:
        cases = [
            _bug_case("a", severity="critical"),
            _bug_case("b", severity="low"),
        ]
        scores = [
            _score("a", caught=True),  # weight=4
            _score("b", caught=False),  # weight=1
        ]
        rate = severity_weighted_catch_rate(scores, cases)
        # 4 / (4+1) = 0.8
        assert abs(rate - 0.8) < 1e-9


class TestMedianLocalizationDistance:
    def test_basic(self) -> None:
        scores = [
            _score("a", caught=True, loc_dist=2),
            _score("b", caught=True, loc_dist=4),
            _score("c", caught=True, loc_dist=6),
            _score("d", caught=False),
        ]
        assert median_localization_distance(scores) == 4.0

    def test_no_catches(self) -> None:
        assert median_localization_distance([_score("a")]) is None


class TestUsefulnessRate:
    def test_ratio(self) -> None:
        scores = [
            _score("a", tp=2, novel=1, fp=2),  # 3 useful / 5 total
            _score("b", tp=1, fp=1),            # 1 useful / 2 total
        ]
        # 4 useful / 7 total (usefulness_rate was removed; use signal_to_noise)
        assert abs(signal_to_noise(scores) - 4 / 7) < 1e-9


class TestFalseAlarmRate:
    def test_rate(self) -> None:
        # We need cases to know which are clean, but false_alarm_rate uses
        # the false_alarm field on CaseScore. We'll mark some as false_alarm.
        scores = [
            _score("clean-1", false_al=True),
            _score("clean-2", false_al=False),
            _score("clean-3", false_al=True),
            _score("bug-1"),  # bug case, not counted
        ]
        cases = [
            _clean_case("clean-1"),
            _clean_case("clean-2"),
            _clean_case("clean-3"),
            _bug_case("bug-1"),
        ]
        # 2/3 clean cases with false alarm
        rate = false_alarm_rate(scores, cases)
        assert abs(rate - 2 / 3) < 1e-9


class TestSignalToNoise:
    def test_ratio(self) -> None:
        scores = [
            _score("a", tp=3, novel=1, fp=2),  # 4 useful / 6
            _score("b", tp=0, fp=2),            # 0 useful / 2
        ]
        # 4 / 8
        assert abs(signal_to_noise(scores) - 0.5) < 1e-9


class TestCostPerBug:
    def test_basic(self) -> None:
        scores = [
            _score("a", caught=True),
            _score("b", caught=True),
            _score("c"),
        ]
        results = [
            ToolResult(case_id="a", tool="t", cost_usd=10.0),
            ToolResult(case_id="b", tool="t", cost_usd=20.0),
            ToolResult(case_id="c", tool="t", cost_usd=30.0),
        ]
        cpb = cost_per_bug(scores, results)
        assert cpb is not None
        assert abs(cpb - 30.0) < 1e-9  # 60/2

    def test_no_catches(self) -> None:
        assert cost_per_bug([_score("a")], [ToolResult(case_id="a", tool="t")]) is None


class TestSliceScores:
    def test_by_repo(self) -> None:
        cases = [_bug_case("a", repo="R/x"), _bug_case("b", repo="R/y")]
        scores = [_score("a"), _score("b")]
        sliced = slice_scores(scores, cases, "repo", "R/x")
        assert len(sliced) == 1
        assert sliced[0].case_id == "a"


class TestBuildComparisonTable:
    def test_columns(self) -> None:
        cases = [_bug_case("a"), _clean_case("b")]
        all_scores = {
            "copilot": [
                _score("a", tool="copilot", caught=True, tp=1, quality=3),
                _score("b", tool="copilot"),
            ],
        }
        all_results = {
            "copilot": [
                ToolResult(case_id="a", tool="copilot", cost_usd=1.0),
                ToolResult(case_id="b", tool="copilot", cost_usd=1.0),
            ],
        }
        table = build_comparison_table(all_scores, all_results, cases)
        assert len(table) == 1
        row = table[0]
        assert row["tool"] == "copilot"
        assert "catch_rate" in row
        assert "ci_lower" in row
        assert "ci_upper" in row
        assert "mean_quality" in row
        assert "false_alarm_rate" in row
        assert "precision" in row
        assert "snr" in row
        assert "cost_per_bug" in row


class TestJudgeFailedExcludedFromQualityMetrics:
    def test_judge_failed_excluded(self) -> None:
        """Judge-failed scores should not affect mean_quality."""
        cases = [_bug_case("a"), _bug_case("b")]
        good = _score("a", tool="copilot", caught=True, tp=1, quality=3)
        failed = _score("b", tool="copilot", caught=False, quality=0)
        failed.judge_failed = True

        all_scores = {"copilot": [good, failed]}
        all_results = {
            "copilot": [
                ToolResult(case_id="a", tool="copilot", cost_usd=1.0),
                ToolResult(case_id="b", tool="copilot", cost_usd=1.0),
            ],
        }
        table = build_comparison_table(all_scores, all_results, cases)
        row = table[0]
        # mean_quality should only include the non-failed score (quality=3)
        assert row["mean_quality"] == 3.0

    def test_caught_metric_still_includes_all(self) -> None:
        """The mechanical caught metric should include all cases."""
        cases = [_bug_case("a"), _bug_case("b")]
        good = _score("a", tool="copilot", caught=True, tp=1, quality=3)
        failed = _score("b", tool="copilot", caught=False, quality=0)
        failed.judge_failed = True

        all_scores = {"copilot": [good, failed]}
        all_results = {
            "copilot": [
                ToolResult(case_id="a", tool="copilot", cost_usd=1.0),
                ToolResult(case_id="b", tool="copilot", cost_usd=1.0),
            ],
        }
        table = build_comparison_table(all_scores, all_results, cases)
        row = table[0]
        # catch_rate = 1/2 (both bug cases counted)
        assert row["catch_rate"] == 0.5


class TestExportCSV:
    def test_writes_file(self, tmp_path: Path) -> None:
        table = [{"tool": "copilot", "catch_rate": 0.5}]
        out = tmp_path / "out.csv"
        export_csv(table, out)
        text = out.read_text()
        assert "tool" in text
        assert "copilot" in text


class TestLoadScores:
    def test_loads(self, tmp_path: Path) -> None:
        s = _score("a", tool="copilot")
        save_score(s, tmp_path / "a__copilot.yaml")
        loaded = load_scores(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].case_id == "a"


class TestRunAnalysis:
    def test_no_charts(self, tmp_path: Path) -> None:
        # Set up minimal directory structure
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()

        s = _score("a", tool="copilot", caught=True, tp=1)
        save_score(s, scores_dir / "a__copilot.yaml")

        from bugeval.io import save_case, save_result

        save_case(_bug_case("a"), cases_dir / "a.yaml")
        save_result(
            ToolResult(case_id="a", tool="copilot", cost_usd=1.0),
            results_dir / "a__copilot.yaml",
        )

        run_analysis(tmp_path, cases_dir, no_charts=True)
        assert (tmp_path / "comparison.csv").exists()


class TestRunAnalysisPairwiseComparisons:
    def test_pairwise_output(self, tmp_path: Path, capsys: object) -> None:
        """Verify permutation test runs when 2+ tools are present."""
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()

        from bugeval.io import save_case, save_result

        save_case(_bug_case("a"), cases_dir / "a.yaml")

        # Two tools with different catch results
        s1 = _score("a", tool="copilot", caught=True, tp=1)
        s2 = _score("a", tool="agent", caught=False, fp=1)
        save_score(s1, scores_dir / "a__copilot.yaml")
        save_score(s2, scores_dir / "a__agent.yaml")

        save_result(
            ToolResult(case_id="a", tool="copilot", cost_usd=1.0),
            results_dir / "a__copilot.yaml",
        )
        save_result(
            ToolResult(case_id="a", tool="agent", cost_usd=1.0),
            results_dir / "a__agent.yaml",
        )

        run_analysis(tmp_path, cases_dir, no_charts=True)

        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "Pairwise Comparisons" in captured.err or "Pairwise Comparisons" in captured.out
        assert "agent vs copilot" in captured.err or "agent vs copilot" in captured.out


class TestSeverityWeightedInTable:
    def test_column_present(self) -> None:
        cases = [_bug_case("a", severity="critical")]
        all_scores = {
            "copilot": [_score("a", tool="copilot", caught=True, tp=1)],
        }
        all_results = {
            "copilot": [ToolResult(case_id="a", tool="copilot", cost_usd=1.0)],
        }
        table = build_comparison_table(all_scores, all_results, cases)
        assert len(table) == 1
        assert "severity_weighted_catch_rate" in table[0]
        assert table[0]["severity_weighted_catch_rate"] == 1.0


class TestContextLevelSlice:
    def test_slice_by_context_level(self) -> None:
        cases = [_bug_case("a"), _bug_case("b")]
        s1 = CaseScore(
            case_id="a", tool="t", caught=True, context_level="diff-only",
        )
        s2 = CaseScore(
            case_id="b", tool="t", caught=False, context_level="diff+repo",
        )
        sliced = slice_scores([s1, s2], cases, "context_level", "diff-only")
        assert len(sliced) == 1
        assert sliced[0].case_id == "a"


class TestLocalizationDistanceInTable:
    def test_present_when_caught(self) -> None:
        cases = [_bug_case("a"), _bug_case("b")]
        all_scores = {
            "copilot": [
                _score("a", tool="copilot", caught=True, loc_dist=3),
                _score("b", tool="copilot", caught=True, loc_dist=7),
            ],
        }
        all_results = {
            "copilot": [
                ToolResult(case_id="a", tool="copilot", cost_usd=1.0),
                ToolResult(case_id="b", tool="copilot", cost_usd=1.0),
            ],
        }
        table = build_comparison_table(all_scores, all_results, cases)
        assert len(table) == 1
        assert "median_localization" in table[0]
        assert table[0]["median_localization"] == 5.0

    def test_none_when_no_catches(self) -> None:
        cases = [_bug_case("a")]
        all_scores = {
            "copilot": [_score("a", tool="copilot", caught=False)],
        }
        all_results = {
            "copilot": [ToolResult(case_id="a", tool="copilot", cost_usd=1.0)],
        }
        table = build_comparison_table(all_scores, all_results, cases)
        assert table[0]["median_localization"] is None


class TestJudgeFailedReportedInAnalysis:
    def test_warning_output(self, tmp_path: Path, capsys: object) -> None:
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()

        from bugeval.io import save_case, save_result

        save_case(_bug_case("a"), cases_dir / "a.yaml")

        s = _score("a", tool="copilot", caught=False)
        s.judge_failed = True
        save_score(s, scores_dir / "a__copilot.yaml")

        save_result(
            ToolResult(case_id="a", tool="copilot", cost_usd=1.0),
            results_dir / "a__copilot.yaml",
        )

        run_analysis(tmp_path, cases_dir, no_charts=True)

        captured = capsys.readouterr()  # type: ignore[union-attr]
        combined = captured.err + captured.out
        assert "judge failures" in combined
        assert "scored as 0/0" in combined
