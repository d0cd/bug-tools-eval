# tests/test_analyze.py
"""Tests for analyze module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bugeval.analyze import (
    aggregate_scores,
    compute_catch_rate,
    compute_snr,
    generate_csv,
    generate_markdown,
)
from bugeval.judge_models import JudgeScore, NoiseStats
from bugeval.models import (
    Category,
    Difficulty,
    ExpectedFinding,
    PRSize,
    Severity,
    TestCase,
    Visibility,
)


def _make_scores(data: list[tuple[str, str, int, float]]) -> list[JudgeScore]:
    """Helper: (case_id, tool, score, snr) → list[JudgeScore]."""
    return [
        JudgeScore(
            test_case_id=cid,
            tool=tool,
            score=score,
            votes=[score, score, score],
            reasoning="test",
            noise=NoiseStats(total_comments=4, true_positives=int(snr * 4), snr=snr),
        )
        for cid, tool, score, snr in data
    ]


def test_compute_catch_rate_basic() -> None:
    scores = _make_scores(
        [
            ("c1", "greptile", 2, 0.5),
            ("c2", "greptile", 0, 0.0),
            ("c3", "greptile", 3, 1.0),
        ]
    )
    rate = compute_catch_rate(scores)
    assert rate == pytest.approx(2 / 3)


def test_compute_catch_rate_empty() -> None:
    assert compute_catch_rate([]) == 0.0


def test_compute_snr_average() -> None:
    scores = _make_scores(
        [
            ("c1", "greptile", 2, 0.5),
            ("c2", "greptile", 1, 0.25),
        ]
    )
    assert compute_snr(scores) == pytest.approx(0.375)


def test_aggregate_scores_groups_by_tool() -> None:
    scores = _make_scores(
        [
            ("c1", "greptile", 2, 0.5),
            ("c2", "coderabbit", 3, 1.0),
            ("c3", "greptile", 0, 0.0),
        ]
    )
    agg = aggregate_scores(scores)
    assert set(agg.keys()) == {"greptile", "coderabbit"}
    assert agg["greptile"]["catch_rate"] == pytest.approx(0.5)
    assert agg["coderabbit"]["catch_rate"] == pytest.approx(1.0)
    assert agg["greptile"]["count"] == 2


def test_generate_csv(tmp_path: Path) -> None:
    scores = _make_scores([("c1", "greptile", 2, 0.5)])
    agg = aggregate_scores(scores)
    path = tmp_path / "scores.csv"
    generate_csv(agg, path)
    content = path.read_text()
    assert "greptile" in content
    assert "catch_rate" in content


def test_generate_markdown(tmp_path: Path) -> None:
    scores = _make_scores([("c1", "greptile", 2, 0.5)])
    agg = aggregate_scores(scores)
    md = generate_markdown(agg)
    assert "| Tool |" in md
    assert "greptile" in md


# ---- Phase 8 additions: slicing + cost metrics ----


def _make_case(
    case_id: str = "case-001",
    category: str = "logic",
    difficulty: str = "medium",
    severity: str = "high",
    pr_size: str = "small",
    language: str = "rust",
) -> TestCase:
    return TestCase(
        id=case_id,
        repo="org/repo",
        base_commit="abc",
        head_commit="def",
        fix_commit="ghi",
        category=Category(category),
        difficulty=Difficulty(difficulty),
        severity=Severity(severity),
        language=language,
        pr_size=PRSize(pr_size),
        description="test case",
        expected_findings=[ExpectedFinding(file="a.rs", line=1, summary="bug")],
    )


def test_load_cases_lookup(tmp_path: Path) -> None:
    from bugeval.analyze import load_cases_lookup

    case = _make_case("case-001")
    (tmp_path / "case-001.yaml").write_text(
        yaml.safe_dump(case.model_dump(mode="json"), sort_keys=False)
    )
    lookup = load_cases_lookup(tmp_path)
    assert "case-001" in lookup
    assert lookup["case-001"].category.value == "logic"


def test_load_cases_lookup_empty_dir(tmp_path: Path) -> None:
    from bugeval.analyze import load_cases_lookup

    assert load_cases_lookup(tmp_path) == {}


def test_load_normalized_lookup(tmp_path: Path) -> None:
    from bugeval.analyze import load_normalized_lookup
    from bugeval.result_models import NormalizedResult, ResultMetadata

    r = NormalizedResult(
        test_case_id="case-001",
        tool="greptile",
        context_level="diff-only",
        metadata=ResultMetadata(cost_usd=0.05),
    )
    (tmp_path / "case-001-greptile.yaml").write_text(
        yaml.safe_dump(r.model_dump(mode="json"), sort_keys=False)
    )
    lookup = load_normalized_lookup(tmp_path)
    assert ("case-001", "greptile") in lookup
    assert lookup[("case-001", "greptile")].metadata.cost_usd == pytest.approx(0.05)


def test_slice_scores_by_dimension() -> None:
    from bugeval.analyze import slice_scores

    scores = _make_scores(
        [
            ("c1", "greptile", 2, 0.5),
            ("c2", "greptile", 0, 0.0),
        ]
    )
    cases = {
        "c1": _make_case("c1", difficulty="easy"),
        "c2": _make_case("c2", difficulty="hard"),
    }
    groups = slice_scores(scores, cases, "difficulty")
    assert set(groups.keys()) == {"easy", "hard"}
    assert len(groups["easy"]) == 1
    assert len(groups["hard"]) == 1


def test_slice_scores_unknown_case() -> None:
    from bugeval.analyze import slice_scores

    scores = _make_scores([("missing-case", "greptile", 2, 0.5)])
    groups = slice_scores(scores, {}, "difficulty")
    assert "unknown" in groups


def test_slice_scores_by_context() -> None:
    from bugeval.analyze import slice_scores_by_context
    from bugeval.result_models import NormalizedResult

    scores = _make_scores(
        [
            ("c1", "greptile", 2, 0.5),
            ("c2", "greptile", 0, 0.0),
        ]
    )
    results = {
        ("c1", "greptile"): NormalizedResult(
            test_case_id="c1", tool="greptile", context_level="diff-only"
        ),
        ("c2", "greptile"): NormalizedResult(
            test_case_id="c2", tool="greptile", context_level="diff+repo"
        ),
    }
    groups = slice_scores_by_context(scores, results)
    assert set(groups.keys()) == {"diff-only", "diff+repo"}


def test_compute_cost_per_tool() -> None:
    from bugeval.analyze import compute_cost_per_tool
    from bugeval.result_models import NormalizedResult, ResultMetadata

    scores = _make_scores(
        [
            ("c1", "greptile", 2, 0.5),
            ("c2", "greptile", 0, 0.0),
        ]
    )
    results = {
        ("c1", "greptile"): NormalizedResult(
            test_case_id="c1",
            tool="greptile",
            metadata=ResultMetadata(cost_usd=0.10),
        ),
        ("c2", "greptile"): NormalizedResult(
            test_case_id="c2",
            tool="greptile",
            metadata=ResultMetadata(cost_usd=0.05),
        ),
    }
    cost = compute_cost_per_tool(scores, results)
    assert cost["greptile"]["total_cost_usd"] == pytest.approx(0.15)
    assert cost["greptile"]["cost_per_review"] == pytest.approx(0.075)
    assert cost["greptile"]["cost_per_detection"] == pytest.approx(0.15)


def test_slice_scores_by_visibility() -> None:
    from bugeval.analyze import slice_scores

    scores = _make_scores(
        [
            ("c1", "greptile", 2, 0.5),
            ("c2", "greptile", 0, 0.0),
        ]
    )
    cases = {
        "c1": TestCase(
            id="c1",
            repo="org/repo",
            base_commit="abc",
            head_commit="def",
            fix_commit="ghi",
            category=Category("logic"),
            difficulty=Difficulty("medium"),
            severity=Severity("high"),
            language="rust",
            pr_size=PRSize("small"),
            description="test",
            expected_findings=[ExpectedFinding(file="a.rs", line=1, summary="bug")],
            visibility=Visibility.public,
        ),
        "c2": TestCase(
            id="c2",
            repo="org/repo",
            base_commit="abc",
            head_commit="def",
            fix_commit="ghi",
            category=Category("logic"),
            difficulty=Difficulty("medium"),
            severity=Severity("high"),
            language="rust",
            pr_size=PRSize("small"),
            description="test",
            expected_findings=[ExpectedFinding(file="a.rs", line=1, summary="bug")],
            visibility=Visibility.private,
        ),
    }
    groups = slice_scores(scores, cases, "visibility")
    assert set(groups.keys()) == {"public", "private"}
    assert len(groups["public"]) == 1
    assert len(groups["private"]) == 1


def test_generate_dx_markdown() -> None:
    from bugeval.analyze import generate_dx_markdown
    from bugeval.result_models import DxAssessment, NormalizedResult

    results = {
        ("c1", "greptile"): NormalizedResult(
            test_case_id="c1",
            tool="greptile",
            dx=DxAssessment(
                actionability=4, false_positive_burden=2, integration_friction=3, response_latency=5
            ),
        ),
    }
    md = generate_dx_markdown(results)
    assert "DX Assessment" in md
    assert "greptile" in md
    assert "4.0" in md


def test_analyze_skips_dx_when_absent() -> None:
    from bugeval.analyze import generate_dx_markdown
    from bugeval.result_models import NormalizedResult

    results = {
        ("c1", "greptile"): NormalizedResult(test_case_id="c1", tool="greptile"),
    }
    md = generate_dx_markdown(results)
    assert md == ""


def test_generate_slice_markdown() -> None:
    from bugeval.analyze import generate_slice_markdown

    scores = _make_scores(
        [
            ("c1", "greptile", 2, 0.5),
            ("c2", "greptile", 0, 0.0),
        ]
    )
    cases = {
        "c1": _make_case("c1", difficulty="easy"),
        "c2": _make_case("c2", difficulty="hard"),
    }
    md = generate_slice_markdown(scores, cases, "difficulty")
    assert "difficulty" in md.lower()
    assert "easy" in md
    assert "hard" in md
    assert "greptile" in md


def test_generate_confidence_band_markdown_basic() -> None:
    from bugeval.analyze import generate_confidence_band_markdown
    from bugeval.result_models import Comment, NormalizedResult

    scores = _make_scores(
        [
            ("c1", "anthropic-api", 2, 0.5),
            ("c2", "anthropic-api", 0, 0.0),
            ("c3", "anthropic-api", 3, 1.0),
        ]
    )
    results: dict = {
        ("c1", "anthropic-api"): NormalizedResult(
            test_case_id="c1",
            tool="anthropic-api",
            comments=[Comment(body="b", confidence=0.6)],
        ),
        ("c2", "anthropic-api"): NormalizedResult(
            test_case_id="c2",
            tool="anthropic-api",
            comments=[Comment(body="b", confidence=0.8)],
        ),
        ("c3", "anthropic-api"): NormalizedResult(
            test_case_id="c3",
            tool="anthropic-api",
            comments=[Comment(body="b", confidence=0.95)],
        ),
    }
    md = generate_confidence_band_markdown(scores, results)
    assert "Confidence Band" in md
    assert "0.5" in md or "[0.5" in md


def test_slice_scores_by_verified() -> None:
    from bugeval.analyze import slice_scores

    scores = _make_scores(
        [
            ("c1", "greptile", 2, 0.5),
            ("c2", "greptile", 0, 0.0),
        ]
    )
    cases = {
        "c1": TestCase(
            id="c1",
            repo="org/repo",
            base_commit="abc",
            head_commit="def",
            fix_commit="ghi",
            category=Category("logic"),
            difficulty=Difficulty("medium"),
            severity=Severity("high"),
            language="rust",
            pr_size=PRSize("small"),
            description="test",
            expected_findings=[ExpectedFinding(file="a.rs", line=1, summary="bug")],
            verified=True,
        ),
        "c2": TestCase(
            id="c2",
            repo="org/repo",
            base_commit="abc",
            head_commit="def",
            fix_commit="ghi",
            category=Category("logic"),
            difficulty=Difficulty("medium"),
            severity=Severity("high"),
            language="rust",
            pr_size=PRSize("small"),
            description="test",
            expected_findings=[ExpectedFinding(file="a.rs", line=1, summary="bug")],
            verified=False,
        ),
    }
    groups = slice_scores(scores, cases, "verified")
    assert set(groups.keys()) == {"True", "False"}
    assert len(groups["True"]) == 1
    assert len(groups["False"]) == 1


def test_generate_confidence_band_markdown_empty_when_no_confidence() -> None:
    """When no comments have confidence data, returns empty string."""
    from bugeval.analyze import generate_confidence_band_markdown
    from bugeval.result_models import Comment, NormalizedResult

    scores = _make_scores([("c1", "greptile", 2, 0.5)])
    results: dict = {
        ("c1", "greptile"): NormalizedResult(
            test_case_id="c1",
            tool="greptile",
            comments=[Comment(body="no confidence here")],
        ),
    }
    md = generate_confidence_band_markdown(scores, results)
    assert md == ""
