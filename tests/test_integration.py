"""Integration tests: synthetic end-to-end pipeline (normalize → analyze)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from bugeval.analyze import aggregate_scores, generate_markdown, load_normalized_lookup
from bugeval.judge_models import JudgeScore, NoiseStats
from bugeval.normalize import normalize_api_result
from bugeval.result_models import NormalizedResult


def _make_scores_dir(run_dir: Path, tool: str, case_id: str, score: int) -> None:
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    js = JudgeScore(
        test_case_id=case_id,
        tool=tool,
        score=score,
        votes=[score, score, score],
        reasoning="automated test",
        noise=NoiseStats(total_comments=2, true_positives=1, snr=0.5),
    )
    (scores_dir / f"{case_id}-{tool}.yaml").write_text(
        yaml.safe_dump(js.model_dump(mode="json"), sort_keys=False)
    )


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    """Synthetic pipeline: normalize → judge scores → analyze."""
    run_dir = tmp_path / "run-test"
    run_dir.mkdir()

    # Step 1: Create raw API findings
    raw_dir = run_dir / "raw" / "case-001-greptile"
    raw_dir.mkdir(parents=True)
    findings = [{"path": "src/main.rs", "line": 42, "body": "potential off-by-one"}]
    (raw_dir / "findings.json").write_text(json.dumps(findings))
    (raw_dir / "metadata.json").write_text(json.dumps({"time_seconds": 1.2, "cost_usd": 0.05}))

    # Step 2: Normalize
    result = normalize_api_result("case-001", "greptile", "diff-only", raw_dir)
    assert result.test_case_id == "case-001"
    assert result.tool == "greptile"
    assert result.metadata.cost_usd == 0.05

    # Write normalized result YAML
    (run_dir / "case-001-greptile.yaml").write_text(
        yaml.safe_dump(result.model_dump(mode="json"), sort_keys=False)
    )

    # Step 3: Create judge scores
    _make_scores_dir(run_dir, "greptile", "case-001", score=3)

    # Step 4: Load normalized lookup and judge scores
    normalized = load_normalized_lookup(run_dir)
    assert ("case-001", "greptile") in normalized

    scores_dir = run_dir / "scores"
    scores: list[JudgeScore] = []
    for path in scores_dir.glob("*.yaml"):
        scores.append(JudgeScore(**yaml.safe_load(path.read_text())))

    # Step 5: Aggregate and generate report
    agg = aggregate_scores(scores)
    assert "greptile" in agg
    assert agg["greptile"]["catch_rate"] == 1.0

    out_dir = run_dir / "analysis"
    out_dir.mkdir()
    md = generate_markdown(agg)
    (out_dir / "report.md").write_text(md)

    assert (out_dir / "report.md").exists()
    report = (out_dir / "report.md").read_text()
    assert "greptile" in report


def test_pipeline_produces_analysis_report(tmp_path: Path) -> None:
    """Verify report contains expected tool name after full aggregation."""
    run_dir = tmp_path / "run-test2"
    run_dir.mkdir()

    tool_name = "coderabbit"

    # Setup normalized result
    r = NormalizedResult(test_case_id="case-002", tool=tool_name, context_level="diff-only")
    (run_dir / f"case-002-{tool_name}.yaml").write_text(
        yaml.safe_dump(r.model_dump(mode="json"), sort_keys=False)
    )

    # Setup judge score
    _make_scores_dir(run_dir, tool_name, "case-002", score=2)

    scores_dir = run_dir / "scores"
    scores: list[JudgeScore] = []
    for path in scores_dir.glob("*.yaml"):
        scores.append(JudgeScore(**yaml.safe_load(path.read_text())))

    agg = aggregate_scores(scores)
    md = generate_markdown(agg)

    assert tool_name in md
    assert "Detection Rate" in md
