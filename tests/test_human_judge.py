"""Tests for human_judge module."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bugeval.human_judge import (
    cohen_kappa,
    compute_kappa_report,
    export_sample,
    import_scores,
    select_sample,
)
from bugeval.judge_models import JudgeScore


def _make_judge_score(case_id: str, tool: str, score: int) -> JudgeScore:
    return JudgeScore(
        test_case_id=case_id,
        tool=tool,
        score=score,
        votes=[score, score, score],
        reasoning="test",
    )


# --- cohen_kappa ---


def test_cohen_kappa_perfect_agreement() -> None:
    assert cohen_kappa([2, 0, 3, 1], [2, 0, 3, 1]) == pytest.approx(1.0)


def test_cohen_kappa_no_agreement() -> None:
    # All disagreements; kappa could be negative
    result = cohen_kappa([0, 0, 0, 0], [3, 3, 3, 3])
    assert result < 1.0


def test_cohen_kappa_empty() -> None:
    assert cohen_kappa([], []) == 0.0


# --- select_sample ---


def test_select_sample_rate() -> None:
    scores = [_make_judge_score(f"c{i:03d}", "greptile", 2) for i in range(20)]
    sample = select_sample(scores, sample_rate=0.25)
    assert len(sample) == 5


def test_select_sample_min_one() -> None:
    scores = [_make_judge_score("c001", "greptile", 2)]
    sample = select_sample(scores, sample_rate=0.10)
    assert len(sample) >= 1


# --- export_sample ---


def test_export_sample_writes_csv(tmp_path: Path) -> None:
    scores = [
        _make_judge_score("c001", "greptile", 2),
        _make_judge_score("c002", "coderabbit", 0),
        _make_judge_score("c003", "greptile", 3),
        _make_judge_score("c004", "coderabbit", 1),
    ]
    out_path = tmp_path / "sample.csv"
    export_sample(scores, run_dir=tmp_path, output_path=out_path, sample_rate=1.0)
    assert out_path.exists()
    rows = list(csv.DictReader(out_path.read_text().splitlines()))
    assert len(rows) == 4
    # Tool names must be blinded
    tool_values = {r["tool"] for r in rows}
    assert "greptile" not in tool_values
    assert "coderabbit" not in tool_values
    # tool_map.yaml must exist
    assert (tmp_path / "human_judge" / "tool_map.yaml").exists()


# --- import_scores ---


def test_import_scores_writes_yaml(tmp_path: Path) -> None:
    # Set up: export first to create tool_map
    scores = [
        _make_judge_score("c001", "greptile", 2),
        _make_judge_score("c002", "greptile", 0),
    ]
    csv_path = tmp_path / "sample.csv"
    export_sample(scores, run_dir=tmp_path, output_path=csv_path, sample_rate=1.0)

    # Edit CSV to add human_score
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    for r in rows:
        r["human_score"] = "2"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    import_scores(csv_path, run_dir=tmp_path)
    human_dir = tmp_path / "human_judge"
    yaml_files = list(human_dir.glob("*.yaml"))
    # Should have score YAMLs + tool_map.yaml
    score_files = [p for p in yaml_files if p.name != "tool_map.yaml"]
    assert len(score_files) == 2


# --- compute_kappa_report ---


def test_compute_kappa_report(tmp_path: Path) -> None:
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    hj_dir = tmp_path / "human_judge"
    hj_dir.mkdir()

    import yaml

    (scores_dir / "c001-greptile.yaml").write_text(
        yaml.safe_dump({"test_case_id": "c001", "tool": "greptile", "score": 2})
    )
    (hj_dir / "c001-greptile.yaml").write_text(
        yaml.safe_dump(
            {"test_case_id": "c001", "tool": "greptile", "human_score": 2, "llm_score": 2}
        )
    )

    result = compute_kappa_report(tmp_path)

    assert "kappa" in result
    assert result["n_pairs"] == 1
    assert result["threshold"] == 0.85
    assert "calibrated" in result
