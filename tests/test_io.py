"""Tests for YAML I/O and checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path

from bugeval.io import (
    load_case,
    load_cases,
    load_checkpoint,
    load_result,
    load_score,
    save_case,
    save_checkpoint,
    save_result,
    save_score,
    write_run_metadata,
)
from bugeval.models import CaseKind, TestCase
from bugeval.result_models import ToolResult
from bugeval.score_models import CaseScore


class TestCaseIO:
    def test_round_trip(self, tmp_path: Path, sample_case: TestCase) -> None:
        p = tmp_path / "case.yaml"
        save_case(sample_case, p)
        loaded = load_case(p)
        assert loaded.id == sample_case.id
        assert loaded.kind == CaseKind.bug
        assert loaded.truth is not None
        assert loaded.truth.blame_confidence == "A"
        assert len(loaded.truth.buggy_lines) == 1
        assert loaded.truth.buggy_lines[0].file == "consensus/src/worker.rs"
        assert loaded.validation is not None
        assert loaded.validation.agreement is True
        assert loaded.issue_bodies == sample_case.issue_bodies
        assert len(loaded.related_prs) == 2
        assert loaded.bug_latency_days == 25

    def test_round_trip_clean(self, tmp_path: Path, clean_case: TestCase) -> None:
        p = tmp_path / "clean.yaml"
        save_case(clean_case, p)
        loaded = load_case(p)
        assert loaded.kind == CaseKind.clean
        assert loaded.truth is None

    def test_load_cases(
        self, tmp_path: Path, sample_case: TestCase, clean_case: TestCase
    ) -> None:
        d = tmp_path / "cases" / "repo"
        d.mkdir(parents=True)
        save_case(sample_case, d / "case-001.yaml")
        save_case(clean_case, d / "case-002.yaml")
        loaded = load_cases(tmp_path / "cases")
        assert len(loaded) == 2
        ids = {c.id for c in loaded}
        assert ids == {"snarkVM-001", "clean-001"}


class TestResultIO:
    def test_round_trip(self, tmp_path: Path, sample_result: ToolResult) -> None:
        p = tmp_path / "result.yaml"
        save_result(sample_result, p)
        loaded = load_result(p)
        assert loaded.case_id == "snarkVM-001"
        assert loaded.tool == "copilot"
        assert len(loaded.comments) == 2
        assert loaded.comments[0].file == "consensus/src/worker.rs"
        assert loaded.time_seconds == 45.2


class TestScoreIO:
    def test_round_trip(self, tmp_path: Path, sample_score: CaseScore) -> None:
        p = tmp_path / "score.yaml"
        save_score(sample_score, p)
        loaded = load_score(p)
        assert loaded.case_id == "snarkVM-001"
        assert loaded.caught is True
        assert loaded.detection_score == 3
        assert len(loaded.comment_scores) == 2


class TestCheckpoint:
    def test_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "checkpoint.json"
        assert load_checkpoint(p) == set()

    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "checkpoint.json"
        done = {"case-001", "case-002", "case-003"}
        save_checkpoint(done, p)
        loaded = load_checkpoint(p)
        assert loaded == done

    def test_idempotent(self, tmp_path: Path) -> None:
        p = tmp_path / "checkpoint.json"
        save_checkpoint({"a", "b"}, p)
        save_checkpoint({"a", "b", "c"}, p)
        assert load_checkpoint(p) == {"a", "b", "c"}


class TestWriteRunMetadata:
    def test_creates_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        write_run_metadata(run_dir, "agent", "diff-only", cases_dir)
        assert (run_dir / "run_metadata.json").exists()

    def test_contains_tool_and_context(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        write_run_metadata(
            run_dir, "greptile", "diff+repo", cases_dir,
            model="claude-opus-4-6", thinking_budget=1024, timeout=600,
        )
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        assert meta["tool"] == "greptile"
        assert meta["context_level"] == "diff+repo"
        assert meta["model"] == "claude-opus-4-6"
        assert meta["thinking_budget"] == 1024
        assert meta["timeout"] == 600
        assert "created_at" in meta
        assert "python_version" in meta

    def test_handles_missing_config(
        self, tmp_path: Path, monkeypatch: object,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        # Change to tmp_path so config/config.yaml does not exist
        monkeypatch.chdir(tmp_path)  # type: ignore[union-attr]
        write_run_metadata(run_dir, "agent", "", cases_dir)
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        assert "config_sha256" not in meta
        assert meta["tool"] == "agent"
