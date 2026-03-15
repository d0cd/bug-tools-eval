"""Tests for prediction_format module — Prediction model and export/import round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from bugeval.prediction_format import (
    Prediction,
    PredictionFinding,
    export_predictions,
    import_predictions,
)
from bugeval.result_models import Comment, NormalizedResult


class TestPredictionFinding:
    def test_minimal_finding(self) -> None:
        f = PredictionFinding(summary="a bug")
        assert f.summary == "a bug"
        assert f.file == ""
        assert f.line == 0
        assert f.confidence is None

    def test_full_finding(self) -> None:
        f = PredictionFinding(
            file="src/main.rs",
            line=42,
            summary="Off-by-one",
            confidence=0.9,
            severity="high",
            category="logic",
            suggested_fix="add +1",
            reasoning="because",
        )
        assert f.file == "src/main.rs"
        assert f.line == 42
        assert f.confidence == 0.9


class TestPrediction:
    def test_minimal_prediction(self) -> None:
        p = Prediction(instance_id="leo-001", tool="my-tool")
        assert p.instance_id == "leo-001"
        assert p.tool == "my-tool"
        assert p.findings == []
        assert p.context_level == ""

    def test_with_findings(self) -> None:
        p = Prediction(
            instance_id="leo-001",
            tool="my-tool",
            context_level="diff+repo",
            findings=[PredictionFinding(summary="bug", file="a.rs", line=1)],
        )
        assert len(p.findings) == 1
        assert p.findings[0].file == "a.rs"

    def test_json_round_trip(self) -> None:
        p = Prediction(
            instance_id="leo-001",
            tool="test",
            findings=[PredictionFinding(summary="bug", file="a.rs", line=1)],
        )
        data = json.dumps(p.model_dump(mode="json"))
        loaded = Prediction(**json.loads(data))
        assert loaded.instance_id == "leo-001"
        assert loaded.findings[0].file == "a.rs"


class TestExportPredictions:
    def test_export_creates_jsonl(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        r = NormalizedResult(
            test_case_id="leo-001",
            tool="greptile",
            context_level="diff-only",
            comments=[Comment(body="bad code", file="a.rs", line=5)],
        )
        (run_dir / "leo-001-greptile.yaml").write_text(
            yaml.safe_dump(r.model_dump(mode="json"), sort_keys=False)
        )

        out_path = tmp_path / "predictions.jsonl"
        export_predictions(run_dir, out_path)

        assert out_path.exists()
        lines = [ln for ln in out_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["instance_id"] == "leo-001"
        assert obj["tool"] == "greptile"
        assert obj["context_level"] == "diff-only"
        assert len(obj["findings"]) == 1
        assert obj["findings"][0]["file"] == "a.rs"
        assert obj["findings"][0]["line"] == 5
        assert obj["findings"][0]["summary"] == "bad code"

    def test_export_skips_checkpoint(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Write a checkpoint.yaml (should be ignored)
        (run_dir / "checkpoint.yaml").write_text("state: {}")
        # Write one real result
        r = NormalizedResult(test_case_id="c1", tool="tool-a")
        (run_dir / "c1-tool-a.yaml").write_text(
            yaml.safe_dump(r.model_dump(mode="json"), sort_keys=False)
        )

        out_path = tmp_path / "out.jsonl"
        export_predictions(run_dir, out_path)
        lines = [ln for ln in out_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["instance_id"] == "c1"

    def test_export_multiple_results(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for i in range(3):
            r = NormalizedResult(test_case_id=f"case-{i:03d}", tool="greptile")
            (run_dir / f"case-{i:03d}-greptile.yaml").write_text(
                yaml.safe_dump(r.model_dump(mode="json"), sort_keys=False)
            )

        out_path = tmp_path / "out.jsonl"
        export_predictions(run_dir, out_path)
        lines = [ln for ln in out_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3


class TestImportPredictions:
    def test_import_creates_normalized_yamls(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        jsonl = tmp_path / "preds.jsonl"
        p = Prediction(
            instance_id="leo-001",
            tool="my-tool",
            context_level="diff+repo",
            findings=[
                PredictionFinding(
                    file="src/main.rs",
                    line=10,
                    summary="null deref",
                    confidence=0.85,
                )
            ],
        )
        jsonl.write_text(json.dumps(p.model_dump(mode="json")) + "\n")

        import_predictions(jsonl, run_dir)

        # Should write a NormalizedResult YAML
        written = list(run_dir.glob("*.yaml"))
        assert len(written) == 1
        data = yaml.safe_load(written[0].read_text())
        r = NormalizedResult(**data)
        assert r.test_case_id == "leo-001"
        assert r.tool == "my-tool"
        assert r.context_level == "diff+repo"
        assert len(r.comments) == 1
        assert r.comments[0].file == "src/main.rs"
        assert r.comments[0].line == 10
        assert r.comments[0].body == "null deref"
        assert r.comments[0].confidence == 0.85

    def test_import_export_round_trip(self, tmp_path: Path) -> None:
        """export → import → export produces identical JSONL."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        r = NormalizedResult(
            test_case_id="leo-001",
            tool="greptile",
            context_level="diff-only",
            comments=[Comment(body="logic error", file="a.rs", line=7, confidence=0.8)],
        )
        (run_dir / "leo-001-greptile.yaml").write_text(
            yaml.safe_dump(r.model_dump(mode="json"), sort_keys=False)
        )

        # Export
        jsonl1 = tmp_path / "preds1.jsonl"
        export_predictions(run_dir, jsonl1)

        # Import into a new dir
        run_dir2 = tmp_path / "run2"
        run_dir2.mkdir()
        import_predictions(jsonl1, run_dir2)

        # Export again
        jsonl2 = tmp_path / "preds2.jsonl"
        export_predictions(run_dir2, jsonl2)

        obj1 = json.loads(jsonl1.read_text().strip())
        obj2 = json.loads(jsonl2.read_text().strip())
        assert obj1["instance_id"] == obj2["instance_id"]
        assert obj1["tool"] == obj2["tool"]
        assert obj1["findings"][0]["file"] == obj2["findings"][0]["file"]
        assert obj1["findings"][0]["summary"] == obj2["findings"][0]["summary"]

    def test_import_skips_blank_lines(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        p = Prediction(instance_id="c1", tool="t1")
        jsonl = tmp_path / "preds.jsonl"
        jsonl.write_text(json.dumps(p.model_dump(mode="json")) + "\n\n")

        import_predictions(jsonl, run_dir)
        written = list(run_dir.glob("*.yaml"))
        assert len(written) == 1
