"""Standardized prediction format for bug-tools-eval.

Allows external tools to benchmark against the dataset by producing a JSONL
file in the Prediction format, which can then be imported and run through the
normalize → judge → analyze pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml
from pydantic import BaseModel, ValidationError

from bugeval.result_models import Comment, NormalizedResult


class PredictionFinding(BaseModel):
    """A single finding in a prediction."""

    file: str = ""
    line: int = 0
    summary: str
    confidence: float | None = None
    severity: str | None = None
    category: str | None = None
    suggested_fix: str | None = None
    reasoning: str | None = None


class Prediction(BaseModel):
    """Standardized prediction schema for one (instance, tool) pair."""

    instance_id: str
    tool: str
    context_level: str = ""
    findings: list[PredictionFinding] = []


def _normalized_to_prediction(result: NormalizedResult) -> Prediction:
    """Convert a NormalizedResult to a Prediction."""
    findings = [
        PredictionFinding(
            file=c.file,
            line=c.line,
            summary=c.body,
            confidence=c.confidence,
            severity=c.severity,
            category=c.category,
            suggested_fix=c.suggested_fix,
            reasoning=c.reasoning,
        )
        for c in result.comments
    ]
    return Prediction(
        instance_id=result.test_case_id,
        tool=result.tool,
        context_level=result.context_level,
        findings=findings,
    )


def _prediction_to_normalized(pred: Prediction) -> NormalizedResult:
    """Convert a Prediction to a NormalizedResult."""
    comments = [
        Comment(
            file=f.file,
            line=f.line,
            body=f.summary,
            confidence=f.confidence,
            severity=f.severity,
            category=f.category,
            suggested_fix=f.suggested_fix,
            reasoning=f.reasoning,
        )
        for f in pred.findings
    ]
    return NormalizedResult(
        test_case_id=pred.instance_id,
        tool=pred.tool,
        context_level=pred.context_level,
        comments=comments,
    )


def export_predictions(run_dir: Path, out_path: Path) -> int:
    """Export NormalizedResult YAMLs from run_dir to a JSONL file.

    Returns the number of predictions written.
    """
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for path in sorted(run_dir.glob("*.yaml")):
            if path.name == "checkpoint.yaml":
                continue
            data = yaml.safe_load(path.read_text()) or {}
            try:
                result = NormalizedResult(**data)
            except ValidationError as exc:
                print(f"Warning: skipping {path.name} — {exc}", file=sys.stderr)
                continue
            pred = _normalized_to_prediction(result)
            f.write(json.dumps(pred.model_dump(mode="json")) + "\n")
            count += 1
    return count


@click.command("export-predictions")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Run directory containing NormalizedResult YAML files.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output JSONL file path.",
)
def export_predictions_cmd(run_dir: Path, output: Path) -> None:
    """Export NormalizedResult YAMLs from a run directory to a predictions JSONL file."""
    count = export_predictions(run_dir, output)
    click.echo(f"Exported {count} prediction(s) → {output}")


@click.command("import-predictions")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input JSONL file containing predictions.",
)
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Run directory to write NormalizedResult YAML files into.",
)
def import_predictions_cmd(input_path: Path, run_dir: Path) -> None:
    """Import predictions from a JSONL file into a run directory as NormalizedResult YAMLs."""
    run_dir.mkdir(parents=True, exist_ok=True)
    count = import_predictions(input_path, run_dir)
    click.echo(f"Imported {count} prediction(s) → {run_dir}")


def import_predictions(jsonl_path: Path, run_dir: Path) -> int:
    """Import predictions from a JSONL file into run_dir as NormalizedResult YAMLs.

    Returns the number of predictions imported.
    """
    count = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            pred = Prediction(**json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            print(f"Warning: skipping line — {exc}", file=sys.stderr)
            continue
        result = _prediction_to_normalized(pred)
        filename = f"{pred.instance_id}-{pred.tool}.yaml"
        (run_dir / filename).write_text(
            yaml.safe_dump(result.model_dump(mode="json"), sort_keys=False)
        )
        count += 1
    return count
