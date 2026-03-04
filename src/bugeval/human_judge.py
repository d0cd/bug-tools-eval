"""Human judge calibration workflow: export blinded CSV, import scores, compute Cohen's kappa."""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Any

import click
import yaml

from bugeval.judge_models import JudgeScore

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_path_component(value: str) -> str:
    """Validate that a path component is safe (alphanumerics, hyphens, underscores only)."""
    if not _SAFE_ID_RE.match(value):
        raise ValueError(f"Unsafe path component: {value!r}")
    return value


def cohen_kappa(scores_a: list[int], scores_b: list[int]) -> float:
    """Compute Cohen's kappa between two raters. Both lists must be same length."""
    n = len(scores_a)
    if n == 0:
        return 0.0
    cats = list(range(4))  # scores 0–3
    po = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n
    pe = sum(
        (sum(1 for a in scores_a if a == k) / n) * (sum(1 for b in scores_b if b == k) / n)
        for k in cats
    )
    return (po - pe) / (1 - pe) if pe < 1.0 else 1.0


def select_sample(scores: list[JudgeScore], sample_rate: float = 0.25) -> list[JudgeScore]:
    """Select a random sample of scores. Always returns at least 1 if scores non-empty."""
    n = max(1, round(len(scores) * sample_rate))
    return random.sample(scores, min(n, len(scores)))


def _make_tool_map(scores: list[JudgeScore]) -> dict[str, str]:
    """Assign anonymous labels (Tool-A, Tool-B, …) to real tool names."""
    tools = sorted({s.tool for s in scores})
    return {t: f"Tool-{chr(65 + i)}" for i, t in enumerate(tools)}


def export_sample(
    scores: list[JudgeScore],
    run_dir: Path,
    output_path: Path,
    sample_rate: float = 0.25,
) -> None:
    """Select sample, blind tool names, write CSV for human raters.

    Writes tool_map.yaml to run_dir/human_judge/ for later de-blinding.
    """
    sample = select_sample(scores, sample_rate)
    tool_map = _make_tool_map(scores)

    hj_dir = run_dir / "human_judge"
    hj_dir.mkdir(exist_ok=True)
    (hj_dir / "tool_map.yaml").write_text(yaml.safe_dump(tool_map, sort_keys=True))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_id", "test_case_id", "tool", "llm_score", "human_score", "notes"],
        )
        writer.writeheader()
        for i, s in enumerate(sample):
            writer.writerow(
                {
                    "row_id": f"{i:04d}",
                    "test_case_id": s.test_case_id,
                    "tool": tool_map.get(s.tool, s.tool),
                    "llm_score": s.score,
                    "human_score": "",
                    "notes": "",
                }
            )


def import_scores(input_path: Path, run_dir: Path) -> None:
    """Read a filled-in human judge CSV and write one YAML per row to run_dir/human_judge/.

    Requires run_dir/human_judge/tool_map.yaml (written by export_sample).
    """
    hj_dir = run_dir / "human_judge"
    tool_map_path = hj_dir / "tool_map.yaml"
    if not tool_map_path.exists():
        raise FileNotFoundError(f"tool_map.yaml missing in {hj_dir} — run export first")

    tool_map: dict[str, str] = yaml.safe_load(tool_map_path.read_text()) or {}
    reverse_map = {v: k for k, v in tool_map.items()}

    rows = list(csv.DictReader(input_path.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        human_score_raw = row.get("human_score", "").strip()
        if not human_score_raw:
            continue
        try:
            human_score = int(human_score_raw)
        except ValueError:
            click.echo(
                f"Warning: invalid human_score '{human_score_raw}' in row {row.get('row_id')}",
                err=True,
            )
            continue

        if human_score not in range(4):  # 0-3 only
            click.echo(
                f"Warning: human_score {human_score} out of range [0-3] in row {row.get('row_id')}",
                err=True,
            )
            continue

        real_tool = reverse_map.get(row["tool"], row["tool"])
        case_id = row["test_case_id"]
        try:
            safe_case_id = _safe_path_component(case_id)
            safe_tool = _safe_path_component(real_tool)
        except ValueError as exc:
            click.echo(f"Warning: skipping row {row.get('row_id')} — {exc}", err=True)
            continue

        try:
            llm_score_val = int(row.get("llm_score", 0))
        except ValueError:
            llm_score_val = 0

        out: dict[str, Any] = {
            "test_case_id": case_id,
            "tool": real_tool,
            "human_score": human_score,
            "llm_score": llm_score_val,
            "notes": row.get("notes", ""),
        }
        out_path = hj_dir / f"{safe_case_id}-{safe_tool}.yaml"
        out_path.write_text(yaml.safe_dump(out, sort_keys=False))


def compute_kappa_report(run_dir: Path) -> dict[str, Any]:
    """Load LLM scores (scores/) and human scores (human_judge/), compute kappa.

    Returns dict with: kappa, n_pairs, threshold, calibrated (bool), pairs.
    """
    scores_dir = run_dir / "scores"
    hj_dir = run_dir / "human_judge"

    if not scores_dir.exists():
        return {"kappa": 0.0, "n_pairs": 0, "threshold": 0.85, "calibrated": False, "pairs": []}
    if not hj_dir.exists():
        return {"kappa": 0.0, "n_pairs": 0, "threshold": 0.85, "calibrated": False, "pairs": []}

    llm_lookup: dict[tuple[str, str], int] = {}
    for path in scores_dir.glob("*.yaml"):
        data = yaml.safe_load(path.read_text()) or {}
        cid = data.get("test_case_id", "")
        tool = data.get("tool", "")
        score = data.get("score", 0)
        if cid and tool:
            llm_lookup[(cid, tool)] = int(score)

    llm_scores: list[int] = []
    human_scores: list[int] = []
    pairs: list[dict[str, Any]] = []

    for path in hj_dir.glob("*.yaml"):
        if path.name == "tool_map.yaml":
            continue
        data = yaml.safe_load(path.read_text()) or {}
        cid = data.get("test_case_id", "")
        tool = data.get("tool", "")
        human_score = data.get("human_score")
        llm_score = llm_lookup.get((cid, tool))
        if human_score is not None and llm_score is not None:
            llm_scores.append(llm_score)
            human_scores.append(int(human_score))
            pairs.append(
                {"test_case_id": cid, "tool": tool, "llm": llm_score, "human": int(human_score)}
            )

    kappa = cohen_kappa(llm_scores, human_scores)
    threshold = 0.85
    return {
        "kappa": kappa,
        "n_pairs": len(pairs),
        "threshold": threshold,
        "calibrated": kappa >= threshold,
        "pairs": pairs,
    }


@click.group("human-judge")
def human_judge() -> None:
    """Human judge calibration: export blinded CSV, import scores, compute kappa."""


@human_judge.command("export")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to run directory (must contain scores/)",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Output CSV path (default: {run_dir}/human_judge/sample.csv)",
)
@click.option("--sample-rate", default=0.25, show_default=True, help="Fraction to sample")
def export_cmd(run_dir: str, output_path: str | None, sample_rate: float) -> None:
    """Export blinded sample CSV for human raters."""
    resolved = Path(run_dir)
    scores = []
    for path in sorted((resolved / "scores").glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        try:
            scores.append(JudgeScore(**data))
        except Exception as exc:
            click.echo(f"Warning: skipping {path.name} — {exc}", err=True)

    if not scores:
        click.echo("No scores found.")
        return

    out = Path(output_path) if output_path else resolved / "human_judge" / "sample.csv"
    out.parent.mkdir(exist_ok=True)
    export_sample(scores, resolved, out, sample_rate)
    click.echo(f"Exported {round(len(scores) * sample_rate)} rows → {out}")
    click.echo(f"Tool map → {resolved / 'human_judge' / 'tool_map.yaml'}")


@human_judge.command("import-scores")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to run directory",
)
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filled-in CSV from human raters",
)
def import_cmd(run_dir: str, input_path: str) -> None:
    """Import human scores from filled-in CSV."""
    import_scores(Path(input_path), Path(run_dir))
    click.echo(f"Scores imported → {Path(run_dir) / 'human_judge'}/")


@human_judge.command("kappa")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to run directory",
)
def kappa_cmd(run_dir: str) -> None:
    """Compute Cohen's kappa between LLM judge and human scores."""
    report = compute_kappa_report(Path(run_dir))
    click.echo(f"Pairs: {report['n_pairs']}")
    click.echo(f"Kappa: {report['kappa']:.3f} (threshold: {report['threshold']})")
    if report["calibrated"]:
        click.echo("CALIBRATED — LLM judge approved for scale-up.")
    else:
        click.echo("NOT CALIBRATED — Review judge prompt and re-run calibration.")
