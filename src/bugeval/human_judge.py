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
from bugeval.pr_eval_models import default_judging, default_scoring
from bugeval.result_models import NormalizedResult

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
    cats = default_scoring().scale
    po = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n
    pe = sum(
        (sum(1 for a in scores_a if a == k) / n) * (sum(1 for b in scores_b if b == k) / n)
        for k in cats
    )
    return (po - pe) / (1 - pe) if pe < 1.0 else 1.0


def select_sample(scores: list[JudgeScore], sample_rate: float | None = None) -> list[JudgeScore]:
    """Stratified random sample by tool, targeting approximately sample_rate of total.

    Each tool contributes proportionally to the sample (at least 1 per tool if possible).
    Difficulty stratification is not applied here because JudgeScore does not carry
    difficulty metadata; callers needing difficulty stratification must pre-filter scores.
    """
    if not scores:
        return []

    resolved_rate = sample_rate if sample_rate is not None else default_judging().human_sample_rate

    # Stratify by tool to ensure proportional tool coverage
    by_tool: dict[str, list[JudgeScore]] = {}
    for s in scores:
        by_tool.setdefault(s.tool, []).append(s)

    total_n = max(1, round(len(scores) * resolved_rate))
    sampled: list[JudgeScore] = []

    # Proportional allocation per tool (at least 1 per tool if possible)
    for tool, tool_scores in by_tool.items():
        tool_n = max(1, round(len(tool_scores) * resolved_rate))
        sampled.extend(random.sample(tool_scores, min(tool_n, len(tool_scores))))

    # Trim to target if over-allocated (can happen with many small strata)
    if len(sampled) > total_n:
        sampled = random.sample(sampled, total_n)

    return sampled


def _make_tool_map(scores: list[JudgeScore]) -> dict[str, str]:
    """Assign anonymous labels (Tool-A, Tool-B, …) to real tool names."""
    tools = sorted({s.tool for s in scores})
    return {t: f"Tool-{chr(65 + i)}" for i, t in enumerate(tools)}


def _format_comments(result: NormalizedResult | None, max_chars: int = 800) -> str:
    """Render tool comments as a readable string for the human rater CSV."""
    if result is None or not result.comments:
        return "(no comments)"
    parts = []
    for i, c in enumerate(result.comments):
        loc = f"{c.file}:{c.line}" if c.file else "PR-level"
        parts.append(f"[{i}] {loc}: {c.body[:200]}")
    joined = " | ".join(parts)
    return joined[:max_chars] + ("…" if len(joined) > max_chars else "")


def export_sample(
    scores: list[JudgeScore],
    run_dir: Path,
    output_path: Path,
    sample_rate: float | None = None,
    results: dict[tuple[str, str], NormalizedResult] | None = None,
) -> None:
    """Select sample, blind tool names, write CSV for human raters.

    Writes tool_map.yaml to run_dir/human_judge/ for later de-blinding.
    The CSV includes a 'tool_comments' column so raters can evaluate the findings
    without knowing which tool produced them.
    """
    sample = select_sample(scores, sample_rate)
    tool_map = _make_tool_map(scores)

    hj_dir = run_dir / "human_judge"
    hj_dir.mkdir(exist_ok=True)
    (hj_dir / "tool_map.yaml").write_text(yaml.safe_dump(tool_map, sort_keys=True))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "test_case_id",
                "tool",
                "llm_score",
                "tool_comments",
                "human_score",
                "notes",
            ],
        )
        writer.writeheader()
        for i, s in enumerate(sample):
            normalized = results.get((s.test_case_id, s.tool)) if results else None
            writer.writerow(
                {
                    "row_id": f"{i:04d}",
                    "test_case_id": s.test_case_id,
                    "tool": tool_map.get(s.tool, s.tool),
                    "llm_score": s.score,
                    "tool_comments": _format_comments(normalized),
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

        if human_score not in default_scoring().scale:
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

    threshold = default_judging().calibration_threshold
    if not scores_dir.exists() or not hj_dir.exists():
        return {
            "kappa": 0.0, "n_pairs": 0, "threshold": threshold, "calibrated": False, "pairs": []
        }

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

    # Load normalized results so the CSV includes tool comments for raters.
    normalized: dict[tuple[str, str], NormalizedResult] = {}
    for path in resolved.glob("*.yaml"):
        if path.name == "checkpoint.yaml":
            continue
        data = yaml.safe_load(path.read_text()) or {}
        try:
            r = NormalizedResult(**data)
            normalized[(r.test_case_id, r.tool)] = r
        except Exception:
            pass

    out = Path(output_path) if output_path else resolved / "human_judge" / "sample.csv"
    out.parent.mkdir(exist_ok=True)
    export_sample(scores, resolved, out, sample_rate, results=normalized)
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
