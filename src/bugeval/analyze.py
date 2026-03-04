# src/bugeval/analyze.py
"""Aggregate judge scores into comparison tables and charts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import click
import yaml

from bugeval.judge_models import JudgeScore
from bugeval.models import TestCase
from bugeval.result_models import NormalizedResult
from bugeval.run_pr_eval import load_cases


def compute_catch_rate(scores: list[JudgeScore]) -> float:
    """Fraction of cases scoring >= 2 (correct-id or better)."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s.score >= 2) / len(scores)


def compute_snr(scores: list[JudgeScore]) -> float:
    """Average signal-to-noise ratio across scores."""
    if not scores:
        return 0.0
    return sum(s.noise.snr for s in scores) / len(scores)


def aggregate_scores(scores: list[JudgeScore]) -> dict[str, dict[str, Any]]:
    """Group scores by tool and compute per-tool metrics."""
    by_tool: dict[str, list[JudgeScore]] = {}
    for s in scores:
        by_tool.setdefault(s.tool, []).append(s)

    result = {}
    for tool, tool_scores in sorted(by_tool.items()):
        dist = {i: sum(1 for s in tool_scores if s.score == i) for i in range(4)}
        result[tool] = {
            "count": len(tool_scores),
            "catch_rate": compute_catch_rate(tool_scores),
            "avg_snr": compute_snr(tool_scores),
            "score_dist": dist,
            "avg_score": sum(s.score for s in tool_scores) / len(tool_scores),
        }
    return result


def generate_markdown(agg: dict[str, dict[str, Any]]) -> str:
    """Produce a markdown comparison table from aggregated scores."""
    lines = [
        "| Tool | Cases | Catch Rate | Avg Score | Avg SNR |",
        "|------|-------|-----------|-----------|---------|",
    ]
    for tool, metrics in agg.items():
        lines.append(
            f"| {tool} | {metrics['count']} "
            f"| {metrics['catch_rate']:.1%} "
            f"| {metrics['avg_score']:.2f} "
            f"| {metrics['avg_snr']:.2f} |"
        )
    return "\n".join(lines)


def generate_csv(agg: dict[str, dict[str, Any]], path: Path) -> None:
    """Write aggregated scores to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tool",
                "count",
                "catch_rate",
                "avg_score",
                "avg_snr",
                "score_0",
                "score_1",
                "score_2",
                "score_3",
            ],
        )
        writer.writeheader()
        for tool, m in agg.items():
            dist = m["score_dist"]
            writer.writerow(
                {
                    "tool": tool,
                    "count": m["count"],
                    "catch_rate": round(m["catch_rate"], 4),
                    "avg_score": round(m["avg_score"], 4),
                    "avg_snr": round(m["avg_snr"], 4),
                    "score_0": dist.get(0, 0),
                    "score_1": dist.get(1, 0),
                    "score_2": dist.get(2, 0),
                    "score_3": dist.get(3, 0),
                }
            )


def load_cases_lookup(cases_dir: Path) -> dict[str, TestCase]:
    """Load all test cases from cases_dir into a dict keyed by case ID."""
    if not cases_dir.exists():
        return {}
    cases = load_cases(cases_dir)
    return {c.id: c for c in cases}


def load_normalized_lookup(run_dir: Path) -> dict[tuple[str, str], NormalizedResult]:
    """Load all NormalizedResult YAMLs from run_dir. Keys are (test_case_id, tool)."""
    from pydantic import ValidationError

    lookup: dict[tuple[str, str], NormalizedResult] = {}
    for path in run_dir.glob("*.yaml"):
        if path.name == "checkpoint.yaml":
            continue
        data = yaml.safe_load(path.read_text()) or {}
        try:
            r = NormalizedResult(**data)
            lookup[(r.test_case_id, r.tool)] = r
        except ValidationError:
            pass
    return lookup


def slice_scores(
    scores: list[JudgeScore],
    cases: dict[str, TestCase],
    dimension: str,
) -> dict[str, list[JudgeScore]]:
    """Group scores by a TestCase categorical field (e.g. 'difficulty', 'category')."""
    groups: dict[str, list[JudgeScore]] = {}
    for s in scores:
        case = cases.get(s.test_case_id)
        if case is None:
            key = "unknown"
        else:
            raw: Any = getattr(case, dimension, "unknown")
            key = raw.value if hasattr(raw, "value") else str(raw)
        groups.setdefault(key, []).append(s)
    return groups


def slice_scores_by_context(
    scores: list[JudgeScore],
    results: dict[tuple[str, str], NormalizedResult],
) -> dict[str, list[JudgeScore]]:
    """Group scores by context_level from the corresponding NormalizedResult."""
    groups: dict[str, list[JudgeScore]] = {}
    for s in scores:
        r = results.get((s.test_case_id, s.tool))
        key = r.context_level if r else "unknown"
        groups.setdefault(key, []).append(s)
    return groups


def compute_cost_per_tool(
    scores: list[JudgeScore],
    results: dict[tuple[str, str], NormalizedResult],
) -> dict[str, dict[str, float]]:
    """Compute per-tool cost_per_review and cost_per_bug_caught."""
    by_tool: dict[str, list[JudgeScore]] = {}
    for s in scores:
        by_tool.setdefault(s.tool, []).append(s)

    out: dict[str, dict[str, float]] = {}
    for tool, tool_scores in by_tool.items():
        total_cost = sum(
            results[(s.test_case_id, s.tool)].metadata.cost_usd
            for s in tool_scores
            if (s.test_case_id, s.tool) in results
        )
        n = len(tool_scores)
        bugs_caught = sum(1 for s in tool_scores if s.score >= 2)
        out[tool] = {
            "total_cost_usd": total_cost,
            "cost_per_review": total_cost / n if n > 0 else 0.0,
            "cost_per_bug_caught": total_cost / bugs_caught if bugs_caught > 0 else 0.0,
        }
    return out


def generate_slice_markdown(
    scores: list[JudgeScore],
    cases: dict[str, TestCase],
    dimension: str,
) -> str:
    """Produce a per-dimension breakdown markdown table (value x tool)."""
    groups = slice_scores(scores, cases, dimension)
    lines = [
        f"## By {dimension.replace('_', ' ').title()}",
        "",
        "| Value | Tool | Cases | Catch Rate | Avg Score |",
        "|-------|------|-------|-----------|-----------|",
    ]
    for value in sorted(groups.keys()):
        agg = aggregate_scores(groups[value])
        for tool, metrics in agg.items():
            lines.append(
                f"| {value} | {tool} | {metrics['count']} "
                f"| {metrics['catch_rate']:.1%} "
                f"| {metrics['avg_score']:.2f} |"
            )
    return "\n".join(lines)


def generate_slice_markdown_context(
    scores: list[JudgeScore],
    results: dict[tuple[str, str], NormalizedResult],
) -> str:
    """Produce a context_level breakdown markdown table."""
    groups = slice_scores_by_context(scores, results)
    lines = [
        "## By Context Level",
        "",
        "| Context | Tool | Cases | Catch Rate | Avg Score |",
        "|---------|------|-------|-----------|-----------|",
    ]
    for value in sorted(groups.keys()):
        agg = aggregate_scores(groups[value])
        for tool, metrics in agg.items():
            lines.append(
                f"| {value} | {tool} | {metrics['count']} "
                f"| {metrics['catch_rate']:.1%} "
                f"| {metrics['avg_score']:.2f} |"
            )
    return "\n".join(lines)


def generate_charts(agg: dict[str, dict[str, Any]], out_dir: Path) -> bool:
    """Write catch_rate.png and score_dist.png to out_dir.

    Returns False if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    tools = list(agg.keys())

    # Catch rate bar chart
    catch_rates = [agg[t]["catch_rate"] for t in tools]
    fig, ax = plt.subplots()
    ax.bar(tools, catch_rates)
    ax.set_ylabel("Catch Rate (score >= 2)")
    ax.set_title("Bug Detection Rate by Tool")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "catch_rate.png", dpi=150)
    plt.close(fig)

    # Score distribution stacked bar
    score_0 = [agg[t]["score_dist"].get(0, 0) for t in tools]
    score_1 = [agg[t]["score_dist"].get(1, 0) for t in tools]
    score_2 = [agg[t]["score_dist"].get(2, 0) for t in tools]
    score_3 = [agg[t]["score_dist"].get(3, 0) for t in tools]
    fig, ax = plt.subplots()
    x = range(len(tools))
    ax.bar(x, score_0, label="0 missed")
    ax.bar(x, score_1, bottom=score_0, label="1 wrong-area")
    bottom_2 = [a + b for a, b in zip(score_0, score_1)]
    ax.bar(x, score_2, bottom=bottom_2, label="2 correct-id")
    bottom_3 = [a + b for a, b in zip(bottom_2, score_2)]
    ax.bar(x, score_3, bottom=bottom_3, label="3 correct+fix")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tools, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by Tool")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "score_dist.png", dpi=150)
    plt.close(fig)

    return True


@click.command("analyze")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to run directory (must contain scores/ subdirectory)",
)
@click.option(
    "--cases-dir",
    default="cases/",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory containing case YAML files (for dimensional slicing)",
)
@click.option("--no-charts", is_flag=True, default=False, help="Skip matplotlib chart generation")
def analyze(run_dir: str, cases_dir: str, no_charts: bool) -> None:
    """Aggregate judge scores into comparison tables and charts."""
    resolved = Path(run_dir)
    scores_dir = resolved / "scores"
    if not scores_dir.exists() or not list(scores_dir.glob("*.yaml")):
        click.echo(f"No score files found in {scores_dir}")
        return

    from pydantic import ValidationError

    scores = []
    for path in sorted(scores_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        try:
            scores.append(JudgeScore(**data))
        except ValidationError as exc:
            click.echo(f"Warning: skipping {path.name} — {exc}", err=True)

    agg = aggregate_scores(scores)
    cases = load_cases_lookup(Path(cases_dir))
    results = load_normalized_lookup(resolved)

    out_dir = resolved / "analysis"
    out_dir.mkdir(exist_ok=True)

    md_lines = [generate_markdown(agg)]

    # Cost metrics (only if any cost data is available)
    cost = compute_cost_per_tool(scores, results)
    if any(m["total_cost_usd"] > 0 for m in cost.values()):
        cost_lines = [
            "\n## Cost Metrics\n",
            "| Tool | Total Cost | Per Review | Per Bug Caught |",
            "|------|-----------|-----------|---------------|",
        ]
        for tool, m in sorted(cost.items()):
            cost_lines.append(
                f"| {tool} | ${m['total_cost_usd']:.4f} "
                f"| ${m['cost_per_review']:.4f} "
                f"| ${m['cost_per_bug_caught']:.4f} |"
            )
        md_lines.append("\n".join(cost_lines))

    # Dimensional slices by TestCase fields
    if cases:
        for dim in ("category", "difficulty", "severity", "pr_size", "language"):
            md_lines.append(generate_slice_markdown(scores, cases, dim))

    # Context-level slice
    if results:
        md_lines.append(generate_slice_markdown_context(scores, results))

    full_report = "\n\n".join(md_lines)
    (out_dir / "report.md").write_text(full_report)
    click.echo(f"Report \u2192 {out_dir / 'report.md'}")

    generate_csv(agg, out_dir / "scores.csv")
    click.echo(f"CSV \u2192 {out_dir / 'scores.csv'}")

    if not no_charts:
        if generate_charts(agg, out_dir):
            click.echo(f"Charts \u2192 {out_dir}/")
        else:
            click.echo("Charts skipped (matplotlib not installed)", err=True)

    click.echo("\n" + generate_markdown(agg))
