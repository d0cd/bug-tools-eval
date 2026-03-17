# src/bugeval/pipeline.py
"""Pipeline command: chain normalize → judge → analyze in one shot."""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from bugeval.analyze import run_analyze
from bugeval.judge import judge_normalized_results
from bugeval.normalize import (
    _parse_raw_dir_name,
    discover_raw_dirs,
    normalize_agent_result,
    normalize_api_result,
    normalize_pr_result,
)
from bugeval.pr_eval_models import EvalConfig, ToolType, load_eval_config


def _run_normalize(run_dir: Path, config_path: Path, context_level: str, dry_run: bool) -> None:
    """Normalize all raw tool outputs found under run_dir/raw/."""
    config: EvalConfig = load_eval_config(config_path)
    tool_types = {t.name: t.type for t in config.tools}
    raw_dirs = discover_raw_dirs(run_dir)

    if not raw_dirs:
        click.echo(f"No raw output directories found in {run_dir / 'raw'}")
        return

    success = 0
    for raw_dir in raw_dirs:
        case_id, tool_name = _parse_raw_dir_name(raw_dir.name)
        tool_type = tool_types.get(tool_name)

        try:
            if tool_type == ToolType.pr:
                result = normalize_pr_result(case_id, tool_name, raw_dir)
            elif tool_type == ToolType.api:
                result = normalize_api_result(case_id, tool_name, context_level, raw_dir)
            elif tool_type == ToolType.agent:
                result = normalize_agent_result(case_id, tool_name, raw_dir)
            else:
                click.echo(f"[skip] {raw_dir.name}: unknown tool type for '{tool_name}'")
                continue

            out_path = run_dir / f"{case_id}-{tool_name}.yaml"
            if dry_run:
                click.echo(f"[dry-run] would write {out_path.name}")
            else:
                out_path.write_text(yaml.safe_dump(result.model_dump(mode="json"), sort_keys=False))
                click.echo(f"[ok] {out_path.name}")
            success += 1
        except Exception as exc:
            click.echo(f"[error] {raw_dir.name}: {exc}", err=True)

    if dry_run:
        click.echo(f"Would normalize {success}/{len(raw_dirs)} results → {run_dir}/")
    else:
        click.echo(f"Normalized {success}/{len(raw_dirs)} results → {run_dir}/")


def _run_judge(
    run_dir: Path, cases_dir: Path, dry_run: bool, via_cli: bool = False, max_concurrent: int = 1
) -> None:
    """Judge all normalized results in run_dir."""
    count = judge_normalized_results(
        run_dir, cases_dir, dry_run, via_cli=via_cli, max_concurrent=max_concurrent
    )
    click.echo(f"Judged {count} result(s).")


def _run_analyze(run_dir: Path, cases_dir: Path, no_charts: bool) -> None:
    """Aggregate judge scores into comparison tables and charts."""
    run_analyze(run_dir, cases_dir, no_charts)


@click.command("pipeline")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to completed run directory",
)
@click.option(
    "--config",
    "config_path",
    default="config/config.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml",
)
@click.option(
    "--cases-dir",
    default="cases/",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory containing case YAML files",
)
@click.option(
    "--context-level",
    default="diff-only",
    show_default=True,
    type=click.Choice(["diff-only", "diff+repo", "diff+repo+domain"]),
    help="Context level used for API tools during normalization",
)
@click.option("--no-charts", is_flag=True, default=False, help="Skip chart generation")
@click.option("--dry-run", is_flag=True, default=False, help="Skip writes and LLM calls")
@click.option(
    "--via-cli",
    is_flag=True,
    default=False,
    help="Use claude CLI subprocess for judging instead of Anthropic API",
)
@click.option(
    "--max-concurrent",
    default=1,
    show_default=True,
    type=int,
    help="Number of cases to judge in parallel",
)
def pipeline(
    run_dir: str,
    config_path: str,
    cases_dir: str,
    context_level: str,
    no_charts: bool,
    dry_run: bool,
    via_cli: bool,
    max_concurrent: int,
) -> None:
    """Normalize → judge → analyze a completed eval run in one shot."""
    resolved = Path(run_dir)
    click.echo("=== Stage 1: normalize ===")
    _run_normalize(resolved, Path(config_path), context_level, dry_run)
    click.echo("=== Stage 2: judge ===")
    _run_judge(resolved, Path(cases_dir), dry_run, via_cli=via_cli, max_concurrent=max_concurrent)
    click.echo("=== Stage 3: analyze ===")
    _run_analyze(resolved, Path(cases_dir), no_charts)
    click.echo("Pipeline complete.")
