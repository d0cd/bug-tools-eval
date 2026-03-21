"""CLI entry point for bugeval."""

from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """Bug-finding evaluation framework."""


@cli.command()
@click.option("--repo", required=True, help="GitHub repo (e.g. ProvableHQ/snarkVM)")
@click.option("--limit", default=200, help="Max PRs to scrape")
@click.option("--since", default="2023-01-01", help="Only PRs merged after this date")
@click.option("--output-dir", default="cases", help="Output directory for case YAMLs")
@click.option("--concurrency", default=1, help="Parallel workers")
def mine(repo: str, limit: int, since: str, output_dir: str, concurrency: int) -> None:
    """Scrape fix PRs and build initial test cases."""
    from pathlib import Path

    from bugeval.mine import mine_repo

    cases = mine_repo(
        repo=repo,
        limit=limit,
        since=since,
        output_dir=Path(output_dir),
        concurrency=concurrency,
    )
    click.echo(f"Mined {len(cases)} cases from {repo}")


@cli.command()
@click.option("--cases-dir", required=True, help="Directory containing case YAMLs")
@click.option("--repo-dir", required=True, help="Path to local repo clone")
@click.option("--concurrency", default=1, help="Parallel workers")
def blame(cases_dir: str, repo_dir: str, concurrency: int) -> None:
    """Find introducing commits via git blame."""
    from pathlib import Path

    from bugeval.blame import blame_cases

    blame_cases(Path(cases_dir), Path(repo_dir), concurrency)


@cli.command("ground-truth")
@click.option("--cases-dir", required=True, help="Directory containing case YAMLs")
@click.option("--repo-dir", required=True, help="Path to local repo clone")
@click.option("--concurrency", default=1, help="Parallel workers")
def ground_truth(cases_dir: str, repo_dir: str, concurrency: int) -> None:
    """Build ground truth via diff intersection."""
    from pathlib import Path

    from bugeval.ground_truth import build_ground_truth

    build_ground_truth(Path(cases_dir), Path(repo_dir), concurrency)


@cli.command()
@click.option("--cases-dir", required=True, help="Directory containing case YAMLs")
@click.option("--repo-dir", default="", help="Path to local repo clone (for diffs)")
@click.option("--models", default="claude,gemini", help="Models for cross-validation")
@click.option("--concurrency", default=5, help="Parallel workers")
@click.option("--dry-run", is_flag=True, help="Validate without calling LLMs")
def validate(
    cases_dir: str, repo_dir: str, models: str, concurrency: int, dry_run: bool
) -> None:
    """Cross-model validation of ground truth."""
    from pathlib import Path

    from bugeval.validate import validate_cases

    model_list = [m.strip() for m in models.split(",")]
    validate_cases(Path(cases_dir), Path(repo_dir), model_list, concurrency, dry_run)


@cli.command("clean-cases")
@click.option("--repo", required=True, help="GitHub repo")
@click.option("--count", default=50, help="Number of clean cases to generate")
@click.option("--cases-dir", default="cases", help="Output directory")
@click.option("--since", default="2023-01-01", help="Only PRs merged after this date")
def clean_cases(repo: str, count: int, cases_dir: str, since: str) -> None:
    """Generate negative control cases (clean PRs)."""
    from pathlib import Path

    from bugeval.clean_cases import mine_clean_cases

    cases = mine_clean_cases(repo, count, Path(cases_dir), since)
    click.echo(f"Generated {len(cases)} clean cases from {repo}")


@cli.command()
@click.option(
    "--tool", required=True,
    help="Tool: copilot, greptile, coderabbit, agent, agent-gemini,"
    " agent-openai, agent-cli-claude, agent-cli-gemini, agent-cli-codex,"
    " agent-sdk",
)
@click.option("--cases-dir", default="cases", help="Test cases directory")
@click.option("--run-dir", required=True, help="Output directory for results")
@click.option(
    "--context",
    default="",
    help="Context level for agent (diff-only, diff+repo, diff+repo+domain)",
)
@click.option("--concurrency", default=1, help="Parallel workers")
@click.option("--timeout", default=300, help="Timeout per case in seconds")
@click.option("--dry-run", is_flag=True, help="Validate setup without running tools")
@click.option("--repo-dir", default="", help="Path to local repo clone")
@click.option(
    "--thinking-budget", default=0, type=int,
    help="Extended thinking budget tokens (0=disabled, agent only)",
)
@click.option(
    "--model", default="",
    help="Model override for agent runners (e.g. claude-opus-4-6)",
)
@click.option(
    "--org", default="",
    help="GitHub org for PR tool forks (copilot, greptile, coderabbit)",
)
@click.option(
    "--docker", is_flag=True,
    help="Run agent in Docker container (allows Bash tool safely)",
)
@click.option(
    "--docker-image", default="bugeval-agent",
    help="Docker image name for --docker mode",
)
def evaluate(
    tool: str,
    cases_dir: str,
    run_dir: str,
    context: str,
    repo_dir: str,
    concurrency: int,
    timeout: int,
    dry_run: bool,
    thinking_budget: int,
    model: str,
    org: str,
    docker: bool,
    docker_image: str,
) -> None:
    """Run a tool against test cases."""
    from pathlib import Path

    from bugeval.evaluate import evaluate_tool

    evaluate_tool(
        tool,
        Path(cases_dir),
        Path(run_dir),
        context,
        Path(repo_dir),
        concurrency,
        timeout,
        dry_run,
        thinking_budget=thinking_budget,
        model=model,
        org=org,
        docker=docker,
        docker_image=docker_image,
    )


@cli.command()
@click.option("--run-dir", required=True, help="Run directory with results")
@click.option("--cases-dir", default="cases", help="Test cases directory")
@click.option("--concurrency", default=1, help="Parallel LLM scoring (not yet used)")
@click.option("--dry-run", is_flag=True, help="Mechanical scoring only, skip LLM")
@click.option(
    "--judge-model", default="claude-haiku-4-5",
    help="Model for LLM judge",
)
def score(
    run_dir: str, cases_dir: str, concurrency: int,
    dry_run: bool, judge_model: str,
) -> None:
    """Score tool results (mechanical + LLM quality)."""
    from pathlib import Path

    from bugeval.score import score_run

    # concurrency is accepted for future use but not yet implemented
    score_run(Path(run_dir), Path(cases_dir), dry_run, judge_model=judge_model)


@cli.command()
@click.option("--run-dir", required=True, help="Run directory with scores")
@click.option("--cases-dir", default="cases", help="Test cases directory")
@click.option("--no-charts", is_flag=True, help="Skip chart generation")
def analyze(run_dir: str, cases_dir: str, no_charts: bool) -> None:
    """Analyze scores and generate comparison report."""
    from pathlib import Path

    from bugeval.analyze import run_analysis

    run_analysis(Path(run_dir), Path(cases_dir), no_charts)


@cli.command("dashboard")
@click.option("--port", default=5000, show_default=True, help="Port to listen on")
@click.option(
    "--cases-dir",
    default="cases",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory containing case YAML files",
)
@click.option(
    "--results-dir",
    default="results",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Root directory for run outputs",
)
@click.option("--debug", is_flag=True, default=False, help="Enable Flask debug mode")
def dashboard(port: int, cases_dir: str, results_dir: str, debug: bool) -> None:
    """Launch the local review dashboard."""
    from pathlib import Path

    from bugeval.dashboard import create_app

    app = create_app(Path(cases_dir), Path(results_dir))
    click.echo(f"Dashboard -> http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=debug)


@cli.command("add-case")
@click.option(
    "--pr-url", required=True,
    help="GitHub PR URL (e.g. https://github.com/owner/repo/pull/123)",
)
@click.option("--cases-dir", default="cases", help="Output directory")
@click.option(
    "--repo-dir", default="",
    help="Local repo clone (for blame + ground truth)",
)
@click.option("--dry-run", is_flag=True, help="Show what would be added without writing")
def add_case(pr_url: str, cases_dir: str, repo_dir: str, dry_run: bool) -> None:
    """Manually add a test case from a fix PR URL."""
    from pathlib import Path

    from bugeval.add_case import add_case_from_pr

    result = add_case_from_pr(
        pr_url,
        Path(cases_dir),
        Path(repo_dir),
        dry_run=dry_run,
    )
    if result is None:
        click.echo("Skipped: duplicate or error.")
    elif dry_run:
        click.echo(f"[dry-run] Would add: {result.id} (PR #{result.fix_pr_number})")
    else:
        click.echo(f"Added: {result.id} (PR #{result.fix_pr_number})")


# Import and register curate command
from bugeval.curate import curate  # noqa: E402

cli.add_command(curate)
