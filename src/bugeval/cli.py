"""Main CLI entry point."""

import click

from bugeval.analyze import analyze
from bugeval.curate import curate
from bugeval.dashboard import dashboard
from bugeval.extract_patch import extract_patch
from bugeval.human_judge import human_judge
from bugeval.judge import judge
from bugeval.manage_forks import manage_forks
from bugeval.merge_cases import merge_cases
from bugeval.mine_candidates_cmd import mine_candidates
from bugeval.normalize import normalize
from bugeval.pipeline import pipeline
from bugeval.prediction_format import export_predictions_cmd, import_predictions_cmd
from bugeval.run_agent_eval import run_agent_eval
from bugeval.run_api_eval import run_api_eval
from bugeval.run_pr_eval import run_pr_eval
from bugeval.scrape_github_cmd import scrape_benchmark, scrape_github
from bugeval.status_cmd import status
from bugeval.validate_cases import validate_cases
from bugeval.validate_env import validate_env


@click.group()
def cli() -> None:
    """bugeval — AI code review tools evaluation framework."""


cli.add_command(scrape_github)
cli.add_command(scrape_benchmark)
cli.add_command(validate_cases)
cli.add_command(extract_patch)
cli.add_command(curate)
cli.add_command(merge_cases)
cli.add_command(manage_forks)
cli.add_command(run_pr_eval)
cli.add_command(run_api_eval)
cli.add_command(run_agent_eval)
cli.add_command(normalize)
cli.add_command(judge)
cli.add_command(pipeline)
cli.add_command(analyze)
cli.add_command(human_judge)
cli.add_command(mine_candidates)
cli.add_command(validate_env)
cli.add_command(status)
cli.add_command(dashboard)
cli.add_command(export_predictions_cmd)
cli.add_command(import_predictions_cmd)
