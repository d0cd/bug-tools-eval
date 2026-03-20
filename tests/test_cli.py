"""Tests for CLI entry point."""

from __future__ import annotations

from click.testing import CliRunner

from bugeval.cli import cli


class TestCLI:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Bug-finding evaluation" in result.output

    def test_mine_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["mine", "--help"])
        assert result.exit_code == 0
        assert "--repo" in result.output

    def test_blame_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["blame", "--help"])
        assert result.exit_code == 0

    def test_ground_truth_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["ground-truth", "--help"])
        assert result.exit_code == 0

    def test_validate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0

    def test_clean_cases_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["clean-cases", "--help"])
        assert result.exit_code == 0

    def test_evaluate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--tool" in result.output

    def test_score_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["score", "--help"])
        assert result.exit_code == 0

    def test_analyze_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_mine_description(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["mine", "--help"])
        assert result.exit_code == 0
        assert "Scrape fix PRs" in result.output
