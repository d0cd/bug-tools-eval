"""CLI command: validate-cases — validate promoted test cases against a repo checkout."""

from __future__ import annotations

from pathlib import Path

import click

from bugeval.git_utils import commit_exists, format_patch, get_changed_files, get_diff_stats
from bugeval.io import load_all_cases, save_case


@click.command("validate-cases")
@click.option(
    "--repo-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the git repository checkout.",
)
@click.option(
    "--cases-dir",
    default="cases",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory containing test case YAML files.",
)
@click.option(
    "--patches-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="If provided, verify that a .patch file exists for each case.",
)
@click.option(
    "--update-stats",
    is_flag=True,
    help="Auto-populate stats (lines_added, files_changed, etc.) and save.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report validation results without writing stats back.",
)
def validate_cases(
    repo_dir: Path,
    cases_dir: Path,
    patches_dir: Path | None,
    update_stats: bool,
    dry_run: bool,
) -> None:
    """Validate test cases in cases-dir against a repo checkout."""
    if not cases_dir.exists():
        click.echo(f"Cases directory not found: {cases_dir}", err=True)
        raise SystemExit(1)

    cases = load_all_cases(cases_dir)
    if not cases:
        click.echo("No cases found.")
        return

    all_passed = True
    for case in cases:
        errors: list[str] = []

        # Check commits exist
        for attr, sha in [
            ("base_commit", case.base_commit),
            ("head_commit", case.head_commit),
            ("fix_commit", case.fix_commit),
        ]:
            if not commit_exists(sha, repo_dir):
                errors.append(f"{attr} not found: {sha[:12]}")

        # Check diff is non-empty (catches base_commit == head_commit)
        warnings: list[str] = []
        if not errors:
            try:
                diff = format_patch(case.base_commit, case.head_commit, repo_dir)
                if not diff.strip():
                    errors.append("empty diff between base_commit and head_commit")
                else:
                    changed = get_changed_files(case.base_commit, case.head_commit, repo_dir)
                    for finding in case.expected_findings:
                        if finding.file not in changed:
                            warnings.append(f"expected finding file '{finding.file}' not in diff")
            except Exception as exc:
                errors.append(f"could not compute diff: {exc}")

        # Check patch file exists (if patches_dir provided)
        if patches_dir is not None:
            patch_path = patches_dir / f"{case.id}.patch"
            if not patch_path.exists():
                errors.append(f"patch file missing: {patch_path}")

        if errors:
            click.echo(f"  FAIL {case.id}: {'; '.join(errors)}")
            all_passed = False
            continue

        for w in warnings:
            click.echo(f"  WARN {case.id}: {w}")

        # Auto-populate stats
        if update_stats and not dry_run:
            stats = get_diff_stats(case.base_commit, case.head_commit, repo_dir)
            updated = case.model_copy(update={"stats": stats})
            save_case(updated, cases_dir / f"{case.id}.yaml")
            click.echo(f"  PASS {case.id} (stats updated)")
        else:
            click.echo(f"  PASS {case.id}")

    if not all_passed:
        raise SystemExit(1)
