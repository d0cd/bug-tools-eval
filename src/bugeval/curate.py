"""Curation pass: auto-detect and exclude bad test cases."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from bugeval.io import load_cases, save_case
from bugeval.models import TestCase

log = logging.getLogger(__name__)

# Reasons for automatic exclusion
REASON_NO_BUGGY_LINES = "no-buggy-lines"
REASON_TOO_MANY_BUGGY_LINES = "too-many-buggy-lines"
REASON_FEATURE_NOT_FIX = "feature-not-fix"
REASON_WRONG_DESCRIPTION = "wrong-bug-description"
REASON_DUPLICATE_INTRODUCING = "duplicate-introducing-pr"

MAX_BUGGY_LINES = 50
FEATURE_KEYWORDS = ["[Feature]", "[Feat]", "feat:"]


def auto_curate_case(case: TestCase) -> str | None:
    """Return an exclusion reason if the case should be excluded, else None."""
    truth = case.truth

    # No buggy lines — can't be mechanically scored
    if truth is not None and not truth.buggy_lines:
        return REASON_NO_BUGGY_LINES

    # Too many buggy lines — likely a refactor
    if truth is not None and len(truth.buggy_lines) > MAX_BUGGY_LINES:
        return REASON_TOO_MANY_BUGGY_LINES

    # Fix PR is actually a feature
    fix_title = case.fix_pr_title
    if any(kw.lower() in fix_title.lower() for kw in FEATURE_KEYWORDS):
        # Only if "fix" is NOT also in the title
        if "fix" not in fix_title.lower():
            return REASON_FEATURE_NOT_FIX

    return None


def find_duplicate_introducing(cases: list[TestCase]) -> set[str]:
    """Find cases that share an introducing PR (keep first, exclude rest)."""
    seen: dict[int | None, str] = {}
    duplicates: set[str] = set()
    for case in cases:
        ipn = case.introducing_pr_number
        if ipn is None:
            continue
        if ipn in seen:
            duplicates.add(case.id)
        else:
            seen[ipn] = case.id
    return duplicates


def curate_cases(
    cases_dir: Path, *, dry_run: bool = False, reset: bool = False,
) -> dict[str, list[str]]:
    """Run curation pass on all cases. Returns {reason: [case_ids]}."""

    all_cases = load_cases(cases_dir, include_excluded=True)
    results: dict[str, list[str]] = {}

    if reset:
        for case in all_cases:
            if case.excluded:
                case.excluded = False
                case.excluded_reason = ""
                if not dry_run:
                    path = _find_case_path(cases_dir, case.id)
                    if path:
                        save_case(case, path)
        return results

    # Find introducing PR duplicates
    dup_ids = find_duplicate_introducing(all_cases)

    for case in all_cases:
        if case.excluded:
            results.setdefault("already-excluded", []).append(case.id)
            continue

        reason = auto_curate_case(case)
        if reason is None and case.id in dup_ids:
            reason = REASON_DUPLICATE_INTRODUCING

        if reason:
            results.setdefault(reason, []).append(case.id)
            if not dry_run:
                case.excluded = True
                case.excluded_reason = reason
                path = _find_case_path(cases_dir, case.id)
                if path:
                    save_case(case, path)

    return results


def _find_case_path(cases_dir: Path, case_id: str) -> Path | None:
    """Find the YAML file for a case ID."""
    for p in cases_dir.rglob("*.yaml"):
        if p.stem == case_id:
            return p
    return None


@click.command()
@click.option("--cases-dir", required=True, help="Directory with case YAMLs")
@click.option("--dry-run", is_flag=True, help="Show what would be excluded without modifying")
@click.option("--reset", is_flag=True, help="Reset all exclusions")
def curate(cases_dir: str, dry_run: bool, reset: bool) -> None:
    """Auto-curate test cases: detect and exclude bad cases."""
    results = curate_cases(Path(cases_dir), dry_run=dry_run, reset=reset)

    if reset:
        click.echo("Reset all exclusions.")
        return

    total_excluded = 0
    for reason, case_ids in sorted(results.items()):
        if reason == "already-excluded":
            click.echo(f"  Already excluded: {len(case_ids)}")
            continue
        action = "Would exclude" if dry_run else "Excluded"
        click.echo(f"  {action} ({reason}): {', '.join(case_ids)}")
        total_excluded += len(case_ids)

    click.echo(f"\n{'Would exclude' if dry_run else 'Excluded'} {total_excluded} cases.")
