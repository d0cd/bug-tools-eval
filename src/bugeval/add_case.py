"""Add a single test case from a GitHub fix PR URL."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from bugeval.blame import populate_blame
from bugeval.ground_truth import compute_buggy_lines
from bugeval.io import save_case
from bugeval.mine import build_case_from_pr, fetch_pr_details_graphql, run_gh
from bugeval.mine import find_duplicate as find_duplicate
from bugeval.models import TestCase

log = logging.getLogger(__name__)

_PR_URL_RE = re.compile(
    r"^https://github\.com/([^/]+)/([^/]+)/pull/(\d+)/?$"
)


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Parse a GitHub PR URL into (owner, repo, number)."""
    m = _PR_URL_RE.match(url)
    if not m:
        raise ValueError(f"Invalid GitHub PR URL: {url}")
    return m.group(1), m.group(2), int(m.group(3))


def _next_case_id(cases_dir: Path, repo_slug: str) -> str:
    existing = sorted(cases_dir.glob(f"{repo_slug}-*.yaml"))
    next_num = len(existing) + 1
    return f"{repo_slug}-{next_num:03d}"


def add_case_from_pr(
    pr_url: str,
    cases_dir: Path,
    repo_dir: Path,
    *,
    dry_run: bool = False,
) -> TestCase | None:
    """Add a single test case from a GitHub fix PR URL.

    Steps:
    1. Parse the PR URL to extract owner/repo and PR number
    2. Check for duplicates (same fix_pr_number)
    3. Fetch PR metadata via gh CLI + GraphQL
    4. Build initial TestCase (using mine.build_case_from_pr)
    5. If repo_dir provided: run blame to find introducing commit
    6. If introducing commit found: build ground truth (buggy lines)
    7. Save case YAML
    """
    owner, repo_name, pr_number = parse_pr_url(pr_url)
    full_repo = f"{owner}/{repo_name}"

    # Dedup check
    dup = find_duplicate(cases_dir, pr_number)
    if dup:
        log.info("Duplicate: PR #%d already exists as %s", pr_number, dup)
        return None

    # Fetch PR data via gh CLI
    fields = (
        "number,title,body,labels,mergeCommit,baseRefName,headRefName,"
        "files,additions,deletions,changedFiles,mergedAt,author,commits,"
        "reviewDecision,statusCheckRollup"
    )
    output = run_gh(
        "pr", "view", str(pr_number), "--repo", full_repo,
        "--json", fields,
    )
    pr_data = json.loads(output)

    # GraphQL enrichment
    graphql_details = fetch_pr_details_graphql(
        owner, repo_name, [pr_number]
    )
    gql = graphql_details.get(pr_number)

    # Build case
    case_id = _next_case_id(cases_dir, repo_name)
    case = build_case_from_pr(
        repo=full_repo,
        pr=pr_data,
        case_id=case_id,
        graphql_data=gql,
    )

    # Blame + ground truth if repo_dir is available
    if repo_dir and repo_dir != Path("") and repo_dir.is_dir():
        case = populate_blame(case, repo_dir)
        if (
            case.truth
            and case.truth.introducing_commit
            and case.fix_commit
        ):
            from bugeval.git_utils import run_git

            try:
                intro_sha = case.truth.introducing_commit
                intro_diff = run_git(
                    "diff", f"{intro_sha}~1", intro_sha,
                    cwd=repo_dir,
                )
                fix_diff = run_git(
                    "diff", f"{case.fix_commit}~1",
                    case.fix_commit,
                    cwd=repo_dir,
                )
                case.truth.buggy_lines = compute_buggy_lines(
                    intro_diff, [fix_diff]
                )
            except Exception:
                log.debug(
                    "Ground truth computation failed for %s",
                    case_id,
                )

    if not dry_run:
        cases_dir.mkdir(parents=True, exist_ok=True)
        save_case(case, cases_dir / f"{case_id}.yaml")
        log.info("Wrote %s", cases_dir / f"{case_id}.yaml")

    return case
