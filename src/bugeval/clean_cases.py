"""Mine clean (non-buggy) PRs as negative control cases."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bugeval.io import load_checkpoint, save_case, save_checkpoint
from bugeval.mine import (
    _compute_pr_size,
    _detect_language,
    _is_non_code_only,
    has_fix_signal,
    run_gh,
)
from bugeval.models import CaseKind, CaseStats, TestCase

log = logging.getLogger(__name__)


def fetch_clean_prs(
    repo: str, count: int, since: str,
) -> list[dict[str, Any]]:
    """Fetch merged PRs that lack fix/bug signals."""
    fields = (
        "number,title,body,labels,mergeCommit,additions,deletions,"
        "changedFiles,files,mergedAt,author"
    )
    args = [
        "pr", "list", "--repo", repo, "--state", "merged",
        "--json", fields, "--limit", str(count * 3),
    ]
    if since:
        args.extend(["--search", f"merged:>{since}"])
    output = run_gh(*args)
    all_prs: list[dict[str, Any]] = json.loads(output)

    results: list[dict[str, Any]] = []
    for pr in all_prs:
        title = str(pr.get("title") or "")
        body = str(pr.get("body") or "")
        labels = [
            str(lbl.get("name", ""))
            for lbl in (pr.get("labels") or [])
        ]
        additions = int(pr.get("additions") or 0)
        deletions = int(pr.get("deletions") or 0)
        total_lines = additions + deletions

        if total_lines < 3 or total_lines > 1000:
            continue

        pr_files = pr.get("files") or []
        file_names = [str(f.get("path", "")) for f in pr_files]
        if _is_non_code_only(file_names):
            continue

        if has_fix_signal(title, body, labels):
            continue

        results.append(pr)
        if len(results) >= count:
            break

    return results


def check_not_subsequently_fixed(
    repo: str, pr: dict[str, Any], window_months: int = 6,
) -> bool:
    """Check that no later fix PR references this PR number."""
    pr_number = int(pr["number"])
    output = run_gh(
        "pr", "list", "--repo", repo, "--state", "merged",
        "--search", f"#{pr_number}",
        "--json", "number,title,body,labels",
        "--limit", "20",
    )
    candidates: list[dict[str, Any]] = json.loads(output)
    for cand in candidates:
        if int(cand["number"]) == pr_number:
            continue
        title = str(cand.get("title") or "")
        body = str(cand.get("body") or "")
        labels = [
            str(lbl.get("name", ""))
            for lbl in (cand.get("labels") or [])
        ]
        if has_fix_signal(title, body, labels):
            return False
    return True


def build_clean_case(
    repo: str, pr: dict[str, Any], case_id: str,
) -> TestCase:
    """Build a TestCase with kind=clean from a non-buggy PR."""
    title = str(pr.get("title") or "")
    body = str(pr.get("body") or "")
    merge_commit = str(
        (pr.get("mergeCommit") or {}).get("oid", "")
    )
    additions = int(pr.get("additions") or 0)
    deletions = int(pr.get("deletions") or 0)
    files_count = int(pr.get("changedFiles") or 0)
    pr_files = pr.get("files") or []
    file_names = [str(f.get("path", "")) for f in pr_files]
    author = str((pr.get("author") or {}).get("login", ""))
    merged_at = str(pr.get("mergedAt") or "")

    return TestCase(
        id=case_id,
        repo=repo,
        kind=CaseKind.clean,
        language=_detect_language(file_names),
        base_commit=merge_commit,
        introducing_pr_number=int(pr["number"]),
        introducing_pr_title=title,
        introducing_pr_body=body,
        introducing_pr_author=author,
        introducing_pr_merge_date=merged_at,
        truth=None,
        stats=CaseStats(
            lines_added=additions,
            lines_deleted=deletions,
            files_changed=files_count or len(file_names),
        ),
        pr_size=_compute_pr_size(additions, deletions),
    )


def mine_clean_cases(
    repo: str,
    count: int,
    output_dir: Path,
    since: str = "2023-01-01",
) -> list[TestCase]:
    """Fetch clean PRs, filter, build cases, and save with checkpointing."""
    _owner, name = repo.split("/", 1)
    repo_slug = name
    repo_dir = output_dir / repo_slug
    repo_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = repo_dir / ".clean_checkpoint.json"
    done = load_checkpoint(checkpoint_path)

    log.info(
        "Fetching clean PRs from %s (count=%d, since=%s)",
        repo, count, since,
    )
    prs = fetch_clean_prs(repo, count * 2, since)
    log.info("Found %d candidate clean PRs", len(prs))

    pending = [
        pr for pr in prs if str(pr["number"]) not in done
    ]
    log.info(
        "Processing %d pending PRs (%d already done)",
        len(pending), len(done),
    )

    existing = sorted(repo_dir.glob(f"{repo_slug}-clean-*.yaml"))
    next_num = len(existing) + 1

    cases: list[TestCase] = []
    for pr in pending:
        if len(cases) >= count:
            break

        if not check_not_subsequently_fixed(repo, pr):
            done.add(str(pr["number"]))
            save_checkpoint(done, checkpoint_path)
            continue

        case_id = f"{repo_slug}-clean-{next_num:03d}"
        case = build_clean_case(repo, pr, case_id)
        save_case(case, repo_dir / f"{case_id}.yaml")
        cases.append(case)

        done.add(str(pr["number"]))
        save_checkpoint(done, checkpoint_path)
        next_num += 1

    log.info("Wrote %d clean cases to %s", len(cases), repo_dir)
    return cases
