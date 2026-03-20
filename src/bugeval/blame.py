"""Git blame analysis to find introducing commits."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from bugeval.git_utils import GitError, run_git
from bugeval.io import load_cases, load_checkpoint, save_case, save_checkpoint
from bugeval.mine import fetch_pr_details_graphql, run_gh
from bugeval.models import CaseKind, GroundTruth, TestCase

log = logging.getLogger(__name__)

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_diff_deleted_lines(diff: str) -> dict[str, list[int]]:
    """Parse unified diff to extract deleted/modified line numbers per file."""
    result: dict[str, list[int]] = {}
    current_file: str | None = None
    old_line = 0

    for line in diff.splitlines():
        if line.startswith("--- a/"):
            current_file = line[6:]
        elif line.startswith("--- /dev/null"):
            current_file = None
        elif line.startswith("+++ "):
            pass  # skip new-file header
        else:
            hunk_match = _HUNK_RE.match(line)
            if hunk_match:
                old_line = int(hunk_match.group(1))
            elif current_file is not None:
                if line.startswith("-"):
                    result.setdefault(current_file, []).append(old_line)
                    old_line += 1
                elif line.startswith("+"):
                    pass  # additions don't move old line counter
                else:
                    old_line += 1

    return result


def parse_diff_added_lines(diff: str) -> dict[str, list[int]]:
    """Parse unified diff to extract added line numbers per file."""
    result: dict[str, list[int]] = {}
    current_file: str | None = None
    new_line = 0

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("+++ /dev/null"):
            current_file = None
        elif line.startswith("--- "):
            pass  # skip old-file header
        else:
            hunk_match = _HUNK_RE.match(line)
            if hunk_match:
                new_line = int(hunk_match.group(2))
            elif current_file is not None:
                if line.startswith("+"):
                    result.setdefault(current_file, []).append(new_line)
                    new_line += 1
                elif line.startswith("-"):
                    pass  # deletions don't move new line counter
                else:
                    new_line += 1

    return result


def run_blame(
    file: str, lines: list[int], cwd: Path, at_rev: str = "HEAD",
) -> dict[int, str]:
    """Run git blame at a specific revision for specific lines."""
    result: dict[int, str] = {}
    for line_num in lines:
        try:
            output = run_git(
                "blame", "-C", "-C", "-C",
                f"-L{line_num},{line_num}",
                "--porcelain", at_rev, "--", file,
                cwd=cwd,
            )
        except GitError:
            log.debug("Blame failed for %s:%d", file, line_num)
            continue

        # First line of porcelain: <sha> <orig_line> <final_line> <count>
        first_line = output.split("\n", 1)[0] if output else ""
        parts = first_line.split()
        if parts:
            result[line_num] = parts[0]

    return result


def walk_merge_commit(sha: str, cwd: Path) -> str:
    """If SHA is a merge commit, resolve to the feature (second parent)."""
    try:
        parents_out = run_git(
            "log", "--format=%P", "-1", sha, cwd=cwd
        )
        parents = parents_out.strip().split()
        if len(parents) >= 2:
            # Merge commit — get second parent
            child = run_git(
                "log", "--format=%H", f"{sha}^2", "-1", cwd=cwd
            )
            return child.strip()
    except GitError:
        pass
    return sha


def file_level_fallback(
    files: list[str], before_sha: str, cwd: Path
) -> str | None:
    """Find most recent commit touching any of the given files before a SHA."""
    try:
        output = run_git(
            "log", "--format=%H", "-1", before_sha,
            "--", *files,
            cwd=cwd,
        )
        sha = output.strip()
        return sha if sha else None
    except GitError:
        return None


def blame_enclosing_function(
    file: str, line: int, cwd: Path, at_sha: str
) -> str | None:
    """For omission bugs: blame the function signature containing the line."""
    try:
        output = run_git(
            "log", "-1", "--format=%H",
            f"-L{line},{line}:{file}",
            at_sha,
            cwd=cwd,
        )
        sha = output.strip()
        return sha if sha else None
    except GitError:
        return None


def _is_initial_commit(sha: str, cwd: Path) -> bool:
    try:
        output = run_git(
            "rev-list", "--count", sha, cwd=cwd
        )
        return output.strip() == "1"
    except GitError:
        # If rev-list fails entirely, check if there are parents
        try:
            parents = run_git(
                "log", "--format=%P", "-1", sha, cwd=cwd
            )
            return not parents.strip()
        except GitError:
            return False


def find_introducing_commit(
    case: TestCase, repo_dir: Path
) -> tuple[str | None, str]:
    """Find the commit that introduced the bug via git blame analysis."""
    fix_sha = case.fix_commit
    if not fix_sha:
        return None, "excluded"

    # Get diff of the fix commit against its parent
    try:
        diff = run_git(
            "diff", f"{fix_sha}~1", fix_sha, cwd=repo_dir
        )
    except GitError:
        return None, "excluded"

    deleted = parse_diff_deleted_lines(diff)
    if not deleted:
        # Pure addition fix (omission bug) — no lines to blame
        # Try enclosing-function blame on added lines first (tier D)
        added = parse_diff_added_lines(diff)
        for file, lines in added.items():
            if lines:
                result = blame_enclosing_function(
                    file, lines[0], repo_dir, f"{fix_sha}~1"
                )
                if result:
                    resolved = walk_merge_commit(result, repo_dir)
                    if not _is_initial_commit(resolved, repo_dir):
                        return resolved, "D"

        # Fall through to file-level fallback (tier C for omission)
        files = list(added.keys())
        if not files:
            try:
                diff_names = run_git(
                    "diff", "--name-only",
                    f"{fix_sha}~1", fix_sha,
                    cwd=repo_dir,
                )
                files = [
                    f for f in diff_names.strip().splitlines() if f
                ]
            except GitError:
                return None, "excluded"

        if files:
            sha = file_level_fallback(
                files, f"{fix_sha}~1", repo_dir
            )
            if sha:
                resolved = walk_merge_commit(sha, repo_dir)
                if _is_initial_commit(resolved, repo_dir):
                    return None, "excluded"
                return resolved, "C"

        return None, "excluded"

    # Blame deleted lines at the commit BEFORE the fix
    all_shas: list[str] = []
    total_lines = sum(len(lines) for lines in deleted.values())
    blamed_count = 0

    for file_path, line_nums in deleted.items():
        blame_result = run_blame(
            file_path, line_nums, cwd=repo_dir, at_rev=f"{fix_sha}~1",
        )
        for _ln, sha in blame_result.items():
            all_shas.append(sha)
            blamed_count += 1

    # If blame failed on most lines → file-level fallback
    if not all_shas or blamed_count < total_lines * 0.5:
        files = list(deleted.keys())
        sha = file_level_fallback(files, f"{fix_sha}~1", repo_dir)
        if sha:
            resolved = walk_merge_commit(sha, repo_dir)
            if _is_initial_commit(resolved, repo_dir):
                return None, "excluded"
            return resolved, "C"
        return None, "excluded"

    # Majority vote
    counts = Counter(all_shas)
    top_sha, top_count = counts.most_common(1)[0]
    ratio = top_count / len(all_shas)

    resolved = walk_merge_commit(top_sha, repo_dir)
    if _is_initial_commit(resolved, repo_dir):
        return None, "excluded"

    if ratio > 0.6:
        return resolved, "A"
    elif ratio >= 0.4:
        return resolved, "B"
    else:
        # Low agreement — still return top but mark as C
        return resolved, "C"


def _compute_latency_days(date_a: str, date_b: str) -> int | None:
    """Compute days between two ISO-8601 date strings. Returns None on parse failure."""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            da = datetime.fromisoformat(date_a.replace("Z", "+00:00"))
            db = datetime.fromisoformat(date_b.replace("Z", "+00:00"))
            return abs((db - da).days)
        except (ValueError, TypeError):
            continue
    return None


def resolve_introducing_pr(case: TestCase, repo: str) -> TestCase:
    """Resolve the introducing commit SHA to its parent PR and populate fields."""
    if not case.truth or not case.truth.introducing_commit:
        return case

    sha = case.truth.introducing_commit
    owner, name = repo.split("/", 1)

    try:
        output = run_gh(
            "api", f"repos/{owner}/{name}/commits/{sha}/pulls",
            "--header", "Accept: application/vnd.github.v3+json",
        )
        prs = json.loads(output)
    except Exception:
        return case

    if not prs:
        return case

    pr = prs[0]
    pr_number = int(pr.get("number", 0))
    if not pr_number:
        return case

    details = fetch_pr_details_graphql(owner, name, [pr_number])
    gql = details.get(pr_number, {})

    commit_messages: list[str] = []
    commit_shas: list[str] = []
    for node in (gql.get("commits") or {}).get("nodes") or []:
        commit_data = node.get("commit") or {}
        msg = commit_data.get("message", "")
        oid = commit_data.get("oid", "")
        if msg:
            commit_messages.append(msg)
        if oid:
            commit_shas.append(oid)

    review_comments: list[str] = []
    for review in (gql.get("reviews") or {}).get("nodes") or []:
        body = str(review.get("body") or "").strip()
        state = str(review.get("state") or "")
        author = str((review.get("author") or {}).get("login", ""))
        if body:
            prefix = f"[{author}:{state}] " if author or state else ""
            review_comments.append(f"{prefix}{body}")
    for thread in (gql.get("reviewThreads") or {}).get("nodes") or []:
        for comment in (thread.get("comments") or {}).get("nodes") or []:
            body = str(comment.get("body") or "").strip()
            if body:
                review_comments.append(body)

    status = gql.get("statusCheckRollup") or {}
    ci_status = str(status.get("state") or "")

    author = str(
        (gql.get("author") or pr.get("user") or {}).get("login", "")
    )

    intro_merge_date = str(
        pr.get("merged_at") or gql.get("mergedAt") or ""
    )

    # Compute bug latency: days between introducing PR merge and fix PR merge
    latency: int | None = None
    if intro_merge_date and case.fix_pr_merge_date:
        latency = _compute_latency_days(intro_merge_date, case.fix_pr_merge_date)

    return case.model_copy(update={
        "introducing_pr_number": pr_number,
        "introducing_pr_title": str(pr.get("title", "")),
        "introducing_pr_body": str(pr.get("body") or ""),
        "introducing_pr_commit_messages": commit_messages,
        "introducing_pr_commit_shas": commit_shas,
        "introducing_pr_author": author,
        "introducing_pr_merge_date": intro_merge_date,
        "introducing_pr_review_comments": review_comments,
        "introducing_pr_ci_status": ci_status,
        "bug_latency_days": latency,
    })


def populate_blame(case: TestCase, repo_dir: Path) -> TestCase:
    """Run blame logic on a TestCase and update truth fields."""
    if case.truth is None:
        case.truth = GroundTruth()

    # Populate fix_pr_numbers from fix_pr_number
    if case.fix_pr_number and case.fix_pr_number not in case.truth.fix_pr_numbers:
        case.truth.fix_pr_numbers = [case.fix_pr_number]

    sha, confidence = find_introducing_commit(case, repo_dir)
    case.truth.introducing_commit = sha
    case.truth.blame_confidence = confidence

    # Set base_commit to parent of introducing commit
    if sha:
        try:
            parent = run_git(
                "rev-parse", f"{sha}~1", cwd=repo_dir
            )
            case.base_commit = parent.strip()
        except GitError:
            pass

    # Resolve introducing commit to its parent PR
    case = resolve_introducing_pr(case, case.repo)

    return case


def blame_cases(
    cases_dir: Path, repo_dir: Path, concurrency: int
) -> None:
    """Load cases and populate blame for those missing introducing_commit."""
    cases = load_cases(cases_dir)
    checkpoint_path = cases_dir / ".blame_checkpoint.json"
    done = load_checkpoint(checkpoint_path)

    pending = [
        c for c in cases
        if c.id not in done
        and c.kind != CaseKind.clean
        and (c.truth is None or not c.truth.introducing_commit)
    ]

    log.info(
        "Blame: %d pending, %d already done, %d total",
        len(pending), len(done), len(cases),
    )

    def process(case: TestCase) -> TestCase:
        return populate_blame(case, repo_dir)

    if concurrency <= 1:
        for case in pending:
            updated = process(case)
            case_path = _find_case_path(cases_dir, case.id)
            if case_path:
                save_case(updated, case_path)
            done.add(case.id)
            save_checkpoint(done, checkpoint_path)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(process, c): c for c in pending
            }
            for future in futures:
                case = futures[future]
                try:
                    updated = future.result()
                    case_path = _find_case_path(
                        cases_dir, case.id
                    )
                    if case_path:
                        save_case(updated, case_path)
                    done.add(case.id)
                    save_checkpoint(done, checkpoint_path)
                except Exception as exc:
                    log.warning(
                        "Blame failed for %s: %s", case.id, exc,
                    )


def _find_case_path(cases_dir: Path, case_id: str) -> Path | None:
    for p in cases_dir.rglob("*.yaml"):
        if p.stem == case_id:
            return p
    return None
