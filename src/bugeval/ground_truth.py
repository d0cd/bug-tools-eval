"""Build ground truth via diff intersection."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from bugeval.blame import parse_diff_deleted_lines
from bugeval.git_utils import GitError, run_git
from bugeval.io import load_cases, load_checkpoint, save_case, save_checkpoint
from bugeval.models import BuggyLine, GroundTruth, TestCase

log = logging.getLogger(__name__)

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

_LINE_DRIFT_TOLERANCE = 3


def parse_diff_added_lines(diff: str) -> dict[str, list[tuple[int, str]]]:
    """Parse unified diff to extract added lines with new-file line numbers."""
    result: dict[str, list[tuple[int, str]]] = {}
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
                new_line = int(hunk_match.group(1))
            elif current_file is not None:
                if line.startswith("+"):
                    content = line[1:]
                    result.setdefault(current_file, []).append(
                        (new_line, content)
                    )
                    new_line += 1
                elif line.startswith("-"):
                    pass  # deletions don't move new-file counter
                else:
                    new_line += 1

    return result


def compute_buggy_lines(
    introducing_diff: str, fix_diffs: list[str]
) -> list[BuggyLine]:
    """Intersect lines added by introducing commit with lines changed by fixes."""
    added = parse_diff_added_lines(introducing_diff)
    if not added:
        return []

    # Collect all deleted lines from fix diffs (union across multiple fixes)
    fix_deleted: dict[str, set[int]] = {}
    for fd in fix_diffs:
        for file_path, line_nums in parse_diff_deleted_lines(fd).items():
            fix_deleted.setdefault(file_path, set()).update(line_nums)

    result: list[BuggyLine] = []
    for file_path, added_lines in added.items():
        if file_path not in fix_deleted:
            continue
        deleted_set = fix_deleted[file_path]
        for line_num, content in added_lines:
            if line_num <= 0:
                continue
            # Exact or tolerance match (skip invalid line 0 in deleted set)
            if any(
                dl > 0 and abs(line_num - dl) <= _LINE_DRIFT_TOLERANCE
                for dl in deleted_set
            ):
                result.append(
                    BuggyLine(file=file_path, line=line_num, content=content)
                )

    return result


_BUG_KEYWORDS = {
    "bug", "fix", "error", "crash", "panic", "wrong", "incorrect",
    "broken", "fail", "issue", "regression", "mishandl", "overflow",
    "underflow", "null", "missing", "invalid",
}


def _looks_like_bug_report(text: str) -> bool:
    """Check if text contains bug-related keywords (not a feature request)."""
    lower = text.lower()
    return any(kw in lower for kw in _BUG_KEYWORDS)


def extract_bug_description(case: TestCase) -> tuple[str, str]:
    """Extract the best bug description from available metadata.

    Priority:
    1. Fix PR body (directly describes what was fixed)
    2. Fix PR title (concise summary of the fix)
    3. Issue body ONLY if it looks like a bug report (not a feature request)
    4. Fix PR commit messages
    5. Fix PR review comments that mention the bug
    """
    # Fix PR body is the most reliable source — it describes the fix
    if case.fix_pr_body and case.fix_pr_body.strip():
        body = case.fix_pr_body.strip()
        # Skip very short bodies (e.g., "LGTM" or just a link)
        if len(body) > 20:
            return body, "pr_body"

    # Fix PR title
    if case.fix_pr_title and case.fix_pr_title.strip():
        return case.fix_pr_title.strip(), "pr_title"

    # Issue bodies — only if they look like actual bug reports
    if case.issue_bodies:
        for _num, body in sorted(
            case.issue_bodies.items(), key=lambda x: len(x[1]), reverse=True,
        ):
            if body.strip() and _looks_like_bug_report(body):
                return body.strip(), "issue"

    # Commit messages
    if case.fix_pr_commit_messages:
        msg = case.fix_pr_commit_messages[0]
        if msg.strip():
            return msg.strip(), "commit_msg"

    # Review comments that mention the bug
    if case.fix_pr_review_comments:
        for comment in case.fix_pr_review_comments:
            if _looks_like_bug_report(comment) and len(comment.strip()) > 30:
                return comment.strip(), "review_comment"

    return "", ""


def compute_metadata(case: TestCase) -> dict[str, Any]:
    """Compute derived metadata (latency, authorship)."""
    meta: dict[str, Any] = {}

    # Bug latency
    if case.introducing_pr_merge_date and case.fix_pr_merge_date:
        try:
            intro_dt = _parse_date(case.introducing_pr_merge_date)
            fix_dt = _parse_date(case.fix_pr_merge_date)
            meta["bug_latency_days"] = (fix_dt - intro_dt).days
        except (ValueError, TypeError):
            pass

    # Same author fix
    fix_author = _get_fix_author(case)
    if case.introducing_pr_author and fix_author:
        meta["same_author_fix"] = (
            case.introducing_pr_author.lower() == fix_author.lower()
        )

    return meta


def _parse_date(date_str: str) -> datetime:
    # Try ISO format with timezone
    date_str = date_str.strip()
    if date_str.endswith("Z"):
        date_str = date_str[:-1] + "+00:00"
    return datetime.fromisoformat(date_str)


def _get_fix_author(case: TestCase) -> str:
    for pr in case.related_prs:
        if pr.role in ("full_fix", "partial_fix") and pr.author:
            return pr.author
    return ""


def classify_bug(case: TestCase) -> dict[str, str]:
    """Heuristically classify bug category, difficulty, and severity."""
    category = ""
    difficulty = ""
    severity = ""

    # Category from keywords in bug_description + PR title
    text = f"{case.bug_description} {case.fix_pr_title}".lower()
    if any(w in text for w in ("race", "concurren", "deadlock", "lock", "mutex", "atomic")):
        category = "concurrency"
    elif any(w in text for w in ("overflow", "underflow", "panic", "crash", "abort")):
        category = "runtime"
    elif any(w in text for w in ("memory", "leak", "use-after", "dangling", "null")):
        category = "memory"
    elif any(w in text for w in ("logic", "incorrect", "wrong", "should be", "off-by")):
        category = "logic"
    elif any(w in text for w in ("security", "vuln", "inject", "xss", "auth")):
        category = "security"
    elif any(w in text for w in ("type", "cast", "convert", "serializ", "deserializ")):
        category = "type"
    else:
        category = "other"

    # Difficulty from stats
    if case.stats:
        total = case.stats.lines_added + case.stats.lines_deleted
        if total < 10:
            difficulty = "easy"
        elif total < 50:
            difficulty = "medium"
        else:
            difficulty = "hard"

    # Severity from issue labels + keywords
    labels_text = " ".join(case.issue_labels).lower()
    if any(w in labels_text for w in ("critical", "p0", "blocker", "security")):
        severity = "critical"
    elif any(w in labels_text for w in ("high", "p1", "important")):
        severity = "high"
    elif any(w in text for w in ("panic", "crash", "data loss", "corrupt")):
        severity = "high"
    elif any(w in labels_text for w in ("low", "p3", "minor")):
        severity = "low"
    else:
        severity = "medium"

    return {"category": category, "difficulty": difficulty, "severity": severity}


def populate_ground_truth(case: TestCase, repo_dir: Path) -> TestCase:
    """Compute ground truth for a case: buggy lines, description, metadata."""
    if case.truth is None:
        case.truth = GroundTruth()

    intro_sha = case.truth.introducing_commit
    if not intro_sha:
        return case

    # Get introducing diff
    try:
        introducing_diff = run_git(
            "diff", f"{intro_sha}~1", intro_sha, cwd=repo_dir
        )
    except GitError:
        log.warning("Cannot get introducing diff for %s", case.id)
        return case

    # Get fix diffs
    fix_diffs: list[str] = []
    for pr_num in case.truth.fix_pr_numbers:
        diff = _get_fix_pr_diff(pr_num, case, repo_dir)
        if diff:
            fix_diffs.append(diff)

    # If no fix PR diffs, try the fix_commit directly
    if not fix_diffs and case.fix_commit:
        try:
            diff = run_git(
                "diff", f"{case.fix_commit}~1", case.fix_commit,
                cwd=repo_dir,
            )
            if diff:
                fix_diffs.append(diff)
        except GitError:
            pass

    if fix_diffs:
        case.truth.buggy_lines = compute_buggy_lines(
            introducing_diff, fix_diffs
        )

    # Bug description
    desc, source = extract_bug_description(case)
    case.bug_description = desc
    case.bug_description_source = source

    # Metadata
    meta = compute_metadata(case)
    if "bug_latency_days" in meta:
        case.bug_latency_days = meta["bug_latency_days"]
    if "same_author_fix" in meta:
        case.same_author_fix = meta["same_author_fix"]

    # Classification metadata
    classification = classify_bug(case)
    if not case.category:
        case.category = classification["category"]
    if not case.difficulty:
        case.difficulty = classification["difficulty"]
    if not case.severity:
        case.severity = classification["severity"]

    return case


def _get_fix_pr_diff(
    pr_number: int, case: TestCase, repo_dir: Path
) -> str:
    # Find the merge commit for this PR from related_prs
    merge_sha: str | None = None
    for pr in case.related_prs:
        if pr.pr_number == pr_number and pr.commit:
            merge_sha = pr.commit
            break

    if not merge_sha and case.fix_commit:
        merge_sha = case.fix_commit

    if not merge_sha:
        return ""

    try:
        return run_git(
            "diff", f"{merge_sha}~1", merge_sha, cwd=repo_dir
        )
    except GitError:
        return ""


def build_ground_truth(
    cases_dir: Path, repo_dir: Path, concurrency: int
) -> None:
    """Load cases, compute ground truth, checkpoint progress."""
    cases = load_cases(cases_dir)
    checkpoint_path = cases_dir / ".ground_truth_checkpoint.json"
    done = load_checkpoint(checkpoint_path)

    pending = [
        c for c in cases
        if c.id not in done
        and c.truth is not None
        and c.truth.introducing_commit
    ]

    log.info(
        "Ground truth: %d pending, %d done, %d total",
        len(pending), len(done), len(cases),
    )

    def process(case: TestCase) -> TestCase:
        return populate_ground_truth(case, repo_dir)

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
                    case_path = _find_case_path(cases_dir, case.id)
                    if case_path:
                        save_case(updated, case_path)
                    done.add(case.id)
                    save_checkpoint(done, checkpoint_path)
                except Exception as exc:
                    log.warning(
                        "Ground truth failed for %s: %s", case.id, exc,
                    )


def _find_case_path(cases_dir: Path, case_id: str) -> Path | None:
    for p in cases_dir.rglob("*.yaml"):
        if p.stem == case_id:
            return p
    return None
