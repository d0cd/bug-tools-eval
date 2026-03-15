"""GitHub scraper using gh CLI to fetch bug issues and PRs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from bugeval.models import (
    Candidate,
    CaseStats,
    ExpectedFinding,
    PRSize,
    ScrapeState,
)


class GhError(Exception):
    """Raised when a gh CLI command fails."""

    def __init__(self, command: list[str], stderr: str) -> None:
        self.command = command
        self.stderr = stderr
        super().__init__(f"gh command failed: {' '.join(command)}\n{stderr}")


def run_gh(*args: str) -> str:
    """Run a gh CLI command and return stdout. Raises GhError on failure."""
    cmd = ["gh", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise GhError(cmd, "command timed out after 60 seconds")
    if result.returncode != 0:
        raise GhError(cmd, result.stderr)
    return result.stdout


def fetch_bug_issues(
    repo: str,
    limit: int = 200,
    labels: list[str] | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch closed bug issues from a repo across multiple labels."""
    if labels is None:
        labels = ["bug", "fix", "regression", "defect"]

    all_issues: list[dict[str, Any]] = []
    seen_numbers: set[int] = set()

    for label in labels:
        try:
            args = [
                "issue",
                "list",
                "--repo",
                repo,
                "--label",
                label,
                "--state",
                "closed",
                "--json",
                "number,title,body,labels,closedAt",
                "--limit",
                str(limit),
            ]
            if since:
                args.extend(["--search", f"created:>{since}"])
            output = run_gh(*args)
            issues: list[dict[str, Any]] = json.loads(output)
            for issue in issues:
                num = issue.get("number")
                if num not in seen_numbers:
                    seen_numbers.add(num)  # type: ignore[arg-type]
                    all_issues.append(issue)
        except GhError:
            continue  # label may not exist in this repo

    return all_issues


def fetch_fix_prs(repo: str, limit: int = 200, since: str | None = None) -> list[dict[str, Any]]:
    """Fetch merged PRs from a repo using paginated gh api calls.

    Uses gh api --paginate to handle repos with many PRs, collecting pages
    until limit is reached or all PRs are fetched.
    """
    # Use gh pr list for structured JSON fields (no --paginate support, but reliable)
    # For simplicity and reliability, use a single large fetch (GitHub limits to 1000).
    args = [
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "merged",
        "--json",
        "number,title,body,labels,mergeCommit,baseRefName,headRefName,files,"
        "additions,deletions,changedFiles",
        "--limit",
        str(limit),
    ]
    if since:
        args.extend(["--search", f"created:>{since}"])
    output = run_gh(*args)
    collected = json.loads(output)

    # If we hit the limit exactly, try fetching older page via date windowing
    if len(collected) == limit and since:
        oldest = min(
            (pr.get("closedAt") or pr.get("mergedAt") or "") for pr in collected if pr
        ) if collected else ""
        if oldest:
            try:
                args2 = [
                    "pr", "list", "--repo", repo, "--state", "merged",
                    "--json",
                    "number,title,body,labels,mergeCommit,baseRefName,headRefName,files,"
                    "additions,deletions,changedFiles",
                    "--limit", str(limit),
                    "--search", f"created:>{since} created:<{oldest[:10]}",
                ]
                output2 = run_gh(*args2)
                page2: list[dict[str, Any]] = json.loads(output2)
                seen = {pr["number"] for pr in collected}
                for pr in page2:
                    if pr["number"] not in seen:
                        collected.append(pr)
                        seen.add(pr["number"])
            except (GhError, KeyError, ValueError):
                pass  # Pagination is best-effort

    return collected


def _batch_fetch_pr_reviews_graphql(
    owner: str, name: str, pr_numbers: list[int]
) -> dict[int, list[dict[str, Any]]]:
    """Fetch reviews + review comments for multiple PRs in a single GraphQL query.

    Returns a mapping of pr_number → list of review/comment dicts.
    Falls back to empty dict on error (caller treats as no reviews).

    GitHub GraphQL lets us alias each PR as pr_{number} and fetch all of them
    in one network round-trip instead of 3 REST calls × N PRs.
    """
    if not pr_numbers:
        return {}

    # Build aliased fragments for each PR number
    fragments = []
    for num in pr_numbers:
        fragments.append(f"""
  pr_{num}: pullRequest(number: {num}) {{
    reviews(first: 50) {{
      nodes {{ body state author {{ login }} }}
    }}
    reviewThreads(first: 50) {{
      nodes {{
        comments(first: 10) {{
          nodes {{ body author {{ login }} }}
        }}
      }}
    }}
    comments(first: 50) {{
      nodes {{ body author {{ login }} }}
    }}
  }}""")

    fragments_str = "".join(fragments)
    query = f'query {{\n  repository(owner: "{owner}", name: "{name}") {{{fragments_str}\n  }}\n}}'

    try:
        output = run_gh("api", "graphql", "-f", f"query={query}")
        data: dict[str, Any] = json.loads(output)
        repo_data = data.get("data", {}).get("repository", {})
    except (GhError, json.JSONDecodeError, KeyError):
        return {}

    result: dict[int, list[dict[str, Any]]] = {}
    for num in pr_numbers:
        key = f"pr_{num}"
        pr_data = repo_data.get(key) or {}
        items: list[dict[str, Any]] = []

        for review in (pr_data.get("reviews") or {}).get("nodes") or []:
            items.append({
                "body": review.get("body") or "",
                "state": review.get("state") or "",
                "_source": "review",
            })

        for thread in (pr_data.get("reviewThreads") or {}).get("nodes") or []:
            for comment in (thread.get("comments") or {}).get("nodes") or []:
                items.append({"body": comment.get("body") or "", "state": "", "_source": "inline"})

        for comment in (pr_data.get("comments") or {}).get("nodes") or []:
            items.append({"body": comment.get("body") or "", "state": "", "_source": "thread"})

        result[num] = items

    return result


def enrich_with_reviews(
    repo: str, candidates: list[Candidate], top_n: int = 50
) -> list[Candidate]:
    """Batch-fetch reviewer comments for the top_n candidates via a single GraphQL query.

    Uses GitHub's GraphQL API to retrieve reviews, inline review thread comments,
    and PR thread comments for all candidates in one round-trip. Candidates where
    a reviewer explicitly identified a bug get a +0.2 confidence boost.
    """
    owner, name = repo.split("/", 1)
    top = candidates[:top_n]
    pr_numbers = [c.pr_number for c in top]

    # GraphQL has a node limit; batch in chunks of 25 to stay well under it
    batch_size = 25
    all_reviews: dict[int, list[dict[str, Any]]] = {}
    for i in range(0, len(pr_numbers), batch_size):
        batch = pr_numbers[i : i + batch_size]
        all_reviews.update(_batch_fetch_pr_reviews_graphql(owner, name, batch))

    enriched = list(candidates)
    for i, cand in enumerate(top):
        reviews = all_reviews.get(cand.pr_number, [])
        if not reviews:
            continue
        rev_signals, rev_notes = extract_reviewer_bug_signals(reviews)
        if rev_signals:
            enriched[i] = cand.model_copy(
                update={
                    "signals": list(dict.fromkeys(cand.signals + rev_signals)),
                    "reviewer_notes": cand.reviewer_notes + rev_notes,
                    "confidence": min(cand.confidence + 0.2, 1.0),
                }
            )
    return enriched


def fetch_pr_diff(repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Fetch per-file diffs for a PR via GitHub API."""
    owner, name = repo.split("/", 1)
    output = run_gh("api", f"repos/{owner}/{name}/pulls/{pr_number}/files")
    return json.loads(output)  # type: ignore[no-any-return]


def fetch_pr_reviews(repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Fetch all reviewer feedback for a PR: reviews, inline comments, and thread comments.

    Combines three GitHub API endpoints to give the richest possible signal
    about whether a reviewer explicitly identified the bug before the fix.
    """
    owner, name = repo.split("/", 1)
    results: list[dict[str, Any]] = []

    for endpoint, source in [
        (f"repos/{owner}/{name}/pulls/{pr_number}/reviews", "review"),
        (f"repos/{owner}/{name}/pulls/{pr_number}/comments", "inline"),
        (f"repos/{owner}/{name}/issues/{pr_number}/comments", "thread"),
    ]:
        try:
            output = run_gh("api", endpoint)
            items: list[dict[str, Any]] = json.loads(output)
            for item in items:
                item["_source"] = source
                results.append(item)
        except GhError:
            pass

    return results


# Keywords that suggest a reviewer explicitly identified a bug
_REVIEWER_BUG_PATTERNS = re.compile(
    r"\b(?:bug|wrong|incorrect|should be|panic|overflow|underflow|"
    r"off.by.one|race\s+condition|memory\s+leak|use.after.free|"
    r"undefined\s+behavior|null\s+pointer|integer\s+overflow|"
    r"crash(?:es|ed|ing)?|deadlock|infinite\s+loop|typo|mistake|errors?|this\s+will\s+fail|"
    r"this\s+breaks|this\s+is\s+broken|not\s+correct|doesn'?t\s+work)\b",
    re.IGNORECASE,
)


def extract_reviewer_bug_signals(
    reviews: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Scan reviewer comments for explicit bug identification.

    Returns (signals, notes) where:
    - signals: string labels to add to Candidate.signals
    - notes: verbatim reviewer snippets to store in Candidate.reviewer_notes
    """
    signals: list[str] = []
    notes: list[str] = []

    for review in reviews:
        body = str(review.get("body") or "")
        if not body.strip():
            continue

        source = review.get("_source", "")
        if _REVIEWER_BUG_PATTERNS.search(body):
            note = body[:300].replace("\n", " ").strip()
            notes.append(f"[{source}] {note}")
            if "reviewer_bug_feedback" not in signals:
                signals.append("reviewer_bug_feedback")

        # Reviewer left a CHANGES_REQUESTED review — strong signal the code was wrong
        state = str(review.get("state") or "")
        if state == "CHANGES_REQUESTED" and "reviewer_changes_requested" not in signals:
            signals.append("reviewer_changes_requested")

    return signals, notes


def _parse_hunk_ranges(patch: str) -> list[tuple[int, int]]:
    """Parse unified diff patch to extract changed line ranges (new-file line numbers)."""
    ranges: list[tuple[int, int]] = []
    pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
    for match in pattern.finditer(patch):
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) else 1
        end = start + max(count - 1, 0)
        ranges.append((start, end))
    return ranges


def extract_expected_findings(
    pr_diff_files: list[dict[str, Any]],
) -> list[ExpectedFinding]:
    """Auto-extract expected findings from PR diff files (marked [auto] for human review)."""
    findings: list[ExpectedFinding] = []
    for file_data in pr_diff_files:
        filename = str(file_data.get("filename", ""))
        patch = str(file_data.get("patch") or "")
        if not patch:
            continue

        ranges = _parse_hunk_ranges(patch)
        for start, _end in ranges[:3]:  # limit to first 3 hunks per file
            summary = f"Changed lines starting at {start}"
            for line in patch.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    summary = line[1:].strip()[:80]
                    break

            findings.append(
                ExpectedFinding(
                    file=filename,
                    line=start,
                    summary=f"[auto] {summary}",
                )
            )

    return findings


def _extract_issue_numbers(text: str) -> set[int]:
    """Extract issue numbers referenced by fix/close patterns."""
    pattern = re.compile(
        r"(?:fix(?:es)?|close[sd]?|resolve[sd]?)\s+#(\d+)",
        re.IGNORECASE,
    )
    return {int(m.group(1)) for m in pattern.finditer(text)}


def link_issues_to_prs(
    issues: list[dict[str, Any]], prs: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Match issues to PRs via closedIssues field, PR body refs, or PR title refs."""
    issue_by_number: dict[int, dict[str, Any]] = {
        issue["number"]: issue
        for issue in issues  # type: ignore[index]
    }
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    matched_issue_numbers: set[int] = set()

    for pr in prs:
        linked_issue: dict[str, Any] | None = None

        # Parse PR body + title for fixes/closes/resolves references
        body = str(pr.get("body") or "")
        title = str(pr.get("title") or "")
        referenced = _extract_issue_numbers(body) | _extract_issue_numbers(title)
        for num in sorted(referenced):
            if num in issue_by_number and num not in matched_issue_numbers:
                linked_issue = issue_by_number[num]
                matched_issue_numbers.add(num)
                break

        if linked_issue is not None:
            pairs.append((linked_issue, pr))

    return pairs


def score_candidate(issue: dict[str, Any], pr: dict[str, Any]) -> tuple[float, list[str]]:
    """Score a candidate (issue+PR pair). Returns (confidence, signals)."""
    confidence = 0.0
    signals: list[str] = []

    # Has bug label
    labels = [str(lbl.get("name", "")) for lbl in (issue.get("labels") or [])]
    bug_labels = {"bug", "fix", "regression", "defect", "bugfix"}
    if any(lbl.lower() in bug_labels for lbl in labels):
        confidence += 0.3
        signals.append("has_bug_label")

    # PR references this issue via body/title
    issue_num = issue.get("number")
    body = str(pr.get("body") or "")
    title = str(pr.get("title") or "")
    referenced = _extract_issue_numbers(body) | _extract_issue_numbers(title)
    if issue_num in referenced:
        confidence += 0.2
        signals.append("pr_references_issue")

    # Small diff
    additions = int(pr.get("additions") or 0)
    deletions = int(pr.get("deletions") or 0)
    if additions + deletions < 200:
        confidence += 0.1
        signals.append("small_diff")

    # Fix keywords in PR title
    pr_title_lower = title.lower()
    fix_keywords = {"fix", "bug", "patch", "correct", "resolve", "regression"}
    if any(kw in pr_title_lower for kw in fix_keywords):
        confidence += 0.1
        signals.append("fix_keywords_in_title")

    # Has any linked issue in body (even if not this specific issue)
    if "pr_references_issue" not in signals and referenced:
        confidence += 0.2
        signals.append("has_linked_issue")


    return min(confidence, 1.0), signals


_EXTENSION_MAP: dict[str, str] = {
    ".rs": "rust",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".cpp": "c++",
    ".cc": "c++",
    ".c": "c",
    ".h": "c",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sol": "solidity",
    ".leo": "leo",
}


def detect_language(files: list[str]) -> str:
    """Detect dominant language from file list based on extensions."""
    counts: dict[str, int] = {}
    for f in files:
        suffix = Path(f).suffix.lower()
        lang = _EXTENSION_MAP.get(suffix)
        if lang:
            counts[lang] = counts.get(lang, 0) + 1

    if not counts:
        return "unknown"
    return max(counts, key=lambda k: counts[k])


def compute_pr_size(additions: int, deletions: int) -> PRSize:
    """Compute PR size bucket from total line change count."""
    total = additions + deletions
    if total < 10:
        return PRSize.tiny
    if total < 50:
        return PRSize.small
    if total < 200:
        return PRSize.medium
    if total < 500:
        return PRSize.large
    return PRSize.xl


def _pr_has_bug_signal(pr: dict[str, Any]) -> bool:
    """Return True if a PR looks like a bug fix even without a linked issue."""
    title = str(pr.get("title") or "").lower()
    body = str(pr.get("body") or "").lower()
    labels = [str(lbl.get("name", "")).lower() for lbl in (pr.get("labels") or [])]
    bug_labels = {"bug", "fix", "regression", "defect", "bugfix", "hotfix"}
    fix_kws = {"fix", "bug", "patch", "correct", "resolve", "regression", "revert"}
    return (
        any(lbl in bug_labels for lbl in labels)
        or any(kw in title for kw in fix_kws)
        or bool(re.search(r"fix(?:es|ed)?\s+#\d+", body))
    )


def build_pr_only_candidates(
    repo: str,
    prs: list[dict[str, Any]],
    existing_pr_numbers: set[int],
) -> list[Candidate]:
    """Build candidates from PRs that have no linked issue but still look like bug fixes.

    Many fast-moving repos ship fixes directly without filing issues first.
    This catches those cases using label + title signal alone.
    """
    candidates: list[Candidate] = []
    for pr in prs:
        num = int(pr["number"])
        if num in existing_pr_numbers:
            continue
        if not _pr_has_bug_signal(pr):
            continue

        additions = int(pr.get("additions") or 0)
        deletions = int(pr.get("deletions") or 0)
        changed_files_count = int(pr.get("changedFiles") or 0)
        pr_files: list[dict[str, Any]] = pr.get("files") or []
        file_names = [str(f.get("path", "")) for f in pr_files]
        labels = [str(lbl.get("name", "")) for lbl in (pr.get("labels") or [])]
        merge_commit = str((pr.get("mergeCommit") or {}).get("oid", ""))

        confidence = 0.1  # base: PR-only is weaker signal than issue+PR
        signals = ["pr_only"]
        title_lower = str(pr.get("title") or "").lower()
        bug_labels = {"bug", "fix", "regression", "defect", "bugfix", "hotfix"}
        if any(lbl in bug_labels for lbl in [lbl_name.lower() for lbl_name in labels]):
            confidence += 0.25
            signals.append("has_bug_label")
        fix_kws = {"fix", "bug", "patch", "correct", "resolve", "regression", "revert"}
        if any(kw in title_lower for kw in fix_kws):
            confidence += 0.1
            signals.append("fix_keywords_in_title")
        if additions + deletions < 200:
            confidence += 0.1
            signals.append("small_diff")

        candidates.append(
            Candidate(
                repo=repo,
                pr_number=num,
                fix_commit=merge_commit,
                base_commit=None,
                head_commit=None,
                confidence=min(confidence, 1.0),
                signals=signals,
                title=str(pr.get("title", "")),
                body=str(pr.get("body") or ""),
                labels=labels,
                files_changed=file_names,
                diff_stats=CaseStats(
                    lines_added=additions,
                    lines_deleted=deletions,
                    files_changed=changed_files_count,
                    hunks=0,
                ),
                expected_findings=[],
                language=detect_language(file_names),
                pr_size=compute_pr_size(additions, deletions),
            )
        )
    return candidates


def fetch_prs_by_label(
    repo: str,
    labels: list[str] | None = None,
    limit: int = 200,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch merged PRs directly by bug label — bypasses issue-linking requirement.

    Many large OSS repos (sentry, grafana, etc.) label PRs directly as bugs
    without always using 'Fixes #N' syntax in the body.
    """
    if labels is None:
        labels = ["bug", "type: bug", "kind/bug", "bugfix", "regression", "type:bug"]

    all_prs: list[dict[str, Any]] = []
    seen: set[int] = set()

    for label in labels:
        try:
            args = [
                "pr", "list", "--repo", repo, "--state", "merged",
                "--json",
                "number,title,body,labels,mergeCommit,baseRefName,headRefName,"
                "files,additions,deletions,changedFiles",
                "--label", label,
                "--limit", str(limit),
            ]
            if since:
                args.extend(["--search", f"created:>{since}"])
            output = run_gh(*args)
            for pr in json.loads(output):
                num = int(pr["number"])
                if num not in seen:
                    seen.add(num)
                    all_prs.append(pr)
        except GhError:
            continue  # label may not exist in this repo

    return all_prs


def build_labeled_pr_candidates(
    repo: str,
    prs: list[dict[str, Any]],
    existing_pr_numbers: set[int],
) -> list[Candidate]:
    """Build high-signal candidates from directly bug-labeled PRs.

    These PRs were explicitly labeled as bugs by maintainers — higher base
    confidence than generic PR-only candidates.
    """
    candidates: list[Candidate] = []
    for pr in prs:
        num = int(pr["number"])
        if num in existing_pr_numbers:
            continue

        additions = int(pr.get("additions") or 0)
        deletions = int(pr.get("deletions") or 0)
        changed_files_count = int(pr.get("changedFiles") or 0)
        pr_files: list[dict[str, Any]] = pr.get("files") or []
        file_names = [str(f.get("path", "")) for f in pr_files]
        labels = [str(lbl.get("name", "")) for lbl in (pr.get("labels") or [])]
        merge_commit = str((pr.get("mergeCommit") or {}).get("oid", ""))
        title = str(pr.get("title") or "")
        body = str(pr.get("body") or "")

        # Base confidence: maintainer explicitly labeled as bug
        confidence = 0.4
        signals = ["labeled_bug"]

        if additions + deletions < 200:
            confidence += 0.1
            signals.append("small_diff")

        fix_kws = {"fix", "bug", "patch", "correct", "resolve", "regression", "revert"}
        if any(kw in title.lower() for kw in fix_kws):
            confidence += 0.1
            signals.append("fix_keywords_in_title")

        # Referenced an issue even if we couldn't link it
        referenced = _extract_issue_numbers(body) | _extract_issue_numbers(title)
        if referenced:
            confidence += 0.1
            signals.append("has_issue_ref")

        candidates.append(
            Candidate(
                repo=repo,
                pr_number=num,
                fix_commit=merge_commit,
                base_commit=None,
                head_commit=None,
                confidence=min(confidence, 1.0),
                signals=signals,
                title=title,
                body=body,
                labels=labels,
                files_changed=file_names,
                diff_stats=CaseStats(
                    lines_added=additions,
                    lines_deleted=deletions,
                    files_changed=changed_files_count,
                    hunks=0,
                ),
                expected_findings=[],
                language=detect_language(file_names),
                pr_size=compute_pr_size(additions, deletions),
            )
        )

    return candidates


def build_candidates(
    repo: str,
    issues: list[dict[str, Any]],
    prs: list[dict[str, Any]],
) -> list[Candidate]:
    """Build Candidate objects from linked issue/PR pairs plus PR-only bug fixes.

    Does not fetch reviewer comments — call enrich_with_reviews() on the
    filtered result set for per-PR review enrichment.
    """
    pairs = link_issues_to_prs(issues, prs)
    candidates: list[Candidate] = []
    matched_pr_numbers: set[int] = set()

    for issue, pr in pairs:
        confidence, signals = score_candidate(issue, pr)

        merge_commit = str((pr.get("mergeCommit") or {}).get("oid", ""))
        additions = int(pr.get("additions") or 0)
        deletions = int(pr.get("deletions") or 0)
        changed_files_count = int(pr.get("changedFiles") or 0)

        pr_files: list[dict[str, Any]] = pr.get("files") or []
        file_names = [str(f.get("path", "")) for f in pr_files]

        candidates.append(
            Candidate(
                repo=repo,
                pr_number=int(pr["number"]),
                fix_commit=merge_commit,
                base_commit=None,
                head_commit=None,
                confidence=confidence,
                signals=signals,
                title=str(pr.get("title", "")),
                body=str(pr.get("body") or ""),
                labels=[str(lbl.get("name", "")) for lbl in (pr.get("labels") or [])],
                files_changed=file_names,
                diff_stats=CaseStats(
                    lines_added=additions,
                    lines_deleted=deletions,
                    files_changed=changed_files_count,
                    hunks=0,
                ),
                expected_findings=[],
                language=detect_language(file_names),
                pr_size=compute_pr_size(additions, deletions),
            )
        )
        matched_pr_numbers.add(int(pr["number"]))

    # Also capture bug-fix PRs that had no linked issue
    candidates.extend(build_pr_only_candidates(repo, prs, matched_pr_numbers))

    return candidates


def enrich_git_candidates_with_github(
    repo: str,
    candidates: list[Candidate],
    top_n: int = 300,
) -> list[Candidate]:
    """Enrich git-mined candidates with real GitHub PR metadata.

    For each candidate in the top_n (by confidence), queries
    ``gh api repos/{owner}/{repo}/commits/{sha}/pulls`` to find the PR that
    merged the commit.  When a PR is found the candidate is updated with the
    real PR number, title, body, labels, and expected_findings from the diff.
    Candidates beyond top_n are returned unchanged.
    """
    to_enrich = candidates[:top_n]
    rest = candidates[top_n:]

    owner, name = repo.split("/", 1)
    enriched: list[Candidate] = []

    for candidate in to_enrich:
        sha = candidate.fix_commit
        try:
            raw = run_gh(
                "api",
                f"repos/{owner}/{name}/commits/{sha}/pulls",
                "--header", "Accept: application/vnd.github.v3+json",
            )
            prs: list[dict[str, Any]] = json.loads(raw)
        except (GhError, json.JSONDecodeError):
            enriched.append(candidate)
            continue

        if not prs:
            enriched.append(candidate)
            continue

        pr = prs[0]
        pr_number = int(pr.get("number", 0))
        if not pr_number:
            enriched.append(candidate)
            continue

        # Fetch richer PR detail (labels, body, additions, deletions, files)
        try:
            detail_raw = run_gh(
                "pr", "view", str(pr_number), "--repo", repo,
                "--json", "number,title,body,labels,additions,deletions,files",
            )
            detail: dict[str, Any] = json.loads(detail_raw)
        except (GhError, json.JSONDecodeError):
            enriched.append(candidate)
            continue

        labels = [str(lbl.get("name", "")) for lbl in (detail.get("labels") or [])]
        has_bug_label = any(
            "bug" in lbl.lower() or "fix" in lbl.lower() for lbl in labels
        )

        # Update signals and confidence
        new_signals = list(candidate.signals)
        conf_delta = 0.0
        if has_bug_label and "has_bug_label" not in new_signals:
            new_signals.append("has_bug_label")
            conf_delta += 0.15
        if pr_number and "github_pr" not in new_signals:
            new_signals.append("github_pr")

        # Fetch expected_findings from diff
        findings = candidate.expected_findings
        try:
            diff_files = fetch_pr_diff(repo, pr_number)
            if diff_files:
                findings = extract_expected_findings(diff_files)
        except Exception:
            pass

        pr_files: list[dict[str, Any]] = detail.get("files") or []
        file_names = [str(f.get("path", "")) for f in pr_files] or candidate.files_changed
        additions = int(detail.get("additions") or candidate.diff_stats.lines_added)
        deletions = int(detail.get("deletions") or candidate.diff_stats.lines_deleted)

        enriched.append(
            candidate.model_copy(
                update={
                    "pr_number": pr_number,
                    "title": str(detail.get("title") or candidate.title),
                    "body": str(detail.get("body") or ""),
                    "labels": labels,
                    "files_changed": file_names,
                    "diff_stats": CaseStats(
                        lines_added=additions,
                        lines_deleted=deletions,
                        files_changed=len(file_names),
                        hunks=candidate.diff_stats.hunks,
                    ),
                    "signals": new_signals,
                    "confidence": min(1.0, candidate.confidence + conf_delta),
                    "expected_findings": findings,
                    "language": detect_language(file_names),
                    "pr_size": compute_pr_size(additions, deletions),
                }
            )
        )

    return enriched + rest


def load_scrape_state(path: Path) -> ScrapeState | None:
    """Load scrape state from a YAML file. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScrapeState(**data)


def save_scrape_state(state: ScrapeState, path: Path) -> None:
    """Save scrape state to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(state.model_dump(mode="json"), f, sort_keys=False)


def filter_already_processed(
    prs: list[dict[str, Any]], state: ScrapeState | None
) -> list[dict[str, Any]]:
    """Filter out PRs that have already been processed."""
    if state is None:
        return prs
    processed = set(state.processed_pr_numbers)
    return [pr for pr in prs if pr.get("number") not in processed]
