"""Core logic for mining bug-fix commits from a local git history."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bugeval.git_utils import GitError, run_git
from bugeval.github_scraper import compute_pr_size, detect_language
from bugeval.models import Candidate, CaseStats, ExpectedFinding

_FIX_KEYWORDS = re.compile(
    r"\b(fix(es|ed|ing)?|bug|issue|revert|patch|correct(ed|ing)?|repair)\b",
    re.IGNORECASE,
)
_ISSUE_REF = re.compile(r"(close[sd]?|fix(e[sd])?|resolve[sd]?)\s*#\d+|#\d+", re.IGNORECASE)

_LLM_MODEL = "claude-haiku-4-5-20251001"
_COMMIT_SEP = "COMMIT_START"


def detect_fix_keywords(message: str) -> list[str]:
    """Return a list of keyword signal strings found in the commit message."""
    signals: list[str] = []
    if _FIX_KEYWORDS.search(message):
        signals.append("keyword:fix")
    if _ISSUE_REF.search(message):
        signals.append("keyword:issue_ref")
    return signals


def score_git_candidate(commit: dict[str, Any], has_introducing: bool) -> tuple[float, list[str]]:
    """Compute confidence 0.0–1.0 and signals from a commit dict.

    Scoring heuristic:
    - +0.25 fix keywords in subject
    - +0.15 issue reference
    - +0.20 small diff (<200 lines)
    - +0.20 introducing commit identified
    - +0.10 few files (1–3)
    - +0.10 merge commit
    Capped at 1.0.
    """
    message: str = commit.get("message", "")
    lines_added: int = commit.get("lines_added", 0)
    lines_deleted: int = commit.get("lines_deleted", 0)
    files: list[str] = commit.get("files", [])
    is_merge: bool = commit.get("is_merge", False)

    signals = detect_fix_keywords(message)
    score = 0.0

    if "keyword:fix" in signals:
        score += 0.25
    if "keyword:issue_ref" in signals:
        score += 0.15
    if (lines_added + lines_deleted) < 200:
        score += 0.20
        signals.append("signal:small_diff")
    if has_introducing:
        score += 0.20
        signals.append("signal:has_introducing")
    if 1 <= len(files) <= 3:
        score += 0.10
        signals.append("signal:few_files")
    if is_merge:
        score += 0.10
        signals.append("signal:merge_commit")

    return min(score, 1.0), signals


def parse_fix_commits(cwd: Path, branch: str, limit: int) -> list[dict[str, Any]]:
    """Parse git log on branch and return commit dicts filtered to potential bug-fixes."""
    log_output = run_git(
        "log",
        f"-{limit}",
        f"--format={_COMMIT_SEP}%n%H%n%s%n%P",
        "--numstat",
        "--first-parent",
        branch,
        cwd=cwd,
    )

    if not log_output.strip():
        return []

    # Split on the sentinel; first element is empty (before first COMMIT_START)
    raw_blocks = log_output.split(_COMMIT_SEP)
    commits: list[dict[str, Any]] = []

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        if len(lines) < 2:
            continue

        sha = lines[0].strip()
        subject = lines[1].strip()
        parents = lines[2].strip() if len(lines) > 2 else ""
        is_merge = len(parents.split()) > 1

        if not sha:
            continue

        # Everything after the header lines is numstat
        files: list[str] = []
        lines_added = 0
        lines_deleted = 0

        for line in lines[3:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                if parts[0] != "-" and parts[1] != "-":
                    try:
                        lines_added += int(parts[0])
                        lines_deleted += int(parts[1])
                    except ValueError:
                        pass
                files.append(parts[2])  # Always include the file path

        # Only include if fix keywords detected
        signals = detect_fix_keywords(subject)
        if not signals:
            continue

        commits.append(
            {
                "sha": sha,
                "message": subject,
                "lines_added": lines_added,
                "lines_deleted": lines_deleted,
                "files": files,
                "is_merge": is_merge,
            }
        )

    return commits


def find_introducing_commit(
    fix_sha: str, files: list[str], cwd: Path, window: int = 200
) -> str | None:
    """Find a commit that last modified the same files before fix_sha.

    Returns the SHA of the most recent commit that touched the same files
    before fix_sha (within window commits), or None if not found.
    """
    if not files:
        return None
    try:
        # Get commits before fix_sha
        log_output = run_git(
            "log",
            "--format=%H",
            f"-{window}",
            f"{fix_sha}^",
            cwd=cwd,
        )
    except GitError:
        return None

    prior_shas = [s.strip() for s in log_output.splitlines() if s.strip()]

    for prior_sha in prior_shas[:window]:
        try:
            stat_output = run_git(
                "diff-tree", "--no-commit-id", "--root", "-r", "--name-only", prior_sha, cwd=cwd
            )
        except GitError:
            continue
        changed = set(stat_output.strip().splitlines())
        if changed.intersection(files):
            return prior_sha

    return None


def build_git_candidates(repo: str, commits: list[dict[str, Any]], cwd: Path) -> list[Candidate]:
    """Assemble Candidate objects from scored commit dicts."""
    candidates: list[Candidate] = []
    for i, commit in enumerate(commits):
        introducing = find_introducing_commit(commit["sha"], commit["files"], cwd)
        confidence, signals = score_git_candidate(commit, has_introducing=introducing is not None)

        files = commit["files"]
        language = detect_language(files)
        pr_size = compute_pr_size(commit["lines_added"], commit["lines_deleted"])

        candidate = Candidate(
            repo=repo,
            pr_number=i,  # no PR number for git-mined commits; use index as placeholder
            fix_commit=commit["sha"],
            base_commit=introducing,
            head_commit=commit["sha"],
            confidence=confidence,
            signals=signals,
            title=commit["message"],
            body="",
            labels=[],
            files_changed=files,
            diff_stats=CaseStats(
                lines_added=commit["lines_added"],
                lines_deleted=commit["lines_deleted"],
                files_changed=len(files),
                hunks=0,
            ),
            expected_findings=[ExpectedFinding(file=f, line=0, summary="") for f in files[:1]]
            if files
            else [],
            language=language,
            pr_size=pr_size,
        )
        candidates.append(candidate)

    return candidates


def llm_link_introducing_commit(
    client: Any,
    fix_sha: str,
    diff: str,
    git_log: str,
) -> str | None:
    """Optional LLM fallback to identify an introducing commit from a diff and log."""
    prompt = (
        f"Given this bug-fix diff (SHA: {fix_sha}):\n\n{diff[:3000]}\n\n"
        f"And this recent git log:\n\n{git_log[:2000]}\n\n"
        "Which commit SHA most likely introduced the bug being fixed? "
        "Reply with ONLY the 40-character SHA or 'unknown'."
    )
    try:
        response = client.messages.create(
            model=_LLM_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if re.match(r"^[0-9a-f]{40}$", text):
            return text
    except Exception:
        pass
    return None
