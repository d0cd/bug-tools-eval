"""Mine fix PRs from GitHub repos and build initial test cases."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from bugeval.io import load_checkpoint, save_case, save_checkpoint
from bugeval.models import CaseKind, CaseStats, PRRelation, ReviewThread, TestCase

log = logging.getLogger(__name__)


class GhError(Exception):
    def __init__(self, command: list[str], stderr: str) -> None:
        self.command = command
        self.stderr = stderr
        super().__init__(
            f"gh command failed: {' '.join(command)}\n{stderr}"
        )


def run_gh(*args: str, timeout: int = 60) -> str:
    """Run a gh CLI command and return stdout."""
    cmd = ["gh", *args]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise GhError(cmd, f"Command timed out after {timeout}s")
    if result.returncode != 0:
        raise GhError(cmd, result.stderr)
    return result.stdout


# --- Fix-keyword detection ---

_FIX_KEYWORDS = re.compile(
    r"\b(fix(es|ed|ing)?|bug|patch|correct(ed|s|ing)?"
    r"|resolve[sd]?|revert)\b",
    re.IGNORECASE,
)

_ISSUE_REF = re.compile(
    r"(close[sd]?|fix(e[sd])?|resolve[sd]?)\s*#(\d+)",
    re.IGNORECASE,
)

_REFERENCE_PATTERN = re.compile(
    r"(?:see|related\s+to|followup\s+to"
    r"|completes?\s+(?:fix\s+from)?)\s*#(\d+)",
    re.IGNORECASE,
)

_PR_CROSS_REF = re.compile(r"#(\d+)", re.IGNORECASE)


def has_fix_signal(title: str, body: str, labels: list[str]) -> bool:
    """Check if PR has bug-fix signals."""
    bug_labels = {
        "bug", "fix", "regression", "defect", "bugfix", "hotfix",
    }
    if any(lbl.lower() in bug_labels for lbl in labels):
        return True
    text = f"{title} {body}"
    return bool(_FIX_KEYWORDS.search(text))


def extract_closing_issues(text: str) -> list[int]:
    """Extract issue numbers from 'fixes #N', 'closes #N' etc."""
    return [int(m.group(3)) for m in _ISSUE_REF.finditer(text)]


def extract_referenced_issues(text: str) -> list[int]:
    """Extract issue numbers from 'see #N', 'related to #N' etc."""
    return [int(m.group(1)) for m in _REFERENCE_PATTERN.finditer(text)]


# --- GitHub API helpers ---


def fetch_fix_prs(
    repo: str, limit: int, since: str
) -> list[dict[str, Any]]:
    """Fetch merged PRs with fix signals."""
    fields = (
        "number,title,body,labels,mergeCommit,baseRefName,headRefName,"
        "files,additions,deletions,changedFiles,mergedAt,author,commits,"
        "reviewDecision,statusCheckRollup"
    )
    args = [
        "pr", "list", "--repo", repo, "--state", "merged",
        "--json", fields, "--limit", str(limit),
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
            results.append(pr)

    return results


def _is_non_code_only(files: list[str]) -> bool:
    non_code_patterns = {
        ".md", ".txt", ".yml", ".yaml", ".toml", ".lock", ".json",
    }
    non_code_dirs = {
        "docs/", ".github/", ".circleci/", ".gitlab-ci",
    }
    for f in files:
        ext = Path(f).suffix.lower()
        if ext not in non_code_patterns and not any(
            f.startswith(d) for d in non_code_dirs
        ):
            return False
    return True


def fetch_pr_details_graphql(
    owner: str, name: str, pr_numbers: list[int],
) -> dict[int, dict[str, Any]]:
    """Batch-fetch rich PR details via GraphQL."""
    if not pr_numbers:
        return {}

    batch_size = 20
    all_results: dict[int, dict[str, Any]] = {}

    for i in range(0, len(pr_numbers), batch_size):
        batch = pr_numbers[i : i + batch_size]
        fragments = []
        for num in batch:
            fragments.append(f"""
  pr_{num}: pullRequest(number: {num}) {{
    number
    title
    body
    createdAt
    mergedAt
    mergeCommit {{ oid }}
    mergeMethod
    statusCheckRollup {{ state }}
    author {{ login }}
    commits(first: 100) {{
      nodes {{ commit {{ oid message }} }}
    }}
    reviews(first: 100) {{
      nodes {{ body state author {{ login }} }}
    }}
    reviewThreads(first: 100) {{
      nodes {{
        path line originalLine isResolved
        comments(first: 20) {{
          nodes {{ body author {{ login }} diffHunk }}
        }}
      }}
    }}
    comments(first: 100) {{
      nodes {{ body author {{ login }} }}
    }}
    closingIssuesReferences(first: 10) {{
      nodes {{
        number title body
        labels(first: 10) {{ nodes {{ name }} }}
      }}
    }}
  }}""")

        joined = "".join(fragments)
        query = (
            f'query {{\n  repository(owner: "{owner}",'
            f' name: "{name}") {{{joined}\n  }}\n}}'
        )
        try:
            output = run_gh("api", "graphql", "-f", f"query={query}")
            data = json.loads(output)
            repo_data = data.get("data", {}).get("repository", {})
        except (GhError, json.JSONDecodeError):
            continue

        for num in batch:
            pr_data = repo_data.get(f"pr_{num}")
            if pr_data:
                all_results[num] = pr_data

    return all_results


def fetch_issue_details(
    repo: str, issue_numbers: list[int],
) -> dict[int, dict[str, Any]]:
    """Fetch issue bodies and labels."""
    results: dict[int, dict[str, Any]] = {}
    for num in issue_numbers:
        try:
            output = run_gh(
                "issue", "view", str(num), "--repo", repo,
                "--json", "number,title,body,labels",
            )
            results[num] = json.loads(output)
        except (GhError, json.JSONDecodeError):
            continue
    return results


# --- PR relationship graph ---


def detect_cross_references(
    prs: list[dict[str, Any]],
) -> dict[int, list[int]]:
    """Detect PR cross-references (mentions of other PRs)."""
    refs: dict[int, list[int]] = {}
    all_numbers = {int(pr["number"]) for pr in prs}
    for pr in prs:
        num = int(pr["number"])
        text = f"{pr.get('title', '')} {pr.get('body', '')}"
        mentioned = {
            int(m)
            for m in _PR_CROSS_REF.findall(text)
            if int(m) in all_numbers and int(m) != num
        }
        if mentioned:
            refs[num] = sorted(mentioned)
    return refs


def detect_reverts(prs: list[dict[str, Any]]) -> dict[int, int]:
    """Detect revert PRs. Returns {reverting_pr: reverted_pr}."""
    reverts: dict[int, int] = {}
    revert_pattern = re.compile(r"revert.*?#(\d+)", re.IGNORECASE)
    for pr in prs:
        title = str(pr.get("title") or "")
        if "revert" in title.lower():
            match = revert_pattern.search(title)
            if match:
                reverts[int(pr["number"])] = int(match.group(1))
    return reverts


def build_pr_relations(
    fix_pr_number: int,
    all_prs_by_number: dict[int, dict[str, Any]],
    cross_refs: dict[int, list[int]],
    reverts: dict[int, int],
) -> list[PRRelation]:
    """Build relationship graph for a fix PR."""
    relations: list[PRRelation] = []
    fix_pr = all_prs_by_number.get(fix_pr_number)
    if not fix_pr:
        return relations

    for ref_num in cross_refs.get(fix_pr_number, []):
        ref_pr = all_prs_by_number.get(ref_num)
        if not ref_pr:
            continue
        role = (
            "revert"
            if reverts.get(fix_pr_number) == ref_num
            else "related"
        )
        relations.append(PRRelation(
            pr_number=ref_num,
            role=role,
            commit=str(
                (ref_pr.get("mergeCommit") or {}).get("oid", "")
            ),
            title=str(ref_pr.get("title", "")),
            merge_date=str(ref_pr.get("mergedAt") or ""),
            author=str(
                (ref_pr.get("author") or {}).get("login", "")
            ),
        ))

    return relations


# --- TestCase construction ---


def _compute_pr_size(additions: int, deletions: int) -> str:
    total = additions + deletions
    if total < 10:
        return "tiny"
    if total < 50:
        return "small"
    if total < 200:
        return "medium"
    if total < 500:
        return "large"
    return "xl"


def _detect_language(files: list[str]) -> str:
    ext_map = {
        ".rs": "rust",
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".go": "go",
        ".java": "java",
        ".leo": "leo",
    }
    counts: dict[str, int] = {}
    for f in files:
        lang = ext_map.get(Path(f).suffix.lower())
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else "unknown"


def build_case_from_pr(
    repo: str,
    pr: dict[str, Any],
    case_id: str,
    graphql_data: dict[str, Any] | None = None,
    issue_data: dict[int, dict[str, Any]] | None = None,
    relations: list[PRRelation] | None = None,
) -> TestCase:
    """Build a TestCase from a fix PR and optional enrichment data."""
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
    labels = [
        str(lbl.get("name", ""))
        for lbl in (pr.get("labels") or [])
    ]
    pr_number = int(pr["number"])

    commit_messages: list[str] = []
    commit_shas: list[str] = []
    review_comments: list[str] = []
    review_threads: list[ReviewThread] = []
    discussion_comments: list[str] = []
    merge_method = ""
    ci_status = ""

    if graphql_data:
        for node in (
            (graphql_data.get("commits") or {}).get("nodes") or []
        ):
            sha = (node.get("commit") or {}).get("oid", "")
            if sha:
                commit_shas.append(sha)
            msg = (node.get("commit") or {}).get("message", "")
            if msg:
                commit_messages.append(msg)

        merge_method = str(graphql_data.get("mergeMethod") or "")
        status = graphql_data.get("statusCheckRollup") or {}
        ci_status = str(status.get("state") or "")

        for review in (
            (graphql_data.get("reviews") or {}).get("nodes") or []
        ):
            rb = str(review.get("body") or "").strip()
            state = str(review.get("state") or "")
            author = str((review.get("author") or {}).get("login", ""))
            if rb:
                prefix = f"[{author}:{state}] " if author or state else ""
                review_comments.append(f"{prefix}{rb}")
        for thread in (
            (graphql_data.get("reviewThreads") or {})
            .get("nodes") or []
        ):
            thread_comments: list[str] = []
            for comment in (
                (thread.get("comments") or {}).get("nodes") or []
            ):
                cb = str(comment.get("body") or "").strip()
                if cb:
                    review_comments.append(cb)
                    thread_comments.append(cb)
            review_threads.append(ReviewThread(
                path=str(thread.get("path") or ""),
                line=int(thread.get("line") or thread.get("originalLine") or 0),
                is_resolved=bool(thread.get("isResolved", False)),
                comments=thread_comments,
            ))

        for comment in (
            (graphql_data.get("comments") or {}).get("nodes") or []
        ):
            db = str(comment.get("body") or "").strip()
            if db:
                discussion_comments.append(db)

    linked_issues: list[int] = extract_closing_issues(
        f"{title} {body}"
    )
    referenced_issues: list[int] = extract_referenced_issues(
        f"{title} {body}"
    )
    issue_bodies: dict[int, str] = {}
    issue_labels_all: list[str] = list(labels)

    if graphql_data:
        closing_refs = (
            (graphql_data.get("closingIssuesReferences") or {})
            .get("nodes") or []
        )
        for node in closing_refs:
            inum = node.get("number")
            if inum and inum not in linked_issues:
                linked_issues.append(inum)
            ibody = str(node.get("body") or "")
            if inum and ibody:
                issue_bodies[inum] = ibody
            for lbl in (
                (node.get("labels") or {}).get("nodes") or []
            ):
                ln = str(lbl.get("name", ""))
                if ln and ln not in issue_labels_all:
                    issue_labels_all.append(ln)

    if issue_data:
        for inum, idata in issue_data.items():
            if inum not in issue_bodies:
                issue_bodies[inum] = str(idata.get("body") or "")
            for lbl in idata.get("labels") or []:
                ln = str(lbl.get("name", ""))
                if ln and ln not in issue_labels_all:
                    issue_labels_all.append(ln)

    # Add the fix PR itself to relations with role="full_fix"
    fix_relation = PRRelation(
        pr_number=pr_number,
        role="full_fix",
        commit=merge_commit,
        title=title,
        merge_date=str(pr.get("mergedAt") or ""),
        author=str((pr.get("author") or {}).get("login", "")),
    )
    all_relations = [fix_relation] + (relations or [])

    return TestCase(
        id=case_id,
        repo=repo,
        kind=CaseKind.bug,
        language=_detect_language(file_names),
        base_commit="",
        fix_commit=merge_commit,
        fix_pr_number=pr_number,
        fix_pr_title=title,
        fix_pr_body=body,
        fix_pr_commit_messages=commit_messages,
        fix_pr_commit_shas=commit_shas,
        fix_pr_merge_date=str(pr.get("mergedAt") or ""),
        fix_pr_review_comments=review_comments,
        fix_pr_review_threads=review_threads,
        fix_pr_discussion_comments=discussion_comments,
        fix_pr_merge_method=merge_method,
        fix_pr_ci_status=ci_status,
        linked_issues=linked_issues,
        issue_bodies=issue_bodies,
        issue_labels=issue_labels_all,
        referenced_issues=referenced_issues,
        related_prs=all_relations,
        stats=CaseStats(
            lines_added=additions,
            lines_deleted=deletions,
            files_changed=files_count or len(file_names),
        ),
        pr_size=_compute_pr_size(additions, deletions),
    )


def find_duplicate(cases_dir: Path, fix_pr_number: int) -> str | None:
    """Return case_id if a case with this fix_pr_number already exists, else None."""
    for p in sorted(cases_dir.rglob("*.yaml")):
        try:
            with open(p) as f:
                data = yaml.safe_load(f)
            if data and data.get("fix_pr_number") == fix_pr_number:
                return str(data.get("id", p.stem))
        except Exception:
            continue
    return None


# --- Orchestration ---


def mine_repo(
    repo: str,
    limit: int,
    since: str,
    output_dir: Path,
    concurrency: int = 1,
) -> list[TestCase]:
    """Mine fix PRs from a repo and write TestCase YAMLs."""
    owner, name = repo.split("/", 1)
    repo_slug = name
    repo_dir = output_dir / repo_slug
    repo_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = repo_dir / ".mine_checkpoint.json"
    done = load_checkpoint(checkpoint_path)

    log.info(
        "Fetching fix PRs from %s (limit=%d, since=%s)",
        repo, limit, since,
    )
    prs = fetch_fix_prs(repo, limit, since)
    log.info("Found %d fix PRs", len(prs))

    prs_by_number = {int(pr["number"]): pr for pr in prs}
    cross_refs = detect_cross_references(prs)
    reverts = detect_reverts(prs)

    pending_prs = [
        pr for pr in prs if str(pr["number"]) not in done
    ]
    log.info(
        "Processing %d pending PRs (%d already done)",
        len(pending_prs), len(done),
    )

    pending_numbers = [int(pr["number"]) for pr in pending_prs]
    graphql_details = fetch_pr_details_graphql(
        owner, name, pending_numbers,
    )

    all_issue_nums: set[int] = set()
    for pr in pending_prs:
        text = f"{pr.get('title', '')} {pr.get('body', '')}"
        all_issue_nums.update(extract_closing_issues(text))
    for _num, gql in graphql_details.items():
        closing_refs = (
            (gql.get("closingIssuesReferences") or {})
            .get("nodes") or []
        )
        for node in closing_refs:
            inum = node.get("number")
            if inum:
                all_issue_nums.add(inum)

    issue_details = (
        fetch_issue_details(repo, sorted(all_issue_nums))
        if all_issue_nums
        else {}
    )

    existing = sorted(repo_dir.glob(f"{repo_slug}-*.yaml"))
    next_num = len(existing) + 1

    cases: list[TestCase] = []
    for pr in pending_prs:
        pr_num = int(pr["number"])

        dup = find_duplicate(repo_dir, pr_num)
        if dup:
            log.info("Skipping PR #%d: duplicate of %s", pr_num, dup)
            done.add(str(pr_num))
            save_checkpoint(done, checkpoint_path)
            continue

        case_id = f"{repo_slug}-{next_num:03d}"
        relations = build_pr_relations(
            pr_num, prs_by_number, cross_refs, reverts,
        )
        gql = graphql_details.get(pr_num)

        case = build_case_from_pr(
            repo=repo,
            pr=pr,
            case_id=case_id,
            graphql_data=gql,
            issue_data=issue_details,
            relations=relations,
        )
        save_case(case, repo_dir / f"{case_id}.yaml")
        cases.append(case)

        done.add(str(pr_num))
        save_checkpoint(done, checkpoint_path)
        next_num += 1

    log.info("Wrote %d new cases to %s", len(cases), repo_dir)
    return cases
