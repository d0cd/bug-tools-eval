"""CLI command: curate — LLM-assisted enrichment of candidate test cases."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import click
import yaml
from anthropic import Anthropic

from bugeval.git_utils import GitError, run_git
from bugeval.io import load_candidates, save_case
from bugeval.models import (
    Candidate,
    Category,
    Difficulty,
    ExpectedFinding,
    Severity,
    TestCase,
)

_CHECKPOINT_FILE = ".curate_checkpoint.json"

# Default system prompt (overridable by config/curate_prompt.md)
_DEFAULT_SYSTEM_PROMPT = """\
You are a software bug analyst classifying bug-fix PRs for an evaluation framework.

Given a PR's metadata and diff, return a JSON object:
{
  "category": "<logic|memory|concurrency|api|type|perf>",
  "difficulty": "<easy|medium|hard>",
  "severity": "<low|medium|high|critical>",
  "description": "<2-3 sentences on the bug and fix>",
  "expected_findings": [{"file": "<path>", "line": <int>, "summary": "<what to flag>"}],
  "head_commit": "<bug-introducing SHA or null>",
  "base_commit": "<parent of head_commit or null>",
  "needs_manual_review": <true/false>
}

expected_findings should identify WHERE THE BUG IS (not the fix).
Return ONLY the JSON object.\
"""


def _load_system_prompt() -> str:
    """Load system prompt from config/curate_prompt.md if it exists."""
    prompt_path = Path("config") / "curate_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return _DEFAULT_SYSTEM_PROMPT


def build_curation_prompt(candidate: Candidate, diff_context: str, git_log: str = "") -> str:
    """Build the user-facing LLM prompt for curating a candidate."""
    findings_yaml = yaml.dump(
        [f.model_dump() for f in candidate.expected_findings],
        default_flow_style=False,
    )

    lines = [
        f"## PR #{candidate.pr_number}: {candidate.title}",
        "",
        f"**Repo:** {candidate.repo}",
        f"**Language:** {candidate.language}",
        (
            f"**PR Size:** {candidate.pr_size} "
            f"({candidate.diff_stats.lines_added}+ / {candidate.diff_stats.lines_deleted}-)"
        ),
        f"**Confidence signals:** {', '.join(candidate.signals)}",
        "",
        "### PR Description",
        candidate.body[:1000] if candidate.body else "(no description)",
        "",
        "### Changed Files",
        ", ".join(candidate.files_changed[:20]),
        "",
        "### Auto-Extracted Expected Findings (may need correction)",
        findings_yaml,
        "",
        "### Diff",
        "```",
        diff_context[:4000] if diff_context else "(no diff available)",
        "```",
    ]

    if git_log:
        lines.extend(
            [
                "",
                "### Git History Before Fix (for bug-introducing commit identification)",
                "```",
                git_log[:2000],
                "```",
            ]
        )

    lines.extend(["", "Classify this bug-fix PR and return the JSON."])
    return "\n".join(lines)


def get_git_context(candidate: Candidate, repo_dir: Path) -> tuple[str, str]:
    """Get diff and git log context from a repo checkout. Returns (diff, git_log)."""
    diff = ""
    git_log = ""

    if not candidate.fix_commit:
        return diff, git_log

    try:
        diff = run_git("show", "--stat", "-p", "--format=", candidate.fix_commit, cwd=repo_dir)
        diff = diff[:5000]
    except GitError:
        pass

    for filepath in candidate.files_changed[:3]:
        try:
            log = run_git(
                "log",
                "--oneline",
                "--follow",
                "-20",
                "--",
                filepath,
                cwd=repo_dir,
            )
            if log.strip():
                git_log += f"# {filepath}\n{log}\n"
        except GitError:
            pass

    return diff, git_log


def parse_llm_response(
    data: dict[str, Any],
    case_id: str,
    candidate: Candidate,
) -> TestCase:
    """Parse LLM response dict into a TestCase."""
    findings = [ExpectedFinding(**f) for f in (data.get("expected_findings") or [])]

    head_commit = data.get("head_commit") or candidate.fix_commit
    base_commit = data.get("base_commit") or f"{candidate.fix_commit}^"

    return TestCase(
        id=case_id,
        repo=candidate.repo,
        base_commit=str(base_commit),
        head_commit=str(head_commit),
        fix_commit=candidate.fix_commit,
        category=Category(data["category"]),
        difficulty=Difficulty(data["difficulty"]),
        severity=Severity(data["severity"]),
        language=candidate.language,
        pr_size=candidate.pr_size,
        description=str(data["description"]),
        expected_findings=findings,
        stats=None,
        needs_manual_review=bool(data.get("needs_manual_review", False)),
    )


def _generate_case_id(repo: str, output_dir: Path) -> str:
    """Generate next available case ID for a repo in the format <repo-slug>-NNN."""
    repo_short = repo.split("/")[-1][:15].replace(".", "-")
    prefix = f"{repo_short}-"
    max_index = 0
    for f in output_dir.glob(f"{prefix}*.yaml"):
        try:
            max_index = max(max_index, int(f.stem[len(prefix) :]))
        except ValueError:
            pass
    return f"{prefix}{max_index + 1:03d}"


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a text response."""
    # Strip code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not brace_match:
        return None

    try:
        return json.loads(brace_match.group(0))  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None


def curate_candidate(
    candidate: Candidate,
    diff_context: str,
    git_log: str,
    case_id: str,
    system_prompt: str,
    model: str = "claude-opus-4-6",
) -> TestCase | None:
    """Curate a single candidate via the Anthropic API."""
    prompt = build_curation_prompt(candidate, diff_context, git_log)
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    from anthropic.types import TextBlock
    text = "".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    )
    data = _extract_json_from_text(text)
    if data is None:
        return None
    try:
        return parse_llm_response(data, case_id, candidate)
    except (KeyError, ValueError):
        return None


@click.command("curate")
@click.option(
    "--candidates",
    "candidates_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to candidates YAML file.",
)
@click.option(
    "--repo-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Path to a git repo checkout for diff and blame context.",
)
@click.option(
    "--output-dir",
    default="cases",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write test case YAML files.",
)
@click.option(
    "--min-confidence",
    default=0.4,
    show_default=True,
    help="Minimum confidence to curate a candidate.",
)
@click.option(
    "--api-delay",
    default=1.0,
    show_default=True,
    type=float,
    help="Seconds to wait between API calls (rate limiting).",
)
@click.option("--dry-run", is_flag=True, help="Print prompts without calling the API.")
@click.option(
    "--limit",
    default=0,
    show_default=True,
    type=int,
    help="Max candidates to process in this run (0 = no limit). Use for safe batching.",
)
@click.option(
    "--fail-after",
    default=5,
    show_default=True,
    type=int,
    help="Abort after this many consecutive errors (prevents runaway on persistent failures).",
)
@click.option(
    "--no-checkpoint",
    is_flag=True,
    default=False,
    help="Ignore existing checkpoint and re-process all candidates.",
)
@click.option(
    "--shard",
    default=None,
    metavar="K/N",
    help="Process shard K of N (0-indexed). E.g. --shard 0/3 takes every 3rd candidate "
         "starting at 0. Use a separate --output-dir per shard to avoid ID conflicts.",
)
def curate(
    candidates_path: Path,
    repo_dir: Path | None,
    output_dir: Path,
    min_confidence: float,
    api_delay: float,
    dry_run: bool,
    limit: int,
    fail_after: int,
    no_checkpoint: bool,
    shard: str | None,
) -> None:
    """LLM-assisted enrichment of candidates into fully specified test cases."""
    candidates = load_candidates(candidates_path)
    filtered = [c for c in candidates if c.confidence >= min_confidence]
    filtered.sort(key=lambda c: c.confidence, reverse=True)

    # Sharding: process every Nth candidate starting at offset K
    if shard is not None:
        try:
            k, n = (int(x) for x in shard.split("/"))
            if n < 1 or k < 0 or k >= n:
                raise ValueError
        except (ValueError, TypeError):
            raise click.BadParameter("must be in format K/N, e.g. 0/3", param_hint="--shard")
        filtered = filtered[k::n]
        click.echo(f"Shard {k}/{n}: {len(filtered)} candidates.")

    # Checkpoint: skip already-processed fix_commits
    checkpoint_path = Path(output_dir) / _CHECKPOINT_FILE
    done_commits: set[str] = set()
    if not no_checkpoint and checkpoint_path.exists():
        try:
            done_commits = set(json.loads(checkpoint_path.read_text()))
            click.echo(f"Resuming: {len(done_commits)} already curated (from checkpoint).")
        except Exception:
            pass
    filtered = [c for c in filtered if c.fix_commit not in done_commits]

    if limit > 0:
        filtered = filtered[:limit]
        click.echo(f"Curating up to {limit} candidates (min_confidence={min_confidence}).")
    else:
        click.echo(f"Curating {len(filtered)} candidates (min_confidence={min_confidence}).")

    if not filtered:
        click.echo("No candidates to curate.")
        return

    system_prompt = _load_system_prompt()

    if dry_run:
        for i, candidate in enumerate(filtered):
            prompt = build_curation_prompt(candidate, diff_context="(dry run — no diff fetched)")
            click.echo(f"\n--- [{i + 1}/{len(filtered)}] PR #{candidate.pr_number} ---")
            click.echo(f"Confidence: {candidate.confidence:.2f} | Signals: {candidate.signals}")
            click.echo(f"Prompt length: {len(prompt)} chars")
            click.echo(prompt[:300] + "...")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    consecutive_errors = 0

    for i, candidate in enumerate(filtered):
        case_id = _generate_case_id(candidate.repo, output_dir)

        diff_context = ""
        git_log = ""
        if repo_dir is not None:
            diff_context, git_log = get_git_context(candidate, repo_dir)

        click.echo(
            f"  [{i + 1}/{len(filtered)}] PR #{candidate.pr_number}: "
            f"{candidate.title[:50]}..."
        )

        try:
            case = curate_candidate(
                candidate=candidate,
                diff_context=diff_context,
                git_log=git_log,
                case_id=case_id,
                system_prompt=system_prompt,
            )
            consecutive_errors = 0  # reset on any non-exception response
        except Exception as e:
            consecutive_errors += 1
            click.echo(
                f"  FAIL PR #{candidate.pr_number}: {e} "
                f"(consecutive errors: {consecutive_errors}/{fail_after})",
                err=True,
            )
            if consecutive_errors >= fail_after:
                click.echo(
                    f"Aborting: {consecutive_errors} consecutive errors — "
                    "check SDK auth or network and resume with checkpoint.",
                    err=True,
                )
                break
            if api_delay > 0:
                time.sleep(api_delay)
            continue

        if case is None:
            click.echo(f"  SKIP {case_id}: could not parse LLM response")
        else:
            case_path = output_dir / f"{case_id}.yaml"
            save_case(case, case_path)
            click.echo(f"  DONE {case_id} → {case_path}")
            success_count += 1

        # Save checkpoint after each candidate
        done_commits.add(candidate.fix_commit)
        checkpoint_path.write_text(json.dumps(sorted(done_commits)))

        if api_delay > 0:
            time.sleep(api_delay)

    click.echo(f"\nCurated {success_count}/{len(filtered)} candidates → {output_dir}/")
