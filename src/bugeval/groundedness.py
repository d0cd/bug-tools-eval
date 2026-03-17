"""CLI command: groundedness-check — verify expected_findings exist in pre-fix diff."""

from __future__ import annotations

import json
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click

from bugeval.agent_cli_runner import run_claude_cli
from bugeval.io import load_case, save_case
from bugeval.models import TestCase

# ---------------------------------------------------------------------------
# Diff hunk extraction
# ---------------------------------------------------------------------------


def _extract_hunk_context(patch: str, target_file: str, target_line: int, window: int = 20) -> str:
    """Return the ±window lines of diff context around target_file:target_line.

    Parses hunk headers to find the right file section, then slices lines around target_line.
    Returns an empty string if the file/line is not found in the patch.
    """
    lines = patch.splitlines()
    in_target_file = False
    current_orig_line = 0
    file_section_lines: list[str] = []

    hunk_header_re = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")

    for line in lines:
        # Detect file headers
        if line.startswith("--- "):
            # Check if this is our target file
            fname = line[4:].strip()
            if fname.endswith(target_file) or target_file in fname:
                in_target_file = True
            else:
                in_target_file = False
            file_section_lines = []
            current_orig_line = 0
            continue

        if not in_target_file:
            continue

        # Parse hunk headers within our file
        hunk_match = hunk_header_re.match(line)
        if hunk_match:
            current_orig_line = int(hunk_match.group(1))
            file_section_lines.append(line)
            continue

        file_section_lines.append(line)

        # Track line numbers in original file
        if line.startswith("-") or (not line.startswith("+") and not line.startswith("\\")):
            current_orig_line += 1

    # Find lines near target_line in file_section_lines
    if not file_section_lines:
        return ""

    # Re-scan to find the window around target_line
    current_orig_line = 0
    best_match_idx = -1
    best_dist = float("inf")

    for idx, line in enumerate(file_section_lines):
        hunk_match = hunk_header_re.match(line)
        if hunk_match:
            current_orig_line = int(hunk_match.group(1))
            continue
        if line.startswith("-") or (not line.startswith("+") and not line.startswith("\\")):
            dist = abs(current_orig_line - target_line)
            if dist < best_dist:
                best_dist = dist
                best_match_idx = idx
            current_orig_line += 1

    if best_match_idx < 0:
        return "\n".join(file_section_lines[:40])

    start = max(0, best_match_idx - window)
    end = min(len(file_section_lines), best_match_idx + window + 1)
    return "\n".join(file_section_lines[start:end])


def _extract_file_section(patch: str, target_file: str) -> str:
    """Return all diff lines for target_file from the patch.

    Returns an empty string if the file is not present in the patch.
    Unlike _extract_hunk_context, this returns the full file section so haiku
    can see all changed lines without a narrow window that might miss context.
    """
    lines = patch.splitlines()
    in_target_file = False
    file_section_lines: list[str] = []

    for line in lines:
        if line.startswith("--- "):
            fname = line[4:].strip()
            if fname.endswith(target_file) or target_file in fname:
                in_target_file = True
                file_section_lines = [line]
            else:
                in_target_file = False
            continue
        if line.startswith("+++ ") and in_target_file:
            file_section_lines.append(line)
            continue
        if not in_target_file:
            continue
        # Stop at the next file header
        if line.startswith("diff --git "):
            break
        file_section_lines.append(line)

    return "\n".join(file_section_lines)


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------


_VERIFY_PROMPT_TEMPLATE = """\
Given this unified diff, does the following issue exist in the ORIGINAL (pre-fix, '-' side) \
code at {file}:{line}?

Issue: {summary}

Diff context:
```
{diff_context}
```

Return ONLY a JSON object with no other text:
{{"exists": true/false, "confidence": 0.0-1.0, "reasoning": "..."}}"""


def _call_haiku_verify(
    target_file: str,
    target_line: int,
    summary: str,
    diff_context: str,
    model: str,
) -> dict[str, Any] | None:
    """Call the claude CLI to verify a finding in the diff."""
    prompt = _VERIFY_PROMPT_TEMPLATE.format(
        file=target_file,
        line=target_line,
        summary=summary,
        diff_context=diff_context[:6000],
    )
    try:
        result = run_claude_cli(
            Path("."),
            prompt,
            max_turns=1,
            timeout_seconds=60,
            model=model,
            allowed_tools="",
        )
        text = result.response_text or ""
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))  # type: ignore[no-any-return]
    except Exception:
        pass
    return None


def _get_diff_for_case(case: TestCase, patches_dir: Path | None) -> str:
    """Get the diff for a test case: use pre-extracted patch or fetch via gh api."""
    # Try pre-extracted patch file first
    if patches_dir is not None:
        patch_file = patches_dir / f"{case.id}.patch"
        if patch_file.exists():
            return patch_file.read_text()

    # Fetch via gh api using pr_number if available
    if case.pr_number:
        repo = case.repo
        owner, name = repo.split("/", 1)
        try:
            cmd = ["gh", "api", f"repos/{owner}/{name}/pulls/{case.pr_number}",
                   "--header", "Accept: application/vnd.github.v3.diff"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, OSError):
            pass

    return ""


# ---------------------------------------------------------------------------
# Core check logic
# ---------------------------------------------------------------------------


def check_case_groundedness(
    case: TestCase,
    patches_dir: Path | None,
    model: str,
    dry_run: bool,
) -> tuple[TestCase, bool | None]:
    """Check whether expected_findings actually exist in the pre-fix diff.

    Returns (updated_case, verdict) where verdict is:
    - True: findings verified to exist
    - False: findings not found (high confidence)
    - None: skipped (no findings, or low confidence)
    """
    if not case.expected_findings:
        return case, None

    diff = _get_diff_for_case(case, patches_dir)
    if not diff:
        return case, None

    finding = case.expected_findings[0]
    # Use the full file section so haiku sees all changed lines, not a narrow window.
    # If the target file isn't in the diff at all, skip rather than using unrelated content.
    diff_context = _extract_file_section(diff, finding.file)
    if not diff_context:
        return case, None

    if dry_run:
        click.echo(
            f"  [dry-run] {case.id}: would verify {finding.file}:{finding.line} "
            f"— '{finding.summary[:60]}'"
        )
        return case, None

    verdict = _call_haiku_verify(finding.file, finding.line, finding.summary, diff_context, model)
    if verdict is None:
        return case, None

    exists = bool(verdict.get("exists", True))
    confidence = float(verdict.get("confidence", 0.5))

    if not exists:
        if confidence >= 0.7:
            flags = list(case.quality_flags)
            if "groundedness-failed" not in flags:
                flags.append("groundedness-failed")
            updated = case.model_copy(
                update={
                    "quality_flags": flags,
                    "needs_manual_review": True,
                }
            )
            return updated, False
        # Low confidence: ambiguous, skip
        return case, None

    return case, True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("groundedness-check")
@click.option(
    "--cases-dir",
    default="cases/final",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Cases directory to check.",
)
@click.option(
    "--patches-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Optional directory of pre-extracted .patch files.",
)
@click.option(
    "--model",
    default="claude-haiku-4-5-20251001",
    show_default=True,
    help="Model to use for verification.",
)
@click.option(
    "--limit",
    default=0,
    show_default=True,
    type=int,
    help="Max cases to check (0 = no limit).",
)
@click.option("--dry-run", is_flag=True, help="Print verdicts without writing files.")
@click.option(
    "--workers",
    default=8,
    show_default=True,
    type=int,
    help="Number of parallel workers (each spawns a claude CLI subprocess).",
)
def groundedness_check(
    cases_dir: Path,
    patches_dir: Path | None,
    model: str,
    limit: int,
    dry_run: bool,
    workers: int,
) -> None:
    """Verify expected_findings actually exist in the pre-fix diff (using haiku)."""
    # Load all cases from subdirs
    all_case_paths: list[Path] = []
    for repo_dir in sorted(cases_dir.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith("."):
            continue
        all_case_paths.extend(sorted(repo_dir.glob("*.yaml")))

    if not all_case_paths:
        click.echo("No cases found.")
        return

    to_check = all_case_paths
    if limit > 0:
        to_check = all_case_paths[:limit]

    click.echo(
        f"Checking groundedness of {len(to_check)} cases "
        f"(model={model}, workers={workers})..."
    )

    passed = failed = skipped = 0
    lock = threading.Lock()

    def _process(path: Path) -> tuple[Path, TestCase | None, TestCase | None, bool | None]:
        try:
            case = load_case(path)
        except Exception as exc:
            click.echo(f"  [warn] Could not load {path.name}: {exc}", err=True)
            return path, None, None, None
        if not case.expected_findings:
            return path, case, None, None
        updated, verdict = check_case_groundedness(case, patches_dir, model, dry_run)
        return path, case, updated, verdict

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process, p): p for p in to_check}
        for future in as_completed(futures):
            path, case, updated, verdict = future.result()
            with lock:
                if case is None:
                    pass  # load error, already reported
                elif not case.expected_findings:
                    skipped += 1
                elif verdict is None:
                    skipped += 1
                elif verdict:
                    passed += 1
                else:
                    failed += 1
                    assert updated is not None
                    click.echo(
                        f"  [FAIL] {case.id}: {case.expected_findings[0].file}:"
                        f"{case.expected_findings[0].line} — flagged groundedness-failed"
                    )
                    if not dry_run:
                        save_case(updated, path)

    click.echo(
        f"\nGroundedness check: {passed} passed, {failed} failed, {skipped} skipped"
    )
    if failed > 0 and not dry_run:
        click.echo(f"{failed} cases marked with quality_flags: ['groundedness-failed']")
