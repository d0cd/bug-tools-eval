"""Scoring: mechanical catch rate + LLM quality judge."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import click

from bugeval.io import (
    load_cases,
    load_checkpoint,
    load_result,
    save_checkpoint,
    save_score,
)
from bugeval.models import GroundTruth, TestCase
from bugeval.result_models import ToolResult
from bugeval.score_models import CaseScore, CommentScore, CommentVerdict

log = logging.getLogger(__name__)

_GENERIC_BODIES = {
    "lgtm",
    "looks good",
    "nit",
    "nice",
    "ok",
    "+1",
    "approved",
    "ship it",
}


def mechanical_catch(
    result: ToolResult,
    truth: GroundTruth | None,
    tolerance: int = 10,
) -> tuple[bool, int | None]:
    """Check if any comment references a buggy line within tolerance."""
    if truth is None or not truth.buggy_lines:
        return False, None

    best_dist: int | None = None
    for comment in result.comments:
        if comment.line == 0:
            continue
        for bl in truth.buggy_lines:
            if _files_match(comment.file, bl.file):
                dist = abs(comment.line - bl.line)
                if dist <= tolerance:
                    if best_dist is None or dist < best_dist:
                        best_dist = dist

    if best_dist is not None:
        return True, best_dist
    return False, None


def classify_comments(
    result: ToolResult,
    truth: GroundTruth | None,
    tolerance: int = 10,
) -> list[CommentScore]:
    """Classify each comment as TP, FP, or low-value."""
    scores: list[CommentScore] = []
    for i, comment in enumerate(result.comments):
        # Skip line=0 (file-level / general comments) — same guard as mechanical_catch
        if comment.line == 0:
            scores.append(CommentScore(
                comment_index=i,
                verdict=CommentVerdict.low_value,
            ))
            continue

        # Check for low-value first
        body_stripped = comment.body.strip().lower()
        if len(comment.body.strip()) < 20 or body_stripped in _GENERIC_BODIES:
            scores.append(CommentScore(
                comment_index=i,
                verdict=CommentVerdict.low_value,
            ))
            continue

        if truth is None or not truth.buggy_lines:
            scores.append(CommentScore(
                comment_index=i,
                verdict=CommentVerdict.fp,
            ))
            continue

        matched = False
        for bl_idx, bl in enumerate(truth.buggy_lines):
            if _files_match(comment.file, bl.file):
                dist = abs(comment.line - bl.line)
                if dist <= tolerance:
                    scores.append(CommentScore(
                        comment_index=i,
                        verdict=CommentVerdict.tp,
                        matched_buggy_line=bl_idx,
                    ))
                    matched = True
                    break

        if not matched:
            scores.append(CommentScore(
                comment_index=i,
                verdict=CommentVerdict.fp,
            ))

    return scores


def build_judge_prompt(
    case: TestCase, result: ToolResult, diff: str = "",
) -> str:
    """Build prompt for LLM quality judge."""
    comments_text = ""
    for i, c in enumerate(result.comments):
        comments_text += (
            f"\n### Comment {i}\n"
            f"File: {c.file}, Line: {c.line}\n"
            f"Body: {c.body}\n"
        )
        if c.suggested_fix:
            comments_text += f"Suggested fix: {c.suggested_fix}\n"

    buggy_lines_text = ""
    if case.truth and case.truth.buggy_lines:
        for bl in case.truth.buggy_lines:
            buggy_lines_text += f"  - {bl.file}:{bl.line} {bl.content}\n"

    diff_section = ""
    if diff:
        diff_section = f"\n## Diff Under Review\n```diff\n{diff}\n```\n"

    return f"""\
You are an expert code review judge. Evaluate the quality of a tool's \
bug-finding review.

## Known Bug
Description: {case.bug_description}
Fix summary: {case.truth.fix_summary if case.truth else 'N/A'}
Buggy lines:
{buggy_lines_text or '  (none)'}

## Review Comments
{comments_text or '(no comments)'}
{diff_section}
## Scoring Rubric

IMPORTANT: Score detection_score and review_quality INDEPENDENTLY.
A review can have high quality (found other real issues, good explanations)
even if it missed the specific known bug (detection_score=0).

**Detection Score (0-3):**
- 0 = missed — tool did not identify the bug at all
- 1 = wrong-area — tool flagged something in the right file but wrong area
- 2 = correct-id — tool correctly identified the bug location
- 3 = correct-id-and-fix — tool identified the bug AND suggested a correct fix

**Review Quality (0-4):**
- 0 = useless — no actionable feedback
- 1 = shallow — vague or generic feedback
- 2 = adequate — identifies the issue with some detail
- 3 = strong — clear identification with good explanation
- 4 = exceptional — precise identification, root cause, and correct fix

**Comment Verdicts:** For each comment, assign one of:
- "TP" — true positive, correctly identifies the known bug
- "TP-novel" — true positive, identifies a real bug NOT in the known ground truth
- "FP" — false positive, incorrect or irrelevant
- "low-value" — generic, vague, or too short to be useful

Respond with ONLY valid JSON (no markdown fences):
{{"detection_score": <0-3>, "review_quality": <0-4>, \
"comment_verdicts": [<verdict for each comment>], \
"reasoning": "<brief explanation>"}}"""


def call_judge(
    prompt: str,
    model: str = "claude-haiku-4-5",
    case_id: str = "",
) -> dict[str, Any]:
    """Call Anthropic API for LLM judging and parse JSON response."""
    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text  # type: ignore[union-attr]
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)  # type: ignore[no-any-return]
    except Exception as exc:
        log.warning("Judge call failed for %s: %s", case_id, exc)
        return {
            "detection_score": 0,
            "review_quality": 0,
            "comment_verdicts": [],
            "reasoning": "LLM judge failed",
            "judge_failed": True,
        }


def score_case(
    case: TestCase,
    result: ToolResult,
    use_llm: bool = True,
    judge_model: str = "claude-haiku-4-5",
) -> CaseScore:
    """Score a single case result (mechanical + optional LLM)."""
    # Clean cases: only check false alarm
    if case.kind == "clean":
        has_bug_comments = len(result.comments) > 0
        comment_scores = classify_comments(result, None)
        fp = sum(1 for cs in comment_scores if cs.verdict == CommentVerdict.fp)
        return CaseScore(
            case_id=case.id,
            tool=result.tool,
            caught=False,
            false_alarm=has_bug_comments,
            comment_scores=comment_scores,
            tp_count=0,
            fp_count=fp,
            potentially_contaminated=result.potentially_contaminated,
            context_level=result.context_level,
        )

    caught, dist = mechanical_catch(result, case.truth)
    comment_scores = classify_comments(result, case.truth)

    tp_count = sum(
        1 for s in comment_scores if s.verdict == CommentVerdict.tp
    )
    fp_count = sum(
        1 for s in comment_scores if s.verdict == CommentVerdict.fp
    )
    novel_count = sum(
        1 for s in comment_scores if s.verdict == CommentVerdict.tp_novel
    )

    detection_score = 0
    review_quality = 0
    reasoning = ""
    judge_failed = False

    if use_llm and case.truth:
        prompt = build_judge_prompt(case, result)
        judge_result = call_judge(prompt, model=judge_model, case_id=case.id)
        detection_score = max(0, min(3, int(judge_result.get("detection_score", 0))))
        review_quality = max(0, min(4, int(judge_result.get("review_quality", 0))))
        reasoning = str(judge_result.get("reasoning", ""))
        judge_failed = bool(judge_result.get("judge_failed", False))

        # Override comment verdicts from LLM if provided
        verdicts = judge_result.get("comment_verdicts", [])
        _VERDICT_MAP = {
            "TP": CommentVerdict.tp,
            "TP-novel": CommentVerdict.tp_novel,
            "FP": CommentVerdict.fp,
            "low-value": CommentVerdict.low_value,
        }
        for j, v in enumerate(verdicts):
            if j < len(comment_scores) and v in _VERDICT_MAP:
                comment_scores[j] = CommentScore(
                    comment_index=j,
                    verdict=_VERDICT_MAP[v],
                    matched_buggy_line=comment_scores[j].matched_buggy_line,
                )

        # Recount after LLM overrides
        tp_count = sum(
            1 for s in comment_scores if s.verdict == CommentVerdict.tp
        )
        fp_count = sum(
            1 for s in comment_scores if s.verdict == CommentVerdict.fp
        )
        novel_count = sum(
            1 for s in comment_scores
            if s.verdict == CommentVerdict.tp_novel
        )
    else:
        # Simple mechanical heuristic for detection score
        if caught and any(
            c.suggested_fix for c in result.comments
        ):
            detection_score = 3
        elif caught:
            detection_score = 2

    return CaseScore(
        case_id=case.id,
        tool=result.tool,
        caught=caught,
        localization_distance=dist,
        detection_score=detection_score,
        review_quality=review_quality,
        comment_scores=comment_scores,
        reasoning=reasoning,
        tp_count=tp_count,
        fp_count=fp_count,
        novel_count=novel_count,
        potentially_contaminated=result.potentially_contaminated,
        context_level=result.context_level,
        judge_failed=judge_failed,
    )


def detect_contamination(result: ToolResult, case: TestCase) -> bool:
    """Check if tool comments overlap suspiciously with fix PR text."""
    fix_texts = [
        case.fix_pr_title,
        case.fix_pr_body,
        *case.fix_pr_commit_messages,
        *case.fix_pr_review_comments,
        *case.fix_pr_discussion_comments,
    ]
    fix_words: set[str] = set()
    for text in fix_texts:
        fix_words.update(_tokenize(text))

    if len(fix_words) < 3:
        return False

    for comment in result.comments:
        comment_words = _tokenize(comment.body)
        if not comment_words:
            continue
        overlap = comment_words & fix_words
        if len(overlap) / len(comment_words) > 0.5:
            return True

    return False


def score_run(
    run_dir: Path,
    cases_dir: Path,
    dry_run: bool,
    judge_model: str = "claude-haiku-4-5",
) -> None:
    """Orchestrator: load results + cases, score each, save scores."""
    results_dir = run_dir / "results"
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = scores_dir / "checkpoint.json"
    done = load_checkpoint(checkpoint_path)

    cases = load_cases(cases_dir)
    case_map = {c.id: c for c in cases}

    result_files = sorted(results_dir.glob("*.yaml"))
    total = len(result_files)
    scored = 0

    for rf in result_files:
        result = load_result(rf)
        key = f"{result.case_id}__{result.tool}"

        if key in done:
            continue

        case = case_map.get(result.case_id)
        if case is None:
            click.echo(f"Warning: no case found for {result.case_id}")
            continue

        result.potentially_contaminated = detect_contamination(result, case)
        cs = score_case(
            case, result, use_llm=not dry_run, judge_model=judge_model,
        )
        save_score(cs, scores_dir / f"{key}.yaml")

        done.add(key)
        save_checkpoint(done, checkpoint_path)
        scored += 1

    click.echo(f"Scored {scored}/{total} results in {scores_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _files_match(comment_file: str, truth_file: str) -> bool:
    """Match file paths, handling partial paths."""
    if comment_file == truth_file:
        return True
    # Ensure match at path separator boundary
    if comment_file.endswith("/" + truth_file) or truth_file.endswith("/" + comment_file):
        return True
    # Also match basename if both are simple filenames
    if "/" not in comment_file and "/" not in truth_file:
        return comment_file == truth_file
    return False


def _tokenize(text: str) -> set[str]:
    """Extract lowercase word tokens (3+ chars) from text."""
    return {
        w.lower()
        for w in re.findall(r"[a-zA-Z_]\w{2,}", text)
    }
