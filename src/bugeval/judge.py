# src/bugeval/judge.py
"""LLM-as-judge: 3× majority vote scoring for normalized results."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import click
import yaml
from anthropic import Anthropic

from bugeval.judge_models import (
    CommentClassification,
    CommentJudgment,
    JudgeScore,
    NoiseStats,
    majority_vote,
)
from bugeval.models import TestCase
from bugeval.result_models import NormalizedResult
from bugeval.run_pr_eval import load_cases

_DEFAULT_JUDGE_PROMPT = """\
You are an impartial judge evaluating whether an AI code review tool identified a known bug.

Scoring rubric:
0 = missed (bug not identified)
1 = wrong-area (right file, wrong issue/line)
2 = correct-id (correct file + approximate line)
3 = correct-id-and-fix (correct ID + actionable fix)

Return ONLY a JSON object: {"score": N, "reasoning": "...", "comment_judgments": \
[{"id": N, "classification": "TP"|"FP"|"low-value", "relevance": "direct"|"adjacent"|"unrelated"}]}
"""


def load_judge_prompt(path: Path | None = None) -> str:
    """Load system prompt from config/judge_prompt.md. Falls back to default."""
    resolved = path or Path("config") / "judge_prompt.md"
    if resolved.exists():
        return resolved.read_text()
    return _DEFAULT_JUDGE_PROMPT


def _extract_judge_json(text: str) -> dict[str, Any] | None:
    """Extract JSON object from judge response."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if not brace:
        return None
    try:
        return json.loads(brace.group(0))  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None


def _build_judge_prompt(case: TestCase, result: NormalizedResult) -> str:
    """Build the user message for the judge."""
    findings_text = "\n".join(
        f"  - file: {f.file}, line: {f.line}, summary: {f.summary}" for f in case.expected_findings
    )
    comments_text = (
        "\n".join(
            f"  [{i}] file={c.file or '(none)'} line={c.line or '?'}: {c.body[:200]}"
            for i, c in enumerate(result.comments)
        )
        or "  (no comments)"
    )

    return (
        f"## Test Case: {case.id}\n"
        f"### Expected Bug\n{findings_text}\n\n"
        f"### Tool Comments ({result.tool}, {len(result.comments)} total)\n"
        f"{comments_text}\n\n"
        f"Score this tool's output 0–3 and classify each comment."
    )


def judge_case(
    case: TestCase,
    result: NormalizedResult,
    system_prompt: str,
    model: str = "claude-opus-4-6",
    n_votes: int = 3,
    dry_run: bool = False,
    client: Anthropic | None = None,
) -> JudgeScore:
    """Run n_votes independent judge calls. Return majority-vote JudgeScore."""
    if dry_run:
        return JudgeScore(
            test_case_id=case.id,
            tool=result.tool,
            score=0,
            votes=[0] * n_votes,
            reasoning="dry-run",
        )

    _client = client or Anthropic()
    user_prompt = _build_judge_prompt(case, result)
    votes: list[int] = []
    parse_failures = 0
    all_valid_judgments: list[list[CommentJudgment]] = []

    for _ in range(n_votes):
        response = _client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],  # type: ignore[arg-type]
            max_tokens=1024,
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break
        data = _extract_judge_json(text)
        if data is None:
            parse_failures += 1
            votes.append(0)
            continue
        votes.append(int(data.get("score", 0)))
        raw_judgments = data.get("comment_judgments", [])
        parsed: list[CommentJudgment] = []
        for j in raw_judgments:
            try:
                parsed.append(
                    CommentJudgment(
                        id=int(j["id"]),
                        classification=CommentClassification(j["classification"]),
                        relevance=str(j.get("relevance", "")),
                    )
                )
            except (KeyError, ValueError):
                pass
        all_valid_judgments.append(parsed)

    score = majority_vote(votes)
    last_judgments = all_valid_judgments[-1] if all_valid_judgments else []
    tp_count = sum(1 for j in last_judgments if j.classification == CommentClassification.tp)
    total = len(result.comments)
    snr = tp_count / total if total > 0 else 0.0

    reasoning = f"Votes: {votes}. Majority: {score}."
    if parse_failures:
        reasoning += f" ({parse_failures}/{n_votes} votes failed to parse.)"

    return JudgeScore(
        test_case_id=case.id,
        tool=result.tool,
        score=score,
        votes=votes,
        reasoning=reasoning,
        comment_judgments=last_judgments,
        noise=NoiseStats(total_comments=total, true_positives=tp_count, snr=snr),
    )


@click.command("judge")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to run output directory (must contain normalized *.yaml files)",
)
@click.option(
    "--cases-dir",
    default="cases/",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory containing case YAML files",
)
@click.option(
    "--config",
    "config_path",
    default="config/config.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml",
)
@click.option("--tools", "tools_filter", default=None, help="Comma-separated tool names")
@click.option("--dry-run", is_flag=True, default=False, help="Skip API calls; score everything 0")
def judge(
    run_dir: str,
    cases_dir: str,
    config_path: str,
    tools_filter: str | None,
    dry_run: bool,
) -> None:
    """Run LLM-as-judge (3× majority vote) on normalized results."""
    resolved = Path(run_dir)
    cases = {c.id: c for c in load_cases(Path(cases_dir))}
    system_prompt = load_judge_prompt()

    # Find normalized result files: *.yaml excluding checkpoint.yaml and scores/ subdir
    candidate_files = [
        p
        for p in resolved.glob("*.yaml")
        if p.name != "checkpoint.yaml" and not p.name.startswith("scores")
    ]

    # Parse candidates; skip files that aren't valid NormalizedResult YAML
    parsed: list[tuple[Path, NormalizedResult]] = []
    for p in candidate_files:
        data = yaml.safe_load(p.read_text()) or {}
        try:
            parsed.append((p, NormalizedResult(**data)))
        except Exception:
            pass

    if not parsed:
        click.echo(f"No normalized results found in {resolved}")
        return

    if tools_filter:
        names = {n.strip() for n in tools_filter.split(",")}
        parsed = [(p, r) for p, r in parsed if any(p.stem.endswith(f"-{n}") for n in names)]

    scores_dir = resolved / "scores"
    scores_dir.mkdir(exist_ok=True)

    api_client = None if dry_run else Anthropic()

    for path, result in sorted(parsed, key=lambda x: x[0]):
        case = cases.get(result.test_case_id)
        if case is None:
            click.echo(f"[skip] {path.name}: case '{result.test_case_id}' not found")
            continue

        click.echo(f"[judging] {path.stem}")
        score = judge_case(
            case, result, system_prompt=system_prompt, dry_run=dry_run, client=api_client
        )
        out = scores_dir / path.name
        out.write_text(yaml.safe_dump(score.model_dump(mode="json"), sort_keys=False))
        click.echo(f"[score={score.score}] {path.stem}")

    click.echo(f"Scores written to {scores_dir}/")
