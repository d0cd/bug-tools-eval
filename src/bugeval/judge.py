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

from bugeval.agent_api_runner import _ANTHROPIC_RETRYABLE, _retry_call
from bugeval.judge_models import (
    CommentClassification,
    CommentJudgment,
    JudgeScore,
    NoiseStats,
    majority_vote,
)
from bugeval.models import TestCase
from bugeval.pr_eval_models import default_judging
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
    comment_lines = []
    for i, c in enumerate(result.comments):
        line = f"  [{i}] file={c.file or '(none)'} line={c.line or '?'}: {c.body[:200]}"
        if c.suggested_fix:
            line += f"\n      Fix: {c.suggested_fix[:200]}"
        comment_lines.append(line)
    comments_text = "\n".join(comment_lines) or "  (no comments)"

    # Tool name is intentionally omitted to prevent judge bias (self-eval pitfall mitigation).
    return (
        f"## Test Case: {case.id}\n"
        f"### Expected Bug\n{findings_text}\n\n"
        f"### Tool Comments ({len(result.comments)} total)\n"
        f"{comments_text}\n\n"
        f"Score this tool's output 0–3 and classify each comment."
    )


def judge_case(
    case: TestCase,
    result: NormalizedResult,
    system_prompt: str,
    model: str | None = None,
    n_votes: int | None = None,
    dry_run: bool = False,
    client: Anthropic | None = None,
) -> JudgeScore:
    """Run n_votes independent judge calls. Return majority-vote JudgeScore."""
    _judging = default_judging()
    resolved_model = model or _judging.model
    resolved_n_votes = n_votes or _judging.llm_calls

    if dry_run:
        return JudgeScore(
            test_case_id=case.id,
            tool=result.tool,
            score=0,
            votes=[0] * resolved_n_votes,
            reasoning="dry-run",
        )

    _client = client or Anthropic()
    user_prompt = _build_judge_prompt(case, result)
    votes: list[int] = []
    parse_failures = 0
    # Parallel lists for successful parses: (score, reasoning, comment_judgments)
    parsed_votes: list[tuple[int, str, list[CommentJudgment]]] = []

    for _ in range(resolved_n_votes):
        response = _retry_call(
            lambda: _client.messages.create(
                model=resolved_model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],  # type: ignore[arg-type]
                max_tokens=2048,
                temperature=0,
            ),
            _ANTHROPIC_RETRYABLE,
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
        vote_score = int(data.get("score", 0))
        vote_reasoning = str(data.get("reasoning", ""))
        votes.append(vote_score)
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
        parsed_votes.append((vote_score, vote_reasoning, parsed))

    score = majority_vote(votes)
    # Use the first parsed vote that matches the majority score for reasoning and
    # comment judgments (consistent: the vote whose score determined the outcome).
    winning_vote = next((pv for pv in parsed_votes if pv[0] == score), None)
    last_judgments = winning_vote[2] if winning_vote else []
    tp_count = sum(1 for j in last_judgments if j.classification == CommentClassification.tp)
    total = len(result.comments)
    snr = tp_count / total if total > 0 else 0.0

    reasoning = (
        winning_vote[1]
        if winning_vote and winning_vote[1]
        else f"Votes: {votes}. Majority: {score}."
    )
    if parse_failures:
        reasoning += f" ({parse_failures}/{resolved_n_votes} votes failed to parse.)"

    return JudgeScore(
        test_case_id=case.id,
        tool=result.tool,
        score=score,
        votes=votes,
        reasoning=reasoning,
        comment_judgments=last_judgments,
        noise=NoiseStats(total_comments=total, true_positives=tp_count, snr=snr),
    )


def judge_normalized_results(
    run_dir: Path,
    cases_dir: Path,
    dry_run: bool = False,
    model: str | None = None,
    tools_filter: str | None = None,
) -> int:
    """Judge all normalized results in run_dir. Returns count of results scored.

    Args:
        run_dir: Directory containing normalized result YAML files.
        cases_dir: Directory containing test case YAML definitions.
        dry_run: If True, skip LLM API calls and assign score 0 to every result.
            Score YAML files are still written to scores/.
        model: Override the judge model (defaults to claude-opus-4-6 inside judge_case).
        tools_filter: Comma-separated tool names to judge; all tools are judged if None.
    """
    cases = {c.id: c for c in load_cases(cases_dir)}
    system_prompt = load_judge_prompt()

    # Find normalized result files: *.yaml excluding checkpoint.yaml
    candidate_files = [p for p in run_dir.glob("*.yaml") if p.name != "checkpoint.yaml"]

    # Parse candidates; skip files that aren't valid NormalizedResult YAML
    parsed: list[tuple[Path, NormalizedResult]] = []
    for p in candidate_files:
        data = yaml.safe_load(p.read_text()) or {}
        try:
            parsed.append((p, NormalizedResult(**data)))
        except (yaml.YAMLError, ValueError) as e:
            click.echo(f"[skip] {p.name}: {e}", err=True)

    if not parsed:
        click.echo(f"No normalized results found in {run_dir}")
        return 0

    if tools_filter:
        names = {n.strip() for n in tools_filter.split(",")}
        parsed = [(p, r) for p, r in parsed if r.tool in names]

    scores_dir = run_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    judge_kwargs: dict[str, Any] = {"dry_run": dry_run}
    if model is not None:
        judge_kwargs["model"] = model
    api_client = None if dry_run else Anthropic()

    count = 0
    for path, result in sorted(parsed, key=lambda x: x[0]):
        case = cases.get(result.test_case_id)
        if case is None:
            click.echo(f"[skip] {path.name}: case '{result.test_case_id}' not found")
            continue

        click.echo(f"[judging] {path.stem}")
        score = judge_case(
            case,
            result,
            system_prompt=system_prompt,
            client=api_client,
            **judge_kwargs,
        )
        out = scores_dir / path.name
        out.write_text(yaml.safe_dump(score.model_dump(mode="json"), sort_keys=False))
        click.echo(f"[score={score.score}] {path.stem}")
        count += 1

    click.echo(f"Scores written to {scores_dir}/")
    return count


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
@click.option("--tools", "tools_filter", default=None, help="Comma-separated tool names")
@click.option("--dry-run", is_flag=True, default=False, help="Skip API calls; score everything 0")
def judge(
    run_dir: str,
    cases_dir: str,
    tools_filter: str | None,
    dry_run: bool,
) -> None:
    """Run LLM-as-judge (3× majority vote) on normalized results."""
    count = judge_normalized_results(
        Path(run_dir),
        Path(cases_dir),
        dry_run,
        None,
        tools_filter,
    )
    click.echo(f"Judged {count} result(s).")
