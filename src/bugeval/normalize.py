# src/bugeval/normalize.py
"""Normalize raw tool outputs to a common schema and save as YAML."""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
import yaml

from bugeval.pr_eval_models import EvalConfig, ToolType, load_eval_config
from bugeval.result_models import Comment, CommentType, NormalizedResult, ResultMetadata


def _parse_line(val: object) -> int:
    """Parse a line number from int, str, or range string like '58-62' (takes start)."""
    if not val:
        return 0
    s = str(val).strip()
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m else 0


def normalize_pr_result(case_id: str, tool: str, raw_dir: Path) -> NormalizedResult:
    """Normalize PR-mode comments.json → NormalizedResult."""
    comments_path = raw_dir / "comments.json"
    if not comments_path.exists():
        raise FileNotFoundError(f"comments.json missing in {raw_dir}")
    raw: list[dict] = json.loads(comments_path.read_text())
    comments = []
    for c in raw:
        source = c.get("source", "")
        if source == "inline_comment":
            comments.append(
                Comment(
                    file=c.get("path", ""),
                    line=_parse_line(c.get("line") or c.get("original_line")),
                    body=c.get("body", ""),
                    type=CommentType.inline,
                )
            )
        else:
            body = c.get("body", "")
            if body:
                comments.append(Comment(body=body, type=CommentType.pr_level))
    return NormalizedResult(test_case_id=case_id, tool=tool, comments=comments)


def normalize_api_result(
    case_id: str, tool: str, context_level: str, raw_dir: Path
) -> NormalizedResult:
    """Normalize API-mode findings.json → NormalizedResult."""
    findings_path = raw_dir / "findings.json"
    if not findings_path.exists():
        raise FileNotFoundError(f"findings.json missing in {raw_dir}")
    raw: list[dict] = json.loads(findings_path.read_text())
    comments = [
        Comment(
            file=item.get("path") or item.get("file", ""),
            line=_parse_line(item.get("line")),
            body=item.get("body") or item.get("summary", ""),
        )
        for item in raw
    ]
    meta_path = raw_dir / "metadata.json"
    metadata = ResultMetadata()
    if meta_path.exists():
        meta_data = json.loads(meta_path.read_text())
        metadata = ResultMetadata(
            time_seconds=meta_data.get("time_seconds", 0.0),
            cost_usd=meta_data.get("cost_usd", 0.0),
        )
    return NormalizedResult(
        test_case_id=case_id,
        tool=tool,
        context_level=context_level,
        comments=comments,
        metadata=metadata,
    )


def normalize_agent_result(case_id: str, tool: str, raw_dir: Path) -> NormalizedResult:
    """Normalize agent-mode findings.json + metadata.json → NormalizedResult."""
    findings_path = raw_dir / "findings.json"
    raw: list[dict] = json.loads(findings_path.read_text()) if findings_path.exists() else []
    comments = [
        Comment(
            file=item.get("file", ""),
            line=_parse_line(item.get("line")),
            body=(
                item.get("summary")
                or item.get("title")
                or item.get("description")
                or item.get("body", "")
            ),
            confidence=item.get("confidence"),
            severity=item.get("severity"),
            category=item.get("category"),
            suggested_fix=item.get("suggested_fix"),
            reasoning=item.get("reasoning"),
        )
        for item in raw
    ]

    meta = {}
    metadata_path = raw_dir / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())

    return NormalizedResult(
        test_case_id=case_id,
        tool=tool,
        context_level=meta.get("context_level", ""),
        comments=comments,
        metadata=ResultMetadata(
            tokens=int(meta.get("token_count", 0)),
            cost_usd=float(meta.get("cost_usd", 0.0)),
            time_seconds=float(meta.get("wall_time_seconds", 0.0)),
        ),
    )


def discover_raw_dirs(run_dir: Path) -> list[Path]:
    """Return all subdirectories of run_dir/raw/."""
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        return []
    return [p for p in raw_dir.iterdir() if p.is_dir()]


# Known tool name suffixes listed longest-first so the most specific match wins.
_KNOWN_TOOLS = [
    "google-api-flash-lite",
    "google-api-flash",
    "openai-api-mini",
    "openai-api-o4",
    "gemini-cli-flash-lite",
    "gemini-cli-flash",
    "claude-cli-sonnet",
    "claude-cli-haiku",
    "claude-cli-opus",
    "codex-cli-mini",
    "codex-cli-o4",
    "claude-code-cli",
    "anthropic-api",
    "graphite-diamond",
    "augment-code",
    "coderabbit",
    "deepsource",
    "greptile",
    "bugbot",
]


def _parse_raw_dir_name(name: str) -> tuple[str, str]:
    """Parse '{case-id}-{tool}' from a raw dir name. Returns (case_id, tool).

    Strategy 1: match against known tool names (handles hyphenated tools correctly).
    Strategy 2: split at digit-to-alpha boundary (e.g. 'repo-042-toolname').
    Strategy 3: last-hyphen fallback.
    """
    # Strategy 1: known tool suffix match
    for tool in _KNOWN_TOOLS:
        suffix = f"-{tool}"
        if name.endswith(suffix):
            case_id = name[: -len(suffix)]
            if case_id:
                return case_id, tool

    # Strategy 2: split at digit-to-alpha boundary
    m = re.match(r"^(.+-\d+)-([a-zA-Z].*)$", name)
    if m:
        return m.group(1), m.group(2)

    # Strategy 3: last-hyphen fallback
    parts = name.rsplit("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]

    return "unknown", name


@click.command("normalize")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to run output directory (e.g. results/run-2026-03-04)",
)
@click.option(
    "--config",
    "config_path",
    default="config/config.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml",
)
@click.option(
    "--context-level",
    default="diff-only",
    show_default=True,
    type=click.Choice(["diff-only", "diff+repo", "diff+repo+domain"]),
    help="Context level used for API tools (not needed for PR or agent tools)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be normalized without writing output.",
)
def normalize(run_dir: str, config_path: str, context_level: str, dry_run: bool) -> None:
    """Normalize raw tool outputs into a common schema YAML per (case × tool)."""
    resolved = Path(run_dir)
    config: EvalConfig = load_eval_config(Path(config_path))

    tool_types = {t.name: t.type for t in config.tools}
    raw_dirs = discover_raw_dirs(resolved)

    if not raw_dirs:
        click.echo(f"No raw output directories found in {resolved / 'raw'}")
        return

    success = 0
    for raw_dir in raw_dirs:
        case_id, tool_name = _parse_raw_dir_name(raw_dir.name)
        tool_type = tool_types.get(tool_name)

        try:
            if tool_type == ToolType.pr:
                result = normalize_pr_result(case_id, tool_name, raw_dir)
            elif tool_type == ToolType.api:
                result = normalize_api_result(case_id, tool_name, context_level, raw_dir)
            elif tool_type == ToolType.agent:
                result = normalize_agent_result(case_id, tool_name, raw_dir)
            else:
                click.echo(f"[skip] {raw_dir.name}: unknown tool type for '{tool_name}'")
                continue

            out_path = resolved / f"{case_id}-{tool_name}.yaml"
            if dry_run:
                click.echo(f"[dry-run] would write {out_path.name}")
            else:
                out_path.write_text(yaml.safe_dump(result.model_dump(mode="json"), sort_keys=False))
                click.echo(f"[ok] {out_path.name}")
            success += 1
        except Exception as exc:
            click.echo(f"[error] {raw_dir.name}: {exc}", err=True)

    if dry_run:
        click.echo(f"Would normalize {success}/{len(raw_dirs)} results → {resolved}/")
    else:
        click.echo(f"Normalized {success}/{len(raw_dirs)} results → {resolved}/")
