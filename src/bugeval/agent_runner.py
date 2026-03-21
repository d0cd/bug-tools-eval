"""Custom Claude agent evaluation runner."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from bugeval.models import TestCase
from bugeval.result_models import Comment, ToolResult

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
COST_CEILING_USD = 2.0
API_TIMEOUT_SECONDS = 120.0

# File-system tools available to all API runners (Anthropic, Gemini, OpenAI).
# Web search is handled per-provider via native server tools (not in this list).
FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path",
                    "default": ".",
                },
            },
        },
    },
    {
        "name": "search_text",
        "description": "Search for a regex pattern across files in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {
                    "type": "string",
                    "description": "Directory to search in",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
]

# Anthropic server-side web search tool (executed by Anthropic, not by us).
ANTHROPIC_WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}

# Backward compat alias
TOOL_DEFS = FILE_TOOLS


def build_system_prompt(context_level: str) -> str:
    """Build system prompt based on context level."""
    base = (
        "You are an expert code reviewer performing a thorough review of a "
        "pull request.\n\n"
        "Your workspace contains:\n"
        "- `diff.patch` — the unified diff of the changes under review\n"
        "- `.pr/description.md` — the PR title and description\n"
        "- `.pr/commits.txt` — commit messages\n\n"
        "Review the changes for bugs, security vulnerabilities, correctness "
        "issues, logic errors, and edge cases. Focus on the CHANGED code in "
        "the diff — look at both what was added and what was removed.\n\n"
    )
    if context_level in ("diff+repo", "diff+repo+domain"):
        base += (
            "You also have access to the full repository. Use the provided "
            "tools to read surrounding code, check callers/callees, and "
            "understand the broader context of the changes.\n\n"
        )
    if context_level == "diff+repo+domain":
        base += (
            "Domain context is available in `.pr/domain.md`. This is a "
            "zero-knowledge cryptography / blockchain project.\n\n"
        )
    base += (
        "You have access to web search for looking up documentation, API "
        "references, known CVEs, and language/library semantics. Use it when "
        "you need to verify behavior or check for known issues.\n\n"
        "IMPORTANT: Do NOT search for the specific repository, PR, commit, "
        "or issue being reviewed. Do NOT visit github.com URLs related to "
        "this project. Web search is for reference material only.\n\n"
        "After your review, report your findings as a JSON array. Each "
        "finding:\n"
        '- "file": the file path\n'
        '- "line": the line number where the issue is\n'
        '- "description": what the problem is and why it matters\n'
        '- "suggested_fix": how to fix it (if you know)\n\n'
        "If you find no issues, return an empty array: []\n"
        "Output the JSON array as your final message."
    )
    return base


def _scrub_fix_references(text: str) -> str:
    """Remove lines that leak fix/bug context from PR body."""
    fix_pattern = re.compile(
        r"(^.*\b(fix(es|ed|ing)?|bug|patch|hotfix|resolv(es|ed|ing)?)\b.*$)"
        r"|(^.*#\d+.*$)",
        re.IGNORECASE | re.MULTILINE,
    )
    return fix_pattern.sub("", text).strip()


def build_user_prompt(
    case: TestCase, diff: str, context_level: str,
    *, inline_diff: bool = False,
) -> str:
    """Build user message directing the agent to review workspace files.

    When inline_diff is True (diff-only API runners that lack file tools),
    the diff content is appended directly so the model can still see it.
    Otherwise the agent is expected to read diff.patch from the workspace.
    """
    parts: list[str] = [
        "Please review the pull request in your workspace.",
        "Start by reading `diff.patch` and `.pr/description.md`.",
    ]
    if context_level in ("diff+repo", "diff+repo+domain"):
        parts.append(
            "Use the repository tools to explore surrounding code for context."
        )
    if context_level == "diff+repo+domain":
        parts.append("Check `.pr/domain.md` for domain-specific guidance.")
    parts.append(
        "Report all bugs, security issues, and correctness problems you find."
    )
    if inline_diff:
        parts.append(f"\n```diff\n{diff}\n```")
    return "\n".join(parts)


def sanitize_diff(diff: str) -> str:
    """Strip identifying metadata from diff for anti-contamination."""
    sha_pattern = re.compile(r"\b[0-9a-f]{7,40}\b")
    lines = diff.splitlines()
    cleaned: list[str] = []
    for line in lines:
        # Strip index lines (contain blob SHAs)
        if line.startswith("index "):
            continue
        # Strip author/date from git log-style headers
        if line.startswith("Author:") or line.startswith("Date:"):
            continue
        # Strip From: headers (patch email format)
        if line.startswith("From:"):
            continue
        # Strip lines that are purely commit SHAs (e.g. "From <sha>")
        if line.startswith("From ") and sha_pattern.search(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


_DOMAIN_HINTS = (
    "This is a zero-knowledge cryptography / blockchain project written "
    "primarily in Rust. Pay special attention to:\n"
    "- Cryptographic correctness (field arithmetic, curve ops)\n"
    "- Consensus safety (state transitions, finality)\n"
    "- Serialization round-trip fidelity\n"
    "- Resource exhaustion / DoS vectors\n"
    "- Unsafe blocks and FFI boundaries\n"
)


def materialize_workspace(
    case: TestCase, diff: str, workspace: Path, context_level: str,
) -> Path:
    """Write PR context and diff as files in the workspace.

    Creates:
      workspace/.pr/description.md  -- scrubbed PR title + body
      workspace/.pr/commits.txt     -- scrubbed commit messages (one per line)
      workspace/diff.patch          -- sanitized unified diff
      workspace/.pr/domain.md       -- domain hints (diff+repo+domain only)

    For diff-only: creates a temp directory with just these files.
    For diff+repo / diff+repo+domain: writes into the existing repo clone.
    """
    if context_level == "diff-only":
        workspace = Path(tempfile.mkdtemp(
            prefix="bugeval-ws-", dir=workspace.parent,
        ))

    pr_dir = workspace / ".pr"
    pr_dir.mkdir(parents=True, exist_ok=True)

    # description.md
    desc_parts: list[str] = []
    if case.introducing_pr_title:
        scrubbed_title = _scrub_fix_references(case.introducing_pr_title)
        if scrubbed_title:
            desc_parts.append(f"# {scrubbed_title}")
    if case.introducing_pr_body:
        scrubbed_body = _scrub_fix_references(case.introducing_pr_body)
        if scrubbed_body:
            desc_parts.append(scrubbed_body)
    (pr_dir / "description.md").write_text(
        "\n\n".join(desc_parts) if desc_parts else "(no description)",
    )

    # commits.txt
    commit_lines: list[str] = []
    if case.introducing_pr_commit_messages:
        for msg in case.introducing_pr_commit_messages:
            scrubbed = _scrub_fix_references(msg)
            if scrubbed.strip():
                commit_lines.append(scrubbed.strip())
    (pr_dir / "commits.txt").write_text(
        "\n".join(commit_lines) if commit_lines else "(no commits)",
    )

    # diff.patch
    (workspace / "diff.patch").write_text(diff)

    # domain.md (only for diff+repo+domain)
    if context_level == "diff+repo+domain":
        (pr_dir / "domain.md").write_text(_DOMAIN_HINTS)

    return workspace


def parse_agent_findings(response: str) -> list[Comment]:
    """Parse agent's response to extract findings as Comments."""
    # Try to extract JSON array from the response
    text = response.strip()
    # Find JSON array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        findings = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(findings, list):
        return []
    comments: list[Comment] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        comments.append(
            Comment(
                file=str(f.get("file", "")),
                line=int(f.get("line", 0)),
                body=str(f.get("description", "")),
                suggested_fix=str(f.get("suggested_fix", "")),
            )
        )
    return comments


_BLOCKED_DIRS = {".git", ".hg", ".svn"}


def _check_path_traversal(target: Path, resolved_repo: Path) -> str | None:
    """Return error string if target escapes repo or accesses .git/, else None."""
    try:
        rel = target.relative_to(resolved_repo)
    except ValueError:
        return "Error: path outside workspace"
    # Block access to VCS internals (prevents reading git history/logs)
    if any(part in _BLOCKED_DIRS for part in rel.parts):
        return "Error: access to version control directories is not allowed"
    return None


def _execute_tool(
    name: str, tool_input: dict[str, Any], repo_dir: Path
) -> str:
    resolved_repo = repo_dir.resolve()
    if name == "read_file":
        target = (repo_dir / tool_input["path"]).resolve()
        if err := _check_path_traversal(target, resolved_repo):
            return err
        if not target.is_file():
            return f"Error: file not found: {tool_input['path']}"
        try:
            return target.read_text(errors="replace")[:50_000]
        except OSError as e:
            return f"Error reading file: {e}"
    elif name == "list_directory":
        path_str = tool_input.get("path", ".")
        target = (repo_dir / path_str).resolve()
        if err := _check_path_traversal(target, resolved_repo):
            return err
        if not target.is_dir():
            return f"Error: directory not found: {path_str}"
        try:
            entries = sorted(p.name for p in target.iterdir())
            return "\n".join(entries[:200])
        except OSError as e:
            return f"Error listing directory: {e}"
    elif name == "search_text":
        pattern = tool_input["pattern"]
        path_str = tool_input.get("path", ".")
        target = (repo_dir / path_str).resolve()
        if err := _check_path_traversal(target, resolved_repo):
            return err
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.rs", "--include=*.toml",
                 "--include=*.py", "--include=*.ts", "--include=*.go",
                 "--include=*.java", "-m", "50", pattern, str(target)],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout[:20_000] or "No matches found."
        except (subprocess.TimeoutExpired, OSError):
            return "Error: search timed out or failed"
    return f"Error: unknown tool {name}"


def _get_file_tools_for_context(context_level: str) -> list[dict[str, Any]]:
    """Return file-system tools for the given context level."""
    if context_level in ("diff+repo", "diff+repo+domain"):
        return FILE_TOOLS
    return []


# Backward compat alias used by tests
_get_tools_for_context = _get_file_tools_for_context


def _calc_cost(usage: Any) -> float:
    # Claude Sonnet 4.6 pricing: $3/$15 per MTok (input/output)
    # Thinking tokens are billed as output tokens at the same rate.
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    # cache_creation_input_tokens and cache_read_input_tokens are ignored for now
    return round(inp * 3.0 / 1_000_000 + out * 15.0 / 1_000_000, 6)


def setup_workspace(
    case: TestCase,
    repo_url: str,
    context_level: str,
    work_dir: Path,
) -> Path | None:
    """Clone repo at base_commit if context requires it, else return None."""
    if context_level == "diff-only":
        return None
    from bugeval.git_utils import clone_at_sha

    dest = work_dir / case.id
    return clone_at_sha(repo_url, dest, case.base_commit)


def _save_transcript(
    messages: list[dict[str, Any]], transcript_dir: Path, case_id: str
) -> str:
    """Serialize messages to JSON and return the file path."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"{case_id}.json"
    # Convert non-serializable content blocks to dicts
    serializable: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.get("role", "")}
        content = msg.get("content")
        if isinstance(content, str):
            entry["content"] = content
        elif isinstance(content, list):
            entry_content: list[Any] = []
            for item in content:
                if isinstance(item, dict):
                    entry_content.append(item)
                elif hasattr(item, "type") and item.type == "thinking":
                    entry_content.append(
                        {"type": "thinking", "thinking": item.thinking}
                    )
                elif hasattr(item, "type") and item.type == "text":
                    entry_content.append(
                        {"type": "text", "text": item.text}
                    )
                elif hasattr(item, "type") and item.type == "tool_use":
                    entry_content.append({
                        "type": "tool_use",
                        "name": item.name,
                        "input": item.input,
                        "id": item.id,
                    })
                else:
                    entry_content.append(str(item))
            entry["content"] = entry_content
        else:
            entry["content"] = str(content)
        serializable.append(entry)
    path.write_text(json.dumps(serializable, indent=2, default=str))
    return str(path)


def run_anthropic_api(
    case: TestCase,
    diff: str,
    repo_dir: Path | None,
    context_level: str,
    max_turns: int = 30,
    timeout: int = 300,
    transcript_dir: Path | None = None,
    thinking_budget: int = 0,
    model: str = "",
) -> ToolResult:
    """Call Anthropic API with multi-turn tool use and collect findings.

    Uses Anthropic's server-side web_search_20250305 tool for web search
    (executed by Anthropic, not locally) plus local file tools for repo access.
    """
    system = build_system_prompt(context_level)
    sanitized = sanitize_diff(diff)
    # File tools (read_file, list_directory, search_text) + Anthropic server web search
    file_tools = _get_file_tools_for_context(context_level)
    tools: list[dict[str, Any]] = list(file_tools) + [ANTHROPIC_WEB_SEARCH_TOOL]

    # Materialize workspace files for the agent to read
    effective_repo = repo_dir
    if repo_dir is not None:
        effective_repo = materialize_workspace(
            case, sanitized, repo_dir, context_level,
        )
    elif context_level == "diff-only":
        tmp_ws = Path(tempfile.mkdtemp(prefix="bugeval-ws-"))
        effective_repo = materialize_workspace(
            case, sanitized, tmp_ws, context_level,
        )

    # diff-only API runners have no file tools, so inline the diff
    inline = context_level == "diff-only"
    user_msg = build_user_prompt(
        case, sanitized, context_level, inline_diff=inline,
    )

    client = anthropic.Anthropic(timeout=API_TIMEOUT_SECONDS)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
    total_cost = 0.0
    start = time.monotonic()

    def _make_result(
        comments: list[Comment] | None = None,
        error: str = "",
    ) -> ToolResult:
        elapsed = time.monotonic() - start
        transcript_path = ""
        if transcript_dir is not None:
            transcript_path = _save_transcript(
                messages, transcript_dir, case.id
            )
        return ToolResult(
            case_id=case.id,
            tool="agent",
            context_level=context_level,
            comments=comments or [],
            time_seconds=round(elapsed, 2),
            cost_usd=total_cost,
            error=error,
            transcript_path=transcript_path,
        )

    try:
        for _turn in range(max_turns):
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                return _make_result(error="Agent timeout exceeded")
            if total_cost > COST_CEILING_USD:
                return _make_result(
                    error=f"Cost ceiling exceeded: ${total_cost:.2f} > ${COST_CEILING_USD}"
                )
            kwargs: dict[str, Any] = {
                "model": model or MODEL,
                "max_tokens": MAX_TOKENS,
                "system": system,
                "messages": messages,
            }
            if thinking_budget > 0:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                kwargs["max_tokens"] = max(MAX_TOKENS, thinking_budget + 4096)
            if tools:
                kwargs["tools"] = tools
            response = client.messages.create(**kwargs)  # type: ignore[arg-type]
            total_cost += _calc_cost(response.usage)

            # Check if model wants to use tools
            if response.stop_reason == "tool_use":
                # Append assistant message
                messages.append(
                    {"role": "assistant", "content": response.content}
                )
                # Execute each tool call
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "thinking":
                        # Thinking blocks are kept in transcript only
                        continue
                    if block_type == "tool_use":
                        if effective_repo is None:
                            result_text = "Error: no repo available"
                        else:
                            result_text = _execute_tool(
                                block.name,  # type: ignore[union-attr]
                                block.input,  # type: ignore[union-attr]
                                effective_repo,
                            )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,  # type: ignore[union-attr]
                            "content": result_text,
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                # Final text response — append to transcript
                messages.append(
                    {"role": "assistant", "content": response.content}
                )
                final_text = ""
                for block in response.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "thinking":
                        # Thinking blocks are kept in transcript only
                        continue
                    if block_type == "text":
                        final_text += block.text  # type: ignore[union-attr]
                comments = parse_agent_findings(final_text)
                return _make_result(comments=comments)
        # Exhausted turns
        return _make_result(
            error=f"Exhausted {max_turns} turns without final response"
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        transcript_path = ""
        if transcript_dir is not None:
            transcript_path = _save_transcript(
                messages, transcript_dir, case.id
            )
        return ToolResult(
            case_id=case.id,
            tool="agent",
            context_level=context_level,
            time_seconds=round(elapsed, 2),
            cost_usd=total_cost,
            error=str(exc),
            transcript_path=transcript_path,
        )


def run_google_api(
    case: TestCase,
    diff: str,
    repo_dir: Path | None,
    context_level: str,
    max_turns: int = 30,
    timeout: int = 300,
    transcript_dir: Path | None = None,
    thinking_budget: int = 0,
    model: str = "",
) -> ToolResult:
    """Call Google Gemini API with multi-turn tool use and collect findings."""
    try:
        from google import genai  # type: ignore[import-untyped]
        from google.genai import types as genai_types  # type: ignore[import-untyped]
    except ImportError:
        return ToolResult(
            case_id=case.id,
            tool="agent-gemini",
            context_level=context_level,
            error="google-genai not installed. Run: pip install google-genai",
        )

    system = build_system_prompt(context_level)
    sanitized = sanitize_diff(diff)
    tools_for_ctx = _get_tools_for_context(context_level)

    # Materialize workspace files for the agent to read
    effective_repo = repo_dir
    if repo_dir is not None:
        effective_repo = materialize_workspace(
            case, sanitized, repo_dir, context_level,
        )
    elif context_level == "diff-only":
        tmp_ws = Path(tempfile.mkdtemp(prefix="bugeval-ws-"))
        effective_repo = materialize_workspace(
            case, sanitized, tmp_ws, context_level,
        )

    inline = context_level == "diff-only"
    user_msg = build_user_prompt(
        case, sanitized, context_level, inline_diff=inline,
    )

    # Convert TOOL_DEFS to Google FunctionDeclaration format
    google_tools: list[Any] = []
    if tools_for_ctx:
        func_decls: list[Any] = []
        for td in tools_for_ctx:
            schema = td["input_schema"].copy()
            # Google expects "properties" at top level; remove JSON Schema extras
            schema.pop("additionalProperties", None)
            func_decls.append(genai_types.FunctionDeclaration(
                name=td["name"],
                description=td["description"],
                parameters=schema,
            ))
        google_tools.append(genai_types.Tool(function_declarations=func_decls))

    # Add Google Search grounding (native server tool — Gemini executes searches).
    # Older SDK versions may not expose GoogleSearch; fall back gracefully.
    try:
        google_search_tool = genai_types.Tool(
            google_search=genai_types.GoogleSearch(),
        )
        google_tools.append(google_search_tool)
    except (AttributeError, TypeError):
        pass  # SDK too old for google_search grounding — skip

    client = genai.Client()
    contents: list[Any] = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=user_msg)],
        )
    ]
    # For transcript saving, keep a parallel list of dicts
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]

    # Gemini 2.5 Flash pricing: $0.15/$0.60 per MTok
    GOOGLE_INPUT_RATE = 0.15 / 1_000_000
    GOOGLE_OUTPUT_RATE = 0.60 / 1_000_000
    total_cost = 0.0
    start = time.monotonic()
    effective_model = model or "gemini-2.5-flash"

    def _make_result(
        comments: list[Comment] | None = None,
        error: str = "",
    ) -> ToolResult:
        elapsed = time.monotonic() - start
        transcript_path = ""
        if transcript_dir is not None:
            transcript_path = _save_transcript(
                messages, transcript_dir, case.id
            )
        return ToolResult(
            case_id=case.id,
            tool="agent-gemini",
            context_level=context_level,
            comments=comments or [],
            time_seconds=round(elapsed, 2),
            cost_usd=total_cost,
            error=error,
            transcript_path=transcript_path,
        )

    try:
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            tools=google_tools or None,
        )
        for _turn in range(max_turns):
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                return _make_result(error="Agent timeout exceeded")
            if total_cost > COST_CEILING_USD:
                return _make_result(
                    error=f"Cost ceiling exceeded: ${total_cost:.2f}"
                )

            response = client.models.generate_content(
                model=effective_model,
                contents=contents,
                config=config,
            )

            # Estimate cost from usage metadata
            usage = getattr(response, "usage_metadata", None)
            if usage:
                inp = getattr(usage, "prompt_token_count", 0) or 0
                out = getattr(usage, "candidates_token_count", 0) or 0
                total_cost += round(
                    inp * GOOGLE_INPUT_RATE + out * GOOGLE_OUTPUT_RATE, 6
                )

            # Check for function calls in response
            candidate = response.candidates[0]  # type: ignore[index]
            content = candidate.content  # type: ignore[union-attr]
            parts: list[Any] = content.parts or []  # type: ignore[union-attr]
            func_calls = [
                p for p in parts if getattr(p, "function_call", None)
            ]

            if func_calls:
                # Record assistant message in transcript
                msg_entry: dict[str, Any] = {"role": "assistant", "content": []}
                for p in parts:
                    fc = getattr(p, "function_call", None)
                    if fc is not None:
                        msg_entry["content"].append({
                            "type": "tool_use",
                            "name": fc.name,  # type: ignore[union-attr]
                            "input": dict(fc.args) if fc.args else {},  # type: ignore[union-attr]
                        })
                    elif getattr(p, "text", None):
                        msg_entry["content"].append(
                            {"type": "text", "text": p.text}
                        )
                messages.append(msg_entry)

                # Add assistant turn to contents
                contents.append(content)

                # Execute tools and build function responses
                func_response_parts: list[Any] = []
                for p in func_calls:
                    fc = p.function_call  # type: ignore[union-attr]
                    fc_name: str = fc.name  # type: ignore[union-attr]
                    fc_args: dict[str, Any] = dict(fc.args) if fc.args else {}  # type: ignore[union-attr]
                    if effective_repo is None:
                        result_text = "Error: no repo available"
                    else:
                        result_text = _execute_tool(
                            fc_name, fc_args, effective_repo,
                        )
                    func_response_parts.append(
                        genai_types.Part.from_function_response(
                            name=fc_name,
                            response={"result": result_text},
                        )
                    )
                contents.append(
                    genai_types.Content(
                        role="user", parts=func_response_parts,
                    )
                )
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "name": getattr(p.function_call, "name", "")}
                        for p in func_calls
                    ],
                })
            else:
                # Final text response
                final_text = ""
                msg_entry_final: dict[str, Any] = {
                    "role": "assistant", "content": [],
                }
                for p in parts:
                    text_val = getattr(p, "text", None)
                    if text_val:
                        final_text += str(text_val)
                        msg_entry_final["content"].append(
                            {"type": "text", "text": str(text_val)}
                        )
                messages.append(msg_entry_final)
                comments = parse_agent_findings(final_text)
                return _make_result(comments=comments)

        return _make_result(
            error=f"Exhausted {max_turns} turns without final response"
        )
    except Exception as exc:
        return ToolResult(
            case_id=case.id,
            tool="agent-gemini",
            context_level=context_level,
            time_seconds=round(time.monotonic() - start, 2),
            cost_usd=total_cost,
            error=str(exc),
            transcript_path=(
                _save_transcript(messages, transcript_dir, case.id)
                if transcript_dir else ""
            ),
        )


def run_openai_api(
    case: TestCase,
    diff: str,
    repo_dir: Path | None,
    context_level: str,
    max_turns: int = 30,
    timeout: int = 300,
    transcript_dir: Path | None = None,
    thinking_budget: int = 0,
    model: str = "",
) -> ToolResult:
    """Call OpenAI API with multi-turn tool use and collect findings."""
    try:
        import openai  # type: ignore[import-untyped]
    except ImportError:
        return ToolResult(
            case_id=case.id,
            tool="agent-openai",
            context_level=context_level,
            error="openai not installed. Run: pip install openai",
        )

    system = build_system_prompt(context_level)
    sanitized = sanitize_diff(diff)
    tools_for_ctx = _get_tools_for_context(context_level)

    # Materialize workspace files for the agent to read
    effective_repo = repo_dir
    if repo_dir is not None:
        effective_repo = materialize_workspace(
            case, sanitized, repo_dir, context_level,
        )
    elif context_level == "diff-only":
        tmp_ws = Path(tempfile.mkdtemp(prefix="bugeval-ws-"))
        effective_repo = materialize_workspace(
            case, sanitized, tmp_ws, context_level,
        )

    inline = context_level == "diff-only"
    user_msg = build_user_prompt(
        case, sanitized, context_level, inline_diff=inline,
    )

    # Convert TOOL_DEFS to OpenAI function tool format.
    # Always include web_search_preview (native server tool — OpenAI executes
    # searches and returns results as regular assistant content).
    openai_tools: list[dict[str, Any]] = [{"type": "web_search_preview"}]
    if tools_for_ctx:
        for td in tools_for_ctx:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": td["name"],
                    "description": td["description"],
                    "parameters": td["input_schema"],
                },
            })

    client = openai.OpenAI()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    # Separate transcript list (includes system for completeness)
    transcript: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    # o4-mini pricing: $1.10/$4.40 per MTok
    OPENAI_INPUT_RATE = 1.10 / 1_000_000
    OPENAI_OUTPUT_RATE = 4.40 / 1_000_000
    total_cost = 0.0
    start = time.monotonic()
    effective_model = model or "o4-mini"

    def _make_result(
        comments: list[Comment] | None = None,
        error: str = "",
    ) -> ToolResult:
        elapsed = time.monotonic() - start
        transcript_path = ""
        if transcript_dir is not None:
            transcript_path = _save_transcript(
                transcript, transcript_dir, case.id
            )
        return ToolResult(
            case_id=case.id,
            tool="agent-openai",
            context_level=context_level,
            comments=comments or [],
            time_seconds=round(elapsed, 2),
            cost_usd=total_cost,
            error=error,
            transcript_path=transcript_path,
        )

    try:
        for _turn in range(max_turns):
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                return _make_result(error="Agent timeout exceeded")
            if total_cost > COST_CEILING_USD:
                return _make_result(
                    error=f"Cost ceiling exceeded: ${total_cost:.2f}"
                )

            kwargs: dict[str, Any] = {
                "model": effective_model,
                "messages": messages,
                "tools": openai_tools,
            }

            response = client.chat.completions.create(**kwargs)

            # Estimate cost from usage
            usage = getattr(response, "usage", None)
            if usage:
                inp = getattr(usage, "prompt_tokens", 0) or 0
                out = getattr(usage, "completion_tokens", 0) or 0
                total_cost += round(
                    inp * OPENAI_INPUT_RATE + out * OPENAI_OUTPUT_RATE, 6
                )

            choice = response.choices[0]  # type: ignore[index]
            message = choice.message
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls" and message.tool_calls:
                # Append assistant message with tool calls
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
                messages.append(assistant_msg)
                transcript.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": tc.function.name,
                            "input": tc.function.arguments,
                            "id": tc.id,
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Execute each tool call and feed results back
                for tc in message.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}
                    if effective_repo is None:
                        result_text = "Error: no repo available"
                    else:
                        result_text = _execute_tool(
                            fn_name, fn_args, effective_repo,
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })
                    transcript.append({
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_call_id": tc.id}
                        ],
                    })
            else:
                # Final text response
                final_text = message.content or ""
                transcript.append({
                    "role": "assistant",
                    "content": final_text,
                })
                comments = parse_agent_findings(final_text)
                return _make_result(comments=comments)

        return _make_result(
            error=f"Exhausted {max_turns} turns without final response"
        )
    except Exception as exc:
        return ToolResult(
            case_id=case.id,
            tool="agent-openai",
            context_level=context_level,
            time_seconds=round(time.monotonic() - start, 2),
            cost_usd=total_cost,
            error=str(exc),
            transcript_path=(
                _save_transcript(transcript, transcript_dir, case.id)
                if transcript_dir else ""
            ),
        )


def _estimate_claude_cli_cost(cost_info: dict[str, Any]) -> float:
    """Estimate cost from Claude CLI JSON output."""
    inp = cost_info.get("input_tokens", 0) or 0
    out = cost_info.get("output_tokens", 0) or 0
    cache_read = cost_info.get("cache_read_input_tokens", 0) or 0
    cache_create = cost_info.get("cache_creation_input_tokens", 0) or 0
    # Sonnet pricing: $3 input, $15 output, $0.30 cache read, $3.75 cache write per MTok
    return round(
        inp * 3.0 / 1e6 + out * 15.0 / 1e6
        + cache_read * 0.30 / 1e6 + cache_create * 3.75 / 1e6,
        6,
    )


def _save_cli_transcript(
    transcript_dir: Path, case_id: str, cli_tool: str,
    prompt: str, output: Any,
) -> str:
    """Save CLI interaction as transcript JSON."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"{case_id}-{cli_tool}.json"
    data = {
        "tool": cli_tool,
        "prompt": prompt[:5000],  # Truncate for sanity
        "output": output if isinstance(output, dict) else str(output)[:10000],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path)


@dataclass(frozen=True)
class _CliConfig:
    """Configuration for a CLI-based agent tool."""

    binary: str
    tool_label: str
    prepend_system: bool
    build_cmd: Callable[[str, str, str], list[str]]
    parse_output: Callable[[str], tuple[str, float]]


def _run_cli_tool(
    config: _CliConfig,
    case: TestCase, diff: str, repo_dir: Path | None,
    context_level: str, timeout: int,
    system_prompt: str, transcript_dir: Path | None = None,
    model: str = "",
) -> ToolResult:
    """Generic CLI runner shared by all CLI-based agent tools."""
    sanitized = sanitize_diff(diff)

    # Materialize workspace files -- CLI tools read from cwd
    effective_repo = repo_dir
    if repo_dir is not None:
        effective_repo = materialize_workspace(
            case, sanitized, repo_dir, context_level,
        )
    elif context_level == "diff-only":
        tmp_ws = Path(tempfile.mkdtemp(prefix="bugeval-ws-"))
        effective_repo = materialize_workspace(
            case, sanitized, tmp_ws, context_level,
        )

    # CLI runners can read files from cwd, so no inline diff needed
    user_prompt = build_user_prompt(
        case, sanitized, context_level, inline_diff=False,
    )

    cmd = config.build_cmd(system_prompt, context_level, model)
    full_prompt = (
        f"{system_prompt}\n\n{user_prompt}"
        if config.prepend_system
        else user_prompt
    )

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            input=full_prompt,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(effective_repo) if effective_repo else None,
        )
        elapsed = time.monotonic() - start

        response_text, cost_usd = config.parse_output(result.stdout)

        transcript_path = ""
        if transcript_dir:
            transcript_path = _save_cli_transcript(
                transcript_dir, case.id, config.binary, full_prompt,
                {"stdout": result.stdout[:5000], "stderr": result.stderr[:2000]}
                if config.binary != "claude"
                else _try_parse_json_or_raw(result.stdout),
            )

        comments = parse_agent_findings(response_text)
        return ToolResult(
            case_id=case.id, tool=config.tool_label,
            context_level=context_level,
            comments=comments,
            time_seconds=round(elapsed, 2),
            cost_usd=cost_usd,
            transcript_path=transcript_path,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return ToolResult(
            case_id=case.id, tool=config.tool_label,
            context_level=context_level,
            time_seconds=round(elapsed, 2),
            error=f"CLI timed out after {timeout}s",
        )
    except FileNotFoundError:
        elapsed = time.monotonic() - start
        return ToolResult(
            case_id=case.id, tool=config.tool_label,
            context_level=context_level,
            time_seconds=round(elapsed, 2),
            error=f"{config.binary} CLI not found on PATH",
        )


def _try_parse_json_or_raw(stdout: str) -> Any:
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return stdout


# ---------------------------------------------------------------------------
# Per-tool command builders and output parsers
# ---------------------------------------------------------------------------

def _claude_build_cmd(
    system_prompt: str, context_level: str, model: str = "",
) -> list[str]:
    cmd = ["claude", "-p", "--output-format", "json"]
    cmd.extend(["--system-prompt", system_prompt])
    if model:
        cmd.extend(["--model", model])
    if context_level == "diff-only":
        cmd.extend(["--disallowedTools", "Read,Edit,Bash,Glob,Grep,Write,WebSearch"])
    else:
        cmd.extend(["--allowedTools", "Read,Glob,Grep,WebSearch"])
    cmd.extend(["--max-turns", "10"])
    return cmd


def _claude_parse_output(stdout: str) -> tuple[str, float]:
    try:
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            return stdout, 0.0
        response_text = parsed.get("result", stdout)
        cost_usd = _estimate_claude_cli_cost(parsed.get("cost", {}))
        return response_text, cost_usd
    except (json.JSONDecodeError, ValueError):
        return stdout, 0.0


def _gemini_build_cmd(
    _system_prompt: str, context_level: str, model: str = "",
) -> list[str]:
    cmd = ["gemini", "-p", "--output-format", "json"]
    if model:
        cmd.extend(["-m", model])
    if context_level != "diff-only":
        cmd.extend(["--yolo"])
    return cmd


def _plain_parse_output(stdout: str) -> tuple[str, float]:
    return stdout, 0.0


def _codex_build_cmd(
    _system_prompt: str, context_level: str, model: str = "",
) -> list[str]:
    sandbox = "read-only" if context_level == "diff-only" else "workspace-write"
    cmd = [
        "codex", "exec", "--json",
        "--sandbox", sandbox,
        "--ask-for-approval", "never",
    ]
    if model:
        cmd.extend(["-m", model])
    return cmd


_CLAUDE_CLI = _CliConfig(
    binary="claude", tool_label="agent-cli-claude",
    prepend_system=False,
    build_cmd=_claude_build_cmd, parse_output=_claude_parse_output,
)
_GEMINI_CLI = _CliConfig(
    binary="gemini", tool_label="agent-cli-gemini",
    prepend_system=True,
    build_cmd=_gemini_build_cmd, parse_output=_plain_parse_output,
)
_CODEX_CLI = _CliConfig(
    binary="codex", tool_label="agent-cli-codex",
    prepend_system=True,
    build_cmd=_codex_build_cmd, parse_output=_plain_parse_output,
)

_CLI_CONFIGS: dict[str, _CliConfig] = {
    "claude": _CLAUDE_CLI,
    "gemini": _GEMINI_CLI,
    "codex": _CODEX_CLI,
}


# ---------------------------------------------------------------------------
# Thin wrappers (preserve existing call signatures for backward compat)
# ---------------------------------------------------------------------------

def _run_claude_cli(
    case: TestCase, diff: str, repo_dir: Path | None,
    context_level: str, timeout: int,
    system_prompt: str, transcript_dir: Path | None = None,
    model: str = "",
) -> ToolResult:
    """Run Claude Code CLI with full flag support."""
    return _run_cli_tool(
        _CLAUDE_CLI, case, diff, repo_dir,
        context_level, timeout, system_prompt, transcript_dir,
        model=model,
    )


def _run_gemini_cli(
    case: TestCase, diff: str, repo_dir: Path | None,
    context_level: str, timeout: int,
    system_prompt: str, transcript_dir: Path | None = None,
    model: str = "",
) -> ToolResult:
    """Run Gemini CLI."""
    return _run_cli_tool(
        _GEMINI_CLI, case, diff, repo_dir,
        context_level, timeout, system_prompt, transcript_dir,
        model=model,
    )


def _run_codex_cli(
    case: TestCase, diff: str, repo_dir: Path | None,
    context_level: str, timeout: int,
    system_prompt: str, transcript_dir: Path | None = None,
    model: str = "",
) -> ToolResult:
    """Run OpenAI Codex CLI."""
    return _run_cli_tool(
        _CODEX_CLI, case, diff, repo_dir,
        context_level, timeout, system_prompt, transcript_dir,
        model=model,
    )


def run_agent_cli(
    case: TestCase,
    diff: str,
    repo_dir: Path | None,
    context_level: str,
    cli_tool: str = "claude",
    timeout: int = 300,
    transcript_dir: Path | None = None,
    model: str = "",
) -> ToolResult:
    """Dispatch to the appropriate CLI runner."""
    system_prompt = build_system_prompt(context_level)
    if cli_tool == "claude":
        return _run_claude_cli(
            case, diff, repo_dir, context_level, timeout,
            system_prompt, transcript_dir, model=model,
        )
    elif cli_tool == "gemini":
        return _run_gemini_cli(
            case, diff, repo_dir, context_level, timeout,
            system_prompt, transcript_dir, model=model,
        )
    elif cli_tool == "codex":
        return _run_codex_cli(
            case, diff, repo_dir, context_level, timeout,
            system_prompt, transcript_dir, model=model,
        )
    return ToolResult(
        case_id=case.id, tool=f"agent-cli-{cli_tool}",
        context_level=context_level,
        error=f"Unknown CLI tool: {cli_tool}",
    )


async def _run_agent_sdk_async(
    case: TestCase,
    diff: str,
    repo_dir: Path | None,
    context_level: str,
    timeout: int = 300,
    transcript_dir: Path | None = None,
    model: str = "",
) -> ToolResult:
    """Run Claude Code via Agent SDK.

    Uses the proven v1 approach: stream messages, capture AssistantMessage
    content for transcript, read final result from ResultMessage.result.
    """
    try:
        from claude_agent_sdk import (  # type: ignore[import-untyped]
            AssistantMessage,
            ClaudeAgentOptions,
            CLIConnectionError,
            CLINotFoundError,
            ResultMessage,
            query,
        )
        from claude_agent_sdk.types import (  # type: ignore[import-untyped]
            TextBlock as _SdkTextBlock,
        )
        from claude_agent_sdk.types import (
            ThinkingBlock as _SdkThinkingBlock,
        )
        from claude_agent_sdk.types import (
            ToolUseBlock as _SdkToolUseBlock,
        )
    except ImportError:
        return ToolResult(
            case_id=case.id,
            tool="agent-sdk",
            context_level=context_level,
            error="claude-agent-sdk not installed. Run: uv add claude-agent-sdk",
        )

    system_prompt = build_system_prompt(context_level)
    sanitized = sanitize_diff(diff)

    # Materialize workspace files — SDK agent reads from cwd
    effective_repo = repo_dir
    if repo_dir is not None:
        effective_repo = materialize_workspace(
            case, sanitized, repo_dir, context_level,
        )
    elif context_level == "diff-only":
        tmp_ws = Path(tempfile.mkdtemp(prefix="bugeval-ws-"))
        effective_repo = materialize_workspace(
            case, sanitized, tmp_ws, context_level,
        )

    # SDK agent reads files from cwd, no inline diff needed
    user_prompt = build_user_prompt(
        case, sanitized, context_level, inline_diff=False,
    )

    # Same tools for all context levels — the workspace content differs, not the tools.
    # diff-only: workspace has only diff.patch + .pr/ files
    # diff+repo: workspace has full repo clone + diff.patch + .pr/ files
    allowed_tools: list[str] = ["Read", "Glob", "Grep", "WebSearch"]

    # Explicitly block dangerous tools
    disallowed = ["Edit", "Write", "Bash", "NotebookEdit"]

    effective_model = model or MODEL
    sdk_kwargs: dict[str, Any] = {
        "system_prompt": system_prompt,
        "model": effective_model,
        "allowed_tools": allowed_tools,
        "disallowed_tools": disallowed,
        "cwd": str(effective_repo) if effective_repo else None,
        "max_turns": 30,
        "permission_mode": "acceptEdits",
        "env": {"CLAUDECODE": ""},  # Allow spawning from nested session
    }
    # max_budget_usd is None by default (no budget cap)
    options = ClaudeAgentOptions(**sdk_kwargs)

    start = time.monotonic()
    total_cost = 0.0
    session_id = ""
    result_text = ""
    transcript_messages: list[dict[str, Any]] = []

    try:
        async for message in query(prompt=user_prompt, options=options):
            if time.monotonic() - start > timeout:
                return ToolResult(
                    case_id=case.id,
                    tool="agent-sdk",
                    context_level=context_level,
                    time_seconds=round(time.monotonic() - start, 2),
                    cost_usd=total_cost,
                    error=f"Agent SDK timeout after {timeout}s",
                )

            if isinstance(message, AssistantMessage):
                # Capture content blocks for transcript
                msg_entry: dict[str, Any] = {"role": "assistant", "content": []}
                for block in message.content:
                    if isinstance(block, _SdkTextBlock):
                        msg_entry["content"].append(
                            {"type": "text", "text": block.text}
                        )
                    elif isinstance(block, _SdkThinkingBlock):
                        msg_entry["content"].append(
                            {"type": "thinking", "thinking": block.thinking}
                        )
                    elif isinstance(block, _SdkToolUseBlock):
                        msg_entry["content"].append({
                            "type": "tool_use",
                            "name": block.name,
                            "input": block.input,
                        })
                transcript_messages.append(msg_entry)

            elif isinstance(message, ResultMessage):
                # Final result — this is where the agent's output lives
                result_text = message.result or ""
                total_cost = message.total_cost_usd or 0.0
                session_id = message.session_id or ""

    except CLINotFoundError as exc:
        return ToolResult(
            case_id=case.id,
            tool="agent-sdk",
            context_level=context_level,
            time_seconds=round(time.monotonic() - start, 2),
            error=f"claude CLI not found: {exc}",
        )
    except CLIConnectionError as exc:
        return ToolResult(
            case_id=case.id,
            tool="agent-sdk",
            context_level=context_level,
            time_seconds=round(time.monotonic() - start, 2),
            error=f"CLI connection error: {exc}",
        )
    except Exception as exc:
        return ToolResult(
            case_id=case.id,
            tool="agent-sdk",
            context_level=context_level,
            time_seconds=round(time.monotonic() - start, 2),
            cost_usd=total_cost,
            error=str(exc),
        )

    elapsed = time.monotonic() - start
    comments = parse_agent_findings(result_text)

    # Save transcript
    transcript_path = ""
    if transcript_dir:
        transcript_dir.mkdir(parents=True, exist_ok=True)
        t_path = transcript_dir / f"{case.id}-sdk.json"
        data = {
            "session_id": session_id,
            "model": effective_model,
            "messages": transcript_messages,
            "result_text": result_text,
            "cost_usd": total_cost,
            "elapsed_seconds": round(elapsed, 2),
        }
        t_path.write_text(json.dumps(data, indent=2, default=str))
        transcript_path = str(t_path)

    return ToolResult(
        case_id=case.id,
        tool="agent-sdk",
        context_level=context_level,
        comments=comments,
        time_seconds=round(elapsed, 2),
        cost_usd=total_cost,
        transcript_path=transcript_path,
    )


def is_docker_available() -> bool:
    """Check if Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_docker_cli_cmd(cli_tool: str, model: str) -> list[str]:
    """Build the inner CLI command to run inside the Docker container."""
    if cli_tool == "claude":
        cmd = [
            "claude", "-p", "--output-format", "json",
            "--max-turns", "30",
            "--dangerously-skip-permissions",
            "--allowedTools",
            "Read,Glob,Grep,Bash,WebSearch",
        ]
        if model:
            cmd.extend(["--model", model])
        return cmd
    elif cli_tool == "gemini":
        cmd = ["gemini", "-p", "--output-format", "json", "--yolo"]
        if model:
            cmd.extend(["-m", model])
        return cmd
    elif cli_tool == "codex":
        cmd = [
            "codex", "exec", "--json",
            "--sandbox", "workspace-write",
        ]
        if model:
            cmd.extend(["-m", model])
        return cmd
    return [cli_tool]


def run_docker(
    case: TestCase,
    diff: str,
    repo_dir: Path | None,
    context_level: str,
    cli_tool: str = "claude",
    timeout: int = 600,
    transcript_dir: Path | None = None,
    model: str = "",
    image: str = "bugeval-agent",
) -> ToolResult:
    """Run a CLI tool inside a Docker container with the workspace mounted.

    The container:
    - Mounts the workspace at /work (read-write)
    - Passes ANTHROPIC_API_KEY from host environment
    - Runs as non-root user 'agent'
    - Is removed after execution (--rm)
    - Has full outbound network (needed for API + WebSearch)
    - Allows Bash tool (safe inside container)
    """
    sanitized = sanitize_diff(diff)
    system_prompt = build_system_prompt(context_level)

    # Materialize workspace
    effective_repo = repo_dir
    if repo_dir is not None:
        effective_repo = materialize_workspace(
            case, sanitized, repo_dir, context_level,
        )
    elif context_level == "diff-only":
        tmp_ws = Path(tempfile.mkdtemp(prefix="bugeval-ws-"))
        effective_repo = materialize_workspace(
            case, sanitized, tmp_ws, context_level,
        )

    user_prompt = build_user_prompt(
        case, sanitized, context_level, inline_diff=False,
    )
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    # Build inner CLI command
    inner_cmd = _build_docker_cli_cmd(cli_tool, model)

    # Build docker command
    workspace_path = str(effective_repo) if effective_repo else "/dev/null"
    docker_cmd: list[str] = [
        "docker", "run", "--rm",
        "-e", "ANTHROPIC_API_KEY",
        "-v", f"{workspace_path}:/work",
        "-w", "/work",
        image,
    ] + inner_cmd

    tool_label = f"agent-docker-{cli_tool}"
    start = time.monotonic()

    try:
        result = subprocess.run(
            docker_cmd,
            input=full_prompt,
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.monotonic() - start

        # Parse output based on CLI tool
        if cli_tool == "claude":
            response_text, cost_usd = _claude_parse_output(result.stdout)
        else:
            response_text, cost_usd = _plain_parse_output(result.stdout)

        transcript_path = ""
        if transcript_dir:
            transcript_path = _save_cli_transcript(
                transcript_dir, case.id, f"docker-{cli_tool}",
                full_prompt,
                _try_parse_json_or_raw(result.stdout)
                if cli_tool == "claude"
                else {"stdout": result.stdout[:5000], "stderr": result.stderr[:2000]},
            )

        comments = parse_agent_findings(response_text)
        return ToolResult(
            case_id=case.id, tool=tool_label,
            context_level=context_level,
            comments=comments,
            time_seconds=round(elapsed, 2),
            cost_usd=cost_usd,
            transcript_path=transcript_path,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return ToolResult(
            case_id=case.id, tool=tool_label,
            context_level=context_level,
            time_seconds=round(elapsed, 2),
            error=f"Docker container timed out after {timeout}s",
        )
    except FileNotFoundError:
        elapsed = time.monotonic() - start
        return ToolResult(
            case_id=case.id, tool=tool_label,
            context_level=context_level,
            time_seconds=round(elapsed, 2),
            error="docker CLI not found on PATH",
        )


def run_agent_sdk(
    case: TestCase,
    diff: str,
    repo_dir: Path | None,
    context_level: str,
    timeout: int = 300,
    transcript_dir: Path | None = None,
    model: str = "",
) -> ToolResult:
    """Use Anthropic Agent SDK for Claude Code session (sync wrapper)."""
    import asyncio

    return asyncio.run(
        _run_agent_sdk_async(
            case, diff, repo_dir, context_level, timeout, transcript_dir,
            model=model,
        )
    )
