"""Anthropic API multi-turn agentic loop runner for agent evaluation."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import anthropic as _anthropic
from anthropic import Anthropic

from bugeval.agent_models import AgentResult
from bugeval.pr_eval_models import default_pricing

T = TypeVar("T")

_ANTHROPIC_RETRYABLE: tuple[type[Exception], ...] = (
    _anthropic.RateLimitError,
    _anthropic.InternalServerError,
    _anthropic.APIConnectionError,
)


def _retry_call(
    fn: Callable[[], T],
    retryable: tuple[type[Exception], ...],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Call fn, retrying on retryable exceptions with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2**attempt))
    raise RuntimeError("unreachable")


AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the repository",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": "List the contents of a directory in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the directory (use '.' for root)",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search for a pattern in repository files using grep.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Grep pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Relative path to search within (use '.' for whole repo)",
                },
            },
            "required": ["pattern", "path"],
        },
    },
    {
        "name": "read_file_range",
        "description": "Read a specific range of lines from a file (1-indexed, inclusive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the repository",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read (1-indexed)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read (inclusive)",
                },
            },
            "required": ["path", "start_line", "end_line"],
        },
    },
    {
        "name": "git_blame",
        "description": "Show git blame for a range of lines in a file to see who wrote each line.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the repository",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to blame (1-indexed)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to blame (inclusive)",
                },
            },
            "required": ["path", "start_line", "end_line"],
        },
    },
]


def _safe_path(rel_path: str, repo_dir: Path) -> Path:
    """Resolve path and verify it stays within repo_dir. Raises ValueError on traversal."""
    resolved = (repo_dir / rel_path).resolve()
    try:
        resolved.relative_to(repo_dir.resolve())
    except ValueError:
        raise ValueError(f"Path traversal detected: {rel_path!r}")
    return resolved


def execute_tool(tool_name: str, tool_input: dict[str, Any], repo_dir: Path) -> str:
    """Execute a tool call within repo_dir. Rejects path traversal."""
    if tool_name == "read_file":
        path = _safe_path(tool_input["path"], repo_dir)
        try:
            content = path.read_text(errors="replace")
            return content[:10000]
        except OSError as e:
            return f"Error reading file: {e}"

    elif tool_name == "list_directory":
        if (repo_dir / tool_input["path"]).is_symlink():
            return "Error: symlinks are not allowed"
        path = _safe_path(tool_input["path"], repo_dir)
        try:
            entries = sorted(os.listdir(path))
            return "\n".join(entries)
        except OSError as e:
            return f"Error listing directory: {e}"

    elif tool_name == "search_code":
        pattern = tool_input["pattern"]
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"
        search_path = _safe_path(tool_input["path"], repo_dir)
        try:
            result = subprocess.run(
                ["grep", "-rn", pattern, str(search_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "search_code timed out"
        return result.stdout[:10000] if result.stdout else "(no matches)"

    elif tool_name == "read_file_range":
        path = _safe_path(tool_input["path"], repo_dir)
        start_line = int(tool_input["start_line"])
        end_line = int(tool_input["end_line"])
        try:
            lines = path.read_text(errors="replace").splitlines()
            selected = lines[max(0, start_line - 1) : end_line]
            return "\n".join(selected)
        except OSError as e:
            return f"Error reading file: {e}"

    elif tool_name == "git_blame":
        path = _safe_path(tool_input["path"], repo_dir)
        start_line = int(tool_input["start_line"])
        end_line = int(tool_input["end_line"])
        try:
            result = subprocess.run(
                ["git", "blame", f"-L{start_line},{end_line}", "--", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=repo_dir,
            )
            return result.stdout[:10000] if result.stdout else result.stderr[:1000] or "(no output)"
        except subprocess.TimeoutExpired:
            return "git_blame timed out"
        except OSError as e:
            return f"Error running git blame: {e}"

    else:
        raise ValueError(f"Unknown tool: {tool_name!r}")


def _parse_api_findings(text: str) -> list[dict[str, Any]]:
    """Extract JSON array of findings from agent text output.

    Handles fenced code blocks and raw JSON. Uses bracket counting
    instead of regex to correctly handle nested arrays/objects.
    """
    # Strip fenced code blocks to get inner content
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    search_text = fence_match.group(1) if fence_match else text

    # Find the outermost [...] using bracket counting
    start = search_text.find("[")
    if start == -1:
        return []

    depth = 0
    for i in range(start, len(search_text)):
        if search_text[i] == "[":
            depth += 1
        elif search_text[i] == "]":
            depth -= 1
            if depth == 0:
                candidate = search_text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return [item for item in parsed if isinstance(item, dict)]
                except json.JSONDecodeError:
                    pass
                return []

    return []


def run_agent_api(
    repo_dir: Path,
    system_prompt: str,
    user_prompt: str,
    max_turns: int = 20,
    model: str = "claude-sonnet-4-6",
    context_level: str = "diff+repo",
) -> AgentResult:
    """Multi-turn agentic loop: send → tool_use? → execute → append result → repeat.

    Uses anthropic.Anthropic() (sync client). Accumulates token usage.
    Terminates on end_turn or max_turns.
    In diff-only mode, no file tools are provided to enforce experimental control.
    """
    client = Anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    conversation: list[dict[str, Any]] = []
    total_tokens = 0
    total_cost_usd = 0.0
    turns = 0
    findings: list[dict[str, Any]] = []
    start = time.monotonic()
    active_tools: list[dict[str, Any]] = AGENT_TOOLS if context_level != "diff-only" else []
    pricing = default_pricing()

    while turns < max_turns:
        response = _retry_call(
            lambda: client.messages.create(
                model=model,
                system=system_prompt,
                messages=messages,  # type: ignore[arg-type]
                tools=active_tools,  # type: ignore[arg-type]
                max_tokens=16384,
                temperature=0,
            ),
            retryable=_ANTHROPIC_RETRYABLE,
        )
        turns += 1
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        total_cost_usd += pricing.estimate_cost(
            model, response.usage.input_tokens, response.usage.output_tokens
        )

        # Record assistant message
        assistant_content = [block.model_dump() for block in response.content]
        messages.append({"role": "assistant", "content": response.content})
        conversation.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            # Extract text from the final response
            for block in response.content:
                if block.type == "text":
                    findings = _parse_api_findings(block.text)
                    break
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        output = execute_tool(block.name, block.input, repo_dir)
                    except Exception as e:
                        output = f"Tool error: {e}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            conversation.append({"role": "user", "content": tool_results})
        elif response.stop_reason == "max_tokens":
            # Attempt to extract findings from truncated response before giving up
            for block in response.content:
                if block.type == "text":
                    findings = _parse_api_findings(block.text)
                    if findings:
                        break
            break
        else:
            # Unknown stop reason — terminate
            break

    wall_time = time.monotonic() - start
    return AgentResult(
        findings=findings,
        conversation=conversation,
        token_count=total_tokens,
        cost_usd=total_cost_usd,
        wall_time_seconds=wall_time,
        turns=turns,
        model=model,
    )
