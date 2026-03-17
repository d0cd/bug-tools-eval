"""Claude Code CLI subprocess runner for agent evaluation."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from bugeval.agent_models import AgentResult


def _parse_cli_token_count(output: str) -> int:
    """Best-effort extraction of token count from CLI stdout/stderr.

    Recognises common patterns:
    - "Total tokens: N" / "Tokens: N"
    - "Input tokens: X" + "Output tokens: Y" (summed)
    Returns 0 when no match is found.
    """
    # "total tokens: N"
    m = re.search(r"total[_ ]tokens?[:\s]+(\d+)", output, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "input tokens: X" + "output tokens: Y"
    input_m = re.search(r"input[_ ]tokens?[:\s]+(\d+)", output, re.IGNORECASE)
    output_m = re.search(r"output[_ ]tokens?[:\s]+(\d+)", output, re.IGNORECASE)
    if input_m and output_m:
        return int(input_m.group(1)) + int(output_m.group(1))
    # "tokens: N"
    m = re.search(r"\btokens?[:\s]+(\d+)", output, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def _parse_stream_json_output(
    stdout: str,
) -> tuple[list[dict[str, Any]], str, int, float, int]:
    """Parse --output-format stream-json JSONL into conversation + result metadata.

    Returns (conversation, result_text, token_count, cost_usd, turns).
    Filters for type=assistant and type=user events to reconstruct the full
    conversation including all tool_use and tool_result blocks. Extracts
    cost/usage from the type=result event.
    """
    conversation: list[dict[str, Any]] = []
    result_text = ""
    token_count = 0
    cost_usd = 0.0
    turns = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if event_type == "assistant":
            msg = event.get("message", {})
            conversation.append({"role": "assistant", "content": msg.get("content", [])})
        elif event_type == "user":
            msg = event.get("message", {})
            conversation.append({"role": "user", "content": msg.get("content", [])})
        elif event_type == "result":
            result_text = str(event.get("result", ""))
            turns = int(event.get("num_turns", 0))
            cost_usd = float(event.get("total_cost_usd", 0.0))
            usage = event.get("usage") or {}
            token_count = (
                int(usage.get("input_tokens", 0))
                + int(usage.get("cache_creation_input_tokens", 0))
                + int(usage.get("cache_read_input_tokens", 0))
                + int(usage.get("output_tokens", 0))
            )

    return conversation, result_text, token_count, cost_usd, turns


def _parse_cli_findings(stdout: str) -> list[dict[str, Any]]:
    """Extract JSON findings array from CLI stdout output.

    Handles both raw stdout and the --output-format json envelope from claude CLI.
    """
    # Unwrap --output-format json envelope if present
    text = stdout
    try:
        outer = json.loads(stdout)
        if isinstance(outer, dict) and "result" in outer:
            text = outer["result"]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array (findings) in the text
    fence_match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1)
    else:
        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not array_match:
            return []
        inner = array_match.group(0)

    try:
        result = json.loads(inner)
        if isinstance(result, list):
            return result  # type: ignore[no-any-return]
        return []
    except json.JSONDecodeError:
        return []


def run_claude_cli(
    repo_dir: Path,
    prompt: str,
    max_turns: int = 10,
    timeout_seconds: int = 300,
    model: str = "claude-sonnet-4-6",
    allowed_tools: str | None = None,
    dangerously_skip_permissions: bool = False,
) -> AgentResult:
    """Run claude --print -p <prompt> --max-turns N in repo_dir.

    Returns AgentResult with stdout, findings, wall_time.
    On timeout: returns AgentResult with error='timeout'.

    Args:
        allowed_tools: Comma-separated tool names, or "" to disable all tools.
            None means use the claude CLI default (all tools enabled).
        dangerously_skip_permissions: Pass --dangerously-skip-permissions.
            Use only when the process is already isolated (e.g. inside Docker).
    """
    cmd = [
        "claude",
        "--print",
        "-p",
        prompt,
        "--max-turns",
        str(max_turns),
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--setting-sources",
        "project,local",
    ]
    if allowed_tools is not None:
        cmd += ["--allowedTools", allowed_tools]
    if dangerously_skip_permissions:
        cmd += ["--dangerously-skip-permissions"]
    start = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        wall_time = time.monotonic() - start
        return AgentResult(
            wall_time_seconds=wall_time,
            model=model,
            error="timeout",
        )

    wall_time = time.monotonic() - start
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        return AgentResult(
            stdout=stdout,
            stderr=stderr,
            wall_time_seconds=wall_time,
            model=model,
            error=f"claude exited with code {result.returncode}: {stderr[:500]}",
        )

    conversation, response_text, token_count, cost_usd, turns = _parse_stream_json_output(stdout)
    findings = _parse_cli_findings(response_text)
    return AgentResult(
        findings=findings,
        conversation=conversation,
        stdout=stdout,
        stderr=stderr,
        token_count=token_count,
        cost_usd=cost_usd,
        turns=turns,
        response_text=response_text,
        wall_time_seconds=wall_time,
        model=model,
    )


def run_gemini_cli(
    repo_dir: Path,
    prompt: str,
    timeout_seconds: int = 300,
    model: str = "gemini-2.5-flash",
) -> AgentResult:
    """Run gemini -p <prompt> -m <model> in repo_dir.

    Returns AgentResult with stdout, findings, wall_time.
    On timeout: returns AgentResult with error='timeout'.
    """
    cmd = ["gemini", "-p", prompt, "-m", model]
    start = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        wall_time = time.monotonic() - start
        return AgentResult(
            wall_time_seconds=wall_time,
            model=model,
            error="timeout",
        )

    wall_time = time.monotonic() - start
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        return AgentResult(
            stdout=stdout,
            stderr=stderr,
            wall_time_seconds=wall_time,
            model=model,
            error=f"gemini exited with code {result.returncode}: {stderr[:500]}",
        )

    findings = _parse_cli_findings(stdout)
    token_count = _parse_cli_token_count(stdout + "\n" + stderr)
    return AgentResult(
        findings=findings,
        stdout=stdout,
        stderr=stderr,
        token_count=token_count,
        wall_time_seconds=wall_time,
        model=model,
    )


def run_codex_cli(
    repo_dir: Path,
    prompt: str,
    timeout_seconds: int = 300,
    model: str = "o4-mini",
) -> AgentResult:
    """Run codex -q <prompt> --model <model> in repo_dir.

    Returns AgentResult with stdout, findings, wall_time.
    On timeout: returns AgentResult with error='timeout'.
    """
    cmd = ["codex", "-q", prompt, "--model", model]
    start = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        wall_time = time.monotonic() - start
        return AgentResult(
            wall_time_seconds=wall_time,
            model=model,
            error="timeout",
        )

    wall_time = time.monotonic() - start
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        return AgentResult(
            stdout=stdout,
            stderr=stderr,
            wall_time_seconds=wall_time,
            model=model,
            error=f"codex exited with code {result.returncode}: {stderr[:500]}",
        )

    findings = _parse_cli_findings(stdout)
    token_count = _parse_cli_token_count(stdout + "\n" + stderr)
    return AgentResult(
        findings=findings,
        stdout=stdout,
        stderr=stderr,
        token_count=token_count,
        wall_time_seconds=wall_time,
        model=model,
    )


def run_claude_cli_docker(
    repo_dir: Path,
    prompt: str,
    max_turns: int = 10,
    timeout_seconds: int = 300,
    model: str = "claude-sonnet-4-6",
    image: str = "bugeval-agent",
    allowed_tools: str | None = None,
    dangerously_skip_permissions: bool = False,
) -> AgentResult:
    """Run claude --print inside a Docker container with repo_dir mounted at /work.

    The container is removed after execution (--rm). The repo directory is
    mounted at /work which is also the working directory.
    No network isolation is applied beyond Docker's default (full outbound access).

    Args:
        allowed_tools: Comma-separated tool names, or "" to disable all tools.
        dangerously_skip_permissions: Pass --dangerously-skip-permissions.
            Safe to use here since Docker provides the isolation boundary.
    """
    claude_cmd = [
        "claude",
        "--print",
        "-p",
        prompt,
        "--max-turns",
        str(max_turns),
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--setting-sources",
        "project,local",
    ]
    if allowed_tools is not None:
        claude_cmd += ["--allowedTools", allowed_tools]
    if dangerously_skip_permissions:
        claude_cmd += ["--dangerously-skip-permissions"]

    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        "ANTHROPIC_API_KEY",  # pass through from host environment
        "-v",
        f"{repo_dir.resolve()}:/work",
        "-w",
        "/work",
        image,
    ] + claude_cmd
    start = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        wall_time = time.monotonic() - start
        return AgentResult(
            wall_time_seconds=wall_time,
            model=model,
            error="timeout",
        )

    wall_time = time.monotonic() - start
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        return AgentResult(
            stdout=stdout,
            stderr=stderr,
            wall_time_seconds=wall_time,
            model=model,
            error=f"claude exited with code {result.returncode}: {stderr[:500]}",
        )

    conversation, response_text, token_count, cost_usd, turns = _parse_stream_json_output(stdout)
    findings = _parse_cli_findings(response_text)
    return AgentResult(
        findings=findings,
        conversation=conversation,
        stdout=stdout,
        stderr=stderr,
        token_count=token_count,
        cost_usd=cost_usd,
        turns=turns,
        response_text=response_text,
        wall_time_seconds=wall_time,
        model=model,
    )
