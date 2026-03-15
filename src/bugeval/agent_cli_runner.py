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


def _parse_cli_findings(stdout: str) -> list[dict[str, Any]]:
    """Extract JSON findings array from CLI stdout output."""
    # Try to find a JSON array (findings) in the output
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stdout, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        array_match = re.search(r"\[.*\]", stdout, re.DOTALL)
        if not array_match:
            return []
        text = array_match.group(0)

    try:
        result = json.loads(text)
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
) -> AgentResult:
    """Run claude --print -p <prompt> --max-turns N in repo_dir.

    Returns AgentResult with stdout, findings, wall_time.
    On timeout: returns AgentResult with error='timeout'.
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
    ]
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
) -> AgentResult:
    """Run claude --print inside a Docker container with repo_dir mounted at /work.

    The container is removed after execution (--rm). The repo directory is
    mounted at /work which is also the working directory.
    No network isolation is applied beyond Docker's default (full outbound access).
    """
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
        "claude",
        "--print",
        "-p",
        prompt,
        "--max-turns",
        str(max_turns),
        "--model",
        model,
    ]
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
