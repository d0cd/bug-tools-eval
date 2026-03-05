"""Claude Code CLI subprocess runner for agent evaluation."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from bugeval.agent_models import AgentResult


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
    return AgentResult(
        findings=findings,
        stdout=stdout,
        stderr=stderr,
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
    """
    cmd = [
        "docker",
        "run",
        "--rm",
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
    return AgentResult(
        findings=findings,
        stdout=stdout,
        stderr=stderr,
        wall_time_seconds=wall_time,
        model=model,
    )
