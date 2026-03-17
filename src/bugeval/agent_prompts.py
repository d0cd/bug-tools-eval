"""Prompt building for in-house agent evaluation."""

from __future__ import annotations

from pathlib import Path

from bugeval.models import TestCase

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert code reviewer specializing in finding bugs in systems programming code.

You will be given a code patch (diff) to review. Your task is to identify bugs introduced in the
patch or pre-existing bugs that the patch reveals.

Analysis process:
1. Understand what the patch does.
2. For each changed function, identify potential issues.
3. Compile only genuine bugs with clear impact.

Return your findings as a JSON array:
```json
[
  {
    "file": "path/to/file.rs",
    "line": 42,
    "summary": "Brief description of the bug",
    "confidence": 0.9,
    "severity": "high",
    "category": "logic",
    "suggested_fix": "Change X to Y",
    "reasoning": "Why this is a bug and what impact it has."
  }
]
```

Severity values: "critical" | "high" | "medium" | "low"
Category values: "logic" | "memory" | "concurrency" | "api-misuse" | "type"
  | "cryptographic" | "constraint"
Confidence: 0.0-1.0; omit findings below 0.5.

If no bugs are found, return: []

Return ONLY the JSON array of findings, no other text.\
"""


def load_agent_prompt(
    path: Path | None = None,
    language: str | None = None,
    config_dir: Path | None = None,
    context_level: str | None = None,
) -> str:
    """Load system prompt, with optional context-level and language overrides.

    Resolution order:
    1. Explicit path= (if provided and exists)
    2. config_dir/agent_prompt_{context_level}.md (if context_level provided and file exists)
    3. config_dir/agent_prompt_{language}.md (if language provided and file exists)
    4. config_dir/agent_prompt.md
    5. Built-in _DEFAULT_SYSTEM_PROMPT
    """
    if path is not None:
        if path.exists():
            return path.read_text()
        return _DEFAULT_SYSTEM_PROMPT

    base = config_dir if config_dir is not None else Path("config")
    if context_level:
        ctx_file = base / f"agent_prompt_{context_level}.md"
        if ctx_file.exists():
            return ctx_file.read_text()

    if language:
        lang_file = base / f"agent_prompt_{language}.md"
        if lang_file.exists():
            return lang_file.read_text()

    generic = base / "agent_prompt.md"
    if generic.exists():
        return generic.read_text()

    return _DEFAULT_SYSTEM_PROMPT


def build_user_prompt(case: TestCase, patch_content: str, context_level: str) -> str:
    """Build the user message for the agent based on context level.

    - diff-only: just the patch
    - diff+repo: patch + instruction to explore the repo
    - diff+repo+domain: patch + repo + domain context (category, severity)
    """
    lines = [
        f"## Case: {case.id}",
        "",
        "### Patch (diff)",
        "```diff",
        patch_content,
        "```",
    ]

    if context_level in ("diff+repo", "diff+repo+domain"):
        lines += [
            "",
            "The full repository is checked out in the current working directory"
            " with the patch applied.",
            "",
            "**Step 1 — Understand the change** (use Read, Grep, Glob as needed):",
            "- Read the functions that contain the changed lines (not the entire file)",
            "- Grep for callers and usages of any function the patch modifies",
            "- Briefly describe: what this patch is trying to accomplish and what"
            "  invariants or contracts the changed code is expected to maintain",
            "",
            "**Step 2 — Find violations**:",
            "- Check whether the patch correctly maintains those invariants",
            "- Look for edge cases, incorrect assumptions, or missing checks",
        ]

    if context_level == "diff+repo+domain":
        lines += [
            "",
            "### Domain Context",
            f"- Category: {case.category}",
            f"- Severity: {case.severity}",
            f"- Language: {case.language}",
            f"- Description: {case.description}",
        ]

    if context_level in ("diff+repo", "diff+repo+domain"):
        lines += [
            "",
            "Walk through your reasoning, then end with the JSON array in a ```json"
            " code block.",
        ]
    else:
        lines += [
            "",
            "Review this patch as you would in a code review. Identify bugs, code smells,"
            " security issues, incomplete changes, or anything a good reviewer would flag."
            " Return your findings as a JSON array.",
        ]

    return "\n".join(lines)
