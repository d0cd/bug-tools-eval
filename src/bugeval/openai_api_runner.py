"""OpenAI SDK multi-turn agentic loop runner for agent evaluation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import openai as _openai
from openai import OpenAI

from bugeval.agent_api_runner import AGENT_TOOLS, _parse_api_findings, _retry_call, execute_tool
from bugeval.agent_models import AgentResult
from bugeval.pr_eval_models import default_pricing

_OPENAI_RETRYABLE: tuple[type[Exception], ...] = (
    _openai.RateLimitError,
    _openai.InternalServerError,
    _openai.APIConnectionError,
)

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in AGENT_TOOLS
]


def run_openai_api(
    repo_dir: Path,
    system_prompt: str,
    user_prompt: str,
    max_turns: int = 20,
    model: str = "o4-mini",
    context_level: str = "diff+repo",
) -> AgentResult:
    """Multi-turn agentic loop using OpenAI SDK (chat completions).

    Uses openai.OpenAI client with chat.completions.create in a manual loop.
    Tool calls are executed and fed back until finish_reason=stop or max_turns.
    In diff-only mode, no file tools are provided.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    active_tools: list[dict[str, Any]] = OPENAI_TOOLS if context_level != "diff-only" else []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    conversation: list[dict[str, Any]] = []
    total_tokens = 0
    total_cost_usd = 0.0
    turns = 0
    findings: list[dict[str, Any]] = []
    start = time.monotonic()
    pricing = default_pricing()

    while turns < max_turns:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": 16384,
            "temperature": 0,
        }
        if active_tools:
            kwargs["tools"] = active_tools

        response = _retry_call(
            lambda: client.chat.completions.create(**kwargs),  # type: ignore[arg-type]
            retryable=_OPENAI_RETRYABLE,
        )
        turns += 1

        if response.usage:
            total_tokens += response.usage.total_tokens or 0
            prompt_toks = getattr(response.usage, "prompt_tokens", 0)
            completion_toks = getattr(response.usage, "completion_tokens", 0)
            if isinstance(prompt_toks, int) and isinstance(completion_toks, int):
                total_cost_usd += pricing.estimate_cost(model, prompt_toks, completion_toks)

        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        msg_dict: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(msg_dict)
        conversation.append({"role": "assistant", "content": message.content})

        if finish_reason == "stop" or not message.tool_calls:
            if message.content:
                findings = _parse_api_findings(message.content)
            break

        if finish_reason == "tool_calls" and message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    tool_input = json.loads(tool_call.function.arguments)
                    output = execute_tool(tool_call.function.name, tool_input, repo_dir)
                except Exception as e:
                    output = f"Tool error: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )
                conversation.append(
                    {"role": "tool", "name": tool_call.function.name, "content": output}
                )
        else:
            if message.content:
                findings = _parse_api_findings(message.content)
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
