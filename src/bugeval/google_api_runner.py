"""Google Gemini SDK multi-turn agentic loop runner for agent evaluation."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import google.genai as genai  # type: ignore[import-untyped]

from bugeval.agent_api_runner import AGENT_TOOLS, _parse_api_findings, _retry_call, execute_tool
from bugeval.agent_models import AgentResult
from bugeval.pr_eval_models import default_pricing

try:
    from google.api_core.exceptions import (  # type: ignore[import]
        InternalServerError as _GoogleInternalServerError,
    )
    from google.api_core.exceptions import (  # type: ignore[import]
        ResourceExhausted as _GoogleResourceExhausted,
    )
    from google.api_core.exceptions import (  # type: ignore[import]
        ServiceUnavailable as _GoogleServiceUnavailable,
    )

    _GOOGLE_RETRYABLE: tuple[type[Exception], ...] = (
        _GoogleResourceExhausted,
        _GoogleServiceUnavailable,
        _GoogleInternalServerError,
    )
except ImportError:
    _GOOGLE_RETRYABLE = ()


def run_google_api(
    repo_dir: Path,
    system_prompt: str,
    user_prompt: str,
    max_turns: int = 20,
    model: str = "gemini-2.5-flash",
    context_level: str = "diff+repo",
) -> AgentResult:
    """Multi-turn agentic loop using Google Gemini SDK (google-genai).

    Uses genai.Client with models.generate_content in a manual loop.
    Tool calls are executed and fed back until no more function calls or max_turns.
    In diff-only mode, no file tools are provided.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key)

    active_tool_defs = AGENT_TOOLS if context_level != "diff-only" else []
    google_tools: list[Any] = []
    if active_tool_defs:
        fn_decls = [
            genai.types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["input_schema"],
            )
            for t in active_tool_defs
        ]
        google_tools = [genai.types.Tool(function_declarations=fn_decls)]

    cfg = genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=google_tools,
        max_output_tokens=16384,
        temperature=0,
    )

    conversation: list[dict[str, Any]] = []
    total_tokens = 0
    total_cost_usd = 0.0
    turns = 0
    findings: list[dict[str, Any]] = []
    start = time.monotonic()
    pricing = default_pricing()

    contents: list[Any] = [
        genai.types.Content(role="user", parts=[genai.types.Part(text=user_prompt)])
    ]

    while turns < max_turns:
        response = _retry_call(
            lambda: client.models.generate_content(model=model, contents=contents, config=cfg),
            retryable=_GOOGLE_RETRYABLE,
        )
        turns += 1

        if not response.candidates:
            break

        candidate = response.candidates[0]
        content = candidate.content
        if content is None:
            break
        parts = content.parts or []

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            total_tokens += getattr(response.usage_metadata, "total_token_count", 0)
            input_toks = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_toks = getattr(response.usage_metadata, "candidates_token_count", 0)
            if isinstance(input_toks, int) and isinstance(output_toks, int):
                total_cost_usd += pricing.estimate_cost(model, input_toks, output_toks)

        fn_call_parts = [p for p in parts if getattr(p, "function_call", None)]

        if not fn_call_parts:
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    findings = _parse_api_findings(text)
                    break
            text_parts = [
                {"text": getattr(p, "text", "")} for p in parts if getattr(p, "text", None)
            ]
            conversation.append({"role": "model", "parts": text_parts})
            break

        fn_responses: list[Any] = []
        call_log: list[dict[str, Any]] = []
        for part in fn_call_parts:
            fc = part.function_call
            if fc is None:
                continue
            fc_name: str = fc.name  # type: ignore[assignment]
            fc_args: dict[str, Any] = dict(fc.args)  # type: ignore[arg-type]
            try:
                output = execute_tool(fc_name, fc_args, repo_dir)
            except Exception as e:
                output = f"Tool error: {e}"
            fn_responses.append(
                genai.types.Part.from_function_response(name=fc_name, response={"result": output})
            )
            call_log.append({"name": fc_name, "result": output})

        conversation.append({"role": "model", "tool_calls": call_log})
        contents.append(content)
        contents.append(genai.types.Content(role="user", parts=fn_responses))

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
