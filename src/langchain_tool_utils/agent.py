from __future__ import annotations

import math

import httpx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .journal import SQLiteJournal
from .middleware import ToolHealingMiddleware


def create_healing_agent(
    *,
    model: str,
    tools,
    journal: SQLiteJournal,
    api: str = "responses",
    read_timeout: float = 60,
    checkpointer=None,
    middleware=(),
    model_options=None,
    **agent_options,
):
    """Switch API per invocation with configurable.api; reuse one thread's state.

    Full function-call history is sent on both APIs. Provider-owned continuation
    IDs and server-side built-in tools are deliberately excluded.
    """
    if api not in {"responses", "chat_completions"}:
        raise ValueError("api must be responses or chat_completions")
    if not math.isfinite(read_timeout) or read_timeout <= 0:
        raise ValueError("read_timeout must be positive")
    tools = list(tools)
    if any(isinstance(item, dict) for item in tools):
        raise ValueError(
            "Use local function tools; provider-owned tools cannot be switched safely"
        )
    options = dict(model_options or {})
    forbidden = {
        "use_responses_api",
        "use_previous_response_id",
        "previous_response_id",
        "output_version",
        "timeout",
        "max_retries",
        "model",
        "model_kwargs",
        "extra_body",
        "extra_query",
        "client",
        "async_client",
        "root_client",
        "root_async_client",
        "context_management",
        "include",
        "reasoning",
        "truncation",
    }
    if forbidden.intersection(options):
        raise ValueError(
            "Protocol, timeout and retry options are managed by create_healing_agent"
        )
    models = {
        name: ChatOpenAI(
            model=model,
            use_responses_api=name == "responses",
            use_previous_response_id=False,
            output_version="responses/v1",
            timeout=httpx.Timeout(read_timeout, connect=10),
            max_retries=0,
            **options,
        )
        for name in ("responses", "chat_completions")
    }
    healing = ToolHealingMiddleware(journal, models=models, default_api=api)
    return create_agent(
        models[api],
        tools,
        middleware=[*middleware, healing],
        checkpointer=checkpointer,
        **agent_options,
    )
