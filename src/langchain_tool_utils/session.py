"""Session policy only: select history and consume the current request."""

from __future__ import annotations

from typing import NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import RemoveMessage, convert_to_messages
from langgraph.config import get_config
from langgraph.types import Overwrite

from .history import UnsafeHistoryError


class ApiSessionState(AgentState):
    session_api: NotRequired[str]
    session_run_active: NotRequired[bool]
    request_messages: NotRequired[list]


class ApiSessionMiddleware(AgentMiddleware):
    """API switches discard history; cleanup is a separate middleware's job.

    Supply new turns in request_messages. Internal session fields are managed by
    this middleware. Serialize runs and external state updates per thread.
    """

    state_schema = ApiSessionState

    def __init__(self, default_api="responses"):
        if default_api not in {"responses", "chat_completions"}:
            raise ValueError("Unsupported API")
        self.default_api = default_api

    def _api(self):
        api = get_config().get("configurable", {}).get("api", self.default_api)
        if api not in {"responses", "chat_completions"}:
            raise ValueError(f"Unsupported API: {api!r}")
        return api

    def before_agent(self, state, runtime):
        api = self._api()
        previous = state.get("session_api")
        history = state.get("messages", [])
        changed = previous != api and (previous is not None or bool(history))
        if changed and state.get("session_run_active"):
            raise UnsafeHistoryError(
                "Finish the interrupted run using its original API before switching"
            )
        incoming = convert_to_messages(state.get("request_messages", []))
        if any(isinstance(message, RemoveMessage) for message in incoming):
            raise ValueError(
                "request_messages accepts messages, not removal directives"
            )
        if changed and not incoming:
            raise ValueError("API switching requires new request_messages")
        selected = incoming if changed else [*history, *incoming]
        return {
            "messages": Overwrite(
                [message.model_copy(deep=True) for message in selected]
            ),
            "request_messages": [],
            "session_api": api,
            "session_run_active": True,
        }

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    def _guard_resume(self, state):
        if state.get("session_api") != self._api():
            raise UnsafeHistoryError(
                "Resume the interrupted run using its original API"
            )

    def before_model(self, state, runtime):
        self._guard_resume(state)

    async def abefore_model(self, state, runtime):
        self.before_model(state, runtime)

    def wrap_model_call(self, request, handler):
        self._guard_resume(request.state)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._guard_resume(request.state)
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        self._guard_resume(request.state)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        self._guard_resume(request.state)
        return await handler(request)

    def after_agent(self, state, runtime):
        return {"session_run_active": False}

    async def aafter_agent(self, state, runtime):
        return self.after_agent(state, runtime)
