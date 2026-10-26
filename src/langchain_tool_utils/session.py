"""Reset conversation history at the invocation boundary when its API changes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import RemoveMessage, convert_to_messages
from langgraph.config import get_config
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .history import UnsafeHistoryError


class ApiSessionState(AgentState):
    session_api: NotRequired[str]


class ApiSessionMiddleware(AgentMiddleware):
    state_schema = ApiSessionState

    def __init__(self, default_api):
        self.default_api = default_api

    def before_model(self, state, runtime):
        requested = get_config().get("configurable", {}).get("api", self.default_api)
        if state.get("session_api") != requested:
            raise UnsafeHistoryError("API changes must enter through ApiSessionAgent")

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)


def prepare_session_input(value, *, previous_api, api, has_history, interrupted):
    """Build a fresh input; never guess which merged messages belong to this turn."""
    if api not in {"responses", "chat_completions"}:
        raise ValueError(f"Unsupported API: {api!r}")
    changed = previous_api != api and (previous_api is not None or has_history)
    if changed and (interrupted or value is None):
        raise UnsafeHistoryError(
            "Finish the interrupted run using its original API before switching"
        )
    if value is None:
        return None
    if not isinstance(value, Mapping):
        # Command resumes need the same API and retain their original input shape.
        if changed:
            raise UnsafeHistoryError("API switching requires a new message input")
        return value
    output = dict(value)
    if "session_api" in output:
        raise ValueError("session_api is managed by ApiSessionAgent")
    if "messages" in output:
        output["messages"] = [
            m.model_copy(deep=True) for m in convert_to_messages(output["messages"])
        ]
    if changed:
        messages = output.get("messages", [])
        if not messages:
            raise ValueError("API switching requires at least one new message")
        if any(isinstance(m, RemoveMessage) for m in messages):
            raise ValueError(
                "API switching accepts new messages, not removal directives"
            )
        output["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]
    output["session_api"] = api
    return output


class ApiSessionAgent:
    """Graph facade: same API keeps history; API changes start a fresh conversation.

    Callers must serialize invocations and state updates for each thread. Execution
    journal entries and non-message state are retained. Without a checkpointer,
    every invocation is already a new conversation.
    """

    def __init__(self, graph, *, default_api, has_checkpointer):
        self.graph = graph
        self.default_api = default_api
        self.has_checkpointer = has_checkpointer

    def __getattr__(self, name):
        if name in {
            "get_state",
            "aget_state",
            "get_state_history",
            "aget_state_history",
            "update_state",
            "aupdate_state",
            "get_graph",
            "aget_graph",
        }:
            return getattr(self.graph, name)
        raise AttributeError(
            f"ApiSessionAgent does not expose {name!r}; use invoke/stream or async equivalents"
        )

    def _prepare(self, value, config, snapshot):
        api = (config or {}).get("configurable", {}).get("api", self.default_api)
        state = snapshot.values if snapshot else {}
        return prepare_session_input(
            value,
            previous_api=state.get("session_api"),
            api=api,
            has_history=bool(state.get("messages")),
            interrupted=bool(snapshot and snapshot.next),
        )

    def invoke(self, input, config=None, **kwargs):
        snapshot = self.graph.get_state(config) if self.has_checkpointer else None
        return self.graph.invoke(
            self._prepare(input, config, snapshot), config, **kwargs
        )

    async def ainvoke(self, input, config=None, **kwargs):
        snapshot = (
            await self.graph.aget_state(config) if self.has_checkpointer else None
        )
        return await self.graph.ainvoke(
            self._prepare(input, config, snapshot), config, **kwargs
        )

    def stream(self, input, config=None, **kwargs):
        snapshot = self.graph.get_state(config) if self.has_checkpointer else None
        yield from self.graph.stream(
            self._prepare(input, config, snapshot), config, **kwargs
        )

    async def astream(self, input, config=None, **kwargs):
        snapshot = (
            await self.graph.aget_state(config) if self.has_checkpointer else None
        )
        async for event in self.graph.astream(
            self._prepare(input, config, snapshot), config, **kwargs
        ):
            yield event
