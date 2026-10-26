from __future__ import annotations

from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from .history import (
    UnsafeHistoryError,
    error_result,
    heal_history,
    pending_result,
    portable_history,
    validate_calls,
)
from .journal import SQLiteJournal


def scope() -> str:
    value = get_config().get("configurable", {}).get("thread_id")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "A nonempty configurable.thread_id is required for execution fencing"
        )
    # Include subgraph namespace so child agents cannot reuse a parent's call IDs.
    namespace = get_config().get("configurable", {}).get("checkpoint_ns", "")
    namespace = namespace.rpartition("|")[0]  # remove the current node task ID
    return f"{len(value)}:{value}:{namespace}"


class ToolHealingMiddleware(AgentMiddleware):
    """Place last so retries/approval middleware cannot bypass the execution fence.

    New invocation: repair checkpoint/input history and close unresolved calls.
    Resume at tools: the execution fence replays durable results or refuses
    running/unknown/external calls. No retries, argument rewriting or success
    fabrication. LangGraph interrupts and cancellation propagate unchanged.
    """

    def __init__(
        self,
        journal: SQLiteJournal,
        *,
        models: dict | None = None,
        default_api="responses",
    ):
        self.journal = journal
        self.models = models
        self.default_api = default_api

    def _heal(self, messages):
        namespace = scope()
        return heal_history(
            messages, lambda owner, call: self.journal.recover(namespace, owner, call)
        )

    def before_agent(self, state, runtime):
        healed = self._heal(state["messages"])
        if healed != state["messages"]:
            return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *healed]}
        return None

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    def before_model(self, state, runtime):
        return self.before_agent(state, runtime)

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)

    def _request(self, request):
        api = get_config().get("configurable", {}).get("api", self.default_api)
        messages = portable_history(self._heal(request.messages), api=api)
        options = {"messages": messages}
        if self.models is not None:
            api = get_config().get("configurable", {}).get("api", self.default_api)
            if api not in self.models:
                raise ValueError(f"Unsupported API: {api!r}")
            options["model"] = self.models[api]
        return request.override(**options)

    def _authorize(self, response, *, require_terminal=False):
        namespace = scope()
        value = (
            response.model_response
            if isinstance(response, ExtendedModelResponse)
            else response
        )
        messages = [value] if isinstance(value, AIMessage) else value.result
        for msg in messages:
            if isinstance(msg, AIMessage):
                validate_calls(msg)
                if require_terminal:
                    meta = msg.response_metadata
                    if "status" in meta:
                        completed = meta["status"] == "completed"
                    else:
                        expected = "tool_calls" if msg.tool_calls else "stop"
                        completed = meta.get("finish_reason") == expected
                    if not completed:
                        raise UnsafeHistoryError(
                            "Model response lacks a successful terminal event; partial output is not executable"
                        )
                if not msg.id:
                    msg.id = str(uuid4())
                for call in msg.tool_calls:
                    self.journal.authorize(namespace, msg, call)
        return response

    def wrap_model_call(self, request, handler):
        return self._authorize(
            handler(self._request(request)),
            require_terminal=isinstance(request.model, ChatOpenAI),
        )

    async def awrap_model_call(self, request, handler):
        return self._authorize(
            await handler(self._request(request)),
            require_terminal=isinstance(request.model, ChatOpenAI),
        )

    def _claim(self, request):
        call = request.tool_call
        # Match the current owner, never an old call elsewhere in the transcript.
        owner = next(
            (
                m
                for m in reversed(request.state["messages"])
                if isinstance(m, AIMessage)
            ),
            None,
        )
        if owner is None or call not in owner.tool_calls:
            raise UnsafeHistoryError("Tool request has no matching assistant owner")
        state, saved = self.journal.claim(scope(), owner, call)
        if state == "execute":
            return owner, None
        return owner, saved if saved is not None else pending_result(call, state)

    def _commit(self, owner, call, result):
        if isinstance(result, Command):
            if not isinstance(result.update, dict) or result.graph is not None:
                raise UnsafeHistoryError("Command tools require local dict updates")
            matches = [
                m
                for m in result.update.get("messages", [])
                if isinstance(m, ToolMessage) and m.tool_call_id == call["id"]
            ]
            if len(matches) != 1:
                raise UnsafeHistoryError(
                    "Command must contain exactly one matching ToolMessage"
                )
        elif not isinstance(result, ToolMessage) or result.tool_call_id != call["id"]:
            raise UnsafeHistoryError("Tool handler returned a mismatched result")
        self.journal.finish(scope(), owner, call, result)
        return result

    def wrap_tool_call(self, request, handler):
        owner, saved = self._claim(request)
        if saved is not None:
            return saved
        try:
            result = handler(request)
        except GraphBubbleUp:
            raise
        except Exception:  # noqa: BLE001 - tool failures become sanitized protocol results
            # Exception text can contain credentials, paths or remote payloads.
            result = error_result(request.tool_call, "execution_error")
        return self._commit(owner, request.tool_call, result)

    async def awrap_tool_call(self, request, handler):
        owner, saved = self._claim(request)
        if saved is not None:
            return saved
        try:
            result = await handler(request)
        except GraphBubbleUp:
            raise
        except Exception:  # noqa: BLE001 - tool failures become sanitized protocol results
            result = error_result(request.tool_call, "execution_error")
        return self._commit(owner, request.tool_call, result)
