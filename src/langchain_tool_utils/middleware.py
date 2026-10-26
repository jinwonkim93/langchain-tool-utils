from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, Overwrite

from .history import (
    UnsafeHistoryError,
    error_result,
    heal_history,
    validate_calls,
)
from .protocols import validate_messages


class ToolHealingMiddleware(AgentMiddleware):
    """Persist clean history, validate complete responses and report tool errors.

    Tool scheduling, retries and checkpoint replay follow LangGraph semantics.
    """

    def __init__(
        self,
        *,
        models: dict | None = None,
        default_api="responses",
    ):
        self.models = models
        self.default_api = default_api

    def _heal(self, messages):
        return heal_history(messages)

    def before_agent(self, state, runtime):
        healed = self._heal(state["messages"])
        if healed != state["messages"]:
            return {"messages": Overwrite(healed)}
        return None

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    def before_model(self, state, runtime):
        return self.before_agent(state, runtime)

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)

    def _request(self, request):
        api = get_config().get("configurable", {}).get("api", self.default_api)
        messages = request.messages
        if api == "chat_completions" and any(
            isinstance(message, AIMessage)
            and isinstance(message.content, list)
            and any(
                block.get("type") in {"reasoning", "function_call"}
                for block in message.content
                if isinstance(block, dict)
            )
            for message in messages
        ):
            raise UnsafeHistoryError(
                "Responses content cannot be injected into a Chat Completions session"
            )
        validate_messages(messages, api=api)
        options = {"messages": messages}
        if self.models is not None:
            api = get_config().get("configurable", {}).get("api", self.default_api)
            if api not in self.models:
                raise ValueError(f"Unsupported API: {api!r}")
            options["model"] = self.models[api]
        return request.override(**options)

    def _validate_response(self, response, *, require_terminal=False):
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
        return response

    def wrap_model_call(self, request, handler):
        return self._validate_response(
            handler(self._request(request)),
            require_terminal=isinstance(request.model, ChatOpenAI),
        )

    async def awrap_model_call(self, request, handler):
        return self._validate_response(
            await handler(self._request(request)),
            require_terminal=isinstance(request.model, ChatOpenAI),
        )

    def _validate_result(self, call, result):
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
        return result

    def wrap_tool_call(self, request, handler):
        try:
            result = handler(request)
        except GraphBubbleUp:
            raise
        except Exception:  # noqa: BLE001 - tool failures become sanitized protocol results
            # Exception text can contain credentials, paths or remote payloads.
            result = error_result(request.tool_call, "execution_error")
        return self._validate_result(request.tool_call, result)

    async def awrap_tool_call(self, request, handler):
        try:
            result = await handler(request)
        except GraphBubbleUp:
            raise
        except Exception:  # noqa: BLE001 - tool failures become sanitized protocol results
            result = error_result(request.tool_call, "execution_error")
        return self._validate_result(request.tool_call, result)
