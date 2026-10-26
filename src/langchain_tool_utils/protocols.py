"""Optional OpenAI wire validation and explicit legacy history projection."""

from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages

from .history import UnsafeHistoryError, _validate_structure, validate_calls


def portable_history(
    messages: Sequence, *, api: str = "chat_completions", max_call_id_length: int = 40
) -> list[BaseMessage]:
    """Project canonical history for the requested API without changing state."""
    if api not in {"responses", "chat_completions"}:
        raise ValueError("Unsupported API")
    _validate_structure(messages, max_call_id_length=max_call_id_length)
    output = []
    for original in convert_to_messages(messages):
        msg = original.model_copy(deep=True)
        if isinstance(msg, AIMessage):
            validate_calls(msg)
            if api == "chat_completions":
                if isinstance(msg.content, list):
                    msg.content = [
                        {"type": "text", "text": block["text"]}
                        if block["type"] == "text"
                        else {"type": "refusal", "refusal": block["refusal"]}
                        for block in msg.content
                        if isinstance(block, dict)
                        and block.get("type") in {"text", "refusal"}
                    ] or ""
                msg.additional_kwargs.pop("tool_calls", None)
                msg.response_metadata = {}
            if (
                not msg.content
                and not msg.tool_calls
                and not msg.additional_kwargs.get("refusal")
            ):
                continue
        output.append(msg)
    _validate_structure(output, max_call_id_length=max_call_id_length)
    return output


def validate_messages(
    messages: Sequence, *, api: str = "responses", max_call_id_length: int = 40
) -> None:
    """Validate canonical history and its actual LangChain wire representation.

    Returns None on success. No network request or model client is created.
    Raises UnsafeHistoryError for invalid or unsupported message content.
    Model capabilities, tool definitions, token limits and credentials are outside
    this message-only contract. The serializer adapter is version-pinned by the
    package dependencies and covered by protocol tests.
    """
    from langchain_openai.chat_models.base import (
        _construct_responses_api_input,
        _convert_message_to_dict,
    )

    projected = portable_history(
        messages, api=api, max_call_id_length=max_call_id_length
    )
    expected = {
        call["id"]
        for msg in projected
        if isinstance(msg, AIMessage)
        for call in msg.tool_calls
    }
    try:
        if api == "responses":
            payload = _construct_responses_api_input(projected)
            wire_calls = [
                item["call_id"]
                for item in payload
                if item.get("type") == "function_call"
            ]
            wire_results = [
                item["call_id"]
                for item in payload
                if item.get("type") == "function_call_output"
            ]
        else:
            payload = [_convert_message_to_dict(msg) for msg in projected]
            wire_calls = [
                call["id"] for item in payload for call in item.get("tool_calls", [])
            ]
            wire_results = [
                item["tool_call_id"] for item in payload if item.get("role") == "tool"
            ]
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError, KeyError) as exc:
        raise UnsafeHistoryError(
            "Messages cannot be serialized for the selected API"
        ) from exc
    if (
        len(wire_calls) != len(expected)
        or set(wire_calls) != expected
        or len(wire_results) != len(expected)
        or set(wire_results) != expected
    ):
        raise UnsafeHistoryError("Serialized tool calls and results do not match")
