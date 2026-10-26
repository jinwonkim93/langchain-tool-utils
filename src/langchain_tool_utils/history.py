"""Deterministic message cleanup. No tool execution and no repair report."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)


class UnsafeHistoryError(ValueError):
    """The remaining content cannot be safely sent using the supported protocol."""


def _pending(message: BaseMessage) -> bool:
    return (
        isinstance(message, ToolMessage)
        and message.additional_kwargs.get("tool_healing", {}).get("state") == "PENDING"
    )


def pending_result(call: dict, reason: str = "result_missing") -> ToolMessage:
    return ToolMessage(
        id="pending_" + hashlib.sha256(call["id"].encode()).hexdigest(),
        tool_call_id=call["id"],
        name=call["name"],
        status="error",  # LangChain has no pending status; content carries state.
        content=json.dumps(
            {
                "state": "PENDING",
                "reason": reason,
                "message": "No final result is available. Check whether the operation is still running or already took effect. Request the tool again when retrying is safe.",
            }
        ),
        additional_kwargs={"tool_healing": {"state": "PENDING"}},
    )


def error_result(call: dict, reason: str) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(
            {
                "state": "ERROR",
                "reason": reason,
                "message": "The tool reported an error. Side effects may already have occurred.",
            }
        ),
        tool_call_id=call["id"],
        name=call["name"],
        status="error",
    )


def validate_calls(message: AIMessage) -> None:
    if isinstance(message, BaseMessageChunk) or message.invalid_tool_calls:
        raise UnsafeHistoryError("Incomplete tool calls cannot be executed")
    seen = set()
    for call in message.tool_calls:
        if (
            not isinstance(call.get("id"), str)
            or not call["id"]
            or not isinstance(call.get("name"), str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", call["name"])
            or not isinstance(call.get("args"), dict)
            or call["id"] in seen
        ):
            raise UnsafeHistoryError("Invalid function call ID, name or arguments")
        try:
            json.dumps(call["args"], allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise UnsafeHistoryError(
                "Function arguments must be JSON serializable"
            ) from exc
        seen.add(call["id"])


def _sync_call_blocks(message: AIMessage, aliases: dict[str, str]) -> None:
    """Keep canonical calls and provider content representations in agreement."""
    calls = {call["id"]: call for call in message.tool_calls}
    message.additional_kwargs.pop("tool_calls", None)
    if "function_call" in message.additional_kwargs:
        raise UnsafeHistoryError("Legacy function_call messages are not supported")
    if not isinstance(message.content, list):
        return
    kept = []
    seen = set()
    for block in reversed(message.content):
        if not isinstance(block, dict) or block.get("type") not in {
            "function_call",
            "tool_call",
        }:
            kept.append(block)
            continue
        field = "call_id" if block["type"] == "function_call" else "id"
        old = block.get(field)
        key = aliases.get(old, old)
        if key not in calls or key in seen:
            continue
        block[field] = key
        block["name"] = calls[key]["name"]
        if field == "call_id":
            block["arguments"] = json.dumps(calls[key]["args"], allow_nan=False)
            if key != old:
                block.pop("id", None)  # A stored provider item cannot be rewritten.
        else:
            block["args"] = calls[key]["args"]
        kept.append(block)
        seen.add(key)
    message.content = list(reversed(kept)) or ""


def clean_messages(
    messages: Sequence,
    *,
    max_call_id_length: int = 40,
    api: str = "responses",
    recover: Callable[[AIMessage, dict], ToolMessage | None] | None = None,
) -> list[BaseMessage]:
    """Return a cleaned, persistable message list without mutating the input.

    Last duplicate message/call wins. Real tool results outrank synthetic PENDING
    results; otherwise the last result wins. Orphans and uncommitted stream chunks
    are removed. Results are placed directly after their owning AI message.
    Missing results become PENDING. Existing error results are preserved.

    The 40-character ASCII call-ID budget is a conservative application policy,
    not a claim that all providers have the same limit. Invalid IDs are mapped
    deterministically on both sides, with aliases retained for late results.
    Only complete local function calls are supported. Unsupported or malformed
    content raises UnsafeHistoryError rather than being silently discarded.
    """
    if not isinstance(max_call_id_length, int) or max_call_id_length < 21:
        raise ValueError("max_call_id_length must be at least 21")
    source = [
        m.model_copy(deep=True)
        for m in convert_to_messages(messages)
        if not isinstance(m, BaseMessageChunk)
    ]

    for msg in source:
        if isinstance(msg.content, list):
            msg.content = [
                {"type": "text", "text": block} if isinstance(block, str) else block
                for block in msg.content
            ]

    # Storage identity is separate from tool-call identity. Missing message IDs
    # are not duplicates and do not need to be invented by this utility.
    winners = {}
    for i, msg in enumerate(source):
        if msg.id is not None:
            previous = winners.get(msg.id)
            if previous is None or not (
                _pending(msg) and not _pending(source[previous])
            ):
                winners[msg.id] = i
    source = [m for i, m in enumerate(source) if m.id is None or winners[m.id] == i]

    # Externally assembled Responses messages may carry calls only in content.
    for msg in source:
        if (
            isinstance(msg, AIMessage)
            and not msg.tool_calls
            and isinstance(msg.content, list)
        ):
            for block in msg.content:
                if not isinstance(block, dict) or block.get("type") not in {
                    "function_call",
                    "tool_call",
                }:
                    continue
                try:
                    args = (
                        json.loads(block["arguments"])
                        if block["type"] == "function_call"
                        else block["args"]
                    )
                    key = (
                        block["call_id"]
                        if block["type"] == "function_call"
                        else block["id"]
                    )
                    msg.tool_calls.append(
                        {
                            "id": key,
                            "name": block["name"],
                            "args": args,
                            "type": "tool_call",
                        }
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    raise UnsafeHistoryError("Invalid tool call content block") from exc

    aliases: dict[str, str] = {}
    for msg in source:
        if isinstance(msg, AIMessage):
            aliases.update(
                msg.additional_kwargs.get("tool_healing", {}).get("call_id_map", {})
            )

    call_winners = {}
    for i, msg in enumerate(source):
        if not isinstance(msg, AIMessage):
            continue
        if msg.invalid_tool_calls:
            raise UnsafeHistoryError(
                "A completed AI message contains invalid tool arguments"
            )
        for j, call in enumerate(msg.tool_calls):
            if not isinstance(call.get("id"), str) or not call["id"]:
                raise UnsafeHistoryError("A complete tool call must have a nonempty ID")
            key = aliases.get(call["id"], call["id"])
            call_winners[key] = (i, j)

    def valid_id(value):
        return len(value) <= max_call_id_length and re.fullmatch(
            r"[A-Za-z0-9_-]+", value
        )

    reserved = {key for key in call_winners if valid_id(key)}
    mapping = {}
    for key in call_winners:
        if valid_id(key):
            mapping[key] = key
            continue
        salt = 0
        while True:
            digest = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
            candidate = "call_" + digest[: min(max_call_id_length - 5, 64)]
            if candidate not in reserved:
                break
            salt += 1
        mapping[key] = candidate
        reserved.add(candidate)
    aliases = {old: mapping.get(new, new) for old, new in aliases.items()}
    aliases.update({old: new for old, new in mapping.items() if old != new})

    aliases_by_id = {}
    for old, new in aliases.items():
        aliases_by_id.setdefault(new, {})[old] = new
    positions = {position: key for key, position in call_winners.items()}
    owners = {}
    for i, msg in enumerate(source):
        if not isinstance(msg, AIMessage):
            continue
        calls = []
        for j, call in enumerate(msg.tool_calls):
            original = call["id"]
            # Winner lookup uses the pre-remapping identity.
            old = positions.get((i, j))
            if old is None:
                continue
            call["id"] = mapping[old]
            call["type"] = "tool_call"
            calls.append(call)
            owners[call["id"]] = (msg, call, original)
        msg.tool_calls = calls
        _sync_call_blocks(msg, aliases)
        if calls:
            relevant = {
                old: new
                for call in calls
                for old, new in aliases_by_id.get(call["id"], {}).items()
            }
            if relevant:
                msg.additional_kwargs.setdefault("tool_healing", {})["call_id_map"] = (
                    relevant
                )
        validate_calls(msg)

    results = {}
    for msg in source:
        if not isinstance(msg, ToolMessage):
            continue
        msg.tool_call_id = aliases.get(msg.tool_call_id, msg.tool_call_id)
        if msg.tool_call_id not in owners:
            continue
        previous = results.get(msg.tool_call_id)
        if previous is None or not (_pending(msg) and not _pending(previous)):
            results[msg.tool_call_id] = msg

    output = []
    used_message_ids = {m.id for m in source if m.id}
    for msg in source:
        if isinstance(msg, ToolMessage):
            continue
        if (
            isinstance(msg, AIMessage)
            and not msg.content
            and not msg.tool_calls
            and not msg.additional_kwargs.get("refusal")
        ):
            continue
        output.append(msg)
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls:
            key = call["id"]
            result = results.get(key)
            if recover and (result is None or _pending(result)):
                original = owners[key][2]
                # Recover using the execution identity, before ID normalization.
                original = next(iter(aliases_by_id.get(key, {})), original)
                saved = recover(msg, {**call, "id": original})
                if saved is not None:
                    result = saved.model_copy(deep=True)
                    if aliases.get(result.tool_call_id, result.tool_call_id) != key:
                        raise UnsafeHistoryError(
                            "Recovered result has the wrong call ID"
                        )
                    result.tool_call_id = key
            if result is None:
                result = pending_result(call)
                while result.id in used_message_ids:
                    result.id += "_"
                used_message_ids.add(result.id)
            result.name = call["name"]
            output.append(result)
    validate_messages(output, api=api, max_call_id_length=max_call_id_length)
    return output


def _validate_structure(messages: Sequence, *, max_call_id_length: int = 40) -> None:
    """Validate the full local-function message history; raise on invalid input.

    This checks message structure, pairing, ordering, content and JSON arguments.
    It does not validate credentials, context length or model-specific features.
    """
    pending = set()
    calls = set()
    ids = set()
    for msg in convert_to_messages(messages):
        if isinstance(msg, BaseMessageChunk) or not isinstance(
            msg, (AIMessage, HumanMessage, SystemMessage, ToolMessage)
        ):
            raise UnsafeHistoryError(
                "Unsupported message type or unfinished stream chunk"
            )
        if msg.id is not None:
            if msg.id in ids:
                raise UnsafeHistoryError("Duplicate message ID")
            ids.add(msg.id)
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id not in pending:
                raise UnsafeHistoryError("Orphan or duplicate tool result")
            pending.remove(msg.tool_call_id)
        else:
            if pending:
                raise UnsafeHistoryError("Tool results must precede the next message")
            if isinstance(msg, AIMessage):
                validate_calls(msg)
                if "function_call" in msg.additional_kwargs:
                    raise UnsafeHistoryError(
                        "Legacy function_call messages are not supported"
                    )
                for call in msg.tool_calls:
                    key = call["id"]
                    if (
                        key in calls
                        or len(key) > max_call_id_length
                        or not re.fullmatch(r"[A-Za-z0-9_-]+", key)
                    ):
                        raise UnsafeHistoryError("Duplicate or invalid tool-call ID")
                    calls.add(key)
                    pending.add(key)
        if isinstance(msg.content, list):
            for block in msg.content:
                if not isinstance(block, dict) or not isinstance(
                    block.get("type"), str
                ):
                    raise UnsafeHistoryError("Content blocks must be typed objects")
                kind = block["type"]
                if kind == "text" and not isinstance(block.get("text"), str):
                    raise UnsafeHistoryError("Text block requires text")
                allowed = {"text"}
                if isinstance(msg, AIMessage):
                    allowed |= {"reasoning", "function_call", "tool_call", "refusal"}
                elif isinstance(msg, HumanMessage):
                    allowed |= {"image_url"}
                if kind not in allowed:
                    raise UnsafeHistoryError(f"Unsupported content block: {kind}")
                if kind in {"function_call", "tool_call"}:
                    key = block.get("call_id" if kind == "function_call" else "id")
                    canonical = next(
                        (call for call in msg.tool_calls if call["id"] == key), None
                    )
                    try:
                        args = (
                            json.loads(block["arguments"])
                            if kind == "function_call"
                            else block["args"]
                        )
                    except (TypeError, ValueError, KeyError) as exc:
                        raise UnsafeHistoryError(
                            "Invalid native tool arguments"
                        ) from exc
                    if (
                        canonical is None
                        or block.get("name") != canonical["name"]
                        or args != canonical["args"]
                    ):
                        raise UnsafeHistoryError(
                            "Native tool block disagrees with canonical call"
                        )
                if kind == "refusal" and not isinstance(block.get("refusal"), str):
                    raise UnsafeHistoryError("Refusal block requires refusal text")
                if kind == "reasoning" and (
                    not isinstance(block.get("summary"), list)
                    or not (
                        isinstance(block.get("id"), str)
                        or isinstance(block.get("encrypted_content"), str)
                    )
                ):
                    raise UnsafeHistoryError(
                        "Responses reasoning requires its ID or encrypted content and summary"
                    )
                if kind == "image_url":
                    image = block.get("image_url")
                    if not isinstance(image, dict) or not isinstance(
                        image.get("url"), str
                    ):
                        raise UnsafeHistoryError(
                            "Image block requires an image_url.url"
                        )
        try:
            json.dumps(msg.content, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise UnsafeHistoryError(
                "Message content must be JSON serializable"
            ) from exc
    if pending:
        raise UnsafeHistoryError("Missing tool results")


def heal_history(messages: Sequence, recover=None) -> list[BaseMessage]:
    """Backward-compatible function name; now returns only cleaned messages."""
    return clean_messages(messages, recover=recover)


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
