import json
from itertools import permutations

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from langchain_tool_utils import (
    UnsafeHistoryError,
    clean_messages,
    heal_history,
    portable_history,
    validate_messages,
)
from langchain_tool_utils.history import pending_result


def ai(*ids, message_id=None, content=""):
    return AIMessage(
        content=content,
        id=message_id,
        tool_calls=[{"id": key, "name": "lookup", "args": {}} for key in ids],
    )


def result(key, content="ok", **kwargs):
    return ToolMessage(content=content, tool_call_id=key, name="lookup", **kwargs)


def test_missing_parallel_result_and_external_messages():
    original = [
        ai("a", "b"),
        result("b"),
        HumanMessage("continue"),
        AIMessage("external"),
    ]
    cleaned = clean_messages(original)
    assert [m.type for m in cleaned] == ["ai", "tool", "tool", "human", "ai"]
    assert json.loads(cleaned[1].content)["state"] == "PENDING"
    assert cleaned[2].content == "ok"
    assert len(original) == 4
    assert clean_messages(cleaned) == cleaned
    assert heal_history(original) == cleaned


def test_orphans_removed_and_late_results_moved():
    cleaned = clean_messages(
        [result("orphan"), ai("a"), HumanMessage("later"), result("a")]
    )
    assert [m.type for m in cleaned] == ["ai", "tool", "human"]
    assert cleaned[1].content == "ok"
    assert clean_messages([result("orphan")]) == []


def test_last_duplicate_message_wins_without_deduping_missing_ids():
    cleaned = clean_messages(
        [
            HumanMessage("old", id="same"),
            HumanMessage("one"),
            HumanMessage("new", id="same"),
            HumanMessage("one"),
        ]
    )
    assert [m.content for m in cleaned] == ["one", "new", "one"]


def test_last_duplicate_call_wins_across_messages():
    first = ai("a", "b", message_id="one", content="first")
    last = ai("a", "a", message_id="two", content="last")
    last.tool_calls[-1]["args"] = {"value": 2}
    cleaned = clean_messages([first, result("a", "actual"), last])
    calls = [c for m in cleaned if isinstance(m, AIMessage) for c in m.tool_calls]
    assert [c["id"] for c in calls] == ["b", "a"]
    assert calls[-1]["args"] == {"value": 2}
    assert cleaned[-1].content == "actual"
    assert first.tool_calls[0]["id"] == "a"  # No mutation.


def test_duplicate_call_only_message_is_removed_when_empty():
    cleaned = clean_messages([ai("a", message_id="old"), ai("a", message_id="new")])
    assert len(cleaned) == 2
    assert cleaned[0].id == "new"


@pytest.mark.parametrize("same_message_id", [True, False])
@pytest.mark.parametrize("pending_last", [True, False])
def test_real_result_always_replaces_pending(same_message_id, pending_last):
    owner = ai("a")
    pending = pending_result(owner.tool_calls[0])
    actual = result("a", "real", id=pending.id if same_message_id else "actual")
    entries = [actual, pending] if pending_last else [pending, actual]
    cleaned = clean_messages([owner, *entries])
    assert len(cleaned) == 2
    assert cleaned[1].content == "real"


def test_last_conflicting_result_wins_and_error_is_preserved():
    error = result("a", "not found", status="error")
    cleaned = clean_messages([ai("a"), result("a", "ok"), error])
    assert cleaned[1] == error


def test_id_mapping_is_stable_and_handles_late_original_id():
    old = "call_" + "x" * 200
    original = ai(old)
    cleaned = clean_messages([original])
    key = cleaned[0].tool_calls[0]["id"]
    assert len(key) <= 40
    assert key == cleaned[1].tool_call_id
    assert original.tool_calls[0]["id"] == old
    assert clean_messages(cleaned) == cleaned
    late = clean_messages([*cleaned, result(old, "late actual")])
    assert len(late) == 2
    assert late[1].tool_call_id == key
    assert late[1].content == "late actual"


def test_id_mapping_does_not_collide_with_existing_valid_id():
    old = "invalid / ID"
    generated = clean_messages([ai(old)])[0].tool_calls[0]["id"]
    cleaned = clean_messages([ai(old, generated)])
    ids = [c["id"] for c in cleaned[0].tool_calls]
    assert len(set(ids)) == 2
    assert ids[1] == generated
    assert clean_messages(cleaned) == cleaned


def test_pending_message_id_does_not_overwrite_user_message():
    owner = ai("a")
    pending = pending_result(owner.tool_calls[0])
    cleaned = clean_messages([HumanMessage("keep", id=pending.id), owner])
    assert cleaned[0].content == "keep"
    assert cleaned[2].id != cleaned[0].id
    assert clean_messages(cleaned) == cleaned


def test_chunk_is_discarded_without_executing_partial_call():
    chunk = AIMessageChunk(
        content="partial",
        tool_call_chunks=[{"name": "lookup", "args": "{", "id": "a", "index": 0}],
    )
    assert clean_messages([HumanMessage("go"), chunk, result("a")]) == [
        HumanMessage("go")
    ]


def test_complete_message_with_invalid_arguments_is_not_guessed():
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "lookup", "args": "{", "id": "a", "error": "bad json"}
        ],
    )
    with pytest.raises(UnsafeHistoryError):
        clean_messages([message])


@pytest.mark.parametrize(
    "args", [{"x": float("nan")}, {"x": float("inf")}, {"x": object()}]
)
def test_non_json_arguments_rejected(args):
    owner = ai("a")
    owner.tool_calls[0]["args"] = args
    with pytest.raises(UnsafeHistoryError):
        clean_messages([owner])


@pytest.mark.parametrize(
    "messages",
    [
        [ai("a")],
        [result("a")],
        [ai("a"), HumanMessage("later"), result("a")],
        [ai("a"), result("a"), result("a")],
    ],
)
def test_validator_checks_full_history_without_repair(messages):
    with pytest.raises(UnsafeHistoryError):
        validate_messages(messages)
    validate_messages(clean_messages(messages))


def test_provider_blocks_are_synchronized_and_reasoning_is_preserved():
    message = ai("a", "a", message_id="owner")
    message.content = [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "a",
            "name": "lookup",
            "arguments": "{}",
        },
        {
            "type": "function_call",
            "id": "fc_2",
            "call_id": "a",
            "name": "lookup",
            "arguments": "{}",
        },
    ]
    message.response_metadata = {"id": "resp_old"}
    cleaned = clean_messages([message])
    assert len(cleaned[0].content) == 2
    assert cleaned[0].content[1]["id"] == "fc_2"
    assert (
        portable_history(cleaned, api="responses")[0].content[0]["type"] == "reasoning"
    )
    projected = portable_history(cleaned, api="chat_completions")[0]
    assert projected.content == ""
    assert projected.response_metadata == {}
    assert message.response_metadata == {"id": "resp_old"}
    assert clean_messages(cleaned) == cleaned


def test_unknown_provider_builtin_rejected_instead_of_silently_discarded():
    with pytest.raises(UnsafeHistoryError):
        clean_messages([AIMessage(content=[{"type": "web_search_call", "id": "ws_1"}])])


def test_durable_pending_replaced_using_recovery_callback():
    cleaned = clean_messages([ai("a")])
    recovered = clean_messages(
        cleaned, recover=lambda owner, call: result(call["id"], "saved")
    )
    assert recovered[1].content == "saved"


def test_reducer_replacement_really_removes_orphans_and_duplicates():
    original = [
        HumanMessage("go", id="h"),
        ai("a", message_id="ai"),
        result("orphan", id="bad"),
        result("a", "old", id="old"),
        result("a", "new", id="new"),
    ]
    cleaned = clean_messages(original)
    stored = add_messages(original, [RemoveMessage(id=REMOVE_ALL_MESSAGES), *cleaned])
    assert stored == cleaned
    assert clean_messages(stored) == stored


def test_all_permutations_normalize_to_valid_idempotent_history():
    # Results can arrive before or after an externally injected Human/AI message.
    entries = [
        ai("a", "b", message_id="ai"),
        result("a", "old"),
        result("a", "new"),
        result("orphan"),
        HumanMessage("external", id="h"),
    ]
    for permutation in permutations(entries):
        cleaned = clean_messages(permutation)
        validate_messages(cleaned)
        assert clean_messages(cleaned) == cleaned


def test_dict_input():
    cleaned = clean_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
                "content": "",
            }
        ]
    )
    assert cleaned[0].tool_calls[0]["id"] == cleaned[1].tool_call_id == "a"


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
def test_final_sdk_payload_has_exactly_one_output_per_call(api):
    from langchain_openai.chat_models.base import (
        _construct_responses_api_input,
        _convert_message_to_dict,
    )

    cleaned = clean_messages(
        [
            ai("a", "b", "a"),
            result("orphan"),
            result("a", "old"),
            HumanMessage("external"),
            result("a", "latest"),
        ],
        api=api,
    )
    validate_messages(cleaned, api=api)
    projected = portable_history(cleaned, api=api)
    if api == "responses":
        payload = _construct_responses_api_input(projected)
        calls = [
            item["call_id"] for item in payload if item.get("type") == "function_call"
        ]
        outputs = [
            item["call_id"]
            for item in payload
            if item.get("type") == "function_call_output"
        ]
    else:
        payload = [_convert_message_to_dict(m) for m in projected]
        calls = [c["id"] for m in payload for c in m.get("tool_calls", [])]
        outputs = [m["tool_call_id"] for m in payload if m["role"] == "tool"]
    assert sorted(calls) == sorted(outputs) == ["a", "b"]


def test_refusal_is_preserved_when_switching_api():
    cleaned = clean_messages(
        [AIMessage(content=[{"type": "refusal", "refusal": "cannot comply"}])]
    )
    projected = portable_history(cleaned, api="chat_completions")
    assert projected[0].content == [{"type": "refusal", "refusal": "cannot comply"}]


def test_string_blocks_are_normalized():
    assert clean_messages([HumanMessage(["one", "two"])])[0].content == [
        {"type": "text", "text": "one"},
        {"type": "text", "text": "two"},
    ]


def test_responses_reasoning_is_kept_in_serialized_payload():
    from langchain_openai.chat_models.base import _construct_responses_api_input

    owner = ai("a")
    reasoning = {
        "type": "reasoning",
        "id": "rs_saved",
        "summary": [],
        "encrypted_content": "opaque",
    }
    owner.content = [reasoning]
    cleaned = clean_messages([owner])
    payload = _construct_responses_api_input(portable_history(cleaned, api="responses"))
    assert reasoning in payload
    assert clean_messages(cleaned) == cleaned


def test_external_responses_call_content_without_canonical_tool_calls():
    message = AIMessage(
        content=[
            {
                "type": "function_call",
                "id": "fc_external",
                "call_id": "external",
                "name": "lookup",
                "arguments": '{"order_id":"123"}',
            }
        ]
    )
    cleaned = clean_messages([message])
    assert cleaned[0].tool_calls[0]["args"] == {"order_id": "123"}
    assert cleaned[1].tool_call_id == "external"
    assert json.loads(cleaned[1].content)["state"] == "PENDING"
    assert clean_messages(cleaned) == cleaned
