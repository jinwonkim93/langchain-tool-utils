"""Independent oracle, exhaustive short histories, and persisted event replays."""

import copy
import itertools
import json
import random

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

from langchain_tool_utils import UnsafeHistoryError, clean_messages, validate_messages
from langchain_tool_utils.history import pending_result


def call(key, value=0):
    return {"id": key, "name": "lookup", "args": {"value": value}}


def is_pending(message):
    return message.additional_kwargs.get("tool_healing", {}).get("state") == "PENDING"


def oracle(messages):
    # Deliberately independent: quadratic scans, no production helpers/maps.
    source = [m for m in messages if not isinstance(m, AIMessageChunk)]
    retained = []
    for m in source:
        if m.id is None:
            retained.append(m)
            continue
        group = [x for x in source if x.id == m.id]
        real = [x for x in group if not is_pending(x)]
        winner = (real or group)[-1]
        if m is winner:
            retained.append(m)
    flattened = [
        (m, c) for m in retained if isinstance(m, AIMessage) for c in m.tool_calls
    ]
    expected = {}
    for owner, c in flattened:
        results = [
            m
            for m in retained
            if isinstance(m, ToolMessage) and m.tool_call_id == c["id"]
        ]
        real = [m for m in results if not is_pending(m)]
        winner = (real or results)[-1] if results else None
        expected[c["id"]] = (owner.content, c["args"], winner)
    return expected


def assert_contract(source, *, budget=40, use_oracle=True):
    snapshot = copy.deepcopy(source)
    cleaned = clean_messages(source, max_call_id_length=budget)
    assert source == snapshot
    # Serialization is a checkpoint boundary, not merely repeated object input.
    restored = messages_from_dict(json.loads(json.dumps(messages_to_dict(cleaned))))
    assert clean_messages(restored, max_call_id_length=budget) == cleaned
    for api in ("responses", "chat_completions"):
        validate_messages(cleaned, api=api, max_call_id_length=budget)
        assert clean_messages(cleaned, api=api, max_call_id_length=budget) == cleaned
    expected = oracle(source) if use_oracle else None
    seen = set()
    i = 0
    while i < len(cleaned):
        owner = cleaned[i]
        assert not isinstance(owner, ToolMessage)
        i += 1
        if not isinstance(owner, AIMessage):
            continue
        for c in owner.tool_calls:
            assert c["id"] not in seen
            seen.add(c["id"])
            result = cleaned[i]
            i += 1
            assert isinstance(result, ToolMessage)
            assert result.tool_call_id == c["id"]
            assert result.name == c["name"]
            if expected is not None:
                content, args, winner = expected[c["id"]]
                assert owner.content == content
                assert c["args"] == args
                if winner is None:
                    assert is_pending(result)
                    assert json.loads(result.content)["state"] == "PENDING"
                else:
                    assert result.content == winner.content
                    assert result.status == winner.status
    if expected is not None:
        assert seen == set(expected)
    return cleaned


def atom(kind, index):
    if kind == 0:
        return HumanMessage(f"human-{index}")
    if kind == 1:
        return AIMessage(f"plain-{index}")
    if kind == 2:
        return AIMessage(f"call-{index}", tool_calls=[call("a", index)])
    if kind == 3:
        return AIMessage(
            f"parallel-{index}", tool_calls=[call("b", index), call("a", index)]
        )
    if kind == 4:
        return ToolMessage(f"success-{index}", tool_call_id="a")
    if kind == 5:
        return ToolMessage(f"error-{index}", tool_call_id="a", status="error")
    if kind == 6:
        return pending_result(call("b"))
    return ToolMessage(f"orphan-{index}", tool_call_id="orphan")


@pytest.mark.parametrize("length", range(5))
def test_exhaustive_short_histories(length):
    # 1 + 8 + 64 + 512 + 4096 = 4681 histories, including repeated symbols.
    for kinds in itertools.product(range(8), repeat=length):
        source = [atom(kind, index) for index, kind in enumerate(kinds)]
        assert_contract(source)


@pytest.mark.parametrize("seed", range(100))
def test_seeded_long_histories_against_independent_oracle(seed):
    rng = random.Random(seed)
    history = [SystemMessage("system")]
    for index in range(rng.randint(20, 80)):
        message = atom(rng.randrange(8), index)
        message.id = rng.choice([None, None, "m0", "m1", "m2", f"unique-{index}"])
        if isinstance(message, AIMessage) and message.tool_calls:
            message.tool_calls = [
                call(rng.choice("abcd"), index) for _ in range(rng.randint(1, 5))
            ]
        elif isinstance(message, ToolMessage):
            message.tool_call_id = rng.choice("abcde")
        history.append(message)
    assert_contract(history)


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("budget", [21, 40, 64])
def test_incremental_checkpoints_late_results_and_api_switches(seed, budget):
    rng = random.Random(seed)
    keys = ["한글 / call", "x" * 200, "normal", "with space", "emoji-🙂"]
    history = [AIMessage("parallel", id="owner", tool_calls=[call(k) for k in keys])]
    history = assert_contract(history, budget=budget, use_oracle=False)
    rng.shuffle(keys)
    for index, key in enumerate(keys):
        history += [
            HumanMessage(f"external-{index}"),
            AIMessage("injected"),
            ToolMessage(
                f"late-{key}",
                tool_call_id=key,
                status="error" if index % 2 else "success",
            ),
        ]
        history = assert_contract(history, budget=budget, use_oracle=False)
        actual = [
            m for m in history if isinstance(m, ToolMessage) and not is_pending(m)
        ]
        assert len(actual) == index + 1
        assert {m.content for m in actual} == {f"late-{k}" for k in keys[: index + 1]}


@pytest.mark.parametrize("kind", ["function_call", "tool_call"])
@pytest.mark.parametrize(
    "field,value",
    [("name", "different"), ("arguments", '{"different":1}'), ("call_id", "different")],
)
def test_validator_rejects_native_call_disagreeing_with_canonical(kind, field, value):
    block = {
        "type": kind,
        "call_id": "a",
        "id": "a",
        "name": "lookup",
        "arguments": '{"value":0}',
        "args": {"value": 0},
    }
    if kind == "tool_call":
        field = {"arguments": "args", "call_id": "id"}.get(field, field)
        if field == "args":
            value = {"different": 1}
    block[field] = value
    owner = AIMessage(content=[block], tool_calls=[call("a")])
    history = [owner, ToolMessage("ok", tool_call_id="a")]
    with pytest.raises(UnsafeHistoryError):
        validate_messages(history)
    assert_contract(history, use_oracle=False)


def test_all_seven_event_permutations():
    events = [
        HumanMessage("external", id="human"),
        AIMessage("old-owner", id="old", tool_calls=[call("a", 1), call("b", 2)]),
        AIMessage("new-owner", id="new", tool_calls=[call("a", 3)]),
        ToolMessage("old-result", tool_call_id="a", id="r1"),
        ToolMessage("failed-result", tool_call_id="a", id="r2", status="error"),
        pending_result(call("a")),
        ToolMessage("orphan", tool_call_id="unknown", id="orphan"),
    ]
    for ordering in itertools.permutations(events):
        assert_contract(list(ordering))


@pytest.mark.parametrize("cut", range(8))
def test_every_stream_cut_and_late_tool_completion(cut):
    events = [
        HumanMessage("start"),
        AIMessageChunk(
            content="partial",
            tool_call_chunks=[{"name": "lookup", "args": "{", "id": "a", "index": 0}],
        ),
        AIMessage("complete", tool_calls=[call("a"), call("b")]),
        ToolMessage("a-done", tool_call_id="a"),
        HumanMessage("injected"),
        AIMessage("external AI"),
        ToolMessage("b-failed", tool_call_id="b", status="error"),
    ]
    saved = assert_contract(events[:cut])
    resumed = assert_contract(saved + events[cut:])
    tools = [m for m in resumed if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.content, m.status) for m in tools] == [
        ("a", "a-done", "success"),
        ("b", "b-failed", "error"),
    ]


def test_id_budget_changes_preserve_original_alias_for_late_results():
    original = "very long / 호출 " * 20
    history = [AIMessage("owner", tool_calls=[call(original)])]
    for budget in [64, 40, 21, 64, 21]:
        history = assert_contract(history, budget=budget, use_oracle=False)
    result = clean_messages(
        [*history, ToolMessage("actual", tool_call_id=original)], max_call_id_length=21
    )
    assert result[1].content == "actual"


def test_legacy_function_call_cannot_pass_direct_validation():
    message = AIMessage(
        "external",
        additional_kwargs={"function_call": {"name": "lookup", "arguments": "{}"}},
    )
    for api in ("responses", "chat_completions"):
        with pytest.raises(UnsafeHistoryError):
            validate_messages([message], api=api)
