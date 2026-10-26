"""Exercise real ChatOpenAI serializers/parsers and create_agent against fake HTTP."""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_tool_utils import (
    SQLiteJournal,
    StreamIdleTimeout,
    UnsafeHistoryError,
    create_healing_agent,
    watch_stream,
)


def response_body(output, *, status="completed"):
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "status": status,
        "error": None,
        "incomplete_details": None,
        "output": output,
        "usage": None,
    }


def chat_body(message, finish="stop"):
    return {
        "id": "chatcmpl_fixture",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }


def setup_agent(tmp_path, handler, *, streaming=False, async_handler=None):
    executed = []

    @tool
    def lookup(value: int) -> str:
        """Look up a number."""
        executed.append(value)
        return str(value * 2)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(async_handler or handler)
    )
    agent = create_healing_agent(
        model="test-model",
        tools=[lookup],
        journal=SQLiteJournal(tmp_path / "calls.sqlite"),
        checkpointer=InMemorySaver(),
        model_options={
            "api_key": "test-key",
            "base_url": "https://fixture.invalid/v1",
            "http_client": client,
            "http_async_client": async_client,
            "streaming": streaming,
        },
    )
    return agent, executed, client, async_client


@pytest.mark.parametrize(
    "first,second",
    [("responses", "chat_completions"), ("chat_completions", "responses")],
)
def test_switch_api_same_checkpoint_and_actual_wire_format(tmp_path, first, second):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        # Only the first model turn invokes a tool.
        tool_turn = len(requests) == 1
        if request.url.path.endswith("/responses"):
            output = (
                [
                    {
                        "type": "function_call",
                        "id": "fc_fixture",
                        "call_id": "call_fixture",
                        "name": "lookup",
                        "arguments": '{"value":3}',
                        "status": "completed",
                    }
                ]
                if tool_turn
                else [
                    {
                        "type": "message",
                        "id": "msg_fixture",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "done", "annotations": []}
                        ],
                    }
                ]
            )
            body = response_body(output)
            body["id"] = f"resp_fixture_{len(requests)}"
            return httpx.Response(200, json=body)
        message = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_fixture",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"value":3}'},
                    }
                ],
            }
            if tool_turn
            else {"role": "assistant", "content": "done"}
        )
        return httpx.Response(
            200, json=chat_body(message, "tool_calls" if tool_turn else "stop")
        )

    agent, executed, client, _ = setup_agent(tmp_path, handler)
    try:
        config = {"configurable": {"thread_id": "switch", "api": first}}
        result = agent.invoke(
            {"request_messages": [HumanMessage("go")]}, config, durability="sync"
        )
        assert [
            m.content for m in result["messages"] if isinstance(m, ToolMessage)
        ] == ["6"]
        config["configurable"]["api"] = second
        agent.invoke(
            {"request_messages": [HumanMessage("again")]}, config, durability="sync"
        )
        assert executed == [3]
        assert requests[-1][0].endswith(
            "/responses" if second == "responses" else "/chat/completions"
        )
        payload = requests[-1][1]
        assert "previous_response_id" not in payload
        history = payload.get("messages", payload.get("input"))
        encoded = json.dumps(history)
        assert "call_fixture" not in encoded
        assert "again" in encoded
        assert "go" not in encoded
        assert agent.get_state(config).values["session_api"] == second
        assert not any(
            isinstance(m, ToolMessage)
            for m in agent.get_state(config).values["messages"]
        )
        assert "fc_fixture" not in encoded
        assert "resp_fixture" not in encoded
    finally:
        client.close()


def chat_chunk(*, finish=None):
    delta = (
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_partial",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"value":3}'},
                }
            ],
        }
        if finish is None
        else {}
    )
    return {
        "id": "chatcmpl_partial",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def sse(value):
    return f"data: {json.dumps(value)}\n\n".encode()


async def test_chat_sse_eof_without_terminal_never_executes_complete_json(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(chat_chunk()),
        )

    agent, executed, client, async_client = setup_agent(
        tmp_path, handler, streaming=True
    )
    try:
        with pytest.raises(UnsafeHistoryError, match="terminal"):
            await agent.ainvoke(
                {"request_messages": [HumanMessage("go")]},
                {"configurable": {"thread_id": "partial", "api": "chat_completions"}},
            )
        assert executed == []
    finally:
        client.close()
        await async_client.aclose()


async def test_responses_sse_eof_without_completed_never_executes(tmp_path):
    item = {
        "id": "fc_partial",
        "type": "function_call",
        "call_id": "call_partial",
        "name": "lookup",
        "arguments": '{"value":3}',
        "status": "in_progress",
    }
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": response_body([], status="in_progress"),
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": item,
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": "fc_partial",
            "delta": '{"value":3}',
        },
    ]

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(map(sse, events)),
        )

    agent, executed, client, async_client = setup_agent(
        tmp_path, handler, streaming=True
    )
    try:
        with pytest.raises(UnsafeHistoryError):
            await agent.ainvoke(
                {"request_messages": [HumanMessage("go")]},
                {"configurable": {"thread_id": "partial", "api": "responses"}},
            )
        assert executed == []
    finally:
        client.close()
        await async_client.aclose()


async def test_sse_stall_closes_transport_and_does_not_commit_model(tmp_path):
    closed = asyncio.Event()

    class StalledStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse(chat_chunk())
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=StalledStream()
        )

    agent, executed, client, async_client = setup_agent(
        tmp_path, None, streaming=True, async_handler=handler
    )
    config = {"configurable": {"thread_id": "stall", "api": "chat_completions"}}
    try:
        events = agent.astream(
            {"request_messages": [HumanMessage("go")]},
            config,
            stream_mode="messages",
            durability="sync",
        )
        with pytest.raises(StreamIdleTimeout):
            async for _ in watch_stream(events, idle_timeout=0.05, total_timeout=1):
                pass
        assert closed.is_set()
        assert executed == []
        state = await agent.aget_state(config)
        assert all(m.type != "ai" for m in state.values["messages"])
    finally:
        client.close()
        await async_client.aclose()


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
def test_truncated_nonstream_response_never_executes(tmp_path, api):
    def handler(request):
        if api == "responses":
            body = response_body(
                [
                    {
                        "id": "fc_partial",
                        "type": "function_call",
                        "call_id": "partial",
                        "name": "lookup",
                        "arguments": '{"value":3}',
                        "status": "completed",
                    }
                ],
                status="incomplete",
            )
        else:
            body = chat_body(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "partial",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"value":3}'},
                        }
                    ],
                },
                "length",
            )
        return httpx.Response(200, json=body)

    agent, executed, client, _ = setup_agent(tmp_path, handler)
    try:
        with pytest.raises(UnsafeHistoryError, match="terminal"):
            agent.invoke(
                {"request_messages": [HumanMessage("go")]},
                {"configurable": {"thread_id": "partial", "api": api}},
            )
        assert executed == []
    finally:
        client.close()


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
async def test_successful_sse_tool_then_final_text(tmp_path, api):
    requests = []

    def handler(request):
        requests.append(request)
        tool_turn = len(requests) == 1
        if api == "chat_completions":
            if tool_turn:
                chunks = [chat_chunk(), chat_chunk(finish="tool_calls")]
            else:
                chunk = chat_chunk(finish="stop")
                chunk["choices"][0]["delta"] = {"role": "assistant", "content": "done"}
                chunks = [chunk]
            data = b"".join(map(sse, chunks)) + b"data: [DONE]\n\n"
        else:
            item = (
                {
                    "id": "fc_sse",
                    "type": "function_call",
                    "call_id": "call_sse",
                    "name": "lookup",
                    "arguments": '{"value":3}',
                    "status": "completed",
                }
                if tool_turn
                else {
                    "id": "msg_sse",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []}
                    ],
                }
            )
            body = response_body([item])
            body["id"] = f"resp_sse_{len(requests)}"
            events = [
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**body, "status": "in_progress", "output": []},
                }
            ]
            if tool_turn:
                events += [
                    {
                        "type": "response.output_item.added",
                        "sequence_number": 1,
                        "output_index": 0,
                        "item": {**item, "arguments": "", "status": "in_progress"},
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": 2,
                        "output_index": 0,
                        "item_id": "fc_sse",
                        "delta": '{"value":3}',
                    },
                    {
                        "type": "response.output_item.done",
                        "sequence_number": 3,
                        "output_index": 0,
                        "item": item,
                    },
                ]
            else:
                events += [
                    {
                        "type": "response.output_item.added",
                        "sequence_number": 1,
                        "output_index": 0,
                        "item": {**item, "content": [], "status": "in_progress"},
                    },
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": 2,
                        "output_index": 0,
                        "content_index": 0,
                        "item_id": "msg_sse",
                        "delta": "done",
                    },
                ]
            events.append(
                {"type": "response.completed", "sequence_number": 4, "response": body}
            )
            data = b"".join(map(sse, events))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=data
        )

    agent, executed, client, async_client = setup_agent(
        tmp_path, handler, streaming=True
    )
    try:
        result = await agent.ainvoke(
            {"request_messages": [HumanMessage("go")]},
            {"configurable": {"thread_id": "complete", "api": api}},
        )
        assert executed == [3]
        assert len(requests) == 2
        assert result["messages"][-1].text == "done"
    finally:
        client.close()
        await async_client.aclose()
