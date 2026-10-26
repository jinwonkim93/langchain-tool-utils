"""Run with PYTHONPATH=src .venv/bin/python examples/clean_messages.py."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_tool_utils import clean_messages

messages = [
    HumanMessage("주문 상태 확인", id="user-1"),
    AIMessage(
        "",
        id="ai-1",
        tool_calls=[
            {"id": "call-1", "name": "lookup", "args": {"order_id": "123"}},
            {"id": "call-2", "name": "lookup", "args": {"order_id": "456"}},
        ],
    ),
    ToolMessage("오래된 결과", tool_call_id="call-1"),
    HumanMessage("외부에서 추가한 메시지", id="user-2"),
    ToolMessage("최신 결과", tool_call_id="call-1"),
    ToolMessage("연결할 호출이 없음", tool_call_id="orphan"),
]
messages = clean_messages(messages)
assert clean_messages(messages) == messages
for message in messages:
    message.pretty_print()
