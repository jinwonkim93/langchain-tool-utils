import asyncio

import pytest
from conftest import ScriptedModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_tool_utils import (
    ToolHealingMiddleware,
)


def call(key="a", name="lookup", args=None):
    return AIMessage(
        content="",
        id=f"owner-{key}",
        tool_calls=[{"id": key, "name": name, "args": args or {}}],
    )


def config(thread="test"):
    return {"configurable": {"thread_id": thread}}


def build(responses, tools, **kwargs):
    return create_agent(
        ScriptedModel(responses=responses),
        tools,
        middleware=[ToolHealingMiddleware()],
        **kwargs,
    )


def test_real_agent_success_and_error():
    executed = []

    @tool
    def lookup() -> str:
        """Look up a record."""
        executed.append(1)
        raise RuntimeError("secret API key")

    agent = build([call(), AIMessage("done")], [lookup])
    output = agent.invoke({"messages": [HumanMessage("go")]}, config())
    results = [m for m in output["messages"] if isinstance(m, ToolMessage)]
    assert executed == [1]
    assert results[0].status == "error"
    assert "secret" not in results[0].content


def test_external_calls_are_closed_without_execution():
    executed = []

    @tool
    def lookup() -> str:
        """Read a record."""
        executed.append(1)
        return "ok"

    agent = build([AIMessage("done")], [lookup])
    result = agent.invoke(
        {"messages": [call(), HumanMessage("external"), AIMessage("injected")]},
        config(),
    )
    assert executed == []
    assert result["messages"][1].status == "error"


def test_checkpoint_resume_before_tool_executes_once():
    executed = []

    @tool
    def lookup() -> str:
        """Read a record."""
        executed.append(1)
        return "ok"

    saver = InMemorySaver()
    agent = build(
        [call(), AIMessage("done")],
        [lookup],
        checkpointer=saver,
        interrupt_before=["tools"],
    )
    agent.invoke({"messages": [HumanMessage("go")]}, config())
    assert executed == []
    result = agent.invoke(None, config())
    assert executed == [1]
    assert result["messages"][-1].content == "done"


def test_new_human_after_crash_heals_checkpoint():
    class ProcessDeath(BaseException):
        pass

    @tool
    def lookup() -> str:
        """Read a record."""
        raise ProcessDeath()

    saver = InMemorySaver()
    agent = build([call()], [lookup], checkpointer=saver)
    with pytest.raises(ProcessDeath):
        agent.invoke({"messages": [HumanMessage("go")]}, config())
    resumed = build([AIMessage("done")], [lookup], checkpointer=saver)
    result = resumed.invoke({"messages": [HumanMessage("new input")]}, config())
    assert [m.type for m in result["messages"]] == [
        "human",
        "ai",
        "tool",
        "human",
        "ai",
    ]


async def test_async_tool_exception_and_cancellation():
    entered = asyncio.Event()

    @tool
    async def lookup() -> str:
        """Wait for a record."""
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    saver = InMemorySaver()
    agent = build([call(), AIMessage("done")], [lookup], checkpointer=saver)
    task = asyncio.create_task(
        agent.ainvoke({"messages": [HumanMessage("go")]}, config())
    )
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    resumed = build([AIMessage("done")], [lookup], checkpointer=saver)
    result = await resumed.ainvoke({"messages": [HumanMessage("continue")]}, config())
    assert (
        next(m for m in result["messages"] if isinstance(m, ToolMessage)).status
        == "error"
    )


def test_interrupt_propagates_and_resumes_normally():
    @tool
    def lookup() -> str:
        """Ask approval."""
        return interrupt("approve?")

    saver = InMemorySaver()
    agent = build([call(), AIMessage("done")], [lookup], checkpointer=saver)
    result = agent.invoke({"messages": [HumanMessage("go")]}, config())
    assert "__interrupt__" in result
    # Normal LangGraph interrupt semantics apply.
    resumed = agent.invoke(Command(resume="approved"), config())
    assert (
        next(m for m in resumed["messages"] if isinstance(m, ToolMessage)).content
        == "approved"
    )


def test_parallel_tools_and_invalid_arguments():
    executed = []

    @tool
    def lookup(value: int) -> str:
        """Read a number."""
        executed.append(value)
        return str(value)

    message = AIMessage(
        "",
        id="parallel",
        tool_calls=[
            {"id": "a", "name": "lookup", "args": {"value": 1}},
            {"id": "b", "name": "lookup", "args": {"value": "bad"}},
            {"id": "c", "name": "missing_tool", "args": {}},
        ],
    )
    agent = build([message, AIMessage("done")], [lookup])
    result = agent.invoke({"messages": [HumanMessage("go")]}, config())
    assert executed == [1]
    results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(results) == 3
    assert [m.status for m in results] == ["success", "error", "error"]


def test_command_result_preserves_state_update():
    from langchain.agents import AgentState
    from langchain.tools import ToolRuntime

    class State(AgentState):
        value: int

    executed = []

    @tool
    def lookup(runtime: ToolRuntime) -> Command:
        """Update graph state."""
        executed.append(1)
        return Command(
            update={
                "value": 42,
                "messages": [
                    ToolMessage(content="updated", tool_call_id=runtime.tool_call_id)
                ],
            }
        )

    saver = InMemorySaver()
    agent = build(
        [call(), AIMessage("done")],
        [lookup],
        checkpointer=saver,
        state_schema=State,
    )
    output = agent.invoke({"messages": [HumanMessage("go")]}, config())
    assert output["value"] == 42
    assert executed == [1]


async def test_async_exception_returns_error():
    @tool
    async def lookup() -> str:
        """Fail asynchronously."""
        raise ValueError("secret")

    agent = build([call(), AIMessage("done")], [lookup])
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, config())
    output = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert output.status == "error"
    assert "secret" not in output.content


def test_tool_retries_are_owned_by_caller():
    from langchain.agents.middleware import AgentMiddleware

    class RepeatHandler(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            first = handler(request)
            second = handler(request)
            assert first.content == second.content
            return second

    executed = []

    @tool
    def lookup() -> str:
        """Apply one operation."""
        executed.append(1)
        return "ok"

    agent = create_agent(
        ScriptedModel(responses=[call(), AIMessage("done")]),
        [lookup],
        middleware=[RepeatHandler(), ToolHealingMiddleware()],
    )
    agent.invoke({"messages": [HumanMessage("go")]}, config())
    assert executed == [1, 1]


def test_pending_is_saved_then_replaced_by_late_external_result():
    import json

    from langchain_tool_utils import validate_messages

    agent = build(
        [AIMessage("waiting"), AIMessage("received")],
        [],
        checkpointer=InMemorySaver(),
    )
    first = agent.invoke({"messages": [call(), HumanMessage("external")]}, config())
    pending = next(m for m in first["messages"] if isinstance(m, ToolMessage))
    assert json.loads(pending.content)["state"] == "PENDING"
    saved = agent.get_state(config()).values["messages"]
    assert pending in saved
    second = agent.invoke(
        {
            "messages": [
                ToolMessage("actual result", tool_call_id="a", id="late"),
                HumanMessage("continue"),
            ]
        },
        config(),
    )
    results = [m for m in second["messages"] if isinstance(m, ToolMessage)]
    assert len(results) == 1
    assert results[0].content == "actual result"
    validate_messages(agent.get_state(config()).values["messages"])


def test_standalone_middleware_needs_no_thread_or_database():
    agent = build([AIMessage("done")], [])
    result = agent.invoke({"messages": [HumanMessage("hello")]})
    assert result["messages"][-1].content == "done"
