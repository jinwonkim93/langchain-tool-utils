import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import ScriptedModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_tool_utils import (
    SQLiteJournal,
    ToolHealingMiddleware,
    UnsafeHistoryError,
)


def call(key="a", name="lookup", args=None):
    return AIMessage(
        content="",
        id=f"owner-{key}",
        tool_calls=[{"id": key, "name": name, "args": args or {}}],
    )


def config(thread="test"):
    return {"configurable": {"thread_id": thread}}


@pytest.fixture
def journal(tmp_path):
    return SQLiteJournal(tmp_path / "journal.sqlite")


def build(journal, responses, tools, **kwargs):
    return create_agent(
        ScriptedModel(responses=responses),
        tools,
        middleware=[ToolHealingMiddleware(journal)],
        **kwargs,
    )


def test_real_agent_success_and_error(journal):
    executed = []

    @tool
    def lookup() -> str:
        """Look up a record."""
        executed.append(1)
        raise RuntimeError("secret API key")

    agent = build(journal, [call(), AIMessage("done")], [lookup])
    output = agent.invoke({"messages": [HumanMessage("go")]}, config())
    results = [m for m in output["messages"] if isinstance(m, ToolMessage)]
    assert executed == [1]
    assert results[0].status == "error"
    assert "secret" not in results[0].content


def test_external_calls_are_closed_without_execution(journal):
    executed = []

    @tool
    def lookup() -> str:
        """Read a record."""
        executed.append(1)
        return "ok"

    agent = build(journal, [AIMessage("done")], [lookup])
    result = agent.invoke(
        {"messages": [call(), HumanMessage("external"), AIMessage("injected")]},
        config(),
    )
    assert executed == []
    assert result["messages"][1].status == "error"


def test_checkpoint_resume_before_tool_executes_once(journal):
    executed = []

    @tool
    def lookup() -> str:
        """Read a record."""
        executed.append(1)
        return "ok"

    saver = InMemorySaver()
    agent = build(
        journal,
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


def test_crash_after_side_effect_never_reexecutes(journal):
    executed = []

    class ProcessDeath(BaseException):
        pass

    @tool
    def lookup() -> str:
        """Change a record."""
        executed.append(1)
        raise ProcessDeath()

    saver = InMemorySaver()
    agent = build(journal, [call(), AIMessage("done")], [lookup], checkpointer=saver)
    with pytest.raises(ProcessDeath):
        agent.invoke({"messages": [HumanMessage("go")]}, config())
    # Reconstruct both graph and journal to model a process restart.
    restarted = build(
        SQLiteJournal(journal.path), [AIMessage("done")], [lookup], checkpointer=saver
    )
    result = restarted.invoke(None, config())
    assert executed == [1]
    assert (
        next(m for m in result["messages"] if isinstance(m, ToolMessage)).status
        == "error"
    )


def test_new_human_after_crash_heals_checkpoint(journal):
    class ProcessDeath(BaseException):
        pass

    @tool
    def lookup() -> str:
        """Read a record."""
        raise ProcessDeath()

    saver = InMemorySaver()
    agent = build(journal, [call()], [lookup], checkpointer=saver)
    with pytest.raises(ProcessDeath):
        agent.invoke({"messages": [HumanMessage("go")]}, config())
    resumed = build(journal, [AIMessage("done")], [lookup], checkpointer=saver)
    result = resumed.invoke({"messages": [HumanMessage("new input")]}, config())
    assert [m.type for m in result["messages"]] == [
        "human",
        "ai",
        "tool",
        "human",
        "ai",
    ]


def test_committed_result_recovered_without_execution(journal):
    owner = call()
    key = "4:test:"
    journal.authorize(key, owner, owner.tool_calls[0])
    assert journal.claim(key, owner, owner.tool_calls[0])[0] == "execute"
    journal.finish(
        key,
        owner,
        owner.tool_calls[0],
        ToolMessage(content="durable", tool_call_id="a"),
    )
    agent = build(journal, [AIMessage("done")], [])
    result = agent.invoke({"messages": [owner, HumanMessage("recover")]}, config())
    assert result["messages"][1].content == "durable"


def test_parallel_claim_is_atomic_and_scopes_isolated(journal):
    owner = call()
    entry = owner.tool_calls[0]
    journal.authorize("one", owner, entry)
    with ThreadPoolExecutor(max_workers=8) as pool:
        states = list(
            pool.map(lambda _: journal.claim("one", owner, entry)[0], range(16))
        )
    assert states.count("execute") == 1
    assert states.count("running") == 15
    assert journal.claim("two", owner, entry)[0] == "unauthorized"
    changed = {**entry, "args": {"other": True}}
    with pytest.raises(UnsafeHistoryError):
        journal.claim("one", owner, changed)


async def test_async_tool_exception_and_cancellation(journal):
    entered = asyncio.Event()

    @tool
    async def lookup() -> str:
        """Wait for a record."""
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    saver = InMemorySaver()
    agent = build(journal, [call(), AIMessage("done")], [lookup], checkpointer=saver)
    task = asyncio.create_task(
        agent.ainvoke({"messages": [HumanMessage("go")]}, config())
    )
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    resumed = build(journal, [AIMessage("done")], [lookup], checkpointer=saver)
    result = await resumed.ainvoke(None, config())
    assert (
        next(m for m in result["messages"] if isinstance(m, ToolMessage)).status
        == "error"
    )


def test_interrupt_propagates_and_requires_reconciliation(journal):
    @tool
    def lookup() -> str:
        """Ask approval."""
        return interrupt("approve?")

    saver = InMemorySaver()
    agent = build(journal, [call(), AIMessage("done")], [lookup], checkpointer=saver)
    result = agent.invoke({"messages": [HumanMessage("go")]}, config())
    assert "__interrupt__" in result
    # Tool entry already ran; replaying an interrupting tool is unsafe by default.
    resumed = agent.invoke(Command(resume="approved"), config())
    assert (
        next(m for m in resumed["messages"] if isinstance(m, ToolMessage)).status
        == "error"
    )


def test_state_injected_at_tools_node_cannot_gain_execution(journal):
    executed = []

    @tool
    def lookup() -> str:
        """Read a record."""
        executed.append(1)
        return "ok"

    saver = InMemorySaver()
    agent = build(journal, [AIMessage("done")], [lookup], checkpointer=saver)
    agent.update_state(
        config(), {"messages": [HumanMessage("external"), call()]}, as_node="model"
    )
    result = agent.invoke(None, config())
    assert executed == []
    assert (
        next(m for m in result["messages"] if isinstance(m, ToolMessage)).status
        == "error"
    )


def test_parallel_tools_and_invalid_arguments(journal):
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
    agent = build(journal, [message, AIMessage("done")], [lookup])
    result = agent.invoke({"messages": [HumanMessage("go")]}, config())
    assert executed == [1]
    results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(results) == 3
    assert [m.status for m in results] == ["success", "error", "error"]


def test_command_result_replay_preserves_state_update(journal):
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
        journal,
        [call(), AIMessage("done")],
        [lookup],
        checkpointer=saver,
        state_schema=State,
    )
    output = agent.invoke({"messages": [HumanMessage("go")]}, config())
    assert output["value"] == 42
    owner = call()
    state, saved = journal.claim("4:test:", owner, owner.tool_calls[0])
    assert state == "done"
    assert isinstance(saved, Command)
    assert saved.update["value"] == 42
    assert journal.recover("4:test:", owner, owner.tool_calls[0]).content == "updated"
    assert executed == [1]


async def test_async_exception_returns_error(journal):
    @tool
    async def lookup() -> str:
        """Fail asynchronously."""
        raise ValueError("secret")

    agent = build(journal, [call(), AIMessage("done")], [lookup])
    result = await agent.ainvoke({"messages": [HumanMessage("go")]}, config())
    output = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert output.status == "error"
    assert "secret" not in output.content


def test_thread_id_is_required(journal):
    agent = build(journal, [AIMessage("done")], [])
    with pytest.raises(ValueError, match="thread_id"):
        agent.invoke({"messages": [HumanMessage("go")]})


def test_outer_middleware_reinvocation_cannot_execute_twice(journal):
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
        middleware=[RepeatHandler(), ToolHealingMiddleware(journal)],
    )
    agent.invoke({"messages": [HumanMessage("go")]}, config())
    assert executed == [1]


def test_pending_is_saved_then_replaced_by_late_external_result(journal):
    import json

    from langchain_tool_utils import validate_messages

    agent = build(
        journal,
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
