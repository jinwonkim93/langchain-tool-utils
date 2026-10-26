import copy

import pytest
from conftest import ScriptedModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_tool_utils import UnsafeHistoryError
from langchain_tool_utils.session import ApiSessionMiddleware


def make_agent(saver=None, **kwargs):
    model = ScriptedModel(responses=[AIMessage("answer")])
    return create_agent(
        model,
        [],
        middleware=[ApiSessionMiddleware("responses")],
        checkpointer=saver or InMemorySaver(),
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.parametrize(
    "first,second",
    [("responses", "chat_completions"), ("chat_completions", "responses")],
)
async def test_session_switch_resets_only_old_messages_and_same_api_retains(
    mode, first, second
):
    agent = make_agent()
    config = {"configurable": {"thread_id": "session", "api": first}}

    async def run(text):
        value = {"request_messages": [HumanMessage(text)]}
        original = copy.deepcopy(value)
        if mode == "invoke":
            agent.invoke(value, config)
        elif mode == "ainvoke":
            await agent.ainvoke(value, config)
        elif mode == "stream":
            list(agent.stream(value, config))
        else:
            [event async for event in agent.astream(value, config)]
        assert value == original
        return agent.get_state(config).values

    await run("old")
    state = await run("same-api")
    assert [m.content for m in state["messages"] if isinstance(m, HumanMessage)] == [
        "old",
        "same-api",
    ]
    config["configurable"]["api"] = second
    state = await run("new-api")
    assert [m.content for m in state["messages"] if isinstance(m, HumanMessage)] == [
        "new-api"
    ]
    assert state["session_api"] == second
    assert state["request_messages"] == []
    assert state["session_run_active"] is False
    config["configurable"]["api"] = first
    state = await run("switch-back")
    assert [m.content for m in state["messages"] if isinstance(m, HumanMessage)] == [
        "switch-back"
    ]


def test_api_identity_survives_new_agent_instance():
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "persisted", "api": "responses"}}
    make_agent(saver).invoke({"request_messages": [HumanMessage("old")]}, config)
    agent = make_agent(saver)
    config["configurable"]["api"] = "chat_completions"
    output = agent.invoke({"request_messages": [HumanMessage("new")]}, config)
    assert [m.content for m in output["messages"]] == ["new", "answer"]


def test_switch_rejected_before_resuming_interrupted_graph():
    agent = make_agent(interrupt_before=["model"])
    config = {"configurable": {"thread_id": "paused", "api": "responses"}}
    agent.invoke({"request_messages": [HumanMessage("old")]}, config)
    before = agent.get_state(config)
    config["configurable"]["api"] = "chat_completions"
    with pytest.raises(UnsafeHistoryError, match="original API"):
        agent.invoke(None, config)
    assert agent.get_state(config).values["messages"] == before.values["messages"]
    config["configurable"]["api"] = "responses"
    output = agent.invoke(None, config)
    assert output["messages"][-1].content == "answer"


def test_new_request_cannot_switch_an_active_run():
    agent = make_agent(interrupt_before=["model"])
    config = {"configurable": {"thread_id": "active", "api": "responses"}}
    agent.invoke({"request_messages": [HumanMessage("old")]}, config)
    config["configurable"]["api"] = "chat_completions"
    with pytest.raises(UnsafeHistoryError, match="original API"):
        agent.invoke({"request_messages": [HumanMessage("new")]}, config)
    state = agent.get_state(config).values
    assert state["session_api"] == "responses"
    assert [m.content for m in state["messages"]] == ["old"]


def test_plain_graph_and_consumed_request_is_not_replayed():
    from langgraph.graph.state import CompiledStateGraph

    agent = make_agent()
    assert isinstance(agent, CompiledStateGraph)
    config = {"configurable": {"thread_id": "consume"}}
    agent.invoke({"request_messages": [HumanMessage("once")]}, config)
    result = agent.invoke({"request_messages": []}, config)
    assert [m.content for m in result["messages"] if isinstance(m, HumanMessage)] == [
        "once"
    ]
    assert result["request_messages"] == []


def test_session_selection_does_not_call_cleanup(monkeypatch):
    from langchain_core.messages import ToolMessage

    from langchain_tool_utils import session

    monkeypatch.setattr(
        session, "get_config", lambda: {"configurable": {"api": "chat_completions"}}
    )
    old = AIMessage("old")
    orphan = ToolMessage("orphan", tool_call_id="no-call")
    update = ApiSessionMiddleware().before_agent(
        {"messages": [old], "session_api": "responses", "request_messages": [orphan]},
        None,
    )
    # Session policy preserves the request as supplied. Cleanup belongs elsewhere.
    assert update["messages"].value == [orphan]
    assert update["request_messages"] == []


def test_switch_requires_separate_request_messages():
    agent = make_agent()
    config = {"configurable": {"thread_id": "missing", "api": "responses"}}
    agent.invoke({"request_messages": [HumanMessage("old")]}, config)
    config["configurable"]["api"] = "chat_completions"
    with pytest.raises(ValueError, match="request_messages"):
        agent.invoke({"request_messages": []}, config)


async def test_async_resume_api_guard_at_tool_node():
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    executed = []

    @tool
    def lookup() -> str:
        """Look up a value."""
        executed.append(True)
        return "ok"

    model = ScriptedModel(
        responses=[
            AIMessage("", tool_calls=[{"id": "a", "name": "lookup", "args": {}}]),
            AIMessage("done"),
        ]
    )
    agent = create_agent(
        model,
        [lookup],
        middleware=[ApiSessionMiddleware()],
        checkpointer=InMemorySaver(),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "tools", "api": "responses"}}
    await agent.ainvoke({"request_messages": [HumanMessage("go")]}, config)
    config["configurable"]["api"] = "chat_completions"
    with pytest.raises(UnsafeHistoryError, match="original API"):
        await agent.ainvoke(None, config)
    assert executed == []
    config["configurable"]["api"] = "responses"
    result = await agent.ainvoke(None, config)
    assert executed == [True]
    assert result["session_run_active"] is False


def test_selection_then_cleanup_persisted_separately(tmp_path):
    from langchain_core.messages import ToolMessage

    from langchain_tool_utils import (
        SQLiteJournal,
        ToolHealingMiddleware,
        validate_messages,
    )

    agent = create_agent(
        ScriptedModel(responses=[AIMessage("answer")]),
        [],
        middleware=[
            ApiSessionMiddleware(),
            ToolHealingMiddleware(SQLiteJournal(tmp_path / "journal.sqlite")),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "clean", "api": "responses"}}
    state = agent.invoke(
        {
            "request_messages": [
                HumanMessage("go"),
                AIMessage(
                    "external", tool_calls=[{"id": "a", "name": "lookup", "args": {}}]
                ),
                HumanMessage("interleaved"),
                ToolMessage("orphan", tool_call_id="orphan"),
            ]
        },
        config,
    )
    validate_messages(state["messages"])
    assert [m.type for m in state["messages"]] == ["human", "ai", "tool", "human", "ai"]
    assert state["request_messages"] == []
    config["configurable"]["api"] = "chat_completions"
    state = agent.invoke(
        {
            "request_messages": [
                HumanMessage("fresh"),
                ToolMessage("late", tool_call_id="a"),
            ]
        },
        config,
    )
    assert [m.content for m in state["messages"]] == ["fresh", "answer"]
    assert agent.get_state(config).values["messages"] == state["messages"]
