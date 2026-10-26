import copy

import pytest
from conftest import ScriptedModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langchain_tool_utils import ApiSessionAgent, UnsafeHistoryError
from langchain_tool_utils.session import ApiSessionMiddleware, prepare_session_input


def make_agent(saver=None, **kwargs):
    model = ScriptedModel(responses=[AIMessage("answer")])
    graph = create_agent(
        model,
        [],
        middleware=[ApiSessionMiddleware("responses")],
        checkpointer=saver or InMemorySaver(),
        **kwargs,
    )
    return ApiSessionAgent(graph, default_api="responses", has_checkpointer=True)


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
        value = {"messages": [HumanMessage(text)]}
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
    config["configurable"]["api"] = first
    state = await run("switch-back")
    assert [m.content for m in state["messages"] if isinstance(m, HumanMessage)] == [
        "switch-back"
    ]


def test_api_identity_survives_new_agent_instance():
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "persisted", "api": "responses"}}
    make_agent(saver).invoke({"messages": [HumanMessage("old")]}, config)
    agent = make_agent(saver)
    config["configurable"]["api"] = "chat_completions"
    output = agent.invoke({"messages": [HumanMessage("new")]}, config)
    assert [m.content for m in output["messages"]] == ["new", "answer"]


def test_switch_rejected_before_resuming_interrupted_graph():
    agent = make_agent(interrupt_before=["model"])
    config = {"configurable": {"thread_id": "paused", "api": "responses"}}
    agent.invoke({"messages": [HumanMessage("old")]}, config)
    before = agent.get_state(config)
    config["configurable"]["api"] = "chat_completions"
    for value in [None, {"messages": [HumanMessage("new")]}]:
        with pytest.raises(UnsafeHistoryError, match="original API"):
            agent.invoke(value, config)
        assert agent.get_state(config).values == before.values
    config["configurable"]["api"] = "responses"
    output = agent.invoke(None, config)
    assert output["messages"][-1].content == "answer"


def test_unknown_legacy_api_resets_existing_history():
    output = prepare_session_input(
        {"messages": [("user", "new")], "custom": 42},
        previous_api=None,
        api="responses",
        has_history=True,
        interrupted=False,
    )
    assert output["messages"][0] == RemoveMessage(id=REMOVE_ALL_MESSAGES)
    assert output["messages"][1].content == "new"
    assert output["custom"] == 42


@pytest.mark.parametrize(
    "value", [{"session_api": "responses"}, {"messages": [RemoveMessage(id="old")]}]
)
def test_switch_rejects_managed_metadata_and_removal_directives(value):
    with pytest.raises(ValueError):
        prepare_session_input(
            value,
            previous_api="responses",
            api="chat_completions",
            has_history=True,
            interrupted=False,
        )


def test_no_implicit_bypass_via_batch_or_event_stream():
    agent = make_agent()
    for method in ("batch", "abatch", "astream_events", "with_config"):
        with pytest.raises(AttributeError):
            getattr(agent, method)


def test_switch_without_new_input_does_not_send_empty_request():
    with pytest.raises(ValueError, match="at least one new message"):
        prepare_session_input(
            {},
            previous_api="responses",
            api="chat_completions",
            has_history=True,
            interrupted=False,
        )
