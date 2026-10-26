"""Clean an interrupted disk-backed history when a new turn starts."""

import os
import subprocess
import sys
from pathlib import Path

from conftest import ScriptedModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_tool_utils import ToolHealingMiddleware


def test_process_death_then_new_turn_closes_missing_result(tmp_path):
    program = r'''
import os, sys
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from conftest import ScriptedModel
from langchain_tool_utils import ToolHealingMiddleware
root = Path(sys.argv[1])
@tool
def lookup() -> str:
    """Apply a durable side effect."""
    with (root / "effects.txt").open("a") as file:
        file.write("applied\n")
        file.flush()
        os.fsync(file.fileno())
    os._exit(73)
with SqliteSaver.from_conn_string(str(root / "graph.sqlite")) as saver:
    agent = create_agent(ScriptedModel(responses=[AIMessage("", id="owner", tool_calls=[{"id":"a","name":"lookup","args":{}}])]),
        [lookup], middleware=[ToolHealingMiddleware()], checkpointer=saver)
    agent.invoke({"messages": [("user", "go")]}, {"configurable":{"thread_id":"crash"}}, durability="sync")
'''
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")]),
    }
    process = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 73, process.stderr

    @tool
    def lookup() -> str:
        """Must not execute again."""
        raise AssertionError("Duplicate side effect")

    with SqliteSaver.from_conn_string(str(tmp_path / "graph.sqlite")) as saver:
        agent = create_agent(
            ScriptedModel(responses=[AIMessage("done")]),
            [lookup],
            middleware=[ToolHealingMiddleware()],
            checkpointer=saver,
        )
        result = agent.invoke(
            {"messages": [HumanMessage("new turn")]},
            {"configurable": {"thread_id": "crash"}},
            durability="sync",
        )
    assert (tmp_path / "effects.txt").read_text() == "applied\n"
    output = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert output.status == "error"
    assert '"state": "PENDING"' in output.content
