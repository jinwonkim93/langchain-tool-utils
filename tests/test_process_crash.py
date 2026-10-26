"""An actual process exit with disk-backed graph and execution journals."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ScriptedModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_tool_utils import SQLiteJournal, ToolHealingMiddleware


@pytest.mark.parametrize("crash_point", ["inside_tool", "after_journal_commit"])
def test_process_death_and_disk_resume(tmp_path, crash_point):
    program = r'''
import os, sys
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from conftest import ScriptedModel
from langchain_tool_utils import SQLiteJournal, ToolHealingMiddleware
root = Path(sys.argv[1])
point = sys.argv[2]
class CrashAfterCommit(SQLiteJournal):
    def finish(self, *args):
        super().finish(*args)
        os._exit(73)
journal = (CrashAfterCommit if point == "after_journal_commit" else SQLiteJournal)(root / "journal.sqlite")
@tool
def lookup() -> str:
    """Apply a durable side effect."""
    with (root / "effects.txt").open("a") as file:
        file.write("applied\n")
        file.flush()
        os.fsync(file.fileno())
    if point == "inside_tool":
        os._exit(73)
    return "committed"
with SqliteSaver.from_conn_string(str(root / "graph.sqlite")) as saver:
    agent = create_agent(ScriptedModel(responses=[AIMessage("", id="owner", tool_calls=[{"id":"a","name":"lookup","args":{}}])]),
        [lookup], middleware=[ToolHealingMiddleware(journal)], checkpointer=saver)
    agent.invoke({"messages": [("user", "go")]}, {"configurable":{"thread_id":"crash"}}, durability="sync")
'''
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")]),
    }
    process = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), crash_point],
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
            middleware=[
                ToolHealingMiddleware(SQLiteJournal(tmp_path / "journal.sqlite"))
            ],
            checkpointer=saver,
        )
        result = agent.invoke(
            None, {"configurable": {"thread_id": "crash"}}, durability="sync"
        )
    assert (tmp_path / "effects.txt").read_text() == "applied\n"
    output = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    if crash_point == "after_journal_commit":
        assert output.content == "committed"
        assert output.status == "success"
    else:
        assert output.status == "error"
