"""Durable execution fence: committed results replay; unknown outcomes never rerun."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from .history import UnsafeHistoryError

_SERDE = JsonPlusSerializer()


def encode_result(result):
    kind, data = _SERDE.dumps_typed(result)
    return json.dumps([kind, base64.b64encode(data).decode("ascii")])


def decode_result(payload):
    kind, data = json.loads(payload)
    return _SERDE.loads_typed((kind, base64.b64decode(data)))


def fingerprint(owner: AIMessage, call: dict) -> str:
    payload = [owner.id, call["name"], call["args"]]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class SQLiteJournal:
    """Use a local durable file, shared by all workers for the same thread.

    Transactions claim a call before its side effect. A crash between the side
    effect and result commit leaves `running`; this deliberately forbids retry.
    Not a distributed exactly-once guarantee or a replacement for API idempotency.
    """

    def __init__(self, path: str | Path):
        if str(path) == ":memory:":
            raise ValueError("Journal requires a durable file")
        self.path = str(path)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS calls (
                scope TEXT NOT NULL, call_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                state TEXT NOT NULL, result TEXT,
                PRIMARY KEY (scope, call_id))""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def authorize(self, scope: str, owner: AIMessage, call: dict) -> None:
        if not owner.id:
            raise UnsafeHistoryError("Model output requires a message ID")
        fp = fingerprint(owner, call)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT fingerprint FROM calls WHERE scope=? AND call_id=?",
                (scope, call["id"]),
            ).fetchone()
            if row and row[0] != fp:
                raise UnsafeHistoryError("Tool call ID collision")
            db.execute(
                "INSERT OR IGNORE INTO calls VALUES (?,?,?,'ready',NULL)",
                (scope, call["id"], fp),
            )

    def claim(
        self, scope: str, owner: AIMessage, call: dict
    ) -> tuple[str, ToolMessage | Command | None]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT fingerprint,state,result FROM calls WHERE scope=? AND call_id=?",
                (scope, call["id"]),
            ).fetchone()
            if row is None:
                return "unauthorized", None
            if row[0] != fingerprint(owner, call):
                raise UnsafeHistoryError("Tool call changed after model generation")
            if row[1] == "ready":
                db.execute(
                    "UPDATE calls SET state='running' WHERE scope=? AND call_id=?",
                    (scope, call["id"]),
                )
                return "execute", None
            return row[1], decode_result(row[2]) if row[2] else None

    def finish(
        self, scope: str, owner: AIMessage, call: dict, result: ToolMessage | Command
    ) -> None:
        payload = encode_result(result)
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE calls SET state='done',result=? WHERE scope=? AND call_id=? AND fingerprint=? AND state='running'",
                (payload, scope, call["id"], fingerprint(owner, call)),
            )
            if cursor.rowcount != 1:
                raise UnsafeHistoryError("Execution fence changed before result commit")

    def recover(self, scope: str, owner: AIMessage, call: dict) -> ToolMessage | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT fingerprint,result FROM calls WHERE scope=? AND call_id=?",
                (scope, call["id"]),
            ).fetchone()
        if not row:
            return None
        if row[0] != fingerprint(owner, call):
            raise UnsafeHistoryError("Journal/history call collision")
        result = decode_result(row[1]) if row[1] else None
        if isinstance(result, Command):
            # Reconstruct the protocol result only; historical repair must not
            # replay arbitrary state changes or graph routing from a Command.
            return next(
                (
                    m
                    for m in result.update.get("messages", [])
                    if isinstance(m, ToolMessage) and m.tool_call_id == call["id"]
                ),
                None,
            )
        return result
