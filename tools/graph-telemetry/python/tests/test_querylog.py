# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-query JSONL log: records carry user/endpoint context,
execution time, row counts, and error status; context survives the executor
hop in `submit`."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Make the `graph_telemetry` package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_telemetry.serve import querylog  # noqa: E402
from graph_telemetry.serve.db import connect  # noqa: E402


@pytest.fixture
def db(tmp_path):
    db = connect(f"kuzu:{tmp_path}/db.kuzu", read_pool_size=2)
    db.ensure_schema()
    yield db
    db.close()


@pytest.fixture
def log_lines(tmp_path):
    log_path = tmp_path / "query_log.jsonl"
    querylog.init(str(log_path))
    yield lambda: [
        json.loads(line) for line in log_path.read_text().splitlines() if line
    ]
    querylog.init(None)


def test_read_query_logged_with_context(db, log_lines):
    querylog.request_ctx.set(
        {"user": "alice", "endpoint": "POST /cypher", "requestId": "req1"}
    )
    db.execute("MATCH (g:Graph) RETURN g.graphId AS id")
    (rec,) = log_lines()
    assert rec["user"] == "alice"
    assert rec["endpoint"] == "POST /cypher"
    assert rec["requestId"] == "req1"
    assert rec["query"].startswith("MATCH (g:Graph)")
    assert rec["queryHash"] == querylog.query_hash(
        "MATCH (g:Graph)  RETURN g.graphId AS id"
    )
    assert rec["status"] == "ok"
    assert rec["write"] is False
    assert rec["rows"] == 0
    assert rec["durationMs"] >= 0


def test_write_and_param_keys(db, log_lines):
    db.execute_write(
        "CREATE (:Graph {graphId: $id, testName: $name})", {"name": "t", "id": "g1"}
    )
    (rec,) = log_lines()
    assert rec["write"] is True
    assert rec["paramsKeys"] == ["id", "name"]


def test_error_status(db, log_lines):
    with pytest.raises(Exception):
        db.execute("MATCH (x:NoSuchTable) RETURN x")
    (rec,) = log_lines()
    assert rec["status"] == "error"
    assert rec["rows"] == 0


def test_context_propagates_through_submit(db, log_lines):
    async def run():
        querylog.request_ctx.set(
            {"user": "bob", "endpoint": "GET /graphs", "requestId": "req2"}
        )
        await db.submit(db.execute, "MATCH (g:Graph) RETURN g.graphId")

    asyncio.run(run())
    (rec,) = log_lines()
    assert rec["user"] == "bob"


def test_disabled_log_is_noop(db, tmp_path):
    querylog.init(None)
    db.execute("MATCH (g:Graph) RETURN g.graphId")
    assert not (tmp_path / "query_log.jsonl").exists()
