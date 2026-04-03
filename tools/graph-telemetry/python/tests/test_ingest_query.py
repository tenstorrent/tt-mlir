# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the graph-telemetry ingest + query layer.

Builds a synthetic v2 telemetry document that exercises every node/edge type,
ingests it into a temporary embedded Kuzu DB, and asserts the motivating
queries (dataflow patterns, attribute lookups, calls, diff) return the expected
results. No compiler build or device is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the `graph_telemetry` package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_telemetry.serve.db import KuzuDB, QueryTimeout, connect  # noqa: E402
from graph_telemetry.serve import ingest as I, service as S  # noqa: E402


def _op(i, parent, name, order=0, **kw):
    return {
        "id": i,
        "parentId": parent,
        "regionIdx": 0,
        "blockIdx": 0,
        "order": order,
        "opName": name,
        "dialect": name.split(".")[0],
        "location": kw.get("loc", f"loc:{i}"),
        "symName": kw.get("symName", ""),
        "isFunc": kw.get("isFunc", False),
        "numArgs": kw.get("numArgs", 0),
        "numResults": kw.get("numResults", 0),
        "isTerminator": kw.get("term", False),
        "attrs": kw.get("attrs", {}),
    }


def _tensor(i, prod, ptype, shape=(1, 256), **kw):
    return {
        "id": i,
        "producerId": prod,
        "producerType": ptype,
        "resultIdx": kw.get("ri", 0),
        "shape": list(shape),
        "rank": len(shape),
        "dtype": kw.get("dtype", "bf16"),
        "attrs": kw.get("attrs", {}),
    }


def _build_snapshot(sid, sidx, tag, suffix, mm_transpose):
    """A snapshot: mesh_shard -> matmul, a reshape->broadcast->reshape->permute
    chain, a const-eval function reached via ttcore.load_cached, plus a @sym
    REFERENCES edge. Node ids carry `suffix` so they are unique per snapshot
    (mirroring the serializer's per-snapshot UUIDs); `location` stays stable so
    the diff can match ops across snapshots."""

    def n(x):
        return x + suffix

    ops = [
        _op(n("m"), "", "builtin.module", loc="loc:m"),
        _op(
            n("ce"),
            n("m"),
            "func.func",
            symName="weights_const_eval_0",
            isFunc=True,
            numArgs=1,
            attrs={"tt.function_type": "const_eval"},
            loc="loc:ce",
        ),
        _op(
            n("f"),
            n("m"),
            "func.func",
            symName="forward",
            isFunc=True,
            numArgs=1,
            loc="loc:f",
        ),
        _op(n("ms"), n("f"), "ttir.mesh_shard", order=0, loc="loc:ms"),
        _op(
            n("mm"),
            n("f"),
            "ttnn.matmul",
            order=1,
            attrs={"transpose_a": mm_transpose},
            loc="loc:mm",
        ),
        _op(n("rs1"), n("f"), "ttir.reshape", order=2, loc="loc:rs1"),
        _op(n("bc"), n("f"), "ttir.broadcast", order=3, loc="loc:bc"),
        _op(n("rs2"), n("f"), "ttir.reshape", order=4, loc="loc:rs2"),
        _op(n("pm"), n("f"), "ttir.permute", order=5, loc="loc:pm"),
        _op(
            n("lc"),
            n("f"),
            "ttcore.load_cached",
            order=6,
            attrs={"callee": "@weights_const_eval_0"},
            loc="loc:lc",
        ),
        _op(n("ret"), n("f"), "func.return", order=7, term=True, loc="loc:ret"),
    ]
    block_args = [
        {"id": n("ba0"), "parentId": n("f"), "regionIdx": 0, "blockIdx": 0, "argIdx": 0}
    ]
    tensors = [
        _tensor(n("tba0"), n("ba0"), "block_arg"),
        _tensor(n("tms"), n("ms"), "op", attrs={"buffer_type": "l1", "grid": "8x8"}),
        _tensor(n("tmm"), n("mm"), "op", shape=(1, 1, 256, 256)),
        _tensor(n("trs1"), n("rs1"), "op"),
        _tensor(n("tbc"), n("bc"), "op"),
        _tensor(n("trs2"), n("rs2"), "op"),
        _tensor(n("tpm"), n("pm"), "op"),
        _tensor(n("tlc"), n("lc"), "op"),
    ]
    edges = [
        {"tensorId": n("tba0"), "consumerId": n("ms"), "operandIdx": 0},
        {"tensorId": n("tms"), "consumerId": n("mm"), "operandIdx": 0},
        {"tensorId": n("tba0"), "consumerId": n("rs1"), "operandIdx": 0},
        {"tensorId": n("trs1"), "consumerId": n("bc"), "operandIdx": 0},
        {"tensorId": n("tbc"), "consumerId": n("rs2"), "operandIdx": 0},
        {"tensorId": n("trs2"), "consumerId": n("pm"), "operandIdx": 0},
        {"tensorId": n("tmm"), "consumerId": n("ret"), "operandIdx": 0},
    ]
    calls = [{"callerId": n("lc"), "calleeId": n("ce")}]
    refs = [{"userId": n("mm"), "targetId": n("f"), "attrName": "some_sym"}]
    return {
        "snapshotId": sid,
        "tag": tag,
        "passName": tag,
        "snapshotIndex": sidx,
        "timestampUs": 1,
        "ops": ops,
        "blockArgs": block_args,
        "tensors": tensors,
        "edges": edges,
        "calls": calls,
        "refs": refs,
    }


@pytest.fixture(scope="module")
def db():
    import tempfile

    tmp = tempfile.mkdtemp()
    doc = {
        "version": 2,
        "graph": {
            "graphId": "r1",
            "modelName": "toy",
            "branch": "main",
            "createdAt": 7,
        },
        "sourcePath": "toy.mlir",
        "graphIndex": 0,
        "snapshots": [
            _build_snapshot("s0", 0, "initial", "_0", "false"),
            _build_snapshot("s1", 1, "final", "_1", "true"),
        ],
    }
    conn = connect(f"kuzu:{tmp}/t.kuzu")
    I.ingest(conn, doc)
    return conn


def _rows(db, cypher, params=None):
    return db.execute(cypher, params)


def test_mesh_shard_feeds_matmul(db):
    rows = _rows(
        db,
        "MATCH (a:Op)-[:FEEDS]->(b:Op) "
        "WHERE a.opName CONTAINS 'mesh_shard' AND b.opName CONTAINS 'matmul' "
        "RETURN a.opName AS a, b.opName AS b",
    )
    assert any(r["a"] == "ttir.mesh_shard" and r["b"] == "ttnn.matmul" for r in rows)


def test_reshape_broadcast_reshape_permute_chain(db):
    rows = _rows(
        db,
        "MATCH (r1:Op)-[:FEEDS]->(bc:Op)-[:FEEDS]->(r2:Op)-[:FEEDS]->(p:Op) "
        "WHERE r1.opName CONTAINS 'reshape' AND bc.opName CONTAINS 'broadcast' "
        "AND r2.opName CONTAINS 'reshape' AND p.opName CONTAINS 'permute' "
        "RETURN r1.id AS r1, p.id AS p",
    )
    assert any(r["r1"].startswith("rs1") and r["p"].startswith("pm") for r in rows)


def test_l1_operand_lookup_via_layout(db):
    # Layout is reified as a shared Layout node, so an L1-operand lookup anchors
    # on HAS_LAYOUT rather than the EAV Attr bag.
    rows = _rows(
        db,
        "MATCH (o:Op)<-[:CONSUMED_BY]-(t:Tensor)-[:HAS_LAYOUT]->(:Layout{bufferType:'l1'}) "
        "RETURN DISTINCT o.opName AS op",
    )
    assert {"op": "ttnn.matmul"} in [dict(r) for r in rows]


def test_typed_shape_query(db):
    rows = _rows(db, "MATCH (t:Tensor) WHERE size(t.shape) = 4 RETURN t.id AS id")
    assert {r["id"] for r in rows} == {"tmm_0", "tmm_1"}


def test_load_cached_is_a_call(db):
    # ttcore.load_cached resolves into the uniform CALLS graph.
    rows = _rows(
        db,
        "MATCH (a:Op)-[:CALLS]->(b:Op) WHERE a.opName = 'ttcore.load_cached' "
        "RETURN b.symName AS callee",
    )
    assert {"callee": "weights_const_eval_0"} in [dict(r) for r in rows]


def test_const_eval_function_retained_and_queryable(db):
    rows = _rows(
        db,
        "MATCH (o:Op)-[:HAS_ATTR]->(:Attr{key:'tt.function_type',val:'const_eval'}) "
        "RETURN o.symName AS name",
    )
    assert {"name": "weights_const_eval_0"} in [dict(r) for r in rows]


def test_references_edge(db):
    rows = _rows(db, "MATCH ()-[r:REFERENCES]->() RETURN r.attrName AS attr")
    assert {"attr": "some_sym"} in [dict(r) for r in rows]


def test_containment_and_funcs(db):
    rows = _rows(
        db,
        "MATCH (f:Op{symName:'forward'})-[:CONTAINS]->(o:Op{snapshotId:'s0'}) "
        "RETURN count(o) AS n",
    )
    assert (
        rows[0]["n"] == 8
    )  # mesh_shard, matmul, 2x reshape, broadcast, permute, load_cached, return


def test_graph_outputs_via_terminator(db):
    rows = _rows(
        db,
        "MATCH (t:Tensor{snapshotId:'s0'})-[:CONSUMED_BY]->(o:Op) WHERE o.isTerminator "
        "RETURN t.id AS id",
    )
    assert {"id": "tmm_0"} in [dict(r) for r in rows]


def test_cypher_createdat_not_rejected(db):
    # Regression: 'createdAt' must not trip the write-keyword guard.
    out = S.run_cypher(db, "MATCH (r:Graph) RETURN r.createdAt AS c")
    assert out["results"][0]["c"] == 7


def test_cypher_rejects_writes(db):
    with pytest.raises(S.BadRequest):
        S.run_cypher(db, "MATCH (o:Op) DELETE o")


def test_service_navigation(db):
    runs = S.list_graphs(db)
    assert any(r["graphId"] == "r1" for r in runs)
    snaps = S.list_graph_snapshots(db, "r1")
    assert len(snaps) == 2
    assert snaps[0]["stats"]["numOps"] == 11
    full = S.get_snapshot(db, "s0")
    assert len(full["ops"]) == 11 and len(full["calls"]) == 1


def test_ci_run_rollup_and_lazy_graphs(db):
    # The fixture's single run has no CI provenance, so it lands in the local
    # ("") bucket. The dashboard expands a bucket by re-querying /graphs with that
    # runId.
    ci_runs = S.list_runs(db)
    assert len(ci_runs) == 1
    row = ci_runs[0]
    assert row["runId"] == ""
    assert row["numGraphs"] == 1
    assert row["latestCreatedAt"] == 7
    assert row["numTests"] == 1  # numTests must survive next to max()/min()

    # Workflow provenance is surfaced in the rollup (empty here -- the fixture
    # has no CI metadata -- but the columns must be present, not dropped).
    assert row["workflowName"] == ""
    assert row["workflowTitle"] == ""

    local = S.list_graphs_for_run(db, "")
    assert [r["graphId"] for r in local] == ["r1"]
    assert S.list_graphs_for_run(db, "nonexistent") == []


def test_get_run(db):
    assert S.get_graph(db, "r1")["modelName"] == "toy"
    with pytest.raises(S.NotFound):
        S.get_graph(db, "missing")


def test_search_runs(db):
    assert [r["graphId"] for r in S.search_graphs(db, "toy")] == ["r1"]
    assert [r["graphId"] for r in S.search_graphs(db, "TOY")] == [
        "r1"
    ]  # case-insensitive
    assert [r["graphId"] for r in S.search_graphs(db, "main")] == ["r1"]  # branch field
    assert S.search_graphs(db, "nomatch") == []
    assert S.search_graphs(db, "") == []  # empty query matches nothing, not all


def test_set_ci_run_workflow():
    # Backfill uses its own DB so it doesn't mutate the shared module fixture.
    import tempfile

    conn = connect(f"kuzu:{tempfile.mkdtemp()}/bf.kuzu")
    doc = {
        "version": 2,
        "graph": {"graphId": "rb", "modelName": "m", "runId": "C1", "createdAt": 1},
        "sourcePath": "x.mlir",
        "graphIndex": 0,
        "snapshots": [_build_snapshot("bs0", 0, "initial", "_0", "false")],
    }
    I.ingest(conn, doc)

    assert S.get_graph(conn, "rb")["workflowName"] == ""
    res = S.set_run_workflow(conn, "C1", "Run Test", "Test ( Suite: x.json )")
    assert res == {"runId": "C1", "updated": 1}
    backfilled = S.get_graph(conn, "rb")
    assert backfilled["workflowName"] == "Run Test"
    assert backfilled["workflowTitle"] == "Test ( Suite: x.json )"
    # A non-matching CI run updates nothing.
    assert S.set_run_workflow(conn, "nope", "x", "y")["updated"] == 0


def test_emitter_minimal_output_round_trips():
    # The C++ emitter writes a `graph` block of only graphId/modelName/createdAt
    # -- all CI provenance is stamped later by push. Feeding exactly that minimal
    # shape through ingest must round-trip: this is the drift the schemas had.
    import tempfile

    conn = connect(f"kuzu:{tempfile.mkdtemp()}/emit.kuzu")
    doc = {
        "version": 2,
        "graph": {"graphId": "g_emit", "modelName": "jit_add", "createdAt": 42},
        "sourcePath": "jit_add.mlir",
        "graphIndex": 0,
        "snapshots": [
            _build_snapshot("es0", 0, "initial", "_0", "false"),
            _build_snapshot("es1", 1, "final", "_1", "true"),
        ],
    }
    summary = I.ingest(conn, doc)
    assert summary == {"graphId": "g_emit", "snapshotsIngested": 2}

    g = S.get_graph(conn, "g_emit")
    assert g["modelName"] == "jit_add" and g["createdAt"] == 42
    # CI fields the emitter no longer sets default to empty, not missing.
    assert g["gitSha"] == "" and g["runId"] == "" and g["branch"] == ""
    assert g["workflowName"] == "" and g["testName"] == ""

    assert [r["graphId"] for r in S.list_graphs(conn) if r["graphId"] == "g_emit"]
    assert len(S.list_graph_snapshots(conn, "g_emit")) == 2

    feeds = conn.execute(
        "MATCH (a:Op)-[:FEEDS]->(b:Op) "
        "WHERE a.opName CONTAINS 'mesh_shard' AND b.opName CONTAINS 'matmul' "
        "RETURN count(*) AS n"
    )
    assert feeds[0]["n"] >= 1

    diff = S.diff_snapshots(conn, "es0", "es1")
    mods = {m["opName"]: m["changes"] for m in diff["ops"]["modified"]}
    assert mods["ttnn.matmul"]["transpose_a"] == {"from": "false", "to": "true"}


def test_diff_detects_attr_change(db):
    diff = S.diff_snapshots(db, "s0", "s1")
    mods = {m["opName"]: m["changes"] for m in diff["ops"]["modified"]}
    assert "ttnn.matmul" in mods
    assert mods["ttnn.matmul"]["transpose_a"] == {"from": "false", "to": "true"}


# ---------------------------------------------------------------------------
# Connection pooling (Plan A)
# ---------------------------------------------------------------------------


def test_read_pool_serves_concurrent_queries(db):
    """Many queries run concurrently against the read pool, all correct.

    Checks out/returns connections across threads without leaking the pool or
    crossing results -- the count is fixed, so every worker must see it.
    """
    import concurrent.futures

    expected = db.execute("MATCH (o:Op) RETURN count(o) AS n")[0]["n"]

    def one():
        return db.execute("MATCH (o:Op) RETURN count(o) AS n")[0]["n"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = [f.result() for f in [pool.submit(one) for _ in range(50)]]

    assert results == [expected] * 50


def test_query_timeout_raises(tmp_path):
    """A query that runs past the per-query timeout surfaces as QueryTimeout."""
    db = KuzuDB(str(tmp_path / "timeout.kuzu"), read_pool_size=2, query_timeout_ms=50)
    try:
        with pytest.raises(QueryTimeout):
            # A large self-join with no early exit far outruns a 50 ms budget.
            db.execute(
                "UNWIND range(1, 100000) AS x "
                "UNWIND range(1, 100000) AS y "
                "RETURN count(*) AS n"
            )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Reified OpKind / Layout nodes + analytics (Plan B)
# ---------------------------------------------------------------------------


def test_op_kind_reified(db):
    # opName is reified: every Op has exactly one HAS_KIND edge to its OpKind.
    matmuls = _rows(
        db,
        "MATCH (:OpKind {name: 'ttnn.matmul'})<-[:HAS_KIND]-(o:Op) "
        "RETURN count(o) AS n",
    )
    assert matmuls[0]["n"] == 2  # one per snapshot

    kinds = {r["name"] for r in _rows(db, "MATCH (k:OpKind) RETURN k.name AS name")}
    assert {"ttnn.matmul", "ttir.mesh_shard", "func.func"} <= kinds


def test_layout_node_dedup(db):
    # tms carries the same layout in both snapshots -> a single shared Layout
    # node, with one HAS_LAYOUT edge per snapshot's tensor.
    layouts = _rows(db, "MATCH (l:Layout) RETURN count(l) AS n")
    assert layouts[0]["n"] == 1
    edges = _rows(db, "MATCH (:Tensor)-[:HAS_LAYOUT]->(:Layout) RETURN count(*) AS n")
    assert edges[0]["n"] == 2


def test_layout_attrs_removed_from_eav(db):
    # Layout fields no longer live as Attr nodes.
    for key in ("buffer_type", "is_tiled", "grid"):
        rows = _rows(
            db,
            "MATCH (a:Attr {key: $k}) RETURN count(a) AS n",
            {"k": key},
        )
        assert rows[0]["n"] == 0, key


def test_snapshot_view_reattaches_layout(db):
    # The snapshot view rebuilds the tensor attr bag from the Layout node.
    snap = S.get_snapshot(db, "s0")
    tms = next(t for t in snap["tensors"] if t["id"] == "tms_0")
    assert tms["attrs"]["buffer_type"] == "l1"
    assert tms["attrs"]["grid"] == "8x8"
    assert tms["attrs"]["is_tiled"] == "false"  # absent in source -> default


def test_analytics_op_kind_counts(db):
    by_graph = S.op_kind_counts(db, graph_id="r1")
    assert any(r["opName"] == "ttnn.matmul" and r["n"] == 1 for r in by_graph)

    by_run = S.op_kind_counts(db, run_id="")
    total = sum(r["n"] for r in by_run if r["opName"] == "ttnn.matmul")
    assert total == 2


def test_analytics_ops_of_kind(db):
    rows = S.ops_of_kind(db, "ttnn.matmul", run_id="")
    assert len(rows) == 2
    assert all(r["location"] == "loc:mm" for r in rows)


def test_analytics_op_layouts(db):
    rows = S.op_kind_layouts(db, "ttir.mesh_shard", run_id="")
    assert rows  # mesh_shard produces tms, which has an L1, non-tiled layout
    row = rows[0]
    assert row["bufferType"] == "l1" and row["isTiled"] is False and row["n"] == 2
