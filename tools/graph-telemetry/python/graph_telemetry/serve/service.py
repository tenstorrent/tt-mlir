# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Transport-agnostic query logic over a GraphDB.

Holds the telemetry queries (runs, snapshots, diff, cypher). Depends only on a
`GraphDB` and returns plain Python data, so the FastAPI app is a thin transport
over it and the same logic could back any other caller.

Errors are raised as `NotFound` / `BadRequest`; callers map them to their own
transport.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .db import GraphDB
from . import queries as Q


class NotFound(Exception):
    """A requested entity does not exist."""


class BadRequest(Exception):
    """The request is malformed or disallowed (e.g. a write Cypher query)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(row: dict, key: str) -> dict:
    """Extract node properties from a result row, dropping Kuzu internals."""
    val = row.get(key, row)
    if isinstance(val, dict):
        if "_properties" in val:
            val = val["_properties"]
        return {k: v for k, v in val.items() if not k.startswith("_")}
    return row


def _attr_map(db: GraphDB, query: str, snapshot_id: str) -> dict[str, dict]:
    """Group HAS_ATTR rows into {ownerId: {key: val}}."""
    out: dict[str, dict] = defaultdict(dict)
    for row in db.execute(query, {"snapshotId": snapshot_id}):
        out[row["ownerId"]][row["key"]] = row["val"]
    return out


def _layout_map(db: GraphDB, snapshot_id: str) -> dict[str, dict]:
    """Group HAS_LAYOUT rows back into per-tensor attr bags.

    Layout lives on a shared Layout node now, but the snapshot/diff view treats
    it as part of the tensor's attribute bag, so reattach it under the original
    emitter key names. isTiled is reconstructed as the string 'true'/'false';
    the string fields are included only when non-empty (the emitter omits them
    when not applicable).
    """
    out: dict[str, dict] = {}
    field_to_key = {
        "bufferType": "buffer_type",
        "memoryLayout": "tensor_memory_layout",
        "shardShape": "shard_shape",
        "grid": "grid",
    }
    for row in db.execute(Q.GET_SNAPSHOT_TENSOR_LAYOUTS, {"snapshotId": snapshot_id}):
        bag = {"is_tiled": "true" if row["isTiled"] else "false"}
        for field, key in field_to_key.items():
            if row[field]:
                bag[key] = row[field]
        out[row["ownerId"]] = bag
    return out


def _diff_dicts(a: dict, b: dict) -> dict:
    """Compare two dicts, returning changed fields as {key: {from, to}}."""
    changes = {}
    for key in set(a) | set(b):
        if a.get(key) != b.get(key):
            changes[key] = {"from": a.get(key), "to": b.get(key)}
    return changes


def _load_snapshot_data(db: GraphDB, snapshot_id: str) -> dict:
    """Load a full snapshot, reattaching the EAV attribute bags."""
    meta_rows = db.execute(Q.GET_SNAPSHOT_META, {"snapshotId": snapshot_id})
    if not meta_rows:
        raise NotFound(f"Snapshot {snapshot_id} not found")
    meta = _node(meta_rows[0], "s")

    op_attrs = _attr_map(db, Q.GET_SNAPSHOT_OP_ATTRS, snapshot_id)
    tensor_attrs = _attr_map(db, Q.GET_SNAPSHOT_TENSOR_ATTRS, snapshot_id)
    tensor_layouts = _layout_map(db, snapshot_id)

    ops = []
    for row in db.execute(Q.GET_SNAPSHOT_OPS, {"snapshotId": snapshot_id}):
        o = _node(row, "o")
        o["attrs"] = dict(op_attrs.get(o.get("id", ""), {}))
        ops.append(o)

    block_args = [
        _node(row, "b")
        for row in db.execute(Q.GET_SNAPSHOT_BLOCKARGS, {"snapshotId": snapshot_id})
    ]

    tensors = []
    for row in db.execute(Q.GET_SNAPSHOT_TENSORS, {"snapshotId": snapshot_id}):
        t = _node(row, "t")
        tid = t.get("id", "")
        attrs = dict(tensor_attrs.get(tid, {}))
        attrs.update(tensor_layouts.get(tid, {}))
        t["attrs"] = attrs
        tensors.append(t)

    edges = db.execute(Q.GET_SNAPSHOT_EDGES, {"snapshotId": snapshot_id})
    calls = db.execute(Q.GET_SNAPSHOT_CALLS, {"snapshotId": snapshot_id})
    refs = db.execute(Q.GET_SNAPSHOT_REFS, {"snapshotId": snapshot_id})

    return {
        **meta,
        "ops": ops,
        "blockArgs": block_args,
        "tensors": tensors,
        "edges": edges,
        "calls": calls,
        "refs": refs,
    }


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def list_graphs(db: GraphDB) -> list[dict]:
    return [_node(row, "r") for row in db.execute(Q.LIST_GRAPHS)]


def get_graph(db: GraphDB, graph_id: str) -> dict:
    rows = db.execute(Q.GET_GRAPH, {"graphId": graph_id})
    if not rows:
        raise NotFound(f"Graph {graph_id} not found")
    return _node(rows[0], "r")


def list_runs(db: GraphDB) -> list[dict]:
    """Roll up graphs by run (CI run) for the dashboard's landing page.

    Each row is one run (empty runId is the "local" bucket): graph count,
    distinct test count, the latest graph's timestamp, and a representative
    branch. numTests comes from a separate query and is merged here (see
    LIST_RUNS for why).
    """
    rows = db.execute(Q.LIST_RUNS)
    test_counts = {r["runId"]: r["numTests"] for r in db.execute(Q.RUN_TEST_COUNTS)}
    for r in rows:
        r["numTests"] = test_counts.get(r["runId"], 0)
    return rows


def list_graphs_for_run(db: GraphDB, run_id: str) -> list[dict]:
    """List the graphs of one run; an empty id selects local graphs."""
    return [
        _node(row, "r") for row in db.execute(Q.LIST_GRAPHS_FOR_RUN, {"runId": run_id})
    ]


def set_run_workflow(
    db: GraphDB, run_id: str, workflow_name: str, workflow_title: str
) -> dict:
    """Backfill a run's workflow name/title across all its graphs."""
    # This is a write (SET); route it through the serialized write connection,
    # not the read pool.
    rows = db.execute_write(
        Q.SET_RUN_WORKFLOW,
        {
            "runId": run_id,
            "workflowName": workflow_name or "",
            "workflowTitle": workflow_title or "",
        },
    )
    return {"runId": run_id, "updated": rows[0]["updated"] if rows else 0}


def search_graphs(db: GraphDB, q: str) -> list[dict]:
    """Substring-search graphs across model/test/run/workflow/branch/sha."""
    q = (q or "").strip().lower()
    if not q:
        return []
    return [_node(row, "r") for row in db.execute(Q.SEARCH_GRAPHS, {"q": q})]


def list_graph_snapshots(db: GraphDB, graph_id: str) -> list[dict]:
    """List snapshots for a graph, with summary stats per snapshot."""
    rows = db.execute(Q.LIST_SNAPSHOTS_FOR_GRAPH, {"graphId": graph_id})
    if not rows:
        raise NotFound(f"Graph {graph_id} not found or has no snapshots")

    results = []
    for row in rows:
        snap = _node(row, "s")
        sid = snap.get("snapshotId", "")

        op_count = db.execute(Q.SNAPSHOT_OP_COUNT, {"snapshotId": sid})
        tensor_count = db.execute(Q.SNAPSHOT_TENSOR_COUNT, {"snapshotId": sid})
        dialects = db.execute(Q.SNAPSHOT_DIALECT_BREAKDOWN, {"snapshotId": sid})

        results.append(
            {
                "snapshotId": sid,
                "tag": snap.get("tag", ""),
                "passName": snap.get("passName", ""),
                "snapshotIndex": snap.get("snapshotIndex", 0),
                "sourcePath": snap.get("sourcePath", ""),
                "graphIndex": snap.get("graphIndex", 0),
                "mlirPath": snap.get("mlirPath", ""),
                "stats": {
                    "numOps": op_count[0]["numOps"] if op_count else 0,
                    "numTensors": tensor_count[0]["numTensors"] if tensor_count else 0,
                    "dialectBreakdown": {r["dialect"]: r["n"] for r in dialects},
                },
            }
        )
    return results


def get_snapshot(db: GraphDB, snapshot_id: str) -> dict:
    return _load_snapshot_data(db, snapshot_id)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def diff_snapshots(db: GraphDB, a: str, b: str) -> dict:
    """Diff two snapshots, matching ops by source location.

    Location matching is approximate: ops sharing a location are matched
    positionally within that location bucket. (Robust cross-snapshot identity
    is a separate, planned mechanism.)
    """
    snap_a = _load_snapshot_data(db, a)
    snap_b = _load_snapshot_data(db, b)

    def bucket(ops: list[dict]) -> dict[tuple, dict]:
        seen: Counter = Counter()
        out = {}
        for op in ops:
            loc = op.get("location", "")
            if not loc:
                continue
            key = (loc, seen[loc])
            seen[loc] += 1
            out[key] = op
        return out

    ops_a, ops_b = bucket(snap_a["ops"]), bucket(snap_b["ops"])
    added = [ops_b[k] for k in ops_b if k not in ops_a]
    removed = [ops_a[k] for k in ops_a if k not in ops_b]
    modified = []
    for k in ops_a:
        if k not in ops_b:
            continue
        changes = _diff_dicts(ops_a[k].get("attrs", {}), ops_b[k].get("attrs", {}))
        if ops_a[k].get("opName") != ops_b[k].get("opName"):
            changes["opName"] = {
                "from": ops_a[k].get("opName"),
                "to": ops_b[k].get("opName"),
            }
        if changes:
            modified.append(
                {
                    "location": k[0],
                    "opName": ops_b[k].get("opName", ""),
                    "changes": changes,
                }
            )

    # Tensor encoding/shape changes, keyed by (producer location, resultIdx).
    def tensor_bucket(snap: dict) -> dict[tuple, dict]:
        loc = {op["id"]: op.get("location", "") for op in snap["ops"]}
        out = {}
        for t in snap["tensors"]:
            ploc = loc.get(t.get("producerId", ""), "")
            if ploc:
                out[(ploc, t.get("resultIdx", 0))] = t
        return out

    ta, tb = tensor_bucket(snap_a), tensor_bucket(snap_b)
    encoding_changed = []
    for k in ta:
        if k not in tb:
            continue
        diff = _diff_dicts(ta[k].get("attrs", {}), tb[k].get("attrs", {}))
        if ta[k].get("shape") != tb[k].get("shape"):
            diff["shape"] = {"from": ta[k].get("shape"), "to": tb[k].get("shape")}
        if diff:
            encoding_changed.append({"location": k[0], "changes": diff})

    return {
        "ops": {"added": added, "removed": removed, "modified": modified},
        "tensors": {
            "added": [tb[k] for k in tb if k not in ta],
            "removed": [ta[k] for k in ta if k not in tb],
            "encodingChanged": encoding_changed,
        },
        "summary": {
            "opsAdded": len(added),
            "opsRemoved": len(removed),
            "opsModified": len(modified),
            "encodingsChanged": len(encoding_changed),
        },
    }


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

# Match write keywords only as whole words, so that read-only queries selecting
# columns like `createdAt` are not falsely rejected.
_WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DROP|REMOVE|COPY|ALTER|INSTALL|ATTACH|EXPORT)\b"
)


def run_cypher(
    db: GraphDB,
    query: str,
    params: dict | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict:
    """Execute a read-only Cypher query with pagination."""
    query = query.strip()
    match = _WRITE_RE.search(query.upper())
    if match:
        raise BadRequest(
            f"Write operation '{match.group(1)}' not allowed in read-only endpoint"
        )

    limit = limit or 500
    offset = offset or 0
    upper = query.upper()

    # Fetch one extra row to detect a next page without a second full-store scan
    # for an exact total. If the caller supplied their own LIMIT we leave the
    # query untouched and can't probe past it, so hasMore stays False.
    paginated = query
    probed = "LIMIT" not in upper
    if probed:
        if "SKIP" not in upper and offset > 0:
            paginated += f" SKIP {offset}"
        paginated += f" LIMIT {limit + 1}"

    results = db.execute(paginated, params)

    has_more = probed and len(results) > limit
    if has_more:
        results = results[:limit]

    return {
        "results": results,
        "hasMore": has_more,
        "limit": limit,
        "offset": offset,
    }


def get_schema(db: GraphDB) -> dict:
    return db.get_schema()


# ---------------------------------------------------------------------------
# Reified-node analytics (B1/B2)
#
# All helpers here scope by traversing from a primary key (Graph.graphId or
# OpKind.name), never by a {graphId: <variable>} correlated-scalar join -- that
# shape is what produced the 183 s pathology (see docs/CONTEXT.md).
# ---------------------------------------------------------------------------


def op_kind_counts(
    db: GraphDB, graph_id: str | None = None, run_id: str | None = None
) -> list[dict]:
    """Op-kind counts, scoped to one graph (graph_id) or a whole run (run_id).

    A graph is scoped by a PK traversal from Graph; a run by traversing from
    each of its Graph PKs. Exactly one scope must be given.
    """
    if graph_id:
        return db.execute(Q.OPKIND_COUNTS_FOR_GRAPH, {"graphId": graph_id})
    if run_id is not None:
        return db.execute(Q.OPKIND_COUNTS_FOR_RUN, {"runId": run_id})
    raise BadRequest("op_kind_counts requires either graphId or runId")


def ops_of_kind(db: GraphDB, op_name: str, run_id: str) -> list[dict]:
    """Every op of `op_name` across a run, anchored on the OpKind PK."""
    if not op_name:
        raise BadRequest("opName is required")
    return db.execute(Q.OPS_OF_KIND_FOR_RUN, {"opName": op_name, "runId": run_id})


def op_kind_layouts(db: GraphDB, op_name: str, run_id: str) -> list[dict]:
    """Layout breakdown (tiled/buffer type) for one op kind, per test in a run."""
    if not op_name:
        raise BadRequest("opName is required")
    return db.execute(Q.OPKIND_LAYOUT_FOR_RUN, {"opName": op_name, "runId": run_id})
