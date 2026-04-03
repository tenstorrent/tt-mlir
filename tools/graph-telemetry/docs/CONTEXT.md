# Graph Telemetry — Context for implementers

Shared background for the optimization plans in this directory
([plan-a-connection-pooling.md](plan-a-connection-pooling.md),
[plan-b-schema-optimizations.md](plan-b-schema-optimizations.md)). Read this
first.

## What the system is

A telemetry subsystem that captures the compiler IR graph at points during
compilation and exposes it as a queryable graph DB for coding agents and a
dashboard.

- **C++ emitter** (tt-mlir): serializes ordered IR snapshots to JSON during the
  pass pipeline. Also runs from the tt-xla PJRT compile path (that is where CI
  telemetry comes from).
- **Python tooling** lives in `tools/graph-telemetry/python/`, package
  `graph_telemetry`:
  - `graph_telemetry.serve` — FastAPI app + embedded **Kuzu** graph DB. Serves a
    REST API + static dashboard. Integration model is plain HTTP (`curl` +
    self-describing `GET /schema`), **not** MCP. Do not add MCP.
  - `graph_telemetry.push` — uploads CI telemetry archives to a running serve.
- The DB is a **derived, rebuildable index**. The JSON + `.mlir` artifacts are
  the source of truth. Any schema change = bump schema, build a **fresh** DB,
  re-ingest. There is no in-place migration and none is needed.

## Code map (all paths under `tools/graph-telemetry/python/graph_telemetry/`)

- `serve/db.py` — `GraphDB` abstract base + `KuzuDB` backend. **Currently holds
  a single `kuzu.Connection` (`self._conn`) shared by every request.**
- `serve/queries.py` — `KUZU_CREATE_NODE_TABLES`, `KUZU_CREATE_REL_TABLES`
  (the DDL), and `SCHEMA_DESCRIPTION` (returned by `GET /schema`).
- `serve/ingest.py` — `ingest`, `ingest_many`, `graph_exists`. Bulk-loads nodes
  via Kuzu `COPY ... FROM df` (pandas).
- `serve/service.py` — all query logic; every function takes a `GraphDB` as its
  first arg. Endpoints are thin wrappers over these.
- `serve/app.py` — FastAPI app. `app.state.db` holds the single `GraphDB`.
  Endpoints are `async def` that call the **synchronous** `service.*` functions
  directly (so DB work runs on the event loop).
- `serve/__main__.py` — CLI entry (`python -m graph_telemetry.serve ...`).

## Data model (current schema)

Node tables: `Graph` (PK `graphId`), `Snapshot` (PK `snapshotId`), `Op`
(PK `id`), `BlockArg` (PK `id`), `Tensor` (PK `id`), `Attr` (PK `id`).
Every `Op`/`Tensor`/`BlockArg` row also denormalizes `graphId` and `snapshotId`
as scalar string properties.

Rel tables: `IN_GRAPH (Snapshot→Graph)`, `IN_SNAPSHOT (Op/BlockArg/Tensor→
Snapshot)`, `CONTAINS (Op→Op)`, `HAS_ARG`, `PRODUCES (Op→Tensor)`,
`CONSUMED_BY (Tensor→Op)`, `FEEDS`, `CALLS (Op→Op)`, `REFERENCES (Op→Op)`,
`HAS_ATTR (Op→Attr, Tensor→Attr)`.

Hierarchy: `Graph ←IN_GRAPH← Snapshot ←IN_SNAPSHOT← Op`. An op's enclosing
function is found via `o.parentId` → the `func.func` `Op`. Layout info
(`is_tiled`, `buffer_type`, `tensor_memory_layout`, `shard_shape`, `grid`) is
currently stored as `Attr` nodes linked to `Tensor` via `HAS_ATTR`.

## Measured scale (the numbers that justify these plans)

Current local DB (`/tmp/gt3.kuzu`, 4 runs / 677 graphs):

| node table | count |
|---|---|
| Graph | 677 |
| Snapshot | 4,062 (~6 / graph) |
| Op | 11,623,621 (~17k / graph, ~2.9k / snapshot) |
| Tensor | 11,560,626 |
| BlockArg | 1,829,905 |
| Attr | 32,531 |

DB on disk: 7.5 GB. Artifact files: 16 GB.

**Target scale:** ~10 concurrent users; ~100 runs × ~100 graphs × ~5 snapshots.
That is ~10,000 graphs ≈ **14.8× current** → ~172M `Op`, ~172M `Tensor`,
~370M nodes, ~1B edges, ~110 GB DB, ~240 GB files.

## Performance findings (ground truth — measure, don't assume)

All timings from a direct read-only `kuzu.Connection` to the real
`/tmp/gt3.kuzu` (Kuzu 0.11.3):

| query | time |
|---|---|
| `MATCH (o:Op) RETURN count(o)` (11.6M) | 0.018 s |
| full scan + `WHERE o.opName='ttnn.mesh_partition'` | 0.088 s |
| `MATCH (o:Op {graphId:'<const>'}) RETURN count(o)` | 0.059 s |
| traversal from Graph PK (`Graph←IN_GRAPH←Snapshot←IN_SNAPSHOT←Op`) | 0.051 s |
| 3-join query scoped to one **constant** graphId | 0.364 s |
| **full-run, `{graphId: g.graphId}` correlated var join + 3 traversal joins + ORDER BY** | **183 s** |
| same logical query, scoped via `IN_GRAPH`/`IN_SNAPSHOT` traversal | **1.4 s** |

Conclusions, verified:

1. **Scans are fast.** 11.6M-row scan+filter = 88 ms; ~1.3 s even at 15× scale.
   Missing secondary indexes are **not** the bottleneck at this scale.
2. **The 183 s pathology** comes from a planner-hostile shape: matching `Op` by a
   **correlated scalar variable** (`{graphId: g.graphId}`) combined with several
   traversal joins and `ORDER BY ... LIMIT`. Rewriting the same query as a
   relationship traversal from the `Graph` primary key is **130× faster** (1.4 s).
3. **Kuzu 0.11.3 has no `CREATE INDEX`** (confirmed: `CREATE INDEX`,
   `ALTER TABLE ... ADD INDEX`, `CALL CREATE_INDEX` all rejected). The idiomatic
   substitute is **reifying a hot property as a node** (PK lookup + adjacency).
   Microbenchmark: reified-node lookup 3 ms vs property scan 18 ms over 2M rows.
4. **Kuzu supports concurrent read connections.** One `kuzu.Database` +
   N `kuzu.Connection`s; 4 parallel scans completed in 0.28 s wall for 4× 0.27 s
   queries — real parallelism. `connection.set_query_timeout(ms)` exists and works.
5. **The serve serializes everything today:** a single shared `Connection`, called
   synchronously inside `async def` handlers, single uvicorn worker, no query
   timeout. One slow query blocks all users and can run unbounded. This is the
   acute, engine-independent problem.

**Decision: stay on Kuzu.** The evidence does not justify an engine switch. The
fixes are (A) serve concurrency + query timeout + disciplined query formulation,
and (B) reify hot properties + cut layout duplication.

## Gotchas

- Kuzu is **single-writer, single-process** for read-write. Do not run two RW
  serves on one DB. Reads can be concurrent (pool of connections in one process);
  multiple OS processes can only open the DB **read-only**.
- Schema changes require a **fresh** DB (`CREATE TABLE IF NOT EXISTS` will not add
  columns/tables to an existing DB). Rebuild by re-ingesting artifacts.
- `service.py`/`queries.py`/`ingest.py`/`db.py` changes require a serve
  **restart**. Static dashboard files are served fresh from disk.
- Cypher reserved word: `in` cannot be a variable name (use `src`, etc.).
- `push` parses the **trailing `-<digits>` token** of each artifact dir name as
  the GitHub job id; keep that contract if you touch ingest naming.
- Run the serve in the background; confirm up by polling
  `curl -sf http://localhost:8321/graphs`.

## Local dev quickstart

```
cd /localdev/dmilinkovic/tt-mlir && source env/activate
cd tools/graph-telemetry/python
python -m pytest tests/                      # unit tests
# run serve (background), fresh or existing DB:
python -m graph_telemetry.serve --db kuzu:/tmp/gtX.kuzu --files-dir /tmp/gtX-files \
  --port 8321 --log-level WARNING &
```
