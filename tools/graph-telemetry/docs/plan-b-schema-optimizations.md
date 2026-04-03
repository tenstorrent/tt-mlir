# Plan B — Schema optimizations

> Read [CONTEXT.md](CONTEXT.md) first. Land **after** Plan A. This makes
> cross-run analytics O(result) without secondary indexes, cuts storage, and
> bakes in query formulation that can't trip the planner pathology.
>
> The DB is derived and rebuildable: every change here = bump schema → fresh DB
> → re-ingest from artifacts. No in-place migration.

## Why (verified — see CONTEXT.md)

- Kuzu 0.11.3 has **no `CREATE INDEX`**. The idiomatic substitute for "index a
  property" is to **reify the property as a node** (PK lookup + adjacency-list
  traversal). Microbenchmark: 3 ms reified lookup vs 18 ms scan over 2M rows; the
  gap widens with scale and, more importantly, reified access is **immune to the
  correlated-scalar-join planner blowup** that caused the 183 s query.
- Layout attributes are **massively duplicated**: 11.6M `Tensor`s each carry
  ~5 `Attr` rows for layout, but the number of *distinct* layouts per graph is
  tiny. Reifying layout dedups this — a storage and ingest win.
- The 183 s pathology must not recur: queries must scope from primary keys via
  relationships, never via `{graphId: <variable>}` correlated joins.

## B1 — Reify `opName` → `OpKind` node  ("index on op name")

- New node table: `OpKind { name STRING, PRIMARY KEY(name) }`.
- New rel table: `HAS_KIND (FROM Op TO OpKind)`.
- Query rewrite: cross-run "all ops named X" goes from a scan to
  `MATCH (:OpKind {name:'ttnn.mesh_partition'})<-[:HAS_KIND]-(o:Op)` — PK lookup +
  adjacency. ~ms, planner-stable.

## B2 — Reify tensor layout → `Layout` node  ("index on tensor properties")

- New node table:
  `Layout { key STRING, isTiled BOOL, bufferType STRING, memoryLayout STRING, shardShape STRING, grid STRING, PRIMARY KEY(key) }`
  where `key` is a deterministic hash/concat of the layout fields.
- New rel table: `HAS_LAYOUT (FROM Tensor TO Layout)`.
- **Replace** the per-tensor layout `Attr` rows (`is_tiled`, `buffer_type`,
  `tensor_memory_layout`, `shard_shape`, `grid`) with a single shared `Layout`
  node per distinct layout. (Keep `Attr`/`HAS_ATTR` for genuinely
  per-op/per-tensor attributes that are not layout; only layout moves out.)
- Query rewrite: "all tiled tensors" / "all L1 tensors" become node-anchored:
  `MATCH (l:Layout {isTiled:true})<-[:HAS_LAYOUT]-(t:Tensor) ...`.

## B3 — Canonical scoped-traversal helpers + `/schema` guidance

- In `service.py`, add helper(s) that always scope by traversal from a primary
  key: `Graph {graphId}` `←IN_GRAPH← Snapshot ←IN_SNAPSHOT← Op` (and the
  `OpKind`/`Layout` anchors from B1/B2 for cross-run analytics). Route the
  dashboard and canned queries through these.
- **Forbid** `MATCH (o:Op {graphId: <variable>})` correlated-scalar joins. Audit
  `service.py` for any remaining ones and rewrite.
- Update `SCHEMA_DESCRIPTION` in `queries.py` (returned by `GET /schema`) with:
  - the new `OpKind`/`Layout` nodes and `HAS_KIND`/`HAS_LAYOUT` edges,
  - 2–3 example traversal queries (per-graph op listing; cross-run op-kind
    lookup; tiled-tensor lookup),
  - an explicit note: *"Scope by traversing from `Graph`/`OpKind`/`Layout`
    primary keys via relationships. Do NOT filter `Op`/`Tensor` by the `graphId`
    scalar inside a join with other traversals — it triggers a catastrophic plan
    (observed 183 s vs 1.4 s for the traversal form)."*
- (Optional B3b) Add a direct `OF_GRAPH (Op→Graph)` / `(Tensor→Graph)` edge for
  one-hop scoping. Low priority: the 2-hop traversal is already ~50 ms.

## File-by-file changes (all under `serve/`)

### `queries.py`
- `KUZU_CREATE_NODE_TABLES`: add `OpKind`, `Layout`.
- `KUZU_CREATE_REL_TABLES`: add `HAS_KIND`, `HAS_LAYOUT`.
- `SCHEMA_DESCRIPTION`: document new nodes/edges + example traversals + the
  "no correlated scalar join" guidance.

### `ingest.py`
- Build distinct `OpKind` rows from the op stream; `COPY` them; build `HAS_KIND`
  edges (Op.id → opName).
- Compute a `Layout.key` per distinct tensor layout; `COPY` distinct `Layout`
  rows; build `HAS_LAYOUT` edges (Tensor.id → layout key).
- Stop emitting the layout `Attr` rows now represented by `Layout`.
- Keep the trailing-job-id artifact-name contract intact (CONTEXT.md gotcha).

### `service.py`
- Add the PK-traversal query helpers (B3).
- Add analytics helpers anchored on `OpKind`/`Layout` (e.g. "op-kind counts per
  test across a run", "tiled vs row-major per test") used by the dashboard/agents.
- Rewrite any layout-attr queries to use `HAS_LAYOUT`.

## Validation

- Re-ingest **one** run into a fresh DB. Compare node/edge counts and DB size to
  the current schema (expect fewer `Attr`, new `OpKind`/`Layout`, smaller DB).
- Benchmark, against the fresh DB, the queries used in analysis (must be ms–low-s):
  - "all `ttnn.mesh_partition` across the run" (via `OpKind`),
  - "tiled vs row-major mesh_partition per test" (via `Layout` + `OpKind`),
  - per-graph op listing (via PK traversal).
- Confirm all dashboard endpoints and `tests/` pass against the new schema.
- Re-run the regression: the old `{graphId: <variable>}` shape must no longer be
  produced by any `service.py` helper.

## Acceptance criteria

- Cross-run op-kind / layout analytics return in **ms–low seconds** at current
  scale, with no full `Op`/`Tensor` scan in the plan.
- `Attr` count drops substantially (layout moved to shared `Layout` nodes);
  DB-on-disk shrinks for the same data.
- `GET /schema` documents the reified model and steers clients to traversals.

## Effort & risk

~1–2 days incl. re-ingest. Low risk: derived DB, rebuildable from artifacts; can
be validated on one run before reprocessing everything.

## Deferred (not in this plan — see the discussion in CONTEXT history)

- **Op dedup across snapshots** (content-hash shared ops): attacks the ~6×
  snapshot duplication; larger change.
- **Ingest speedup**: 46 min/run today (per-archive HTTP + single writer) —
  batch `COPY` / parallelize. Offline path, not user-facing.
- **Retention**: keep last N runs hot, archive older to the `.mlir` files; caps
  the working set to the rolling 100-run window.
