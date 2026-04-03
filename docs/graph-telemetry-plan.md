# mlir-graph-telemetry — Architecture (as built)

## Context

A tt-mlir tool that captures the IR graph at points of interest during
compilation and exposes it as a queryable graph database for coding agents.
Everything lives in the tt-mlir repo. The compiler emits JSON telemetry files;
separate Python tools ingest them into a graph DB and serve an HTTP query API. A
coding agent queries that API directly (e.g. `curl`), bootstrapping from the
self-describing `GET /schema`; the dashboard is a read-only view over the same
endpoints.

The design goal is to let an agent answer questions like:

- *Why didn't fusing pattern X fire between these two snapshots?*
- *Are there any `mesh_shard` ops feeding into matmuls?*
- *Are there any `reshape -> broadcast -> reshape -> permute` chains?*

### Graph vs. job vs. run

Three grouping concepts, kept deliberately separate (`run → job → test → graph`):

- A **graph** is one compilation of one model — the unit the C++ emitter writes
  (one JSON file, an ordered list of snapshots). It is the `Graph` node.
- A **job** is one CI matrix leg (one device/arch + category, e.g. *"test n150
  … and large (tt-ubuntu-2204-n150-stable, run_forge_models, 4)"*). One job runs
  pytest and produces many graphs. It is *not* a node: it is the `jobId`/`jobName`
  columns on `Graph`. The job is what splits a single test's graphs across
  devices — the same model compiled on n150 vs p150 are distinct graphs under
  distinct jobs, differing only in the target `system_desc`.
- A **run** is a CI run that fanned out into many jobs. It is the `runId` column,
  surfaced by the `/runs` rollup. Graphs pushed without CI provenance fall into
  the empty-string "local" bucket (and skip the job level in the dashboard).

The C++ emitter records compilation facts — `graphId`, `modelName`, `createdAt`,
and `testName` (the test/invocation that triggered this compile, which only the
frontend knows at emit time). Run-level CI/GitHub provenance — `runId`,
`branch`, `gitSha`, `workflowName`, `workflowTitle` — is the same across every
graph in a run and is stamped later by `mlir-graph-push`. One GitHub-aware
layer, in Python.

---

## Architecture

```
ttmlir-opt model.mlir \
  --graph-telemetry-dir=/tmp/telemetry \
  --graph-telemetry-graph-id="my-graph" \
  --ttir-to-ttnn-backend-pipeline
    │
    ▼
/tmp/telemetry/my-graph.json   (+ /tmp/telemetry/my-graph/*.mlir sidecars)
    │
    ├─ mlir-graph-push *.json --url http://serve:8321   (CI: stamps provenance)
    │
    ├─ mlir-graph-serve --db kuzu:... --ingest-dir ...   (local dev)
    │
    ▼
mlir-graph-serve (FastAPI) → Kuzu (local) or shared graph DB (CI)
    │
    ▼ (HTTP: REST + dashboard)
coding agent (curl) / browser
```

The same emitter runs from `ttmlir-opt` and from the tt-xla frontend (it is a
small, stable C++ API: `TTGraphTelemetrySession` + `TTGraphTelemetryOptions`). A
compilation that spans several PassManagers (e.g. SHLO, SHLO→TTIR, TTIR→TTNN)
writes a single graph via a shared session.

The query surface is plain HTTP (`service.py` over a `GraphDB`), so a coding
agent needs nothing installed: it hits the endpoints with `curl`, starting at
`GET /schema` to learn the node/edge model and example queries, then `POST
/cypher` for arbitrary traversals.

---

## Design principles

1. **The core is generic MLIR.** The serializer is a uniform reflection of
   `Operation → Region → Block → {Operation, BlockArgument}`. It contains no
   dialect names and no op-specific logic. Modules and functions are just ops.
2. **TT knowledge lives only in visitors.** A small plugin set decorates ops and
   tensors with TT-specific, decomposed fields. Removing them yields a correct,
   dialect-agnostic graph.
3. **Everything is queryable by default.** Open-ended attributes are first-class
   `Attr` nodes (EAV), so any attribute is directly matchable in Cypher without
   knowing the schema up front. Only universal, typed fields (tensor
   shape/rank/dtype, op flags) are top-level columns.
4. **The emitter is CI-agnostic.** It records compilation facts only; provenance
   is a push concern (see *Graph vs. run*).

---

## Data Model

Six node types. Modules, functions, and ops collapse into a single `Op` — the
hierarchy is carried entirely by the `CONTAINS` edge plus region/block indices.

### Node types

**`:Graph`** — one compilation of one model
```
graphId (PK), modelName, testName, createdAt  — set by the emitter
gitSha, runId, branch,                         — empty from the emitter;
jobId, jobName,                                  stamped by mlir-graph-push
workflowName, workflowTitle                      (jobId/jobName per artifact)
```

**`:Snapshot`** — IR state captured at a point of interest
```
snapshotId (PK), graphId, tag, passName, snapshotIndex, timestampUs,
sourcePath, graphIndex, mlirPath
```

**`:Op`** — every operation, modules and functions included
```
id (PK), snapshotId, graphId, parentId
regionIdx, blockIdx, orderIdx     — position within the parent block
opName, dialect, location
mlirLine                          — 1-based line of this op in the snapshot's printed MLIR
symName, isFunc, numArgs, numResults, isTerminator   — generic, interface-derived
```
`mlirLine` is captured for free during serialization via MLIR's
`AsmState::LocationMap` (the printer records each op's output line), so a query
result can be turned into a dashboard deep link straight to the op's line.
`symName` comes from `SymbolOpInterface`; `isFunc`/`numArgs`/`numResults` from
`FunctionOpInterface`; `isTerminator` from the trait. No dialect names are ever
checked.

**`:BlockArg`** — a block argument (function params, region/loop vars)
```
id (PK), snapshotId, graphId, parentId, regionIdx, blockIdx, argIdx
```

**`:Tensor`** — a first-class SSA value (op result or block-arg output)
```
id (PK), snapshotId, graphId, producerId, producerType, resultIdx
shape (INT64[]), rank, dtype          — typed, always present
```
Non-tensor values (e.g. `!ttnn.device`) are also represented here with empty
shape; "Tensor" means "SSA value".

**`:Attr`** — a shared, content-addressed attribute (EAV)
```
id (PK = sha1(key, val)), key, val
```
Every op/tensor attribute (except elided bulk data) becomes one `Attr`, shared
across all owners with the same `(key, val)`. Visitor-decomposed fields (e.g.
`buffer_type`, `grid`) land here too, so they are queryable uniformly.

`graphId` is denormalized onto `Op`/`BlockArg`/`Tensor`/`Snapshot` so
graph-scoped queries are a column filter rather than a traversal back to the
`Graph` node.

### Relationships

```
(:Snapshot)-[:IN_GRAPH]->(:Graph)
(:Op | :BlockArg | :Tensor)-[:IN_SNAPSHOT]->(:Snapshot)
(:Op)-[:CONTAINS {regionIdx, blockIdx, orderIdx}]->(:Op)
(:Op)-[:HAS_ARG {regionIdx, blockIdx, argIdx}]->(:BlockArg)
(:Op | :BlockArg)-[:PRODUCES {resultIdx}]->(:Tensor)
(:Tensor)-[:CONSUMED_BY {operandIdx}]->(:Op)
(:Op | :BlockArg)-[:FEEDS {operandIdx, resultIdx}]->(:Op)   — derived shortcut
(:Op)-[:CALLS]->(:Op)                — CallOpInterface or a `callee` symbol attr
(:Op)-[:REFERENCES {attrName}]->(:Op)                        — other symbol refs
(:Op | :Tensor)-[:HAS_ATTR]->(:Attr)
```

`FEEDS` is a derived producer→consumer shortcut over
`PRODUCES`/`CONSUMED_BY`, materialized at ingest so dataflow pattern queries are
half as long:

```
(:Op)-[:FEEDS]->(:Op)   ==   (:Op)-[:PRODUCES]->(:Tensor)-[:CONSUMED_BY]->(:Op)
```

`CALLS` is the uniform call graph. It is populated from `CallOpInterface` (e.g.
`func.call`) *or* from the conventional `callee` symbol attribute, so call-like
ops that don't implement the interface — notably `ttcore.load_cached`, which
invokes a cached const-eval function — appear as calls too. Other symbol-ref
attributes become `REFERENCES`. Const-eval functions are **not** filtered out,
so these call edges resolve to real callee nodes; they remain identifiable via
their `tt.function_type = "const_eval"` attribute in the attr bag.

Graph outputs are not a separate concept: they are the tensors
`CONSUMED_BY` an op whose `isTerminator` is true (symmetric with block-arg
inputs).

### Hierarchy

```
Graph
 └─ Snapshot[] (one per point of interest, ordered by snapshotIndex)
      └─ Op tree (CONTAINS), BlockArgs, Tensors, dataflow edges, calls, refs
```

There is no separate pipeline entity — the ordered list of snapshots within a
graph captures pipeline evolution.

---

## Visitor Pattern (C++)

The serializer walks the op tree and calls visitors to decide what to keep and
how to represent it. The base class provides dialect-agnostic behavior:

```cpp
class TelemetryVisitor {
public:
  // Per-op: populate attrs; return false to skip this op AND its subtree.
  virtual bool visitOp(Operation *op, OpData &data) { return true; }

  // Per-value (result or block arg): base extracts shape/dtype/rank.
  virtual void visitValue(Value value, TensorData &data);

  // Per-attribute: return false to suppress default stringification.
  virtual bool visitAttr(StringRef name, Attribute attr,
                         llvm::StringMap<std::string> &attrs) { return true; }
};
```

The serializer itself handles dense-attr elision (bulk `ElementsAttr` data is
summarized, never stringified) and generic interface fields before visitors run.

### tt-mlir visitors (the entire TT surface)

- **`TTNNLayoutVisitor`** — `visitValue` decomposes `TTNNLayoutAttr` into
  `buffer_type`, `tensor_memory_layout`, `grid`, `shard_shape`, `is_tiled`
  (written into the tensor attr bag).
- **`MemoryConfigVisitor`** — `visitAttr` decomposes `MemoryConfigAttr` into
  `<name>.buffer_type` / `<name>.tensor_memory_layout`.

Visitors are wired up in `RegisterVisitors.cpp`. No visitor prunes ops: the
graph is a complete reflection of the IR (const-eval functions included).
`visitOp`-based filtering remains supported by the interface if a future visitor
wants it.

To enrich coverage (e.g. mesh/shard specifics, conv configs, reshape/permute
dims) add visitors here — never in the core serializer.

### Registry chaining

- `visitOp`: first visitor returning false short-circuits (op + subtree skipped).
- `visitValue`: all visitors run in order, each augments the data.
- `visitAttr`: first visitor returning false stops the chain (attr handled).

---

## JSON Telemetry Format (version 2)

The emitter writes a `graph` block of compilation facts only (`graphId`,
`modelName`, `testName`, `createdAt`). Run-level provenance fields are absent
here and added by `mlir-graph-push` before ingest.

```json
{
  "version": 2,
  "graph": { "graphId": "my-graph", "modelName": "jit_add",
             "testName": "tests/test_mnist.py::test_inference[bf16]",
             "createdAt": 1711900000000 },
  "sourcePath": "model.mlir",
  "graphIndex": 0,
  "snapshots": [
    {
      "snapshotId": "uuid", "tag": "initial", "passName": "initial",
      "snapshotIndex": 0, "timestampUs": 1711900000000,
      "mlirPath": "my-graph/0_initial.mlir",
      "ops": [
        { "id": "uuid", "parentId": "uuid",
          "regionIdx": 0, "blockIdx": 0, "order": 1,
          "opName": "ttnn.linear", "dialect": "ttnn", "location": "model.py:42",
          "mlirLine": 17,
          "symName": "", "isFunc": false, "numArgs": 0, "numResults": 0,
          "isTerminator": false,
          "attrs": {"transpose_a": "false"} }
      ],
      "blockArgs": [
        { "id": "uuid", "parentId": "uuid",
          "regionIdx": 0, "blockIdx": 0, "argIdx": 0 }
      ],
      "tensors": [
        { "id": "uuid", "producerId": "uuid", "producerType": "op",
          "resultIdx": 0, "shape": [1, 256], "dtype": "bf16", "rank": 2,
          "attrs": {"buffer_type": "dram", "grid": "1x1", "is_tiled": "true"} }
      ],
      "edges":  [ { "tensorId": "uuid", "consumerId": "uuid", "operandIdx": 0 } ],
      "calls":  [ { "callerId": "uuid", "calleeId": "uuid" } ],
      "refs":   [ { "userId": "uuid", "targetId": "uuid", "attrName": "..." } ]
    }
  ]
}
```

Each graph is written as `<graphId>.json`; the printed MLIR for each snapshot is
written alongside under `<graphId>/<index>_<tag>.mlir` and referenced by
`mlirPath`. `mlir-graph-push` preserves this layout so the sidecars survive the
upload and `/files` can serve them.

---

## File Layout

```
tt-mlir/
├── include/ttmlir/Telemetry/
│   ├── GraphTypes.h           — GraphMeta, SnapshotData, OpData, TensorData, ...
│   ├── IRSerializer.h         — serialize(Operation*, registry) → SnapshotData
│   ├── TelemetryVisitor.h     — visitor interface + registry
│   └── TelemetryWriter.h      — accumulate snapshots, write JSON
├── include/ttmlir/Support/
│   └── TTGraphTelemetryInstrumentation.h
├── lib/Telemetry/
│   ├── IRSerializer.cpp       — uniform recursive walk, calls/refs resolution
│   ├── TelemetryWriter.cpp    — UUID gen + JSON output
│   ├── TelemetryVisitor.cpp   — base visitValue + registry chaining
│   └── Visitors/              — TT-specific plugin layer
│       ├── TTNNLayoutVisitor.cpp
│       ├── MemoryConfigVisitor.cpp
│       └── RegisterVisitors.cpp
├── lib/Support/
│   └── TTGraphTelemetryInstrumentation.cpp   — PassManager instrumentation
└── tools/graph-telemetry/python/
    ├── pyproject.toml
    └── graph_telemetry/
        ├── serve/   — __main__ (CLI), app (FastAPI), db (Kuzu), ingest,
        │             queries, service, static/ (dashboard)
        └── push/    — mlir-graph-push CLI
```

The telemetry libraries are `add_mlir_library` targets exported in
`TTMLIRTargets`, and `TTMLIRSupport` links them publicly, so every consumer of
the instrumentation resolves the symbols.

The Python tooling is a single `graph_telemetry` namespace package. Console
scripts: `mlir-graph-serve`, `mlir-graph-push`. From a source checkout without
installing, run the modules from `tools/graph-telemetry/python` (e.g. `python -m
graph_telemetry.serve`).

---

## Storage / indexing rationale

- **EAV over MAP.** Kuzu `MAP` columns require `map_extract(o.attrs,'k')=['v']`
  (list-wrapped, error-prone for an LLM) and don't bind from dict parameters.
  EAV gives natural pattern-match Cypher
  (`(o:Op)-[:HAS_ATTR]->(:Attr{key:'buffer_type',val:'l1'})`) and trivial
  UNWIND ingest. Joins are irrelevant at this data scale.
- **Typed columns only where arithmetic is needed.** `shape INT64[]`, `rank`,
  `dtype`, and the boolean op flags are top-level so queries like
  `size(t.shape)=4` and `t.shape[1]=256` work. (Kuzu lists are 1-indexed.)
- **No queryability config, no indexes.** Everything is queryable by default;
  there is no per-op/per-attr allowlist to maintain (which would silently hide
  forgotten attributes). At ~10⁴–10⁵ nodes per graph, scans are sub-millisecond,
  so DB indexes are deferred until a latency problem actually exists.

---

## mlir-graph-serve API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ingest` | Accept one telemetry JSON document, load into DB |
| POST | `/ingest/archive` | Accept a zip (JSON + `.mlir` sidecars) for a whole run |
| GET | `/graphs` | List all graphs |
| GET | `/graphs/{graph_id}` | Graph metadata |
| GET | `/graphs/{graph_id}/snapshots` | Snapshots (with op/tensor counts + dialect breakdown) |
| GET | `/runs` | CI-run rollup (one row per `runId`; empty = local bucket) |
| POST | `/runs/{run_id}/workflow` | Backfill workflow name/title onto a run's graphs |
| GET | `/search?q=` | Search graphs by model/test/branch/etc. |
| GET | `/snapshots/{snapshot_id}` | Full snapshot: ops, block args, tensors (with attrs), edges, calls, refs |
| GET | `/diff?a=&b=` | Diff two snapshots |
| POST | `/cypher` | Read-only Cypher with pagination (rejects writes via whole-word match) |
| GET | `/schema` | Schema description + example queries |
| GET | `/files/{path}` | Serve a snapshot's `.mlir` sidecar |
| GET | `/` | Dashboard (static SPA) |

### Dashboard URLs

The dashboard (`GET /`) is a deep-linkable SPA. The detail view encodes its
state in the hash:

```
#graph=<graphId>[&snapshot=<snapshotIndex>][&line=<n>]
```

`snapshot` selects a snapshot by its stable `snapshotIndex`; `line` is a 1-based
line in that snapshot's printed MLIR, which the view highlights and scrolls to.
Combined with `Op.mlirLine`, a Cypher result for an op yields a direct link to
its line, e.g. `#graph=<g>&snapshot=2&line=<op.mlirLine>`. (The legacy
`#run=<graphId>` form is still accepted.)

### Diff

Ops are matched across snapshots by source location, bucketed positionally
within a location to tolerate collisions. The diff reports ops
added/removed/modified (attr + opName changes) and tensors
added/removed/encodingChanged (attr + shape changes), plus a summary count.

> Location-based matching is approximate. Robust cross-snapshot (and
> cross-graph) node identity — e.g. content-derived structural IDs and a
> `SAME_AS` edge — is a planned follow-up; it is what would let the diff and
> "why didn't pattern X fire" queries become exact.

---

## Agent access (plain HTTP, no install)

A coding agent queries the API directly with `curl` — nothing to install, no
extra protocol layer. The flow:

1. `GET /schema` — self-describing: returns the node/edge model and example
   Cypher, so the agent learns the graph without prior knowledge.
2. `POST /cypher` with `{"query": "...", "params": {...}, "limit": N}` — arbitrary
   read-only traversals (writes are rejected). Pagination via `limit`/`offset`,
   with `hasMore` in the response.
3. Navigation shortcuts: `GET /graphs`, `GET /graphs/{id}/snapshots`,
   `GET /snapshots/{id}`, `GET /diff?a=&b=`.

A result that returns `o.graphId`, `s.snapshotIndex` and `o.mlirLine` can be
turned straight into a dashboard deep link (see *Dashboard URLs*).

---

## Example queries (validated on a real TTIR→TTNN MLP)

```cypher
-- mesh_shard feeding a matmul/linear
MATCH (a:Op)-[:FEEDS]->(b:Op)
WHERE a.opName CONTAINS 'mesh_shard' AND b.opName CONTAINS 'matmul'
RETURN a.location, b.location

-- reshape -> broadcast -> reshape -> permute chain
MATCH (r1:Op)-[:FEEDS]->(bc:Op)-[:FEEDS]->(r2:Op)-[:FEEDS]->(p:Op)
WHERE r1.opName CONTAINS 'reshape' AND bc.opName CONTAINS 'broadcast'
  AND r2.opName CONTAINS 'reshape' AND p.opName CONTAINS 'permute'
RETURN r1.location, p.location

-- where in the pipeline did matmul fuse into linear?
MATCH (o:Op)-[:IN_SNAPSHOT]->(s:Snapshot)
WHERE o.opName IN ['ttir.matmul', 'ttnn.linear']
RETURN s.snapshotIndex, s.passName, o.opName, count(o) ORDER BY s.snapshotIndex

-- ops with an L1 operand at a given snapshot
MATCH (o:Op {snapshotId: $sid})<-[:CONSUMED_BY]-(t:Tensor)
MATCH (t)-[:HAS_ATTR]->(:Attr {key: 'buffer_type', val: 'l1'})
RETURN DISTINCT o.opName, o.location

-- 4D tensors
MATCH (t:Tensor) WHERE size(t.shape) = 4 RETURN t.id, t.shape
```

---

## Future work

- **Exact cross-snapshot node identity.** Content/structure-derived node IDs plus
  `SAME_AS` edges, to replace location-bucket diff matching and make
  "why didn't pattern X fire" exact.
- **Server Docker image.** The `graph_telemetry` package is a self-contained
  build context (no compiler build needed), so containerizing `mlir-graph-serve`
  is straightforward when a shared deployment is wanted.
```
