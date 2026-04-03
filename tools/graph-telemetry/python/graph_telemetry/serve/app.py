# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application for mlir-graph-serve.

This is a thin HTTP transport over `service.py`, where all query logic lives.
The dashboard, a coding agent (via curl), and the push uploader all talk to
these endpoints; `GET /schema` is self-describing so a client can bootstrap the
node/edge model and example Cypher without prior knowledge.
"""

from __future__ import annotations

import io
import json
import logging
import secrets
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import GraphDB, PoolExhausted, QueryTimeout, connect
from .ingest import ingest as ingest_telemetry, ingest_many, graph_exists
from . import querylog, service

logger = logging.getLogger(__name__)

app = FastAPI(title="mlir-graph-serve", version="0.1.0")
app.state.files_root = None
app.state.write_secret = None


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def init_db(
    dsn: str,
    read_pool_size: int | None = None,
    query_timeout_ms: int = 30000,
) -> None:
    """Connect to the graph DB and store it on app state."""
    db = connect(dsn, read_pool_size=read_pool_size, query_timeout_ms=query_timeout_ms)
    db.ensure_schema()
    app.state.db = db


def get_db() -> GraphDB:
    """Get the database from app state."""
    return app.state.db


def init_files_root(path: str | None) -> None:
    """Set the directory the /files route serves snapshot .mlir dumps from.

    This is the telemetry/ingest directory; a Snapshot's mlirPath is relative
    to it.
    """
    app.state.files_root = Path(path).resolve() if path else None
    if app.state.files_root is not None:
        app.state.files_root.mkdir(parents=True, exist_ok=True)


def init_write_secret(secret: str | None) -> None:
    """Require `secret` as a bearer token on mutating endpoints; None opens them.

    Reads are always open: the telemetry is derived from an open-source repo's
    CI, so the secret only protects ingest integrity (non-idempotent loads,
    zip extraction to disk), not confidentiality.
    """
    app.state.write_secret = secret or None


def init_query_log(path: str | None) -> None:
    """Enable the per-query JSONL log at `path`; None disables it."""
    querylog.init(path)


# Mutating endpoints require the write secret; everything else is open.
_WRITE_PREFIXES = ("/ingest", "/runs")


@app.middleware("http")
async def _request_middleware(request: Request, call_next):
    # Query-log context for every DB call made under this request; one HTTP
    # request that fans out into several Cypher queries shares one requestId.
    # Attribution is honor-system: the X-User header if sent, else client IP.
    client = request.client.host if request.client else ""
    ctx = {
        "user": request.headers.get("x-user", "") or client or "anonymous",
        "endpoint": f"{request.method} {request.url.path}",
        "requestId": uuid.uuid4().hex[:12],
    }
    querylog.request_ctx.set(ctx)
    secret: str | None = app.state.write_secret
    if (
        secret is not None
        and request.url.path.startswith(_WRITE_PREFIXES)
        and request.method != "GET"
    ):
        header = request.headers.get("authorization", "")
        token = header[len("Bearer ") :] if header.startswith("Bearer ") else ""
        if not secrets.compare_digest(token, secret):
            return JSONResponse(
                status_code=401, content={"detail": "write secret required"}
            )
    return await call_next(request)


async def _handle(fn: Callable, *args, **kwargs):
    """Run a service call off the event loop, mapping errors to HTTP codes.

    The blocking Kuzu work runs on the DB's executor (bounded by the read pool)
    so a slow query can't stall the event loop. QueryTimeout (408) and
    PoolExhausted (503) are mapped by the app-wide exception handlers below.
    """
    try:
        return await get_db().submit(fn, *args, **kwargs)
    except service.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except service.BadRequest as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.exception_handler(QueryTimeout)
async def _query_timeout_handler(request: Request, exc: QueryTimeout):
    return JSONResponse(status_code=408, content={"detail": str(exc)})


@app.exception_handler(PoolExhausted)
async def _pool_exhausted_handler(request: Request, exc: PoolExhausted):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CypherRequest(BaseModel):
    query: str
    params: dict[str, Any] | None = None
    limit: int | None = None
    offset: int | None = None


class WorkflowMeta(BaseModel):
    workflowName: str = ""
    workflowTitle: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/ingest")
async def ingest_endpoint(body: dict[str, Any]):
    """Accept a single telemetry JSON document and load it into the database."""
    db = get_db()
    graph_id = (body.get("graph") or {}).get("graphId", "")
    if graph_id and await db.submit(graph_exists, db, graph_id):
        raise HTTPException(
            status_code=409, detail=f"graph '{graph_id}' already ingested"
        )
    return await db.submit(ingest_telemetry, db, body)


@app.post("/ingest/archive")
async def ingest_archive_endpoint(request: Request):
    """Ingest a whole run (CI run) as one zip.

    The zip carries the telemetry directory layout: `<graph>.json` documents
    plus their `<graph>/<index>_<tag>.mlir` sidecars (exactly a GitHub Actions
    artifact of the telemetry dir). Every member is extracted under the files
    root (so its mlirPath resolves), then every JSON is ingested -- JSON and
    MLIR arrive together in a single request, for all graphs in the run.
    """
    if app.state.files_root is None:
        raise HTTPException(status_code=400, detail="no files root configured")
    try:
        archive = zipfile.ZipFile(io.BytesIO(await request.body()))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="body is not a valid zip")

    json_docs: list[bytes] = []
    files_written = 0
    for name in archive.namelist():
        if name.endswith("/"):
            continue
        target = _resolve_in_files_root(name)  # rejects zip-slip paths
        target.parent.mkdir(parents=True, exist_ok=True)
        data = archive.read(name)
        target.write_bytes(data)
        files_written += 1
        if name.endswith(".json"):
            json_docs.append(data)

    docs = []
    for data in json_docs:
        try:
            docs.append(json.loads(data))
        except json.JSONDecodeError:
            continue

    # Pre-flight: reject if any run is already ingested, before writing any
    # rows -- ingestion is not idempotent, so a re-pushed archive fails cleanly
    # with 409 instead of a duplicate-key crash mid-way. Both the dup probe and
    # the load run off the event loop so a large archive doesn't stall reads.
    db = get_db()

    def _find_dups():
        return sorted(
            d["graph"]["graphId"]
            for d in docs
            if d.get("graph", {}).get("graphId")
            and graph_exists(db, d["graph"]["graphId"])
        )

    dups = await db.submit(_find_dups)
    if dups:
        raise HTTPException(
            status_code=409, detail=f"already ingested: {', '.join(dups)}"
        )

    graphs = await db.submit(ingest_many, db, docs)
    return {
        "graphs": graphs,
        "graphsIngested": len(graphs),
        "filesWritten": files_written,
    }


@app.get("/graphs")
async def list_graphs(run_id: str | None = Query(None, alias="runId")):
    """List graphs.

    With no `runId`, returns every graph. With `runId` present (empty string
    selects the local bucket), returns just that run's graphs -- the dashboard
    uses this to lazily expand one run at a time.
    """
    if run_id is not None:
        return await _handle(service.list_graphs_for_run, get_db(), run_id)
    return await _handle(service.list_graphs, get_db())


@app.get("/runs")
async def list_runs():
    """Run (CI run) rollup for the dashboard's landing page (one row per run)."""
    return await _handle(service.list_runs, get_db())


@app.post("/runs/{run_id}/workflow")
async def set_run_workflow(run_id: str, body: WorkflowMeta):
    """Backfill workflow name/title onto every graph of an existing run."""
    return await _handle(
        service.set_run_workflow,
        get_db(),
        run_id,
        body.workflowName,
        body.workflowTitle,
    )


@app.get("/search")
async def search_graphs(q: str = Query("")):
    """Substring-search graphs across their human-meaningful fields."""
    return await _handle(service.search_graphs, get_db(), q)


@app.get("/graphs/{graph_id}")
async def get_graph(graph_id: str):
    return await _handle(service.get_graph, get_db(), graph_id)


@app.get("/graphs/{graph_id}/snapshots")
async def list_graph_snapshots(graph_id: str):
    return await _handle(service.list_graph_snapshots, get_db(), graph_id)


@app.get("/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: str):
    return await _handle(service.get_snapshot, get_db(), snapshot_id)


@app.get("/diff")
async def diff_snapshots(a: str = Query(...), b: str = Query(...)):
    return await _handle(service.diff_snapshots, get_db(), a, b)


@app.post("/cypher")
async def run_cypher(body: CypherRequest):
    return await _handle(
        service.run_cypher,
        get_db(),
        body.query,
        body.params,
        body.limit,
        body.offset,
    )


@app.get("/analytics/op-kinds")
async def analytics_op_kinds(
    graph_id: str | None = Query(None, alias="graphId"),
    run_id: str | None = Query(None, alias="runId"),
    op_name: str | None = Query(None, alias="opName"),
):
    """Op-kind analytics, anchored on reified nodes (no full Op scan).

    - `opName` + `runId`: every op of that kind across the run (locations).
    - `graphId` or `runId` alone: op-kind counts for that graph / run.
    """
    if op_name:
        if run_id is None:
            raise HTTPException(status_code=400, detail="opName requires runId")
        return await _handle(service.ops_of_kind, get_db(), op_name, run_id)
    return await _handle(service.op_kind_counts, get_db(), graph_id, run_id)


@app.get("/analytics/op-layouts")
async def analytics_op_layouts(
    op_name: str = Query(..., alias="opName"),
    run_id: str = Query(..., alias="runId"),
):
    """Layout breakdown (tiled/buffer type) for one op kind, per test in a run."""
    return await _handle(service.op_kind_layouts, get_db(), op_name, run_id)


@app.get("/schema")
async def get_schema():
    return await _handle(service.get_schema, get_db())


def _resolve_in_files_root(file_path: str):
    """Resolve `file_path` under the configured files root, rejecting traversal."""
    root = app.state.files_root
    if root is None:
        raise HTTPException(status_code=400, detail="no files root configured")
    target = (root / file_path).resolve()
    if root not in target.parents:
        raise HTTPException(status_code=400, detail="invalid path")
    return target


@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    """Serve a snapshot's MLIR dump by its Snapshot.mlirPath (relative path)."""
    target = _resolve_in_files_root(file_path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target, media_type="text/plain")


@app.get("/")
async def root():
    """Redirect to the dashboard."""
    return RedirectResponse(url="/ui/")


# Static dashboard (read-only view over the same service-layer endpoints).
# Mounted last so it never shadows the API routes above.
app.mount(
    "/ui",
    StaticFiles(directory=str(Path(__file__).parent / "static"), html=True),
    name="ui",
)
