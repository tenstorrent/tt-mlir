# Plan A — Connection pooling & serve concurrency

> Read [CONTEXT.md](CONTEXT.md) first. This plan is engine-independent of the
> schema work and should land **first**. It fixes the acute outage and the
> 10-concurrent-user target. No schema change, no re-ingest.

## Problem

`serve/db.py` `KuzuDB` holds **one** `kuzu.Connection` (`self._conn`) shared by
all requests. `serve/app.py` endpoints are `async def` that call the
**synchronous** `service.*` functions directly, on a **single** uvicorn worker,
with **no query timeout**. Consequences (all observed):

- Every request is fully serialized through one connection on the event loop.
- One slow query blocks all users; a pathological query ran **183 s** and pegged
  the serve (had to be killed by PID). Subsequent requests appeared to "hang"
  because they were queued behind it on the one connection.
- This breaks at ~2 concurrent users, not 10.

## Facts this design relies on (verified — see CONTEXT.md)

- One `kuzu.Database` + N `kuzu.Connection`s give **real read parallelism**
  (4 parallel scans → 0.28 s wall for 4× 0.27 s queries).
- `connection.set_query_timeout(ms)` exists and works.
- Well-formed queries are 50 ms–1.4 s, so a modest pool easily covers 10 users.
- Kuzu is single-writer; reads concurrent within one process. Keep **1 uvicorn
  worker** (multiple workers can't RW-open the embedded DB). Concurrency comes
  from the in-process pool, not from worker processes.

## Design

```
KuzuDB
  ├── _db : kuzu.Database                 (one; holds the RW lock)
  ├── _read_pool : queue.Queue[Connection]  (N read conns, each with timeout set)
  ├── _write_conn : Connection            (one; ingest/DDL serialize here)
  ├── _write_lock : threading.Lock
  └── _executor : ThreadPoolExecutor(max_workers=N)
```

- **Reads**: check out a connection from `_read_pool`, run, return it (context
  manager). Each read connection has `set_query_timeout(query_timeout_ms)`.
- **Writes** (`ingest`, `ensure_schema`, `bulk_copy_rel`, `execute_write`): use
  `_write_conn` under `_write_lock` (Kuzu single-writer).
- **Event loop**: endpoints `await loop.run_in_executor(db._executor, fn)` so the
  blocking Kuzu call runs off the event loop. (Alternative: convert read
  endpoints to sync `def` and let FastAPI's anyio threadpool run them — but an
  explicit executor keeps thread count aligned with the connection pool, which is
  what bounds real DB parallelism.)
- **Timeout → HTTP**: map Kuzu's interrupt/timeout exception to a typed
  `QueryTimeout`; surface as **HTTP 408**. Pool exhaustion / checkout timeout →
  **HTTP 503**.

## File-by-file changes

### `serve/db.py`
- `KuzuDB.__init__(db_path, read_pool_size=min(8, os.cpu_count()), query_timeout_ms=30000)`:
  - create `self._db`, then `read_pool_size` `Connection`s into a `queue.Queue`,
    calling `set_query_timeout(query_timeout_ms)` on each.
  - create `self._write_conn`, `self._write_lock`, `self._executor`.
- Add `@contextmanager _acquire_read()` that `get(timeout=...)`/`put`s a conn;
  raise a typed `PoolExhausted` on checkout timeout.
- `execute(cypher, params)` → use `_acquire_read()`; on Kuzu timeout raise
  `QueryTimeout`. Keep the existing row→dict normalization logic.
- `execute_write` / `bulk_copy_rel` / `ensure_schema` → run on `_write_conn`
  under `_write_lock`.
- Add `async def execute_async(...)` helper, or expose `_executor` so `app.py`
  can `run_in_executor`. (Pick one; document it.)
- Define exceptions `QueryTimeout(Exception)`, `PoolExhausted(Exception)`.

### `serve/app.py`
- On startup, the executor lives on the `KuzuDB`. Wrap each `service.*` call:
  `result = await loop.run_in_executor(db._executor, lambda: service.fn(db, ...))`.
  (Or convert the read endpoints to sync `def`.) Keep `/ingest*` on the write
  path; they must not use the read pool.
- Add exception handlers: `QueryTimeout` → 408 with a clear message;
  `PoolExhausted` → 503; keep existing `NotFound`/`BadRequest` mappings.

### `serve/__main__.py`
- New CLI flags: `--read-pool-size` (default `min(8, cpu)`),
  `--query-timeout-ms` (default 30000). Thread them into `KuzuDB`.

### `serve/service.py`
- No logic change required for Plan A, but see Plan B-B3: ensure the canonical
  queries scope via PK traversal so the pool isn't wasted on 183 s shapes. If
  Plan B is not yet done, at minimum audit `service.py` for any
  `{graphId: <variable>}` correlated joins and rewrite them as traversals.

## Validation

- Unit: `tests/` still pass; add a test that two queries run concurrently
  against the pool, and that a deliberately slow/looping query raises
  `QueryTimeout`.
- Load test (script with `asyncio`+`httpx` or `ab`): 10 clients looping the fast
  per-graph query (`/graphs/{id}/snapshots`) while 1 client fires a heavy query.
  Assert: fast-query p95 stays in the tens of ms; the heavy query returns 408 at
  the timeout and does **not** stall the others.
- Sanity: confirm ingest still works (write path) while reads are served.

## Acceptance criteria

- 10 concurrent fast queries complete with p95 ≈ single-query latency (proves
  parallelism, not serialization).
- No query can run longer than `--query-timeout-ms`; over-long queries 408.
- Ingest (write) and reads coexist without deadlock.

## Effort & risk

~0.5–1 day. Low risk: no schema/data change, fully reversible. Main subtlety is
correctly separating the single write connection (locked) from the read pool.
