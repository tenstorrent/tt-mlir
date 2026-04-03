# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GraphDB abstraction layer backed by embedded Kuzu.

Kuzu is single-writer but allows concurrent read connections against one
`kuzu.Database`. `KuzuDB` exploits that: a pool of read connections (each with a
per-query timeout) serves reads in parallel, while a single write connection
under a lock serializes ingest/DDL. Blocking Kuzu calls are dispatched off the
asyncio event loop via a `ThreadPoolExecutor` so one slow query can no longer
stall every other request (see docs/plan-a-connection-pooling.md).
"""

from __future__ import annotations

import contextvars
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from . import queries as Q
from . import querylog

logger = logging.getLogger(__name__)


class QueryTimeout(Exception):
    """A read query exceeded the configured per-query timeout."""


class PoolExhausted(Exception):
    """No read connection became available within the checkout timeout."""


class GraphDB(ABC):
    """Abstract base class for graph database backends."""

    @abstractmethod
    def execute(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a read query and return rows as list of dicts."""
        ...

    @abstractmethod
    def execute_write(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a write query (CREATE, MERGE, etc.); return any RETURN rows."""
        ...

    @abstractmethod
    def bulk_copy_rel(
        self,
        table: str,
        rows: list[dict],
        from_tbl: str | None = None,
        to_tbl: str | None = None,
    ) -> None:
        """Bulk-create relationships from `rows` (each {f, t, ...props}).

        Far faster than UNWIND ... MATCH ... CREATE for large batches, which
        the planner executes as a full node-table scan + cross product. `from_tbl`
        / `to_tbl` name the endpoint node tables for multi-pair rel tables.
        """
        ...

    @abstractmethod
    def bulk_copy_node(self, table: str, rows: list[dict], columns: list[str]) -> None:
        """Bulk-create nodes from `rows`, projected to `columns` in table order.

        Kuzu's COPY is far faster than UNWIND ... CREATE for large batches and,
        unlike UNWIND/CREATE, its cost does not grow as the table fills. Only for
        CREATE-only tables (no dedup); MERGE'd tables stay on the UNWIND path.
        """
        ...

    @abstractmethod
    async def submit(self, fn, *args, **kwargs):
        """Run a blocking DB call off the event loop on the backend's executor."""
        ...

    @abstractmethod
    def get_schema(self) -> dict:
        """Return the DB schema description."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
    def ensure_schema(self) -> None:
        """Create node/relationship tables and indexes if they don't exist."""
        ...


class KuzuDB(GraphDB):
    """Embedded Kuzu database with a read-connection pool and a single writer."""

    def __init__(
        self,
        db_path: str,
        read_pool_size: int | None = None,
        query_timeout_ms: int = 30000,
    ) -> None:
        import kuzu

        if read_pool_size is None:
            read_pool_size = min(8, os.cpu_count() or 1)
        self._read_pool_size = read_pool_size
        self._query_timeout_ms = query_timeout_ms
        # Wait at most one full query-timeout (plus a small margin) for a free
        # read connection before giving up; a saturated pool surfaces as 503
        # rather than blocking the caller forever.
        self._checkout_timeout_s = query_timeout_ms / 1000.0 + 5.0

        self._db = kuzu.Database(db_path)

        # N read connections, each capped by the query timeout.
        self._read_pool: queue.Queue = queue.Queue()
        for _ in range(read_pool_size):
            conn = kuzu.Connection(self._db)
            conn.set_query_timeout(query_timeout_ms)
            self._read_pool.put(conn)

        # One write connection; Kuzu is single-writer so all writes serialize
        # here under the lock.
        self._write_conn = kuzu.Connection(self._db)
        self._write_lock = threading.Lock()

        # One extra worker beyond the read pool so a write can proceed even when
        # every read connection is busy.
        self._executor = ThreadPoolExecutor(
            max_workers=read_pool_size + 1, thread_name_prefix="kuzu-db"
        )
        self._schema_created = False

    @staticmethod
    def _is_timeout(exc: Exception) -> bool:
        """Heuristic: Kuzu reports a timeout as a generic interrupt error."""
        msg = str(exc).lower()
        return "interrupt" in msg or "timeout" in msg or "timed out" in msg

    @staticmethod
    def _rows(result) -> list[dict]:
        """Normalize a Kuzu QueryResult into a list of plain dicts."""
        column_names = result.get_column_names()
        rows = []
        while result.has_next():
            row = result.get_next()
            row_dict = {}
            for i, name in enumerate(column_names):
                val = row[i]
                # Kuzu returns node values as dicts already in some cases
                if hasattr(val, "__dict__"):
                    val = dict(val.__dict__)
                row_dict[name] = val
            rows.append(row_dict)
        return rows

    @contextmanager
    def _acquire_read(self):
        """Check a read connection out of the pool, returning it on exit."""
        try:
            conn = self._read_pool.get(timeout=self._checkout_timeout_s)
        except queue.Empty:
            raise PoolExhausted(
                f"no read connection available within "
                f"{self._checkout_timeout_s:.0f}s (pool size {self._read_pool_size})"
            )
        try:
            yield conn
        finally:
            self._read_pool.put(conn)

    def execute(self, cypher: str, params: dict | None = None) -> list[dict]:
        with self._acquire_read() as conn:
            start = time.perf_counter()
            try:
                result = conn.execute(cypher, parameters=params or {})
                rows = self._rows(result)
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000.0
                if self._is_timeout(e):
                    logger.warning(
                        "Kuzu query timed out after %d ms: %s",
                        self._query_timeout_ms,
                        cypher,
                    )
                    querylog.log(cypher, params, elapsed, 0, "timeout", write=False)
                    raise QueryTimeout(
                        f"query exceeded {self._query_timeout_ms} ms timeout"
                    ) from e
                logger.error("Kuzu query failed: %s\nQuery: %s", e, cypher)
                querylog.log(cypher, params, elapsed, 0, "error", write=False)
                raise
            elapsed = (time.perf_counter() - start) * 1000.0
            querylog.log(cypher, params, elapsed, len(rows), "ok", write=False)
            return rows

    def execute_write(self, cypher: str, params: dict | None = None) -> list[dict]:
        with self._write_lock:
            start = time.perf_counter()
            try:
                result = self._write_conn.execute(cypher, parameters=params or {})
                rows = self._rows(result)
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000.0
                logger.error("Kuzu write failed: %s\nQuery: %s", e, cypher)
                querylog.log(cypher, params, elapsed, 0, "error", write=True)
                raise
            elapsed = (time.perf_counter() - start) * 1000.0
            querylog.log(cypher, params, elapsed, len(rows), "ok", write=True)
            return rows

    def bulk_copy_rel(
        self,
        table: str,
        rows: list[dict],
        from_tbl: str | None = None,
        to_tbl: str | None = None,
    ) -> None:
        if not rows:
            return
        import pandas as pd

        # `df` is resolved by Kuzu's pandas replacement scan from this frame's
        # locals, so the variable must be named `df`. Columns are positional:
        # first two are the FROM/TO primary keys, the rest are rel properties.
        df = pd.DataFrame(rows)  # noqa: F841 - referenced by name in COPY
        spec = f' (from="{from_tbl}", to="{to_tbl}")' if from_tbl else ""
        cypher = f"COPY {table} FROM df{spec}"
        with self._write_lock:
            start = time.perf_counter()
            try:
                self._write_conn.execute(cypher)
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000.0
                logger.error("Kuzu COPY into %s failed: %s", table, e)
                querylog.log(cypher, None, elapsed, len(rows), "error", write=True)
                raise
            elapsed = (time.perf_counter() - start) * 1000.0
            querylog.log(cypher, None, elapsed, len(rows), "ok", write=True)

    def bulk_copy_node(self, table: str, rows: list[dict], columns: list[str]) -> None:
        if not rows:
            return
        import pandas as pd

        # `df` is resolved by Kuzu's pandas replacement scan from this frame's
        # locals; COPY matches columns positionally, so project to table order.
        df = pd.DataFrame(rows, columns=columns)  # noqa: F841 - named in COPY
        cypher = f"COPY {table} FROM df"
        with self._write_lock:
            start = time.perf_counter()
            try:
                self._write_conn.execute(cypher)
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000.0
                logger.error("Kuzu COPY into %s failed: %s", table, e)
                querylog.log(cypher, None, elapsed, len(rows), "error", write=True)
                raise
            elapsed = (time.perf_counter() - start) * 1000.0
            querylog.log(cypher, None, elapsed, len(rows), "ok", write=True)

    def ensure_schema(self) -> None:
        if self._schema_created:
            return
        with self._write_lock:
            for stmt in Q.KUZU_CREATE_NODE_TABLES:
                self._write_conn.execute(stmt)
            for stmt in Q.KUZU_CREATE_REL_TABLES:
                self._write_conn.execute(stmt)
        self._schema_created = True
        logger.info("Kuzu schema created/verified")

    async def submit(self, fn, *args, **kwargs):
        import asyncio

        loop = asyncio.get_running_loop()
        # run_in_executor does not propagate contextvars; copy the caller's
        # context so the query log sees the request's user/endpoint.
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            self._executor, lambda: ctx.run(fn, *args, **kwargs)
        )

    def get_schema(self) -> dict:
        return Q.SCHEMA_DESCRIPTION

    def close(self) -> None:
        # Kuzu connections don't need explicit close (Database closes on GC);
        # shut the executor down so worker threads don't outlive the process.
        self._executor.shutdown(wait=False)


def connect(
    dsn: str,
    *,
    read_pool_size: int | None = None,
    query_timeout_ms: int = 30000,
) -> GraphDB:
    """Parse a DSN string and return a GraphDB backend.

    Supported formats:
      - kuzu:./path/to/db       (relative path)
      - kuzu:/absolute/path     (absolute path)

    A bare filesystem path (no scheme) is also accepted and treated as Kuzu.
    """
    db_path = dsn[len("kuzu:") :] if dsn.startswith("kuzu:") else dsn
    # Kuzu creates the DB file but not its parent (e.g. /data/db on a fresh volume).
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    logger.info("Connecting to Kuzu at %s", db_path)
    return KuzuDB(
        db_path, read_pool_size=read_pool_size, query_timeout_ms=query_timeout_ms
    )
