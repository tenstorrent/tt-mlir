# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-query JSONL log for offline schema optimization.

Every Cypher execution is appended as one JSON line carrying the issuing user,
endpoint, query text and hash, duration, row count, and status. The log lives
on the data volume next to the DB and rotates daily; it is the input for
"p95 by query / which queries time out" analysis when tuning the Kuzu schema.

The HTTP layer sets `request_ctx` (user/endpoint/request id) and the DB layer
emits one line per execute, so a single HTTP request that fans out into several
Cypher queries shares one request id. Service-layer queries are fixed
templates, so `queryHash` groups them exactly; param values are deliberately
not logged (only their keys) -- ingest payloads would bloat the log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler

# Set per request by the auth middleware; GraphDB.submit copies the calling
# context onto the executor thread so DB-layer logging sees it.
request_ctx: ContextVar[dict | None] = ContextVar("gt_request_ctx", default=None)

_logger: logging.Logger | None = None


def init(path: str | None) -> None:
    """Open the query log at `path` (daily rotation); None disables logging."""
    global _logger
    if not path:
        _logger = None
        return
    handler = TimedRotatingFileHandler(path, when="midnight", backupCount=30, utc=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("graph_telemetry.querylog")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    _logger = logger


def query_hash(cypher: str) -> str:
    """Hash of the whitespace-normalized query text, for grouping templates."""
    normalized = re.sub(r"\s+", " ", cypher).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def log(
    cypher: str,
    params: dict | None,
    duration_ms: float,
    rows: int,
    status: str,
    write: bool,
) -> None:
    """Append one query record; no-op when logging is disabled."""
    if _logger is None:
        return
    ctx = request_ctx.get() or {}
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "user": ctx.get("user", ""),
        "endpoint": ctx.get("endpoint", ""),
        "requestId": ctx.get("requestId", ""),
        "queryHash": query_hash(cypher),
        "query": cypher,
        "paramsKeys": sorted(params) if params else [],
        "durationMs": round(duration_ms, 2),
        "rows": rows,
        "status": status,
        "write": write,
    }
    _logger.info(json.dumps(record, separators=(",", ":")))
