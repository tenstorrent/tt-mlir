# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for mlir-graph-serve."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mlir-graph-serve: Graph telemetry query server"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Kuzu DSN string, e.g. 'kuzu:./telemetry.kuzu'",
    )
    parser.add_argument(
        "--files-dir",
        help="Directory to serve snapshot .mlir dumps from via /files; also "
        "the extraction root for /ingest/archive",
    )
    parser.add_argument(
        "--write-secret",
        default=os.environ.get("GRAPH_TELEMETRY_TOKEN", ""),
        help="Bearer secret required on mutating endpoints (ingest); reads "
        "stay open. Default: $GRAPH_TELEMETRY_TOKEN (omit to leave writes "
        "open, e.g. local dev)",
    )
    parser.add_argument(
        "--query-log",
        help="JSONL file logging every query with user and execution time, "
        "rotated daily; the input for offline schema optimization (omit to "
        "disable)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8321, help="Bind port (default: 8321)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--read-pool-size",
        type=int,
        default=None,
        help="Concurrent read connections in the pool (default: min(8, cpu count))",
    )
    parser.add_argument(
        "--query-timeout-ms",
        type=int,
        default=30000,
        help="Per-read-query timeout in milliseconds; over-long reads return "
        "HTTP 408 (default: 30000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Import here to avoid import-time side effects
    import uvicorn

    from .app import app, init_db, init_files_root, init_query_log, init_write_secret

    init_write_secret(args.write_secret)
    if not args.write_secret:
        logging.warning("No --write-secret; ingest endpoints are open")
    init_query_log(args.query_log)
    init_db(
        args.db,
        read_pool_size=args.read_pool_size,
        query_timeout_ms=args.query_timeout_ms,
    )
    # Snapshot.mlirPath is relative to the files root; ingestion is explicit
    # via /ingest and /ingest/archive only.
    init_files_root(args.files_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
