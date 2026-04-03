# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Write-secret enforcement: reads are open, mutations need the shared secret;
the query log attributes requests via the honor-system X-User header."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_telemetry.serve import querylog  # noqa: E402


def test_write_secret_and_attribution():
    from fastapi.testclient import TestClient

    from graph_telemetry.serve.app import (
        app,
        init_db,
        init_files_root,
        init_query_log,
        init_write_secret,
    )

    tmp = tempfile.mkdtemp()
    log_path = Path(tmp) / "query_log.jsonl"
    init_db(f"kuzu:{tmp}/ws.kuzu")
    init_files_root(tmp)
    init_write_secret("s3cret")
    init_query_log(str(log_path))
    client = TestClient(app, raise_server_exceptions=False)
    try:
        # Reads are open, with and without identity.
        assert client.get("/graphs").status_code == 200
        assert client.get("/graphs", headers={"X-User": "alice"}).status_code == 200

        # Mutations need the exact secret.
        body = json.dumps({"workflowName": "w", "workflowTitle": "t"})
        assert client.post("/runs/x/workflow", content=body).status_code == 401
        assert (
            client.post(
                "/runs/x/workflow",
                content=body,
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/runs/x/workflow",
                content=body,
                headers={"Authorization": "Bearer s3cret"},
            ).status_code
            != 401
        )

        # GETs under write prefixes stay open.
        assert client.get("/runs").status_code == 200

        # X-User lands in the query log; without it, client IP is used.
        users = {json.loads(l)["user"] for l in log_path.read_text().splitlines()}
        assert "alice" in users
        assert "testclient" in users  # TestClient's request.client.host
    finally:
        init_write_secret(None)
        init_query_log(None)
        app.state.db.close()
