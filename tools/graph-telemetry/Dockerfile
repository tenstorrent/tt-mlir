# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained image for mlir-graph-serve. The serve is pure Python with no
# build dependency on the compiler, so the build context is just the package
# directory:
#
#   docker build -f Dockerfile python/

FROM python:3.12-slim

# Layer the dependencies before the source so code-only changes rebuild fast.
COPY pyproject.toml /opt/graph-telemetry/
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" kuzu httpx pandas

COPY . /opt/graph-telemetry/
RUN pip install --no-cache-dir /opt/graph-telemetry

# All persistent state lives under /data (DB, .mlir artifacts) -- mount a
# volume there; the image itself stays disposable and secret-free. The write
# secret arrives via $GRAPH_TELEMETRY_TOKEN at runtime.
VOLUME /data
EXPOSE 8321

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8321/ui/', timeout=4)"

CMD ["mlir-graph-serve", \
     "--db", "kuzu:/data/db/telemetry.kuzu", \
     "--files-dir", "/data/files", \
     "--host", "0.0.0.0", "--port", "8321"]
