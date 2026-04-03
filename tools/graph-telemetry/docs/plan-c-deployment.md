# Plan C: internal deployment

Status: in progress. Covers the gaps between the current docker-compose setup
and the IT/SOC hosting checklist for an internal, handful-of-users service.

Done so far: the write-secret security model (reads open, mutating endpoints
behind `$GRAPH_TELEMETRY_TOKEN`, honor-system `X-User` attribution); the GHCR
build/push workflow and the compose switch to a published image; and the
liveness primitives (restart policy + healthcheck). Remaining items are either
external requests (VM, TLS sign-off, Zabbix) or need a build environment to
resolve pins.

## Security model (decided)

The data is compiler IR telemetry from an open-source repo's CI -- nothing
confidential. Per the SOC guidance, an internal VM behind Tailscale/VPN with no
public exposure does not need SSO; this is deliberately not OAuth2-Proxy +
Entra.

- Reads: open to anyone on the internal network.
- Writes (`POST /ingest*`, `POST /runs/*`): require the shared bearer secret
  (`$GRAPH_TELEMETRY_TOKEN`). Protects ingest integrity (non-idempotent loads,
  archive extraction to disk), not confidentiality. Held by CI (repo secret)
  and operators.
- Attribution: honor-system `X-User` header (curl/clients), client IP
  fallback; recorded in the query log.

Revisit (move behind OAuth2-Proxy + Entra) only if it becomes public-facing or
ingests anything non-public.

## Remaining work

### 1. Hosting

- Request a dedicated internal VM (not login/compute), Tailscale-only.
- One exposed TCP port (8321, plain HTTP). Flag it in the hosting request; no
  websockets or other sockets are used.
- Sizing: 2-4 vCPU / 8 GB RAM; disk dominated by `/data/files` artifact growth
  (~50-200 MB per CI run dir) -- start at 100 GB, monitor. The DB is a derived
  index of the artifacts; only `/data/files` needs backup.
- Document who maintains it (SSH) and that app access = network access.

### 2. TLS

Checklist asks for TLS even internally. Smallest version: nginx sidecar in the
compose with a Vault/ACME cert, 8443 -> serve:8321, serve bound to localhost.
With Tailscale-only access this is low value; confirm with IT whether plain
HTTP is acceptable for a non-confidential internal app before adding it.

### 3. Image supply chain

- **Done:** `.github/workflows/build-graph-telemetry-image.yml` builds the image
  and pushes to `ghcr.io/tenstorrent/tt-mlir/graph-telemetry` on pushes to
  `main` that touch `tools/graph-telemetry/python/**` or the Dockerfile (plus
  manual `workflow_dispatch`). Tags: immutable `sha-<12>` always, `latest` on
  `main`.
- **Done:** compose now defaults to `image:` (`:${GT_IMAGE_TAG:-latest}`) and
  pulls the published image; the `build:` stanza stays as a local fallback.
  `docker compose pull && docker compose up -d` to deploy; set `GT_IMAGE_TAG`
  to pin a specific build.
- **Needs a build environment** (no registry/PyPI access at authoring time):
  pin Python deps (`pip-compile`/`pip freeze` into a constraints file consumed
  by the Dockerfile) and pin the base image by digest
  (`python:3.12-slim@sha256:...`). The Dockerfile still installs an unpinned
  dep list duplicated from `pyproject.toml`; collapse that into the pinned
  constraints file when resolving.

### 4. Monitoring

- **Done (already in place):** container restart-on-fail (`restart:
  unless-stopped` in compose) and the 30s Dockerfile healthcheck cover
  liveness.
- Register the VM in Zabbix. *(external)*
- App-level signals come from the query log JSONL (per-query duration, user,
  timeouts) -- duckdb/pandas offline analysis; Prometheus export later if
  needed.

### 5. Pre-deploy checks

- `GRAPH_TELEMETRY_TOKEN` set on the VM and in repo secrets; not in compose,
  image, or repo.
- Verify mutating endpoints require the secret from a second host.
- Pre-commit + tests pass; healthcheck green.
