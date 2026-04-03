#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""mlir-graph-push: Upload telemetry (JSON + MLIR sidecars) to mlir-graph-serve.

Each argument is sent as one zip to the /ingest/archive endpoint, so a whole
run's graphs and their .mlir files arrive in a single request:

  - a directory  -> zipped recursively (the telemetry output dir, or an
                    unpacked GitHub artifact)
  - a .zip file  -> forwarded as-is (e.g. a GitHub Actions artifact)
  - a .json file -> zipped together with its `<stem>/` sidecar dir, if present

CI/GitHub provenance that the compiler can't know at emit time -- the workflow
name and run title, the CI run id, branch, and commit -- is resolved here, from
explicit flags else the Actions environment, and patched into each document's
`graph` block before upload. The emitter stays CI-agnostic; this is the one
GitHub-aware layer.
"""

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import httpx


def collect_targets(patterns: list[str]) -> list[Path]:
    """Resolve arguments to a list of targets (directories, zips, or files)."""
    targets: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.exists():
            targets.append(p)
        else:
            targets.extend(Path(".").glob(pattern))
    return sorted(set(targets))


def resolve_provenance(args: argparse.Namespace) -> dict[str, str]:
    """Resolve the run-level CI/GitHub provenance to stamp onto each graph.

    Precedence: explicit flags, then the GitHub Actions environment. The run id,
    branch, workflow name and commit come from the standard environment
    variables; the run's display title is not in the environment, so it is
    fetched from the Actions API (public repos need no token). Anything that
    can't be resolved stays empty (the server treats empty as "no provenance").

    Per-graph `testName` is intentionally not here: it is compilation context
    the emitter records at emit time, not a run-level fact this tool can know.
    """
    env = os.environ.get
    repo = args.repo or env("GITHUB_REPOSITORY", "")
    run_id = args.run_id or env("GITHUB_RUN_ID", "")
    # The workflow name and run title both live on the same Actions API object;
    # fetch once and let flags/env take precedence over what we resolve.
    name, title = ("", "")
    if not (args.workflow_name and args.workflow_title):
        name, title = _fetch_run_meta(repo, run_id)
    return {
        "runId": run_id,
        "branch": args.branch or env("GITHUB_REF_NAME", ""),
        "gitSha": args.git_sha or env("GITHUB_SHA", ""),
        "workflowName": args.workflow_name or env("GITHUB_WORKFLOW", "") or name,
        "workflowTitle": args.workflow_title or title,
    }


def _fetch_run_meta(repo: str, run_id: str) -> tuple[str, str]:
    """Fetch a run's workflow name and run title from the GitHub Actions API, as
    `(workflowName, workflowTitle)`.

    The run object carries the run's `display_title` and its `workflow_id`; the
    workflow object carries the workflow's `name` (the `name:` in the yaml, e.g.
    "Run Test"). Public repos need no auth; `GITHUB_TOKEN` is used if present.
    Returns empties on any failure."""
    if not (repo and run_id):
        return ("", "")
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base = f"https://api.github.com/repos/{repo}"
    try:
        run = httpx.get(f"{base}/actions/runs/{run_id}", headers=headers, timeout=15)
        run.raise_for_status()
        run = run.json()
        title = run.get("display_title", "") or ""
        name = ""
        if wf_id := run.get("workflow_id"):
            wf = httpx.get(
                f"{base}/actions/workflows/{wf_id}", headers=headers, timeout=15
            )
            wf.raise_for_status()
            name = wf.json().get("name", "") or ""
        return (name, title)
    except httpx.HTTPError:
        return ("", "")


def _fetch_jobs(repo: str, run_id: str) -> dict[str, str]:
    """Map GitHub job id -> canonical job name for a run (e.g. "test / test n150
    ... (tt-ubuntu-2204-n150-stable, run_forge_models, 4)"). Public repos need no
    auth. Returns {} on any failure."""
    if not (repo and run_id):
        return {}
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
    jobs: dict[str, str] = {}
    try:
        page = 1
        while True:
            r = httpx.get(
                base,
                params={"per_page": 100, "page": page},
                headers=headers,
                timeout=15,
            )
            r.raise_for_status()
            batch = r.json().get("jobs", [])
            for j in batch:
                jobs[str(j["id"])] = j.get("name", "") or ""
            if len(batch) < 100:
                break
            page += 1
    except httpx.HTTPError:
        return jobs
    return jobs


def _job_meta(target: Path, jobs: dict[str, str]) -> dict[str, str]:
    """Per-artifact job provenance. CI artifact names end in the GitHub job id
    (e.g. `graph-telemetry-...-<jobId>`); pull it out and resolve the canonical
    name. Empty for non-CI targets, so `_patch_graph` leaves the doc untouched."""
    job_id = target.name.rsplit("-", 1)[-1]
    if not job_id.isdigit():
        return {}
    return {"jobId": job_id, "jobName": jobs.get(job_id, "")}


def _patch_graph(data: bytes, meta: dict[str, str]) -> bytes:
    """Fill empty workflow fields in a telemetry doc's `graph` block from `meta`.

    Existing non-empty values win, so an explicit value in the document is never
    overwritten. Non-JSON payloads pass through untouched.
    """
    try:
        doc = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return data
    graph = doc.get("graph")
    if not isinstance(graph, dict):
        return data
    changed = False
    for key, val in meta.items():
        if val and not graph.get(key):
            graph[key] = val
            changed = True
    return json.dumps(doc).encode() if changed else data


def _members(target: Path) -> list[tuple[str, bytes]]:
    """Enumerate (arcname, bytes) for a target, preserving the telemetry layout
    (`<graph>.json` + `<graph>/<file>.mlir`) so each snapshot's mlirPath resolves."""
    if target.suffix == ".zip":
        with zipfile.ZipFile(target) as z:
            return [(n, z.read(n)) for n in z.namelist() if not n.endswith("/")]
    out: list[tuple[str, bytes]] = []
    if target.is_dir():
        for f in sorted(target.rglob("*")):
            if f.is_file():
                out.append((f.relative_to(target).as_posix(), f.read_bytes()))
    elif target.suffix == ".json":
        out.append((target.name, target.read_bytes()))
        sidecar = target.with_suffix("")  # <parent>/<graph>/
        if sidecar.is_dir():
            for f in sorted(sidecar.rglob("*")):
                if f.is_file():
                    arc = Path(sidecar.name) / f.relative_to(sidecar)
                    out.append((arc.as_posix(), f.read_bytes()))
    return out


def build_archive(target: Path, meta: dict[str, str] | None = None) -> bytes:
    """Produce zip bytes for a target, patching run-level CI `meta` into every
    JSON document's graph block. `_patch_graph` only fills empty fields, so a
    doc's own values (e.g. the emitter's `testName`) are left untouched."""
    meta = meta or {}
    members = _members(target)
    members = [
        (name, _patch_graph(data, meta)) if name.endswith(".json") else (name, data)
        for name, data in members
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def push_target(
    url: str,
    target: Path,
    timeout: int,
    meta: dict[str, str] | None = None,
    token: str = "",
) -> bool:
    """Send one target as a zip to /ingest/archive. Returns True on success."""
    try:
        blob = build_archive(target, meta)
    except (OSError, zipfile.BadZipFile) as e:
        print(f"  {target.name}: FAIL (could not read: {e})", file=sys.stderr)
        return False

    headers = {"Content-Type": "application/zip"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = httpx.post(
            f"{url}/ingest/archive",
            content=blob,
            headers=headers,
            timeout=timeout,
        )
        r.raise_for_status()
        body = r.json()
        print(
            f"  {target.name}: OK (graphs={body.get('graphsIngested')}, "
            f"files={body.get('filesWritten')})"
        )
        return True
    except httpx.ConnectError:
        print(
            f"  {target.name}: FAIL (connection refused — is mlir-graph-serve "
            f"running at {url}?)",
            file=sys.stderr,
        )
    except httpx.HTTPStatusError as e:
        print(
            f"  {target.name}: FAIL (HTTP {e.response.status_code}: "
            f"{e.response.text})",
            file=sys.stderr,
        )
    except httpx.TimeoutException:
        print(f"  {target.name}: FAIL (request timed out)", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Push telemetry (JSON + MLIR) to mlir-graph-serve"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Telemetry directories, .zip artifacts, .json files, or globs",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="mlir-graph-serve URL (e.g. http://serve:8321)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GRAPH_TELEMETRY_TOKEN", ""),
        help="Write-scoped bearer token (default: $GRAPH_TELEMETRY_TOKEN)",
    )
    parser.add_argument(
        "--workflow-name",
        default="",
        help="CI workflow name (default: $GITHUB_WORKFLOW)",
    )
    parser.add_argument(
        "--workflow-title",
        default="",
        help="CI run display title (default: fetched from the GitHub Actions API)",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="CI run id grouping these graphs (default: $GITHUB_RUN_ID)",
    )
    parser.add_argument(
        "--repo",
        default="",
        help="owner/repo for fetching the run title (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--branch",
        default="",
        help="CI branch (default: $GITHUB_REF_NAME)",
    )
    parser.add_argument(
        "--git-sha",
        default="",
        help="Commit the graphs were compiled from (default: $GITHUB_SHA)",
    )
    args = parser.parse_args()

    targets = collect_targets(args.paths)
    if not targets:
        print("Error: no telemetry found", file=sys.stderr)
        sys.exit(1)

    meta = resolve_provenance(args)
    if any(meta.values()):
        print(
            f"Workflow: {meta['workflowName'] or '—'} / "
            f"{meta['workflowTitle'] or '—'}"
        )
        print(
            f"Run: {meta['runId'] or '—'}  branch: {meta['branch'] or '—'}  "
            f"sha: {meta['gitSha'][:12] or '—'}"
        )
    # Each artifact is one CI job; resolve job names once and stamp the matching
    # one (by the job id in the artifact name) onto that artifact's graphs.
    jobs = _fetch_jobs(
        args.repo or os.environ.get("GITHUB_REPOSITORY", ""), meta["runId"]
    )
    print(f"Pushing {len(targets)} archive(s) to {args.url}")
    succeeded = sum(
        push_target(
            args.url, t, args.timeout, {**meta, **_job_meta(t, jobs)}, args.token
        )
        for t in targets
    )
    failed = len(targets) - succeeded

    print(f"\nSummary: {succeeded} succeeded, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
