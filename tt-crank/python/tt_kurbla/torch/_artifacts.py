"""Collecting the artifacts a run produced — today the IR of each compiled graph.

Each compiled graph contributes an `Artifact` while a `collect_artifacts` context
is open; closing it writes one directory under `$TT_KURBLA_ARTIFACTS_DIR`, plus an
`artifacts.json` index of what is in it.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import _native

# Index of a dump directory: what was compiled, and with which options.
_ARTIFACTS_JSON = "artifacts.json"


class Artifact:
    def __init__(
        self,
        compile_options: dict[str, Any],
        compile_result: _native.CompileResult,
        graph_kind: str | None = None,
    ):
        self.compile_options = compile_options
        self.compile_result = compile_result
        # `forward` / `backward` / `inference`, when the caller could tell.
        self.graph_kind = graph_kind


def _artifacts_root() -> Path:
    # Load path from the env config.
    return Path(_native.artifacts_dir())


def _path_safe(name: str) -> str:
    """`name` with everything that isn't path-safe folded to `_`. A collection name
    comes from the caller (a pytest node name, a benchmark label) and becomes a path
    component."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")


def _unique(name: str, taken: Callable[[str], bool]) -> str:
    """`name`, suffixed `_1`, `_2`, ... until `taken` says it is free.

    Two dumps of the same collection within the same second must not merge into
    one directory.
    """
    candidate, n = name, 1
    while taken(candidate):
        candidate = f"{name}_{n}"
        n += 1
    return candidate


def _collection_dir_name(collection_name: str, stamp: dt.datetime) -> str:
    """`<collection>_<UTC timestamp>`: sorts chronologically, and successive runs
    of the same collection sit side by side rather than clobbering each other.

    Takes the timestamp rather than reading the clock so the directory name and
    the `timestamp` recorded in `artifacts.json` cannot disagree.
    """
    return f"{_path_safe(collection_name)}_{stamp.strftime('%Y%m%d_%H%M%S')}"


class _ArtifactsDumperContext:
    """One collection's worth of artifacts.

    Single use: built when a `collect_artifacts` block opens, dumped and
    thrown away when it closes. Nothing to reset, so nothing can be left stale.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.artifacts: list[Artifact] = []

    def register_artifact(self, artifact: Artifact):
        self.artifacts.append(artifact)

    def dump(self) -> Path | None:
        """Write every registered artifact, plus an `artifacts.json` index, into a
        fresh directory.

        Returns the directory written, or `None` when nothing was collected - an
        empty timestamped directory is pure noise.
        """
        if not self.artifacts:
            return None

        stamp = dt.datetime.now(dt.timezone.utc)
        root = _artifacts_root()
        name = _unique(
            _collection_dir_name(self.collection_name, stamp),
            lambda candidate: (root / candidate).exists(),
        )
        out_dir = root / name
        # No `exist_ok`: `_unique` just established the path is free, so a
        # collision here is a real race and should be loud rather than a silent
        # merge of two runs' IR.
        out_dir.mkdir(parents=True)

        graphs: list[dict[str, Any]] = []
        for index, artifact in enumerate(self.artifacts):
            # Registration order, so a model's forward comes before its backward.
            stem = f"graph_{index}" if artifact.graph_kind is None else f"graph_{index}_{artifact.graph_kind}"

            result = artifact.compile_result
            entry: dict[str, Any] = {"graph": stem}
            for field, suffix, ir in (
                ("ttir", ".ttir.mlir", result.ttir),
                ("ttnn", ".ttnn.mlir", result.program.ttnn_ir()),
            ):
                # An empty IR string means there is nothing to write, and an empty
                # file reads as "this compiled to nothing" — so leave the file out
                # and record the absence as an explicit `null` in `artifacts.json`.
                if ir:
                    (out_dir / f"{stem}{suffix}").write_text(ir, encoding="utf-8")
                entry[field] = f"{stem}{suffix}" if ir else None

            entry["cache_hit"] = result.cache_hit
            entry["compile_options"] = artifact.compile_options
            graphs.append(entry)

        # Named `contents` rather than `artifacts` so it doesn't read as the
        # `Artifact` list it was built from.
        contents = {
            "collection": self.collection_name,
            "timestamp": stamp.isoformat(),
            "graphs": graphs,
        }
        (out_dir / _ARTIFACTS_JSON).write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")

        return out_dir


# Installed for the duration of a `collect_artifacts` block and `None`
# the rest of the time: being set *is* being active, so there is no separate flag
# that can disagree with it.
_global_artifacts_dumper_context: _ArtifactsDumperContext | None = None


@contextmanager
def collect_artifacts(collection_name: str):
    """Collect the artifacts produced in this context, dumped on exit into
    `$TT_KURBLA_ARTIFACTS_DIR/<collection_name>_<timestamp>/`.
    """
    global _global_artifacts_dumper_context

    assert _global_artifacts_dumper_context is None, "artifacts dumper is already active!"
    assert _path_safe(collection_name), f"collection name has no path-safe characters: {collection_name!r}"
    context = _ArtifactsDumperContext(collection_name)
    _global_artifacts_dumper_context = context

    try:
        yield
    finally:
        # Uninstall before dumping, so we leave with the dumper uninstalled no
        # matter whether the dump throws. Skipped if `dump_artifacts()` in the body
        # already wrote this context out.
        if _global_artifacts_dumper_context is context:
            _global_artifacts_dumper_context = None
            context.dump()


def is_artifacts_dumper_active() -> bool:
    return _global_artifacts_dumper_context is not None


def register_artifact(artifact: Artifact):
    assert _global_artifacts_dumper_context is not None, "artifacts dumper is not active!"

    _global_artifacts_dumper_context.register_artifact(artifact)


def dump_artifacts() -> Path | None:
    """Write out what has been collected so far and stop collecting, rather than
    waiting for the enclosing `collect_artifacts` block to close."""
    global _global_artifacts_dumper_context

    assert _global_artifacts_dumper_context is not None, "artifacts dumper is not active!"
    context = _global_artifacts_dumper_context
    _global_artifacts_dumper_context = None

    return context.dump()
