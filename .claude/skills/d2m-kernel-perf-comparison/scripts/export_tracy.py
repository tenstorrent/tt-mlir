#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Export deterministic CSV views from a raw Tracy capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def _file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _run_export(command: list[str], output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as output_file:
        subprocess.run(command, stdout=output_file, check=True)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--csvexport",
        type=Path,
        help="Path to tracy-csvexport; defaults to the active PATH.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tool = args.csvexport or Path(shutil.which("tracy-csvexport") or "")
    if not tool.is_file():
        parser.error("tracy-csvexport was not found; pass --csvexport")
    if not args.capture.is_file():
        parser.error(f"capture does not exist: {args.capture}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    times_path = args.output_dir / "tracy_ops_times.csv"
    data_path = args.output_dir / "tracy_ops_data.csv"
    manifest_path = args.output_dir / "tracy_export_manifest.json"
    existing = [
        path for path in (times_path, data_path, manifest_path) if path.exists()
    ]
    if existing and not args.force:
        parser.error(f"refusing to overwrite existing outputs: {existing}")

    times_command = [
        str(tool),
        "-u",
        "-t",
        "TT_DNN",
        "-x",
        "CompileProgram,HWCommandQueue_write_buffer",
        str(args.capture),
    ]
    data_command = [str(tool), "-m", "-s", ";", str(args.capture)]
    _run_export(times_command, times_path)
    _run_export(data_command, data_path)

    manifest = {
        "capture": _file_record(args.capture),
        "csvexport": _file_record(tool),
        "commands": [times_command, data_command],
        "outputs": [_file_record(times_path), _file_record(data_path)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
