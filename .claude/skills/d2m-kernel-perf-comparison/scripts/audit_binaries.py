#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Record executable provenance and check logical main-program parity."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import _ttmlir_runtime as runtime

_DATA_TYPE_BYTES = {
    "BFloat16": 2,
    "Float16": 2,
    "Float32": 4,
    "Int32": 4,
    "UInt32": 4,
    "UInt16": 2,
    "UInt8": 1,
}


def _parse_json(raw: str) -> object:
    return json.loads(re.sub(r"\binf\b", "Infinity", re.sub(r"\bnan\b", "NaN", raw)))


def _descriptor_summary(entry: dict) -> dict:
    desc = entry["desc"]
    memory_desc = desc["layout"]["memory_desc"]
    data_type = memory_desc["data_type"]
    size_bytes = entry.get("size")
    if size_bytes is None and data_type in _DATA_TYPE_BYTES:
        size_bytes = math.prod(desc["shape"]) * _DATA_TYPE_BYTES[data_type]
    return {
        "shape": desc["shape"],
        "data_type": data_type,
        "size_bytes": size_bytes,
    }


def _type_counts(value: object) -> dict[str, int]:
    counts = collections.Counter()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if "type_type" in node:
                counts[str(node["type_type"])] += 1
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return dict(sorted(counts.items()))


def _walk(value: object):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _buffer_payload_bytes(buffer: dict) -> int | None:
    desc = buffer.get("desc", {})
    data_type = desc.get("data_type")
    element_bytes = _DATA_TYPE_BYTES.get(data_type)
    if element_bytes is None:
        return None
    if "host_volume" in desc:
        return int(desc["host_volume"]) * element_bytes
    if "shape" in desc:
        return math.prod(desc["shape"]) * element_bytes
    return None


def _transfer_summary(flatbuffer: dict) -> dict:
    commands = []
    for node in _walk(flatbuffer):
        if not isinstance(node, dict):
            continue
        command_type = node.get("type_type")
        if command_type not in (
            "EnqueueWriteBufferCommand",
            "EnqueueReadBufferCommand",
        ):
            continue

        command = node["type"]
        src = command["src"]
        dst = command["dst"]
        host_buffer = src if command_type == "EnqueueWriteBufferCommand" else dst
        commands.append(
            {
                "direction": (
                    "host_to_device"
                    if command_type == "EnqueueWriteBufferCommand"
                    else "device_to_host"
                ),
                "payload_bytes": _buffer_payload_bytes(host_buffer),
                "source_global_id": src.get("global_id"),
                "source_shape": src.get("desc", {}).get("shape"),
                "destination_global_id": dst.get("global_id"),
                "destination_shape": dst.get("desc", {}).get("shape"),
                "location": node.get("loc"),
            }
        )

    by_direction = {}
    for direction in ("host_to_device", "device_to_host"):
        selected = [
            command for command in commands if command["direction"] == direction
        ]
        payloads = [
            command["payload_bytes"]
            for command in selected
            if command["payload_bytes"] is not None
        ]
        by_direction[direction] = {
            "command_count": len(selected),
            "total_payload_bytes": sum(payloads),
            "unknown_payload_count": len(selected) - len(payloads),
        }
    return {"by_direction": by_direction, "commands": commands}


def _audit(path: Path) -> dict:
    binary = runtime.binary.load_binary_from_path(str(path))
    flatbuffer = _parse_json(binary.as_json())
    programs = []
    for index in range(binary.get_num_programs()):
        inputs = _parse_json(binary.get_program_inputs_as_json(index))
        outputs = _parse_json(binary.get_program_outputs_as_json(index))
        program = {
            "index": index,
            "name": binary.get_program_name(index),
            "inputs": [_descriptor_summary(entry) for entry in inputs],
            "outputs": [_descriptor_summary(entry) for entry in outputs],
        }
        if path.suffix != ".ttm":
            ops_json = binary.get_program_ops_as_json(index)
            if ops_json:
                program["operation_type_counts"] = _type_counts(_parse_json(ops_json))
        programs.append(program)

    system_desc = flatbuffer["system_desc"]
    result = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "version": binary.version,
        "ttmlir_git_hash": binary.ttmlir_git_hash,
        "program_count": binary.get_num_programs(),
        "programs": programs,
        "target": {
            "chip_arches": sorted(
                {chip_desc["arch"] for chip_desc in system_desc["chip_descs"]}
            ),
            "physical_grid_sizes": sorted(
                {
                    (chip_desc["grid_size"]["y"], chip_desc["grid_size"]["x"])
                    for chip_desc in system_desc["chip_descs"]
                }
            ),
            "main_mesh_shape": flatbuffer["programs"][0]["mesh_shape"],
        },
        "flatbuffer_type_counts": _type_counts(flatbuffer),
    }
    if path.suffix == ".ttm":
        result["ttmetal_transfers"] = _transfer_summary(flatbuffer)
    return result


def _parity_checks(audit: list[dict]) -> list[dict[str, str]]:
    if len(audit) < 2:
        return []
    reference = audit[0]["programs"][0]
    checks = []
    for candidate in audit[1:]:
        main = candidate["programs"][0]
        for field in ("inputs", "outputs"):
            matches = main[field] == reference[field]
            checks.append(
                {
                    "name": f"main_{field}_match:{Path(candidate['path']).name}",
                    "status": "pass" if matches else "fail",
                    "detail": f"reference={Path(audit[0]['path']).name}",
                }
            )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binaries", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = [_audit(path) for path in args.binaries]
    report = {
        "runtime_module": runtime.__file__,
        "binaries": audit,
        "checks": _parity_checks(audit),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
