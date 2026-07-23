#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reconcile wall, Tracy host, device, and binary performance manifests."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return value


def _trailing_index(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if not match:
        raise ValueError(f"invocation name lacks a trailing index: {name}")
    return int(match.group(1))


def _steady(invocations: dict) -> list[dict]:
    ordered = [
        value
        for _, value in sorted(
            invocations.items(), key=lambda item: _trailing_index(item[0])
        )
    ]
    if len(ordered) < 2:
        raise ValueError("at least one cold and one steady invocation are required")
    return ordered[1:]


def _median(invocations: list[dict], field: str) -> float:
    return statistics.median(float(invocation[field]) for invocation in invocations)


def _zone_median(invocations: list[dict], name: str) -> float:
    totals = []
    for invocation in invocations:
        matching = [
            zone
            for zone in invocation["top_zones_by_inclusive_time"]
            if zone["name"] == name
        ]
        if len(matching) != 1:
            raise ValueError(
                f"expected exactly one aggregate for zone {name!r}, found {len(matching)}"
            )
        totals.append(float(matching[0]["inclusive_ms"]))
    return statistics.median(totals)


def _wall_result(manifest: dict, backend: str) -> dict:
    matches = [
        result
        for result in manifest["results"]
        if (backend == "d2m" and result["backend"] == "D2M-JIT")
        or (backend == "ttnn" and "TTNN" in result["backend"])
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {backend} wall result, found {len(matches)}")
    return matches[0]


def _binary(manifest: dict, suffix: str) -> dict:
    matches = [
        binary
        for binary in manifest["binaries"]
        if Path(binary["path"]).suffix == suffix
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} binary, found {len(matches)}")
    return matches[0]


def _relative_error(measured: float, reference: float) -> float:
    return abs(measured - reference) / reference


def _stability(samples: list[float]) -> float:
    median = statistics.median(samples)
    return max(abs(sample - median) / median for sample in samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2m-device", type=Path, required=True)
    parser.add_argument("--ttnn-device", type=Path, required=True)
    parser.add_argument("--d2m-host", type=Path, required=True)
    parser.add_argument("--ttnn-host", type=Path, required=True)
    parser.add_argument("--wall", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = {
        "d2m": _load(args.d2m_device),
        "ttnn": _load(args.ttnn_device),
    }
    host = {
        "d2m": _load(args.d2m_host),
        "ttnn": _load(args.ttnn_host),
    }
    wall_manifest = _load(args.wall)
    binary_manifest = _load(args.binaries)
    wall = {
        backend: _wall_result(wall_manifest, backend) for backend in ("d2m", "ttnn")
    }
    binaries = {
        "d2m": _binary(binary_manifest, ".ttm"),
        "ttnn": _binary(binary_manifest, ".ttnn"),
    }

    metrics = {}
    checks = []
    for backend in ("d2m", "ttnn"):
        host_steady = _steady(host[backend]["invocations"])
        device_steady = _steady(device[backend]["invocations"])
        profiled_submit_ms = _median(host_steady, "host_submit_ms")
        profiler_read_ms = _zone_median(host_steady, "ReadMeshDeviceProfilerResults")
        corrected_submit_ms = profiled_submit_ms - profiler_read_ms
        wall_submit_ms = float(wall[backend]["phase_medians_ms"]["submit_wait_ms"])
        correction_error = _relative_error(corrected_submit_ms, wall_submit_ms)
        samples = [float(sample) for sample in wall[backend]["samples_ms"]]
        stability = _stability(samples)

        metrics[backend] = {
            "wall_total_ms": float(wall[backend]["median_ms"]),
            "wall_submit_wait_ms": wall_submit_ms,
            "profiled_host_submit_ms": profiled_submit_ms,
            "profiled_profiler_read_zone_ms": profiler_read_ms,
            "profiled_submit_minus_profiler_read_ms": corrected_submit_ms,
            "corrected_submit_relative_error": correction_error,
            "device_span_ms": _median(device_steady, "device_fw_span_ms"),
            "device_kernel_row_sum_ms": _median(
                device_steady, "summed_kernel_duration_ms"
            ),
            "device_inter_row_gap_ms": _median(device_steady, "inter_row_gap_ms"),
            "wall_max_relative_deviation": stability,
        }
        checks.extend(
            (
                {
                    "name": f"{backend}_profile_correction_matches_wall_submit",
                    "status": "pass" if correction_error <= 0.1 else "warn",
                    "detail": f"relative error: {correction_error:.3%}",
                },
                {
                    "name": f"{backend}_wall_sample_stability",
                    "status": "pass" if stability <= 0.1 else "warn",
                    "detail": f"max deviation from median: {stability:.3%}",
                },
                {
                    "name": f"{backend}_wall_binary_hash_matches_manifest",
                    "status": (
                        "pass"
                        if wall[backend]["binary"].get("sha256")
                        == binaries[backend]["sha256"]
                        else "fail"
                    ),
                    "detail": f"sha256: {binaries[backend]['sha256']}",
                },
            )
        )

    d2m_host_steady = _steady(host["d2m"]["invocations"])
    d2m_write_ms = _zone_median(d2m_host_steady, "EnqueueWriteBufferCommand")
    d2m_write_bytes = binaries["d2m"]["ttmetal_transfers"]["by_direction"][
        "host_to_device"
    ]["total_payload_bytes"]
    ttnn_input_bytes = sum(
        descriptor["size_bytes"]
        for descriptor in binaries["ttnn"]["programs"][0]["inputs"]
    )
    ttnn_input_ms = float(wall["ttnn"]["phase_medians_ms"]["input_enqueue_ms"])
    ratios = {
        "wall_total_d2m_over_ttnn": (
            metrics["d2m"]["wall_total_ms"] / metrics["ttnn"]["wall_total_ms"]
        ),
        "device_span_d2m_over_ttnn": (
            metrics["d2m"]["device_span_ms"] / metrics["ttnn"]["device_span_ms"]
        ),
        "device_kernel_row_sum_d2m_over_ttnn": (
            metrics["d2m"]["device_kernel_row_sum_ms"]
            / metrics["ttnn"]["device_kernel_row_sum_ms"]
        ),
    }
    transfer_diagnostics = {
        "d2m_serialized_h2d_bytes": d2m_write_bytes,
        "d2m_profiled_enqueue_write_inclusive_ms": d2m_write_ms,
        "d2m_effective_serialized_write_gb_per_s": (
            d2m_write_bytes / (d2m_write_ms * 1_000_000)
        ),
        "ttnn_logical_input_bytes": ttnn_input_bytes,
        "ttnn_wall_input_enqueue_ms": ttnn_input_ms,
        "ttnn_effective_input_enqueue_gb_per_s": (
            ttnn_input_bytes / (ttnn_input_ms * 1_000_000)
        ),
        "interpretation": (
            "The D2M rate uses serialized physical flatbuffer payload and an "
            "inclusive Tracy zone. The TTNN rate uses logical inputs and the "
            "wall input-enqueue envelope. They are diagnostics, not a direct "
            "bandwidth benchmark."
        ),
    }
    checks.extend(binary_manifest.get("checks", []))

    report = {
        "sources": {
            name: str(getattr(args, name)) for name in vars(args) if name != "output"
        },
        "metrics": metrics,
        "ratios": ratios,
        "transfer_diagnostics": transfer_diagnostics,
        "checks": checks,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
