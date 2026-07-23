#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Audit Tracy host zones inside synchronized ttrt invocation windows."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROGRAM_METADATA_PREFIX = "MLIR_PROGRAM_METADATA;"


def _parse_metadata(value: str) -> dict:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"program metadata is not a dictionary: {value!r}")
    if "program_index" not in parsed or "loop_number" not in parsed:
        raise ValueError(f"program metadata lacks program/loop identity: {value!r}")
    return parsed


def _load_markers(path: Path) -> dict[tuple[int, int], list[int]]:
    markers: dict[tuple[int, int], list[int]] = defaultdict(list)
    with path.open(errors="replace") as trace_file:
        for line in trace_file:
            if not line.startswith(PROGRAM_METADATA_PREFIX):
                continue
            message, timestamp = line.rstrip().rsplit(";", 1)
            metadata = _parse_metadata(message.removeprefix(PROGRAM_METADATA_PREFIX))
            identity = (
                int(metadata["program_index"]),
                int(metadata["loop_number"]),
            )
            markers[identity].append(int(timestamp))
    if not markers:
        raise ValueError(f"no {PROGRAM_METADATA_PREFIX[:-1]} markers found: {path}")
    return dict(markers)


def _load_host_results(path: Path) -> dict[tuple[int, int], dict[str, int]]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"run results must contain a JSON list: {path}")

    results = {}
    for entry in entries:
        for program_name, loops in entry.get("program_results", {}).items():
            program_index = int(program_name.removeprefix("program_index_"))
            for loop_name, values in loops.items():
                loop_number = int(loop_name.removeprefix("loop_"))
                results[(program_index, loop_number)] = {
                    "submit_ns": int(values["total_submit_duration_ns"]),
                    "output_ns": int(values["total_get_outputs_duration_ns"]),
                    "total_ns": int(
                        values["total_submit_plus_get_outputs_duration_ns"]
                    ),
                }
    if not results:
        raise ValueError(f"run results contain no program loop timings: {path}")
    return results


def _build_windows(
    markers: dict[tuple[int, int], list[int]],
    host_results: dict[tuple[int, int], dict[str, int]],
) -> dict[tuple[int, int], dict]:
    if set(markers) != set(host_results):
        raise ValueError(
            "program/loop identities differ between trace markers and run results: "
            f"markers_only={sorted(set(markers) - set(host_results))}, "
            f"results_only={sorted(set(host_results) - set(markers))}"
        )

    windows = {}
    for identity, timestamps in markers.items():
        start_ns = min(timestamps)
        timing = host_results[identity]
        windows[identity] = {
            "start_ns": start_ns,
            "end_ns": start_ns + timing["total_ns"],
            "marker_count": len(timestamps),
            "last_marker_ns": max(timestamps),
            "last_marker_offset_ms": (max(timestamps) - start_ns) / 1_000_000,
            "host_submit_ms": timing["submit_ns"] / 1_000_000,
            "host_output_ms": timing["output_ns"] / 1_000_000,
            "host_total_ms": timing["total_ns"] / 1_000_000,
        }
    return windows


def _load_zones(path: Path) -> list[dict]:
    zones = []
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            start_ns = int(row["ns_since_start"])
            duration_ns = int(row["exec_time_ns"])
            if duration_ns < 0:
                raise ValueError(f"negative Tracy zone duration: {row}")
            zones.append(
                {
                    "name": row["zone_name"] or row["name"] or "<unnamed>",
                    "thread": row["thread"],
                    "start_ns": start_ns,
                    "end_ns": start_ns + duration_ns,
                }
            )
    if not zones:
        raise ValueError(f"Tracy times CSV has no zones: {path}")
    return zones


def _union_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def _summarize_window(window: dict, zones: list[dict], top: int) -> dict:
    start_ns = window["start_ns"]
    end_ns = window["end_ns"]
    by_name: dict[str, list[int]] = defaultdict(list)
    by_thread: dict[str, list[tuple[int, int]]] = defaultdict(list)
    overlapping_rows = 0

    for zone in zones:
        clipped_start = max(start_ns, zone["start_ns"])
        clipped_end = min(end_ns, zone["end_ns"])
        if clipped_start >= clipped_end:
            continue
        duration_ns = clipped_end - clipped_start
        overlapping_rows += 1
        by_name[zone["name"]].append(duration_ns)
        by_thread[zone["thread"]].append((clipped_start, clipped_end))

    top_zones = []
    for name, durations in sorted(
        by_name.items(), key=lambda item: sum(item[1]), reverse=True
    )[:top]:
        top_zones.append(
            {
                "name": name,
                "count": len(durations),
                "inclusive_ms": sum(durations) / 1_000_000,
                "median_us": statistics.median(durations) / 1_000,
                "max_ms": max(durations) / 1_000_000,
            }
        )

    thread_coverage = []
    for thread, intervals in sorted(
        by_thread.items(),
        key=lambda item: _union_duration(item[1].copy()),
        reverse=True,
    ):
        covered_ns = _union_duration(intervals)
        thread_coverage.append(
            {
                "thread": thread,
                "zone_rows": len(intervals),
                "covered_ms": covered_ns / 1_000_000,
                "window_fraction": covered_ns / (end_ns - start_ns),
            }
        )

    return {
        **window,
        "tracy_zone_rows_overlapping_window": overlapping_rows,
        "top_zones_by_inclusive_time": top_zones,
        "per_thread_zone_interval_union": thread_coverage,
    }


def _build_checks(windows: dict[tuple[int, int], dict]) -> list[dict[str, str]]:
    checks = []
    by_program: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for (program, loop), window in windows.items():
        by_program[program].append((loop, window))
    marker_counts = {
        window["marker_count"]
        for loop_windows in by_program.values()
        for _, window in sorted(loop_windows)[1:]
    }
    checks.append(
        {
            "name": "steady_invocation_marker_count_stability",
            "status": "pass" if len(marker_counts) <= 1 else "warn",
            "detail": f"steady marker counts: {sorted(marker_counts)}",
        }
    )

    late_markers = [
        identity
        for identity, window in windows.items()
        if window["last_marker_ns"] > window["end_ns"]
    ]
    checks.append(
        {
            "name": "metadata_markers_within_host_envelope",
            "status": "pass" if not late_markers else "fail",
            "detail": f"violations: {late_markers}",
        }
    )

    alignment_errors = []
    for program, loop_windows in by_program.items():
        ordered = sorted(loop_windows)
        for (loop, window), (next_loop, next_window) in zip(ordered, ordered[1:]):
            delta_ns = next_window["start_ns"] - window["end_ns"]
            tolerance_ns = max(5_000_000, int(window["host_total_ms"] * 50_000))
            if abs(delta_ns) > tolerance_ns:
                alignment_errors.append(
                    {
                        "program": program,
                        "loop": loop,
                        "next_loop": next_loop,
                        "delta_ms": delta_ns / 1_000_000,
                    }
                )
    checks.append(
        {
            "name": "consecutive_loop_window_alignment",
            "status": "pass" if not alignment_errors else "warn",
            "detail": f"violations: {alignment_errors}",
        }
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("times_csv", type=Path)
    parser.add_argument("trace_data", type=Path)
    parser.add_argument("run_results", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")

    markers = _load_markers(args.trace_data)
    windows = _build_windows(markers, _load_host_results(args.run_results))
    zones = _load_zones(args.times_csv)
    summaries = {
        f"program_{program}_loop_{loop}": _summarize_window(window, zones, args.top)
        for (program, loop), window in sorted(windows.items())
    }
    report = {
        "times_csv": str(args.times_csv),
        "trace_data": str(args.trace_data),
        "run_results": str(args.run_results),
        "interpretation": (
            "Zone durations are inclusive and may be nested or overlap. "
            "Do not add top-zone times. Per-thread coverage is an interval "
            "union of exported Tracy zones, not CPU utilization."
        ),
        "invocations": summaries,
        "checks": _build_checks(windows),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
