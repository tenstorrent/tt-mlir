#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Audit device profiler rows without treating their sum as makespan."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

START_CYCLE = "DEVICE FW START CYCLE"
END_CYCLE = "DEVICE FW END CYCLE"
FW_DURATION_NS = "DEVICE FW DURATION [ns]"
KERNEL_DURATION_NS = "DEVICE KERNEL DURATION [ns]"
CALL_COUNT = "GLOBAL CALL COUNT"
PROGRAM_METADATA = "PROGRAM_METADATA"

_DEVICE_OP_CALL_COUNT_RE = re.compile(r",\s*(\d+)\s*(?:->|`)")


def _parse_metadata(value: str | None) -> dict:
    if not value:
        return {}
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"program metadata is not a dictionary: {value!r}")
    return parsed


def _load_trace_metadata(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}

    current_metadata: dict = {}
    metadata_by_call_count: dict[int, dict] = {}
    with path.open(errors="replace") as trace_file:
        for line in trace_file:
            if line.startswith("MLIR_PROGRAM_METADATA;"):
                current_metadata = _parse_metadata(line.split(";", 2)[1])
                continue

            if "TT_DNN_DEVICE_OP" not in line:
                continue
            match = _DEVICE_OP_CALL_COUNT_RE.search(line)
            if match and current_metadata:
                metadata_by_call_count[int(match.group(1))] = current_metadata

    return metadata_by_call_count


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"profiler CSV has no data rows: {path}")

    required = {START_CYCLE, END_CYCLE, KERNEL_DURATION_NS, CALL_COUNT}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"profiler CSV is missing columns: {sorted(missing)}")
    return rows


def _partition_by_metadata(
    rows: list[dict[str, str]], metadata_by_call_count: dict[int, dict]
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    unmatched = []
    for row in rows:
        metadata = _parse_metadata(row.get(PROGRAM_METADATA))
        if not metadata:
            metadata = metadata_by_call_count.get(int(row[CALL_COUNT]), {})

        if "program_index" not in metadata or "loop_number" not in metadata:
            unmatched.append(row)
            continue

        key = f"program_{metadata['program_index']}_" f"loop_{metadata['loop_number']}"
        groups[key].append(row)
    return dict(groups), unmatched


def _infer_repeated_invocations(
    rows: list[dict[str, str]], repetitions: int
) -> dict[str, list[dict[str, str]]]:
    rows_by_call_count: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_call_count.setdefault(int(row[CALL_COUNT]), []).append(row)

    invalid = {
        call_count: len(call_rows)
        for call_count, call_rows in rows_by_call_count.items()
        if len(call_rows) != repetitions
    }
    if invalid:
        preview = dict(list(invalid.items())[:8])
        raise ValueError(
            "cannot infer invocations: every call-count group must contain "
            f"exactly {repetitions} rows; mismatches include {preview}"
        )

    return {
        f"inferred_invocation_{invocation}": [
            call_rows[invocation] for call_rows in rows_by_call_count.values()
        ]
        for invocation in range(repetitions)
    }


def _partition_contiguous_invocations(
    rows: list[dict[str, str]], rows_per_invocation: int
) -> dict[str, list[dict[str, str]]]:
    if len(rows) % rows_per_invocation:
        raise ValueError(
            f"cannot split {len(rows)} rows into blocks of {rows_per_invocation}"
        )

    start_cycles = [int(row[START_CYCLE]) for row in rows]
    if any(left > right for left, right in zip(start_cycles, start_cycles[1:])):
        raise ValueError(
            "cannot use contiguous partitioning: device start cycles are not "
            "monotonic in CSV order"
        )

    call_counts = [int(row[CALL_COUNT]) for row in rows]
    if any(left >= right for left, right in zip(call_counts, call_counts[1:])):
        raise ValueError(
            "cannot use contiguous partitioning: global call counts are not "
            "strictly increasing"
        )

    return {
        f"inferred_invocation_{index // rows_per_invocation}": rows[
            index : index + rows_per_invocation
        ]
        for index in range(0, len(rows), rows_per_invocation)
    }


def _summarize_rows(
    rows: list[dict[str, str]], clock_mhz: float | None
) -> dict[str, int | float | None]:
    intervals = sorted((int(row[START_CYCLE]), int(row[END_CYCLE])) for row in rows)
    invalid_intervals = sum(start > end for start, end in intervals)
    if invalid_intervals:
        raise ValueError(f"found {invalid_intervals} device intervals with end < start")

    covered_cycles = 0
    covered_start, covered_end = intervals[0]
    for start, end in intervals[1:]:
        if start > covered_end:
            covered_cycles += covered_end - covered_start
            covered_start, covered_end = start, end
        else:
            covered_end = max(covered_end, end)
    covered_cycles += covered_end - covered_start

    first_start = intervals[0][0]
    last_end = max(end for _, end in intervals)
    span_cycles = last_end - first_start
    fw_sum_ns = sum(int(row.get(FW_DURATION_NS) or 0) for row in rows)
    kernel_sum_ns = sum(int(row[KERNEL_DURATION_NS]) for row in rows)

    result: dict[str, int | float | None] = {
        "rows": len(rows),
        "first_device_fw_start_cycle": first_start,
        "last_device_fw_end_cycle": last_end,
        "device_fw_span_cycles": span_cycles,
        "profiled_fw_interval_union_cycles": covered_cycles,
        "inter_row_gap_cycles": span_cycles - covered_cycles,
        "summed_fw_duration_ms": fw_sum_ns / 1_000_000,
        "summed_kernel_duration_ms": kernel_sum_ns / 1_000_000,
    }
    if clock_mhz is None:
        result["device_fw_span_ms"] = None
        result["profiled_fw_interval_union_ms"] = None
        result["inter_row_gap_ms"] = None
    else:
        cycles_per_ms = clock_mhz * 1_000
        result["device_fw_span_ms"] = span_cycles / cycles_per_ms
        result["profiled_fw_interval_union_ms"] = covered_cycles / cycles_per_ms
        result["inter_row_gap_ms"] = (span_cycles - covered_cycles) / cycles_per_ms
    return result


def _load_host_results(path: Path | None) -> dict[tuple[int, int], dict]:
    if path is None:
        return {}
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"run results must contain a JSON list: {path}")

    host_results = {}
    for entry in entries:
        for program_name, loops in entry.get("program_results", {}).items():
            program_index = int(program_name.removeprefix("program_index_"))
            for loop_name, values in loops.items():
                loop_number = int(loop_name.removeprefix("loop_"))
                host_results[(program_index, loop_number)] = values
    return host_results


def _group_identity(name: str) -> tuple[int, int] | None:
    metadata_match = re.fullmatch(r"program_(\d+)_loop_(\d+)", name)
    if metadata_match:
        return int(metadata_match.group(1)), int(metadata_match.group(2))
    inferred_match = re.fullmatch(r"inferred_invocation_(\d+)", name)
    if inferred_match:
        return 0, int(inferred_match.group(1))
    return None


def _attach_host_results(
    summaries: dict[str, dict], host_results: dict[tuple[int, int], dict]
) -> None:
    for name, summary in summaries.items():
        identity = _group_identity(name)
        values = host_results.get(identity) if identity is not None else None
        if values is None:
            continue

        submit_ns = int(values["total_submit_duration_ns"])
        output_ns = int(values["total_get_outputs_duration_ns"])
        total_ns = int(values["total_submit_plus_get_outputs_duration_ns"])
        summary["host_submit_wait_ms"] = submit_ns / 1_000_000
        summary["host_output_ms"] = output_ns / 1_000_000
        summary["host_submit_plus_output_ms"] = total_ns / 1_000_000
        if summary["device_fw_span_ms"] is not None:
            summary["host_envelope_minus_device_span_ms"] = (
                total_ns / 1_000_000 - summary["device_fw_span_ms"]
            )


def _build_checks(
    summaries: dict[str, dict], unmatched_summary: dict | None
) -> list[dict[str, str]]:
    checks = []
    row_counts = {summary["rows"] for summary in summaries.values()}
    checks.append(
        {
            "name": "invocation_row_count_stability",
            "status": "pass" if len(row_counts) <= 1 else "warn",
            "detail": f"row counts: {sorted(row_counts)}",
        }
    )

    checks.append(
        {
            "name": "unmatched_profiler_rows",
            "status": "pass" if unmatched_summary is None else "warn",
            "detail": (
                "none"
                if unmatched_summary is None
                else f"{unmatched_summary['rows']} rows lack invocation metadata"
            ),
        }
    )

    host_pairs = [
        (name, summary)
        for name, summary in summaries.items()
        if "host_submit_plus_output_ms" in summary
        and summary["device_fw_span_ms"] is not None
    ]
    if host_pairs:
        violations = [
            name
            for name, summary in host_pairs
            if summary["device_fw_span_ms"]
            > summary["host_submit_plus_output_ms"] * 1.05
        ]
        checks.append(
            {
                "name": "device_span_within_synchronized_host_envelope",
                "status": "pass" if not violations else "fail",
                "detail": (
                    f"checked {len(host_pairs)} invocations with 5% clock tolerance; "
                    f"violations: {violations}"
                ),
            }
        )

    spans = [
        summary["device_fw_span_ms"]
        for _, summary in sorted(summaries.items())
        if summary["device_fw_span_ms"] is not None
    ]
    steady_spans = spans[1:]
    if len(steady_spans) >= 2:
        median_span = statistics.median(steady_spans)
        max_deviation = max(
            abs(span - median_span) / median_span for span in steady_spans
        )
        checks.append(
            {
                "name": "steady_device_span_stability",
                "status": "pass" if max_deviation <= 0.1 else "warn",
                "detail": f"max deviation from median: {max_deviation:.3%}",
            }
        )

    gap_fractions = {
        name: summary["inter_row_gap_cycles"] / summary["device_fw_span_cycles"]
        for name, summary in summaries.items()
        if summary["device_fw_span_cycles"]
    }
    checks.append(
        {
            "name": "inter_row_gap_fraction",
            "status": "info",
            "detail": ", ".join(
                f"{name}={fraction:.1%}"
                for name, fraction in sorted(gap_fractions.items())
            ),
        }
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--trace-data",
        type=Path,
        help="Matching tracy_ops_data.csv used to recover loop metadata.",
    )
    partition_group = parser.add_mutually_exclusive_group()
    partition_group.add_argument(
        "--repetitions",
        type=int,
        help=(
            "Infer operation-major invocation order when every call-count "
            "group repeats exactly this many times."
        ),
    )
    partition_group.add_argument(
        "--rows-per-invocation",
        type=int,
        help=(
            "Split an invocation-major report into contiguous blocks of this "
            "validated row count."
        ),
    )
    parser.add_argument(
        "--clock-mhz",
        type=float,
        help="Device clock used to convert firmware cycles to milliseconds.",
    )
    parser.add_argument(
        "--run-results",
        type=Path,
        help="Matching ttrt result JSON containing synchronized host envelopes.",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON report here.")
    args = parser.parse_args()

    if args.repetitions is not None and args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.rows_per_invocation is not None and args.rows_per_invocation <= 0:
        parser.error("--rows-per-invocation must be positive")
    if args.clock_mhz is not None and args.clock_mhz <= 0:
        parser.error("--clock-mhz must be positive")

    rows = _load_rows(args.csv)
    trace_metadata = _load_trace_metadata(args.trace_data)
    groups, unmatched = _partition_by_metadata(rows, trace_metadata)

    if args.repetitions is not None:
        if groups:
            raise ValueError(
                "--repetitions cannot be combined with rows that already have "
                "program/loop metadata"
            )
        groups = _infer_repeated_invocations(unmatched, args.repetitions)
        unmatched = []
    elif args.rows_per_invocation is not None:
        if groups:
            raise ValueError(
                "--rows-per-invocation cannot be combined with rows that "
                "already have program/loop metadata"
            )
        groups = _partition_contiguous_invocations(unmatched, args.rows_per_invocation)
        unmatched = []

    summaries = {
        name: _summarize_rows(invocation_rows, args.clock_mhz)
        for name, invocation_rows in sorted(groups.items())
    }
    unmatched_summary = (
        _summarize_rows(unmatched, args.clock_mhz) if unmatched else None
    )
    _attach_host_results(summaries, _load_host_results(args.run_results))

    report = {
        "source_csv": str(args.csv),
        "trace_data": str(args.trace_data) if args.trace_data else None,
        "run_results": str(args.run_results) if args.run_results else None,
        "partition": {
            "repetitions": args.repetitions,
            "rows_per_invocation": args.rows_per_invocation,
        },
        "clock_mhz": args.clock_mhz,
        "total_rows": len(rows),
        "invocations": summaries,
        "unmatched_rows": unmatched_summary,
        "checks": _build_checks(summaries, unmatched_summary),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
