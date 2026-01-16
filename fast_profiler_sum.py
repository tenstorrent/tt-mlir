#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fast profiler analyzer: Extract and sum only trace replay device kernel durations.
Only sums operations AFTER the trace capture phase (ops with no device duration).
Outputs only the total in nanoseconds.
"""

import csv
import sys
from pathlib import Path


def get_trace_replay_sum(csv_path, debug=False):
    """
    Read CSV, find trace capture boundary, sum only trace replay device kernel times.
    Returns total in nanoseconds.

    The profiler output has 3 phases of equal size:
    1. Warmup: ops with device duration
    2. Trace Capture: ops with NO device duration (host only)
    3. Trace Replay: ops with device duration (the ones we want to sum)
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter out TOTAL/AVERAGE rows
    data_rows = [row for row in rows if row["OP CODE"] not in ["TOTAL", "AVERAGE"]]

    # Find trace capture phase (ops with host duration but NO device duration)
    capture_start_idx = None
    capture_end_idx = None

    for idx, row in enumerate(data_rows):
        host_dur = row["HOST DURATION [ns]"].strip()
        device_dur = row["DEVICE KERNEL DURATION [ns]"].strip()

        # First op with no device duration = trace capture starts
        if host_dur and not device_dur and capture_start_idx is None:
            capture_start_idx = idx
            if debug:
                print(f"DEBUG: Trace capture starts at index {idx}", file=sys.stderr)

        # First op with device duration after capture = trace replay starts
        if (
            capture_start_idx is not None
            and host_dur
            and device_dur
            and capture_end_idx is None
        ):
            capture_end_idx = idx
            if debug:
                print(f"DEBUG: Trace replay starts at index {idx}", file=sys.stderr)
            break

    if capture_start_idx is None or capture_end_idx is None:
        if debug:
            print(
                "DEBUG: Could not find trace capture/replay boundary!", file=sys.stderr
            )
        return 0

    # Calculate the size of the capture phase
    capture_size = capture_end_idx - capture_start_idx

    # Trace replay should be the same size as capture phase
    replay_end_idx = capture_end_idx + capture_size

    if debug:
        print(f"DEBUG: Capture phase size: {capture_size} operations", file=sys.stderr)
        print(
            f"DEBUG: Replay phase: indices {capture_end_idx} to {replay_end_idx-1}",
            file=sys.stderr,
        )

    # Sum only trace replay operations (from capture_end_idx to replay_end_idx)
    total_device_ns = 0
    replay_count = 0

    if debug:
        print(f"\nDEBUG: Summing trace replay operations:", file=sys.stderr)
        print(
            f"DEBUG: {'Index':<8} {'OP CODE':<30} {'Device Duration (ns)':<20} {'Running Total (ns)':<20}",
            file=sys.stderr,
        )
        print(f"DEBUG: {'-'*80}", file=sys.stderr)

    for idx in range(capture_end_idx, min(replay_end_idx, len(data_rows))):
        row = data_rows[idx]
        device_dur = row["DEVICE KERNEL DURATION [ns]"].strip()

        if device_dur:
            duration = int(device_dur)
            total_device_ns += duration
            replay_count += 1

            if debug:
                op_code = row["OP CODE"][:28]  # Truncate long op codes
                print(
                    f"DEBUG: {idx:<8} {op_code:<30} {duration:<20} {total_device_ns:<20}",
                    file=sys.stderr,
                )

    if debug:
        print(f"DEBUG: {'-'*80}", file=sys.stderr)
        print(f"DEBUG: Total replay operations: {replay_count}", file=sys.stderr)
        print(
            f"DEBUG: Total device kernel time: {total_device_ns} ns ({total_device_ns/1e6:.3f} ms)",
            file=sys.stderr,
        )
        print(
            f"DEBUG: Average per op: {total_device_ns/replay_count if replay_count > 0 else 0:.1f} ns\n",
            file=sys.stderr,
        )

    return total_device_ns


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    csv_path = sys.argv[1]

    # Check for debug flag
    debug = len(sys.argv) > 2 and sys.argv[2] == "--debug"

    if not Path(csv_path).exists():
        sys.exit(1)

    # Output only the sum
    print(get_trace_replay_sum(csv_path, debug=debug))
