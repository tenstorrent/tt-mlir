#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fast profiler analyzer: Extract and sum only trace replay device kernel durations.
Only sums operations AFTER the trace capture phase (ops with no device duration).
Handles multiple trace replay iterations and returns the average.
Outputs only the average total in nanoseconds.
"""

import csv
import sys
from pathlib import Path


def get_trace_replay_sum(csv_path, debug=False, num_iterations=10):
    """
    Read CSV, find trace capture boundary, sum trace replay device kernel times,
    and return the average across multiple iterations.
    Returns average total in nanoseconds.

    The profiler output has 3 phases:
    1. Warmup: ops with device duration
    2. Trace Capture: ops with NO device duration (host only)
    3. Trace Replay: ops with device duration, repeated num_iterations times

    With num_iterations=10, the trace replay phase will be 10x the capture phase size.
    We divide the replay ops into blocks, sum each block, and return the average.
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

    # Count total replay operations (all ops with device duration after capture)
    total_replay_ops = 0
    for idx in range(capture_end_idx, len(data_rows)):
        row = data_rows[idx]
        device_dur = row["DEVICE KERNEL DURATION [ns]"].strip()
        if device_dur:
            total_replay_ops += 1

    # Calculate operations per iteration
    # This handles any number of operations (4, 6, or other)
    if total_replay_ops % num_iterations != 0:
        if debug:
            print(
                f"WARNING: Total replay ops ({total_replay_ops}) not evenly divisible by iterations ({num_iterations})",
                file=sys.stderr,
            )

    ops_per_iteration = total_replay_ops // num_iterations
    replay_end_idx = capture_end_idx + total_replay_ops

    if debug:
        print(f"DEBUG: Capture phase size: {capture_size} operations", file=sys.stderr)
        print(f"DEBUG: Number of iterations: {num_iterations}", file=sys.stderr)
        print(
            f"DEBUG: Total replay operations found: {total_replay_ops}", file=sys.stderr
        )
        print(f"DEBUG: Operations per iteration: {ops_per_iteration}", file=sys.stderr)
        print(
            f"DEBUG: Replay phase: indices {capture_end_idx} to {replay_end_idx-1}",
            file=sys.stderr,
        )

    # Divide replay operations into num_iterations blocks and sum each
    iteration_sums = []

    for iteration in range(num_iterations):
        iteration_start = capture_end_idx + (iteration * ops_per_iteration)
        iteration_end = iteration_start + ops_per_iteration
        iteration_sum = 0

        if debug:
            print(
                f"\nDEBUG: === Iteration {iteration + 1}/{num_iterations} ===",
                file=sys.stderr,
            )
            print(
                f"DEBUG: Processing indices {iteration_start} to {iteration_end-1}",
                file=sys.stderr,
            )
            print(
                f"DEBUG: {'Index':<8} {'OP CODE':<30} {'Device Duration (ns)':<20}",
                file=sys.stderr,
            )
            print(f"DEBUG: {'-'*60}", file=sys.stderr)

        for idx in range(iteration_start, min(iteration_end, len(data_rows))):
            row = data_rows[idx]
            device_dur = row["DEVICE KERNEL DURATION [ns]"].strip()

            if device_dur:
                duration = int(device_dur)
                iteration_sum += duration

                if debug:
                    op_code = row["OP CODE"][:28]
                    print(
                        f"DEBUG: {idx:<8} {op_code:<30} {duration:<20}",
                        file=sys.stderr,
                    )

        iteration_sums.append(iteration_sum)

        if debug:
            print(
                f"DEBUG: Iteration {iteration + 1} total: {iteration_sum} ns ({iteration_sum/1e6:.3f} ms)",
                file=sys.stderr,
            )

    # Calculate average
    if iteration_sums:
        avg_device_ns = sum(iteration_sums) / len(iteration_sums)
    else:
        avg_device_ns = 0

    if debug:
        print(f"\nDEBUG: {'-'*80}", file=sys.stderr)
        print(f"DEBUG: All iteration sums (ns): {iteration_sums}", file=sys.stderr)
        print(
            f"DEBUG: Total across all iterations: {sum(iteration_sums)} ns",
            file=sys.stderr,
        )
        print(
            f"DEBUG: Average per iteration: {avg_device_ns:.0f} ns ({avg_device_ns/1e6:.3f} ms)",
            file=sys.stderr,
        )
        min_iter = min(iteration_sums) if iteration_sums else 0
        max_iter = max(iteration_sums) if iteration_sums else 0
        print(
            f"DEBUG: Min: {min_iter} ns, Max: {max_iter} ns, Variance: {max_iter - min_iter} ns",
            file=sys.stderr,
        )

    return int(avg_device_ns)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: fast_profiler_sum.py <csv_file> [--debug] [--iterations N]",
            file=sys.stderr,
        )
        sys.exit(1)

    csv_path = sys.argv[1]

    # Check for debug flag and iterations flag
    debug = "--debug" in sys.argv
    num_iterations = 10  # Default value

    # Parse --iterations flag
    if "--iterations" in sys.argv:
        try:
            iter_idx = sys.argv.index("--iterations")
            if iter_idx + 1 < len(sys.argv):
                num_iterations = int(sys.argv[iter_idx + 1])
            else:
                print("Error: --iterations requires a number", file=sys.stderr)
                sys.exit(1)
        except (ValueError, IndexError):
            print("Error: --iterations must be followed by an integer", file=sys.stderr)
            sys.exit(1)

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Output only the average sum
    print(get_trace_replay_sum(csv_path, debug=debug, num_iterations=num_iterations))
