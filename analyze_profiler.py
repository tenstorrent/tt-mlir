# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Analyzes TTNN profiler CSV files to extract trace replay performance metrics.
Skips warmup and trace capture phases, only sums actual execution times.
"""

import csv
import sys
from pathlib import Path


def analyze_profiler_csv(csv_path):
    """
    Analyzes a profiler CSV file and extracts trace replay metrics.

    Expected structure:
    - First N ops: Warmup (any ops with device duration before trace capture)
    - Middle N ops: Trace capture (has host duration, NO device duration)
    - Last N ops: Trace replay (ops with device duration AFTER trace capture)
    """

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Find the trace capture boundary (first op with no device duration)
    capture_start_idx = None
    capture_end_idx = None

    for idx, row in enumerate(rows):
        # Skip total/average rows
        if row["OP CODE"] in ["TOTAL", "AVERAGE"]:
            continue

        host_dur = row["HOST DURATION [ns]"].strip()
        device_dur = row["DEVICE KERNEL DURATION [ns]"].strip()

        # First operation with no device duration marks start of trace capture
        if host_dur and not device_dur and capture_start_idx is None:
            capture_start_idx = idx

        # First operation with device duration after capture marks end of capture
        if capture_start_idx is not None and host_dur and device_dur:
            capture_end_idx = idx
            break

    # Separate operations into phases
    warmup_ops = []
    capture_ops = []
    replay_ops = []

    for idx, row in enumerate(rows):
        # Skip total/average rows
        if row["OP CODE"] in ["TOTAL", "AVERAGE"]:
            continue

        host_dur = row["HOST DURATION [ns]"].strip()
        device_dur = row["DEVICE KERNEL DURATION [ns]"].strip()

        # Before capture: warmup
        if capture_start_idx is not None and idx < capture_start_idx:
            if host_dur and device_dur:
                warmup_ops.append(row)
        # During capture
        elif (
            capture_start_idx is not None
            and capture_end_idx is not None
            and idx >= capture_start_idx
            and idx < capture_end_idx
        ):
            if host_dur and not device_dur:
                capture_ops.append(row)
        # After capture: replay
        elif capture_end_idx is not None and idx >= capture_end_idx:
            if host_dur and device_dur:
                replay_ops.append(row)

    print(f"\n{'='*70}")
    print(f"Profiler Analysis: {Path(csv_path).name}")
    print(f"{'='*70}")

    print(f"\nPhase Breakdown:")
    print(f"  Warmup operations:        {len(warmup_ops)}")
    print(f"  Trace capture operations: {len(capture_ops)}")
    print(f"  Trace replay operations:  {len(replay_ops)}")

    # Analyze trace replay operations
    if replay_ops:
        print(f"\n{'-'*70}")
        print(f"TRACE REPLAY ANALYSIS (Performance Metrics)")
        print(f"{'-'*70}")

        print(f"\nTrace Replay Operations (lines being summed):")
        print(
            f"{'Index':<8}{'Host Duration (ns)':<20}{'Device Duration (ns)':<22}{'Device (μs)':<15}"
        )
        print(f"{'-'*65}")

        total_device_ns = 0
        total_host_ns = 0
        device_times = []

        for idx, row in enumerate(replay_ops, 1):
            device_ns = int(row["DEVICE KERNEL DURATION [ns]"])
            host_ns = int(row["HOST DURATION [ns]"])

            print(f"{idx:<8}{host_ns:<20}{device_ns:<22}{device_ns/1e3:<15.2f}")

            total_device_ns += device_ns
            total_host_ns += host_ns
            device_times.append(device_ns)

        avg_device_ns = total_device_ns / len(replay_ops)
        avg_host_ns = total_host_ns / len(replay_ops)

        print(f"\nDevice Kernel Execution:")
        print(f"  Total time:    {total_device_ns:,} ns ({total_device_ns/1e6:.3f} ms)")
        print(f"  Average/op:    {avg_device_ns:,.1f} ns ({avg_device_ns/1e3:.2f} μs)")
        print(
            f"  Min/op:        {min(device_times):,} ns ({min(device_times)/1e3:.2f} μs)"
        )
        print(
            f"  Max/op:        {max(device_times):,} ns ({max(device_times)/1e3:.2f} μs)"
        )

        print(f"\nHost Overhead:")
        print(f"  Total time:    {total_host_ns:,} ns ({total_host_ns/1e6:.3f} ms)")
        print(f"  Average/op:    {avg_host_ns:,.1f} ns ({avg_host_ns/1e3:.2f} μs)")

        # Categorize by operation duration (likely matmul vs element-wise)
        heavy_ops = [t for t in device_times if t > 20000]  # > 20μs
        light_ops = [t for t in device_times if t <= 20000]  # <= 20μs

        if heavy_ops and light_ops:
            print(f"\nOperation Classification:")
            print(
                f"  Heavy ops (>20μs): {len(heavy_ops)} ops, avg {sum(heavy_ops)/len(heavy_ops)/1e3:.2f} μs (likely matmuls)"
            )
            print(
                f"  Light ops (≤20μs): {len(light_ops)} ops, avg {sum(light_ops)/len(light_ops)/1e3:.2f} μs (likely element-wise)"
            )

    else:
        print("\n⚠️  WARNING: No trace replay operations found!")

    print(f"\n{'='*70}\n")

    return {
        "warmup_count": len(warmup_ops),
        "capture_count": len(capture_ops),
        "replay_count": len(replay_ops),
        "total_device_ns": total_device_ns if replay_ops else 0,
        "avg_device_ns": avg_device_ns if replay_ops else 0,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_profiler.py <csv_file>")
        print("\nExample:")
        print(
            "  python analyze_profiler.py third_party/tt-metal/src/tt-metal/generated/profiler/resnet_jit_4_32.csv"
        )
        sys.exit(1)

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    analyze_profiler_csv(csv_path)
