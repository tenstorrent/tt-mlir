#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Calculate performance metrics from profiling data.

This script analyzes the profiling data collected in the prof directory
and calculates key performance metrics even when host-side operation tracking
is not available.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional


def calculate_perf_from_cpp_device_report(csv_path: Path) -> Dict[str, Any]:
    """Calculate performance metrics from cpp_device_perf_report.csv.

    This is the primary source of device-side performance data and contains
    aggregated kernel timing information.
    """
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}

    results = {"source": str(csv_path), "operations": [], "summary": {}}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        total_kernel_duration = 0
        total_fw_duration = 0
        total_ops = 0

        for row in reader:
            op_data = {}

            # Extract key performance metrics
            if (
                "DEVICE KERNEL DURATION [ns]" in row
                and row["DEVICE KERNEL DURATION [ns]"]
            ):
                kernel_dur = int(row["DEVICE KERNEL DURATION [ns]"])
                op_data["kernel_duration_ns"] = kernel_dur
                op_data["kernel_duration_ms"] = kernel_dur / 1_000_000
                total_kernel_duration += kernel_dur

            if "DEVICE FW DURATION [ns]" in row and row["DEVICE FW DURATION [ns]"]:
                fw_dur = int(row["DEVICE FW DURATION [ns]"])
                op_data["fw_duration_ns"] = fw_dur
                op_data["fw_duration_ms"] = fw_dur / 1_000_000
                total_fw_duration += fw_dur

            # Core utilization
            if "CORE COUNT" in row and row["CORE COUNT"]:
                op_data["cores_used"] = int(row["CORE COUNT"])

            if (
                "AVAILABLE WORKER CORE COUNT" in row
                and row["AVAILABLE WORKER CORE COUNT"]
            ):
                op_data["available_cores"] = int(row["AVAILABLE WORKER CORE COUNT"])

            # Per-RISC breakdown
            risc_metrics = {}
            for risc in ["BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2", "ERISC"]:
                key = f"DEVICE {risc} KERNEL DURATION [ns]"
                if key in row and row[key]:
                    duration = int(row[key])
                    risc_metrics[risc.lower()] = {
                        "duration_ns": duration,
                        "duration_ms": duration / 1_000_000,
                    }

            if risc_metrics:
                op_data["risc_breakdown"] = risc_metrics

            # Device info
            if "DEVICE ID" in row and row["DEVICE ID"]:
                op_data["device_id"] = int(row["DEVICE ID"])

            if "DEVICE ARCH" in row and row["DEVICE ARCH"]:
                op_data["device_arch"] = row["DEVICE ARCH"]

            # Op identification
            if "GLOBAL CALL COUNT" in row and row["GLOBAL CALL COUNT"]:
                op_data["op_id"] = int(row["GLOBAL CALL COUNT"])

            if "OP NAME" in row and row["OP NAME"]:
                op_data["op_name"] = row["OP NAME"]

            results["operations"].append(op_data)
            total_ops += 1

    # Calculate summary statistics
    results["summary"] = {
        "total_operations": total_ops,
        "total_kernel_duration_ns": total_kernel_duration,
        "total_kernel_duration_ms": total_kernel_duration / 1_000_000,
        "total_kernel_duration_s": total_kernel_duration / 1_000_000_000,
        "total_fw_duration_ns": total_fw_duration,
        "total_fw_duration_ms": total_fw_duration / 1_000_000,
        "total_fw_duration_s": total_fw_duration / 1_000_000_000,
        "avg_kernel_duration_ms": (total_kernel_duration / total_ops / 1_000_000)
        if total_ops > 0
        else 0,
    }

    return results


def calculate_perf_from_device_log(
    csv_path: Path, freq_mhz: int = 1000
) -> Dict[str, Any]:
    """Calculate performance metrics from profile_log_device.csv.

    This provides more detailed per-core, per-RISC timing information but
    requires more processing.
    """
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}

    results = {"source": str(csv_path), "zones": [], "summary": {}}

    # Parse the header line to get architecture and frequency
    with open(csv_path, "r") as f:
        first_line = f.readline().strip()
        if "CHIP_FREQ[MHz]:" in first_line:
            parts = first_line.split(",")
            for part in parts:
                if "CHIP_FREQ[MHz]:" in part:
                    freq_mhz = int(part.split(":")[1].strip())

    # Read zone data
    with open(csv_path, "r") as f:
        # Skip the header comment line
        f.readline()

        reader = csv.DictReader(f)

        zone_durations = {}
        core_activity = {}

        for row in reader:
            zone_name = row.get("zone name", "")
            zone_type = row.get("type", "")
            risc_type = row.get("RISC processor type", "")
            core_loc = (row.get("core_x", ""), row.get("core_y", ""))

            if zone_type == "ZONE_START":
                key = (zone_name, risc_type, core_loc)
                zone_durations[key] = int(row.get("time[cycles since reset]", 0))

            elif zone_type == "ZONE_END" and zone_name:
                key = (zone_name, risc_type, core_loc)
                if key in zone_durations:
                    start_cycle = zone_durations[key]
                    end_cycle = int(row.get("time[cycles since reset]", 0))
                    duration_cycles = end_cycle - start_cycle
                    duration_ns = (duration_cycles * 1000) / freq_mhz

                    zone_key = f"{zone_name}_{risc_type}"
                    if zone_key not in core_activity:
                        core_activity[zone_key] = []

                    core_activity[zone_key].append(
                        {
                            "core": core_loc,
                            "duration_cycles": duration_cycles,
                            "duration_ns": duration_ns,
                            "duration_ms": duration_ns / 1_000_000,
                        }
                    )

                    del zone_durations[key]

    # Aggregate by zone type
    zone_summary = {}
    for zone_key, measurements in core_activity.items():
        durations_ms = [m["duration_ms"] for m in measurements]
        zone_summary[zone_key] = {
            "count": len(measurements),
            "total_ms": sum(durations_ms),
            "avg_ms": sum(durations_ms) / len(durations_ms),
            "min_ms": min(durations_ms),
            "max_ms": max(durations_ms),
        }

    results["zone_summary"] = zone_summary
    results["summary"] = {
        "total_zones_measured": sum(s["count"] for s in zone_summary.values()),
        "unique_zone_types": len(zone_summary),
        "chip_freq_mhz": freq_mhz,
    }

    return results


def main():
    """Main entry point."""
    prof_dir = Path("prof")

    if not prof_dir.exists():
        print(f"Error: {prof_dir} directory not found")
        return 1

    print("=" * 80)
    print("Performance Analysis from Profiling Data")
    print("=" * 80)

    # Try to find the most recent report
    logs_dir = prof_dir / ".logs"
    reports_dirs = sorted(prof_dir.glob("reports/*"), reverse=True)

    # Method 1: Use cpp_device_perf_report.csv (most reliable)
    cpp_report = logs_dir / "cpp_device_perf_report.csv"
    if cpp_report.exists():
        print(f"\n📊 Analyzing C++ Device Performance Report")
        print(f"   Source: {cpp_report}")
        print("-" * 80)

        results = calculate_perf_from_cpp_device_report(cpp_report)

        if "error" not in results:
            summary = results["summary"]
            print(f"\n✅ Performance Summary:")
            print(f"   Total Operations: {summary['total_operations']}")
            print(
                f"   Total Kernel Duration: {summary['total_kernel_duration_ms']:.3f} ms ({summary['total_kernel_duration_s']:.6f} s)"
            )
            print(
                f"   Total FW Duration: {summary['total_fw_duration_ms']:.3f} ms ({summary['total_fw_duration_s']:.6f} s)"
            )
            print(
                f"   Average Kernel Duration: {summary['avg_kernel_duration_ms']:.3f} ms"
            )

            print(f"\n📋 Operation Details:")
            for i, op in enumerate(results["operations"]):
                print(f"\n   Operation {i}:")
                print(f"      Op ID: {op.get('op_id', 'N/A')}")
                print(
                    f"      Kernel Duration: {op.get('kernel_duration_ms', 0):.3f} ms"
                )
                print(f"      FW Duration: {op.get('fw_duration_ms', 0):.3f} ms")
                print(
                    f"      Cores Used: {op.get('cores_used', 'N/A')} / {op.get('available_cores', 'N/A')}"
                )
                print(
                    f"      Device: {op.get('device_arch', 'N/A')} (ID: {op.get('device_id', 'N/A')})"
                )

                if "risc_breakdown" in op:
                    print(f"      RISC Breakdown:")
                    for risc, metrics in op["risc_breakdown"].items():
                        print(
                            f"         {risc.upper()}: {metrics['duration_ms']:.3f} ms"
                        )

            # Save detailed results
            output_file = prof_dir / "perf_analysis.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Detailed results saved to: {output_file}")
        else:
            print(f"   ❌ Error: {results['error']}")
    else:
        print(f"\n⚠️  cpp_device_perf_report.csv not found at {cpp_report}")

    # Method 2: Use profile_log_device.csv (more detailed but requires more processing)
    device_log = logs_dir / "profile_log_device.csv"
    if device_log.exists():
        print(f"\n📊 Analyzing Device Log (Detailed)")
        print(f"   Source: {device_log}")
        print("-" * 80)

        results = calculate_perf_from_device_log(device_log)

        if "error" not in results:
            summary = results["summary"]
            print(f"\n✅ Device Log Summary:")
            print(f"   Total Zones Measured: {summary['total_zones_measured']}")
            print(f"   Unique Zone Types: {summary['unique_zone_types']}")
            print(f"   Chip Frequency: {summary['chip_freq_mhz']} MHz")

            if "zone_summary" in results and results["zone_summary"]:
                print(f"\n📋 Zone Performance:")
                for zone_name, metrics in sorted(
                    results["zone_summary"].items(),
                    key=lambda x: x[1]["total_ms"],
                    reverse=True,
                ):
                    print(f"      {zone_name}:")
                    print(
                        f"         Count: {metrics['count']}, Total: {metrics['total_ms']:.3f} ms"
                    )
                    print(
                        f"         Avg: {metrics['avg_ms']:.6f} ms, Min: {metrics['min_ms']:.6f} ms, Max: {metrics['max_ms']:.6f} ms"
                    )
        else:
            print(f"   ❌ Error: {results['error']}")
    else:
        print(f"\n⚠️  profile_log_device.csv not found at {device_log}")

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
