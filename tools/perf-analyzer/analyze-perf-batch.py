#!/usr/bin/env python3

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Advanced performance analysis tool for batch test results.

This script provides deeper analysis of performance data collected by
batch-perf-test.sh, including:
- Statistical analysis (percentiles, stddev)
- Performance anomaly detection
- Comparison across multiple batch runs
- Visualization (if matplotlib available)
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_summary_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load the performance summary CSV."""
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistical measures for a list of values."""
    if not values:
        return {}

    values = sorted(values)
    n = len(values)

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    stddev = variance**0.5

    # Percentiles
    def percentile(p):
        k = (n - 1) * p
        f = int(k)
        c = k - f
        if f + 1 < n:
            return values[f] + c * (values[f + 1] - values[f])
        return values[f]

    return {
        "mean": mean,
        "stddev": stddev,
        "median": percentile(0.5),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": min(values),
        "max": max(values),
    }


def analyze_performance(rows: List[Dict[str, str]]) -> None:
    """Generate detailed performance analysis."""
    passed_rows = [r for r in rows if r["status"] == "PASSED"]

    if not passed_rows:
        print("No successful tests to analyze")
        return

    # Extract metrics
    total_times = [
        float(r["total_device_time_ms"])
        for r in passed_rows
        if float(r["total_device_time_ms"]) > 0
    ]
    kernel_counts = [
        int(r["kernel_count"]) for r in passed_rows if int(r["kernel_count"]) > 0
    ]
    avg_kernel_times = [
        float(r["avg_kernel_time_ms"])
        for r in passed_rows
        if float(r["avg_kernel_time_ms"]) > 0
    ]
    max_kernel_times = [
        float(r["max_kernel_time_ms"])
        for r in passed_rows
        if float(r["max_kernel_time_ms"]) > 0
    ]

    print("# Advanced Performance Analysis")
    print()

    # Total device time statistics
    if total_times:
        print("## Total Device Time Statistics (ms)")
        print()
        stats = calculate_statistics(total_times)
        print(f"- Count: {len(total_times)}")
        print(f"- Mean: {stats['mean']:.3f}ms")
        print(f"- Median: {stats['median']:.3f}ms")
        print(f"- Std Dev: {stats['stddev']:.3f}ms")
        print(f"- Min: {stats['min']:.3f}ms")
        print(f"- Max: {stats['max']:.3f}ms")
        print(f"- P25: {stats['p25']:.3f}ms")
        print(f"- P75: {stats['p75']:.3f}ms")
        print(f"- P90: {stats['p90']:.3f}ms")
        print(f"- P95: {stats['p95']:.3f}ms")
        print(f"- P99: {stats['p99']:.3f}ms")
        print()

    # Kernel count statistics
    if kernel_counts:
        print("## Kernel Count Statistics")
        print()
        stats = calculate_statistics([float(k) for k in kernel_counts])
        print(f"- Total kernels: {sum(kernel_counts)}")
        print(f"- Mean per test: {stats['mean']:.1f}")
        print(f"- Median: {stats['median']:.1f}")
        print(f"- Std Dev: {stats['stddev']:.2f}")
        print(f"- Min: {int(stats['min'])}")
        print(f"- Max: {int(stats['max'])}")
        print()

    # Average kernel time statistics
    if avg_kernel_times:
        print("## Average Kernel Time Statistics (ms)")
        print()
        stats = calculate_statistics(avg_kernel_times)
        print(f"- Mean: {stats['mean']:.3f}ms")
        print(f"- Median: {stats['median']:.3f}ms")
        print(f"- Std Dev: {stats['stddev']:.3f}ms")
        print(f"- Min: {stats['min']:.3f}ms")
        print(f"- Max: {stats['max']:.3f}ms")
        print()

    # Detect outliers (tests with unusually high times)
    if total_times and len(total_times) > 3:
        print("## Performance Outliers")
        print()
        stats = calculate_statistics(total_times)
        threshold = stats["p95"]

        outliers = [
            r for r in passed_rows if float(r["total_device_time_ms"]) > threshold
        ]

        if outliers:
            print(f"Tests exceeding P95 threshold ({threshold:.3f}ms):")
            print()
            for r in outliers:
                test_name = r["test_name"].split("::")[-1]
                time = float(r["total_device_time_ms"])
                pct_over = ((time - stats["mean"]) / stats["mean"]) * 100
                print(f"- **{test_name}**: {time:.2f}ms ({pct_over:+.1f}% vs mean)")
            print()
        else:
            print("No significant outliers detected.")
            print()

    # Efficiency analysis (total time per kernel)
    if total_times and kernel_counts and len(total_times) == len(kernel_counts):
        print("## Efficiency Analysis")
        print()

        # Calculate time per kernel for each test
        time_per_kernel = [
            (float(r["total_device_time_ms"]) / int(r["kernel_count"]), r["test_name"])
            for r in passed_rows
            if int(r["kernel_count"]) > 0
        ]

        if time_per_kernel:
            time_per_kernel.sort(key=lambda x: x[0])

            # Most efficient (lowest time per kernel)
            print("### Most Efficient Tests (lowest ms/kernel):")
            print()
            for time, name in time_per_kernel[:5]:
                test_name = name.split("::")[-1]
                print(f"- {test_name}: {time:.3f}ms/kernel")
            print()

            # Least efficient (highest time per kernel)
            print("### Least Efficient Tests (highest ms/kernel):")
            print()
            for time, name in reversed(time_per_kernel[-5:]):
                test_name = name.split("::")[-1]
                print(f"- {test_name}: {time:.3f}ms/kernel")
            print()


def compare_runs(csv_paths: List[Path]) -> None:
    """Compare performance across multiple batch runs."""
    if len(csv_paths) < 2:
        print("Need at least 2 CSV files to compare")
        return

    print("# Multi-Run Comparison")
    print()

    # Load all runs
    runs = []
    for i, csv_path in enumerate(csv_paths, 1):
        rows = load_summary_csv(csv_path)
        runs.append(
            {
                "name": f"Run {i} ({csv_path.parent.name})",
                "path": csv_path,
                "rows": rows,
            }
        )

    # Find common tests
    test_names_per_run = [set(r["test_name"] for r in run["rows"]) for run in runs]
    common_tests = set.intersection(*test_names_per_run)

    print(f"Found {len(common_tests)} common tests across {len(runs)} runs")
    print()

    if not common_tests:
        print("No common tests found for comparison")
        return

    # Compare performance for common tests
    print("## Performance Comparison (Common Tests)")
    print()
    print(
        "| Test Name | "
        + " | ".join(f"Run {i}" for i in range(1, len(runs) + 1))
        + " | Delta (%) |"
    )
    print("|-----------|" + "|".join("----------" for _ in runs) + "|-----------|")

    for test_name in sorted(common_tests):
        times = []
        for run in runs:
            row = next((r for r in run["rows"] if r["test_name"] == test_name), None)
            if row and row["status"] == "PASSED":
                time = float(row["total_device_time_ms"])
                times.append(time if time > 0 else None)
            else:
                times.append(None)

        # Skip if not all runs have valid data
        if None in times:
            continue

        # Calculate delta (% change from first run)
        delta = ((times[-1] - times[0]) / times[0]) * 100 if times[0] > 0 else 0
        delta_str = f"{delta:+.1f}%"

        short_name = test_name.split("::")[-1]
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."

        time_cols = " | ".join(f"{t:.2f}ms" for t in times)
        print(f"| {short_name} | {time_cols} | {delta_str} |")

    print()


def export_json(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Export performance data as JSON."""
    data = {
        "tests": [],
        "summary": {
            "total_tests": len(rows),
            "passed": sum(1 for r in rows if r["status"] == "PASSED"),
            "failed": sum(1 for r in rows if r["status"] == "FAILED"),
        },
    }

    for row in rows:
        data["tests"].append(
            {
                "name": row["test_name"],
                "status": row["status"],
                "total_device_time_ms": float(row["total_device_time_ms"])
                if row["total_device_time_ms"]
                else 0,
                "kernel_count": int(row["kernel_count"]) if row["kernel_count"] else 0,
                "avg_kernel_time_ms": float(row["avg_kernel_time_ms"])
                if row["avg_kernel_time_ms"]
                else 0,
                "max_kernel_time_ms": float(row["max_kernel_time_ms"])
                if row["max_kernel_time_ms"]
                else 0,
                "min_kernel_time_ms": float(row["min_kernel_time_ms"])
                if row["min_kernel_time_ms"]
                else 0,
                "output_dir": row["output_dir"],
            }
        )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze performance data from batch-perf-test.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single run
  %(prog)s prof_batch/performance_summary.csv

  # Compare multiple runs
  %(prog)s --compare run1/performance_summary.csv run2/performance_summary.csv

  # Export to JSON
  %(prog)s prof_batch/performance_summary.csv --export-json results.json
        """,
    )

    parser.add_argument(
        "csv_files", nargs="+", type=Path, help="Performance summary CSV file(s)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple runs (requires 2+ CSV files)",
    )
    parser.add_argument(
        "--export-json", type=Path, metavar="FILE", help="Export data to JSON file"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Write output to file instead of stdout"
    )

    args = parser.parse_args()

    # Validate input files
    for csv_file in args.csv_files:
        if not csv_file.exists():
            print(f"Error: {csv_file} not found", file=sys.stderr)
            sys.exit(1)

    # Redirect output if requested
    if args.output:
        sys.stdout = open(args.output, "w")

    try:
        if args.compare:
            # Comparison mode
            compare_runs(args.csv_files)
        else:
            # Single run analysis
            rows = load_summary_csv(args.csv_files[0])
            analyze_performance(rows)

            if args.export_json:
                export_json(rows, args.export_json)
    finally:
        if args.output:
            sys.stdout.close()


if __name__ == "__main__":
    main()
