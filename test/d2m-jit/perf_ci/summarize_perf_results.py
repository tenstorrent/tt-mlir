#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Read the perf_results.json produced by the d2m-jit perf_ci conftest and
# write one Superset-format JSON report per benchmark case.
#
# Usage:
#   python test/d2m-jit/perf_ci/summarize_perf_results.py RUN_DIR [--output-dir DIR] [--job-id ID]

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _measurement(name: str, value: float, step_name: str) -> dict[str, Any]:
    return {
        "measurement_name": name,
        "value": value,
        "iteration": 1,
        "step_name": step_name,
        "step_warm_up_num_iterations": 0,
        "target": -1,
        "device_power": -1.0,
        "device_temperature": -1.0,
    }


def generate_reports(
    raw: list[dict[str, Any]],
    out_dir: Path,
    job_suffix: str,
    quiet: bool,
) -> int:
    file_count = 0
    for r in raw:
        name = r["pattern"]
        shape = r["shape"]
        dtype = r["dtype"]
        d2m_ns = r["d2m_duration_ns"]
        ttnn_ns = r["ttnn_duration_ns"]

        measurements = []
        if d2m_ns > 0:
            measurements.append(_measurement("d2m_kernel_duration_ns", d2m_ns, name))
        if ttnn_ns > 0:
            measurements.append(_measurement("ttnn_kernel_duration_ns", ttnn_ns, name))
        if d2m_ns > 0 and ttnn_ns > 0:
            ratio = round(ttnn_ns / d2m_ns, 4)
            measurements.append(_measurement("perf_ratio", ratio, name))

        report = {
            "project": "tt-mlir",
            "model": name,
            "model_type": "d2m_vs_ttnn",
            "run_type": "d2m_benchmark",
            "precision": dtype,
            "config": {
                "shape": shape,
                "pattern": name,
            },
            "measurements": measurements,
        }

        filename = f"perf_d2m_{name}_{shape}_{dtype}{job_suffix}.json"
        filepath = out_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        file_count += 1
        if not quiet:
            print(f"  Wrote {filepath.name} ({len(measurements)} measurements)")

    if not quiet:
        print(f"Wrote {file_count} report(s) to {out_dir}")
    return file_count


def main():
    parser = argparse.ArgumentParser(
        description="Summarize d2m-jit perf results into JSON reports for Superset."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing perf_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write JSON reports (default: run_dir)",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="GitHub job ID to append to filenames",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Error: not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    results_path = run_dir / "perf_results.json"
    if not results_path.exists():
        print(f"Error: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(results_path, encoding="utf-8") as f:
        all_results: list[dict] = json.load(f)

    if not all_results:
        print("Error: perf_results.json is empty", file=sys.stderr)
        sys.exit(1)

    raw = [
        r
        for r in all_results
        if r.get("d2m_duration_ns", 0) > 0 or r.get("ttnn_duration_ns", 0) > 0
    ]
    skipped = len(all_results) - len(raw)

    if not args.quiet:
        print(f"Results: {results_path} ({len(all_results)} tests, {skipped} skipped)")
        for r in all_results:
            d2m = r.get("d2m_duration_ns", 0)
            ttnn = r.get("ttnn_duration_ns", 0)
            status = "" if d2m > 0 or ttnn > 0 else " [SKIPPED]"
            print(f"  {r['pattern']}: d2m={d2m}ns ttnn={ttnn}ns{status}")

    if not raw:
        print("Error: no tests produced perf data", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.output_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    job_suffix = f"_{args.job_id}" if args.job_id else ""

    generate_reports(raw, out_dir, job_suffix, args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
