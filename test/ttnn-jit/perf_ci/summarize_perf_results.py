#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Read the perf_results.json produced by conftest.py (via the perf_device
# fixture and ReadDeviceProfiler / get_latest_programs_perf_data), group
# JIT vs non-JIT by case (op, dtype, memory_config_id) and write one JSON
# report per case for Superset ingestion.
#
# Usage:
#   python test/ttnn-jit/perf_ci/summarize_perf_results.py RUN_DIR [--output-dir DIR] [--job-id ID]

import argparse
import json
import sys
from pathlib import Path
from typing import Any

UNARY_OPS = frozenset({"abs", "exp"})


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
    """Group raw results by case, write JSON reports. Returns file count."""
    groups: dict[tuple, dict[str, Any]] = {}
    for r in raw:
        key = (r["op"], r["h"], r["w"], r["dtype"], r.get("memory_config_id") or "")
        if key not in groups:
            groups[key] = {
                "op": r["op"],
                "shape": f"{r['h']}x{r['w']}",
                "dtype": r["dtype"],
                "memory_config_id": r.get("memory_config_id") or "",
                "math_fidelity_jit": "",
                "math_fidelity_ttnn": "",
                "jit_duration_ns": None,
                "ttnn_duration_ns": None,
            }
        g = groups[key]
        if r["jit"]:
            g["jit_duration_ns"] = r["duration_ns"]
            g["math_fidelity_jit"] = r.get("math_fidelity", "")
        else:
            g["ttnn_duration_ns"] = r["duration_ns"]
            g["math_fidelity_ttnn"] = r.get("math_fidelity", "")

    file_count = 0
    for key in sorted(groups.keys()):
        g = groups[key]
        op = g["op"]
        dtype = g["dtype"]
        mem_cfg = g["memory_config_id"]
        jit_ns = g["jit_duration_ns"]
        ttnn_ns = g["ttnn_duration_ns"]
        is_unary = op in UNARY_OPS

        measurements = []
        if jit_ns is not None:
            measurements.append(_measurement("jit_kernel_duration_ns", jit_ns, op))
        if ttnn_ns is not None:
            measurements.append(_measurement("ttnn_kernel_duration_ns", ttnn_ns, op))
        if jit_ns is not None and ttnn_ns is not None and jit_ns > 0:
            ratio = round(ttnn_ns / jit_ns, 4)
            measurements.append(_measurement("perf_ratio", ratio, op))

        report = {
            "project": "tt-mlir",
            "model": op,
            "model_type": "jit_vs_ttnn",
            "run_type": "op_benchmark",
            "precision": dtype,
            "config": {
                "input_a_shape": g["shape"],
                "input_b_shape": None if is_unary else g["shape"],
                "input_a_memory_config": mem_cfg,
                "input_b_memory_config": None if is_unary else mem_cfg,
                "math_fidelity_jit": g["math_fidelity_jit"],
                "math_fidelity_ttnn": g["math_fidelity_ttnn"],
            },
            "measurements": measurements,
        }

        filename = f"perf_{op}_{g['shape']}_{dtype}_{mem_cfg}{job_suffix}.json"
        filepath = out_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        file_count += 1
        if not quiet:
            print(f"  Wrote {filepath.name} ({len(measurements)} measurements)")

    if not quiet:
        print(f"Wrote {file_count} report(s) from {len(groups)} case(s) to {out_dir}")
    return file_count


def main():
    parser = argparse.ArgumentParser(
        description="Summarize JIT perf results into JSON reports for Superset."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory produced by run_perf_collect.sh (contains perf_results.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write individual JSON reports (default: run_dir)",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="GitHub job ID to append to filenames (required for CI collect_data)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print progress",
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

    # Filter out entries with 0 duration (tests that failed before running ops).
    raw = [r for r in all_results if r.get("duration_ns", 0) > 0]
    skipped = len(all_results) - len(raw)

    if not args.quiet:
        print(f"Results: {results_path} ({len(all_results)} tests, {skipped} skipped)")
        for r in all_results:
            label = "JIT" if r["jit"] else "TTNN"
            status = "" if r.get("duration_ns", 0) > 0 else " [SKIPPED]"
            print(
                f"  [{label}] {r['op']}: {r['duration_ns']} ns "
                f"({r['num_programs']} program(s), dtype={r['dtype']!r}){status}"
            )

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
