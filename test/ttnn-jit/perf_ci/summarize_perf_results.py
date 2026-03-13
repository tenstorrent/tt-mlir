#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Read all ops_perf_results_*.csv under a run directory (from run_perf_collect.sh),
# group JIT vs non-JIT by case (op, shape, dtype, memory_config_id) and write one
# JSON report per case with structured fields for Superset ingestion.
#
# Each report becomes its own benchmark_run row in Superset with clean filterable
# columns (model=op, precision=dtype, config=memory/shape/fidelity) and simple
# measurement names (jit_kernel_duration_ns, ttnn_kernel_duration_ns, perf_ratio).
#
# Usage:
#   python test/ttnn-jit/perf_ci/summarize_perf_results.py RUN_DIR [--output-dir DIR] [--job-id ID]
#
# Example:
#   python test/ttnn-jit/perf_ci/summarize_perf_results.py generated/jit_perf_reports/run_20250309_123456
#   python test/ttnn-jit/perf_ci/summarize_perf_results.py generated/jit_perf_reports/run_20250309_123456 --job-id 66822899875

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

DEVICE_KERNEL_DURATION_COL = "DEVICE KERNEL DURATION [ns]"
MATH_FIDELITY_COL = "MATH FIDELITY"
OUTPUT_0_DATATYPE_COL = "OUTPUT_0_DATATYPE"
INPUT_0_DATATYPE_COL = "INPUT_0_DATATYPE"

UNARY_OPS = frozenset({"abs", "exp"})

MEMORY_CONFIG_IDS = ("dram_interleaved", "l1_interleaved")


def find_result_csvs(run_dir: Path):
    """Yield (test_id, csv_path) for each ops_perf_results_*.csv under run_dir."""
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        return
    for test_dir in run_dir.iterdir():
        if not test_dir.is_dir():
            continue
        test_id = test_dir.name
        reports_dir = test_dir / "reports"
        if not reports_dir.is_dir():
            continue
        for ts_dir in reports_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            for csv_path in ts_dir.glob("ops_perf_results_*.csv"):
                yield test_id, csv_path


def parse_test_id(test_id: str) -> Optional[dict]:
    """
    Parse test_id into jit, op, h, w, and optionally memory_config_id.
    Supports: 'True-abs-256-256' (4 parts), 'True-bf16-abs-256-256' (5),
    'True-dram_interleaved-bf16-abs-256-256' (6).
    """
    parts = test_id.split("-")
    if len(parts) < 4:
        return None
    jit = parts[0].lower() == "true"
    memory_config_id: Optional[str] = None
    if len(parts) == 6 and parts[1] in MEMORY_CONFIG_IDS:
        memory_config_id = parts[1]
        op = parts[3]
        try:
            h, w = int(parts[4]), int(parts[5])
        except (ValueError, IndexError):
            return None
    elif len(parts) == 5:
        op = parts[2]
        try:
            h, w = int(parts[3]), int(parts[4])
        except (ValueError, IndexError):
            return None
    elif len(parts) == 4:
        op = parts[1]
        try:
            h, w = int(parts[2]), int(parts[3])
        except (ValueError, IndexError):
            return None
    else:
        try:
            h, w = int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            return None
        op = "-".join(parts[1:-2])
    return {"jit": jit, "op": op, "h": h, "w": w, "memory_config_id": memory_config_id}


def read_csv_duration_and_meta(csv_path: Path) -> Optional[tuple[int, str, str]]:
    """
    Read CSV: sum DEVICE KERNEL DURATION [ns], and from first data row return
    (duration_ns, dtype, math_fidelity). dtype/math_fidelity may be empty if
    column missing.
    """
    total = 0
    found_duration = False
    dtype = ""
    math_fidelity = ""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if DEVICE_KERNEL_DURATION_COL in fieldnames:
                val = row.get(DEVICE_KERNEL_DURATION_COL, "").strip()
                if val and val != "-":
                    try:
                        total += int(float(val))
                        found_duration = True
                    except (ValueError, TypeError):
                        pass
            if not dtype and (
                OUTPUT_0_DATATYPE_COL in fieldnames
                or INPUT_0_DATATYPE_COL in fieldnames
            ):
                dtype = (
                    row.get(OUTPUT_0_DATATYPE_COL)
                    or row.get(INPUT_0_DATATYPE_COL)
                    or ""
                ).strip()
            if not math_fidelity and MATH_FIDELITY_COL in fieldnames:
                math_fidelity = (row.get(MATH_FIDELITY_COL) or "").strip()
    if not found_duration:
        return None
    return (total, dtype, math_fidelity)


def make_case_key(
    op: str, h: int, w: int, dtype: str, memory_config_id: Optional[str]
) -> tuple:
    """Immutable key to group JIT and non-JIT runs of the same case."""
    return (op, h, w, dtype, memory_config_id or "")


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


def main():
    parser = argparse.ArgumentParser(
        description="Summarize JIT perf run CSVs into one JSON report per (op, dtype, memory_config) test case for Superset."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory produced by run_perf_collect.sh (contains test_id/reports/...)",
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

    out_dir = (args.output_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    job_suffix = f"_{args.job_id}" if args.job_id else ""

    raw: list[dict[str, Any]] = []
    for test_id, csv_path in find_result_csvs(run_dir):
        parsed = parse_test_id(test_id)
        if not parsed:
            if not args.quiet:
                print(f"Skip (bad test_id): {test_id}", file=sys.stderr)
            continue
        result = read_csv_duration_and_meta(csv_path)
        if result is None:
            if not args.quiet:
                print(f"Skip (no duration): {csv_path}", file=sys.stderr)
            continue
        duration_ns, dtype, math_fidelity = result
        raw.append(
            {
                "test_id": test_id,
                "jit": parsed["jit"],
                "op": parsed["op"],
                "h": parsed["h"],
                "w": parsed["w"],
                "memory_config_id": parsed.get("memory_config_id"),
                "duration_ns": duration_ns,
                "dtype": dtype,
                "math_fidelity": math_fidelity,
            }
        )
        if not args.quiet:
            print(
                f"  {test_id}: {duration_ns} ns (dtype={dtype!r}, math_fidelity={math_fidelity!r})"
            )

    groups: dict[tuple, dict[str, Any]] = {}
    for r in raw:
        key = make_case_key(
            r["op"], r["h"], r["w"], r["dtype"], r.get("memory_config_id")
        )
        if key not in groups:
            groups[key] = {
                "op": r["op"],
                "h": r["h"],
                "w": r["w"],
                "shape": f"{r['h']}x{r['w']}",
                "dtype": r["dtype"],
                "math_fidelity_jit": "",
                "math_fidelity_ttnn": "",
                "memory_config_id": r.get("memory_config_id") or "",
                "jit_duration_ns": None,
                "ttnn_duration_ns": None,
            }
        g = groups[key]
        if r["jit"]:
            g["jit_duration_ns"] = r["duration_ns"]
            g["math_fidelity_jit"] = r["math_fidelity"]
        else:
            g["ttnn_duration_ns"] = r["duration_ns"]
            g["math_fidelity_ttnn"] = r["math_fidelity"]

    file_count = 0
    for key in sorted(groups.keys()):
        g = groups[key]
        op = g["op"]
        dtype = g["dtype"]
        mem_cfg = g["memory_config_id"]
        shape = g["shape"]
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

        config = {
            "input_a_shape": shape,
            "input_b_shape": None if is_unary else shape,
            "input_a_memory_config": mem_cfg,
            "input_b_memory_config": None if is_unary else mem_cfg,
            "math_fidelity_jit": g["math_fidelity_jit"],
            "math_fidelity_ttnn": g["math_fidelity_ttnn"],
        }

        report = {
            "project": "tt-mlir",
            "model": op,
            "model_type": "jit_vs_ttnn",
            "run_type": "op_benchmark",
            "precision": dtype,
            "config": config,
            "measurements": measurements,
        }

        filename = f"perf_{op}_{dtype}_{mem_cfg}{job_suffix}.json"
        filepath = out_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        file_count += 1
        if not args.quiet:
            print(f"  Wrote {filepath.name} ({len(measurements)} measurements)")

    if not args.quiet:
        print(f"Wrote {file_count} report(s) from {len(groups)} case(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
