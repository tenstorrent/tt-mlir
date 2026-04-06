#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Two-phase grid sweep profiler.

Phase 1 - compile flatbuffers (run once):
  pytest benchmark/sweep/test_grid_perf_sweep.py \
      --path benchmark/sweep [test filter flags]

Phase 2 - profile each flatbuffer with tracy (this script):
  python benchmark/sweep/run_sweep.py [--fb-root benchmark/sweep]

Results land in benchmark/sweep/prof_per_config/<test_name>/reports/.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TRACY_TOOLS = ROOT / "build" / "python_packages" / "ttrt" / "runtime"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fb-root",
        default="benchmark/sweep",
        help="Root dir that contains builder-artifacts/ (default: benchmark/sweep)",
    )
    parser.add_argument(
        "--output-root",
        default="benchmark/sweep/prof_per_config",
        help="Where to write per-config tracy reports (default: benchmark/sweep/prof_per_config)",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Only profile flatbuffers whose path contains this substring",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    args = parser.parse_args()

    fb_root = ROOT / args.fb_root
    output_root = ROOT / args.output_root

    flatbuffers = sorted(
        (fb_root / "builder-artifacts").glob("*/*/ttmetal_compiled.ttm")
    )
    if not flatbuffers:
        print(f"No flatbuffers found under {fb_root}/builder-artifacts/")
        print("Run pytest first:")
        print(f"  pytest benchmark/sweep/test_grid_perf_sweep.py --path {args.fb_root}")
        sys.exit(1)

    if args.filter:
        flatbuffers = [fb for fb in flatbuffers if args.filter in str(fb)]

    print(f"Found {len(flatbuffers)} flatbuffers to profile.")

    for i, fb in enumerate(flatbuffers):
        config_name = fb.parent.name
        output_dir = output_root / config_name
        print(f"\n[{i+1}/{len(flatbuffers)}] {config_name}")

        cmd = [
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-v",
            "--output-folder",
            str(output_dir),
            "--tracy-tools-folder",
            str(TRACY_TOOLS),
            "-m",
            "ttrt",
            "run",
            "--enable-perf-trace",
            str(fb),
        ]

        if args.dry_run:
            print("  " + " ".join(cmd))
            continue

        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"  WARNING: tracy exited with code {result.returncode}")

    if not args.dry_run:
        print(f"\nAll done. Reports in {output_root}/")
        _print_summary(output_root)


def _print_summary(output_root: Path):
    import csv

    rows = []
    for report in sorted(output_root.glob("*/reports/*/ops_perf_results_*.csv")):
        config = report.parent.parent.parent.name
        try:
            with open(report) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(
                        {
                            "config": config,
                            "op": row.get("OP CODE", ""),
                            "cores": row.get("CORE COUNT", ""),
                            "kernel_ns": row.get("DEVICE KERNEL DURATION [ns]", ""),
                            "dm_ns": row.get("DEVICE NCRISC KERNEL DURATION [ns]", ""),
                            "compute_ns": row.get(
                                "DEVICE TRISC1 KERNEL DURATION [ns]", ""
                            ),
                            "cb_wait_ns": row.get(
                                "DEVICE COMPUTE CB WAIT FRONT [ns]", ""
                            ),
                        }
                    )
        except Exception:
            pass

    if not rows:
        print("(no ops_perf_results found — device logs may be missing)")
        return

    print(
        f"\n{'config':<60} {'cores':>6} {'kernel_ns':>12} {'dm_ns':>12} {'compute_ns':>12} {'cb_wait_ns':>12}"
    )
    print("-" * 120)
    for r in rows:
        print(
            f"{r['config']:<60} {r['cores']:>6} {r['kernel_ns']:>12} {r['dm_ns']:>12} {r['compute_ns']:>12} {r['cb_wait_ns']:>12}"
        )


if __name__ == "__main__":
    main()
