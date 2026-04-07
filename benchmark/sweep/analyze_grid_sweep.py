#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Analyze grid sweep profiling results from test_k_sharing_sweep and
test_virtual_grid_sweep.

Usage:
  python benchmark/sweep/analyze_grid_sweep.py \
      [--prof-root benchmark/sweep/prof_per_config]
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def load_all_results(prof_root: Path) -> list[dict]:
    rows = []
    for config_dir in sorted(prof_root.iterdir()):
        if not config_dir.is_dir():
            continue
        report_csvs = sorted(config_dir.glob("reports/*/ops_perf_results_*.csv"))
        if not report_csvs:
            continue
        report = report_csvs[-1]
        try:
            with open(report) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    def safe_int(v):
                        try:
                            return int(v)
                        except (ValueError, TypeError):
                            return None

                    kernel_ns = safe_int(row.get("DEVICE KERNEL DURATION [ns]"))
                    dm_ns = safe_int(row.get("DEVICE NCRISC KERNEL DURATION [ns]"))
                    compute_ns = safe_int(row.get("DEVICE TRISC1 KERNEL DURATION [ns]"))
                    cores = safe_int(row.get("CORE COUNT"))
                    cb_wait = safe_int(row.get("DEVICE COMPUTE CB WAIT FRONT [ns]"))

                    if kernel_ns is None:
                        continue

                    rows.append({
                        "config": config_dir.name,
                        "cores": cores,
                        "kernel_ns": kernel_ns,
                        "dm_ns": dm_ns,
                        "compute_ns": compute_ns,
                        "cb_wait_ns": cb_wait,
                    })
        except Exception as e:
            print(f"Warning: could not read {report}: {e}")
    return rows


def aggregate_by_config(rows: list[dict]) -> dict[str, dict]:
    """Sum kernel durations per config (one flatbuffer = one config)."""
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)

    agg = {}
    for config, group in sorted(by_config.items()):
        total_kernel = sum(r["kernel_ns"] for r in group if r["kernel_ns"])
        total_dm = sum(r["dm_ns"] or 0 for r in group)
        total_compute = sum(r["compute_ns"] or 0 for r in group)
        total_cb_wait = sum(r["cb_wait_ns"] or 0 for r in group)
        max_cores = max(r["cores"] or 0 for r in group)
        n_dispatches = len(group)

        agg[config] = {
            "total_kernel_ns": total_kernel,
            "total_dm_ns": total_dm,
            "total_compute_ns": total_compute,
            "total_cb_wait_ns": total_cb_wait,
            "max_cores": max_cores,
            "n_dispatches": n_dispatches,
        }
    return agg


def print_table(title: str, configs: list[str], agg: dict):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    print(f"{'config':<65} {'cores':>5} {'kernel':>10} {'DM':>10} {'compute':>10} {'cb_wait':>10} {'#ops':>5}")
    print("-" * 120)
    for c in configs:
        if c not in agg:
            continue
        a = agg[c]
        print(
            f"{c:<65} {a['max_cores']:>5} "
            f"{a['total_kernel_ns']:>10} {a['total_dm_ns']:>10} "
            f"{a['total_compute_ns']:>10} {a['total_cb_wait_ns']:>10} "
            f"{a['n_dispatches']:>5}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prof-root",
        default="benchmark/sweep/prof_per_config",
        help="Root dir with per-config tracy reports",
    )
    args = parser.parse_args()
    prof_root = Path(args.prof_root)

    rows = load_all_results(prof_root)
    if not rows:
        print(f"No results found in {prof_root}")
        return

    agg = aggregate_by_config(rows)
    all_configs = sorted(agg.keys())

    # --- K-sharing: isolated matmul ---
    isolated = [c for c in all_configs if "isolated_matmul" in c]
    if isolated:
        wide = [c for c in isolated if "wide" in c]
        narrow = [c for c in isolated if "narrow" in c]
        print_table("Isolated Matmul — Wide (320x320 @ 320x320)", wide, agg)
        print_table("Isolated Matmul — Narrow (320x320 @ 320x64)", narrow, agg)

    # --- K-sharing: shared operand ---
    shared = [c for c in all_configs if "shared_operand[" in c and "override" not in c]
    if shared:
        print_table("Shared Operand (W feeds wide + narrow matmul)", shared, agg)

    # --- K-sharing: grid override ---
    override = [c for c in all_configs if "shared_operand_grid_override" in c]
    if override:
        print_table("Shared Operand — Grid Override Strategies", override, agg)

    # --- Virtual grid ---
    vgrid = [c for c in all_configs if "vgrid_multiply[" in c]
    if vgrid:
        print_table("Virtual Grid — Dimension Ordering", vgrid, agg)

    vgrid_constrained = [c for c in all_configs if "vgrid_multiply_constrained" in c]
    if vgrid_constrained:
        print_table("Virtual Grid — Constrained Device", vgrid_constrained, agg)

    # --- Summary comparisons ---
    print(f"\n{'='*80}")
    print(" Key Comparisons")
    print(f"{'='*80}")

    def compare(label, config_a, config_b):
        if config_a in agg and config_b in agg:
            a = agg[config_a]["total_kernel_ns"]
            b = agg[config_b]["total_kernel_ns"]
            if b > 0:
                ratio = a / b
                print(f"  {label}: {a}ns vs {b}ns = {ratio:.2f}x")

    # Wide matmul: g10x10 vs g5x5
    compare(
        "Wide matmul g10x10 vs g5x5",
        "test_isolated_matmul[ttmetal-g10x10-wide]",
        "test_isolated_matmul[ttmetal-g5x5-wide]",
    )
    compare(
        "Narrow matmul g10x10 vs g5x5",
        "test_isolated_matmul[ttmetal-g10x10-narrow]",
        "test_isolated_matmul[ttmetal-g5x5-narrow]",
    )
    # Shared operand: g10x10 vs g5x5
    compare(
        "Shared operand (wide+narrow) g10x10 vs g5x5",
        "test_shared_operand[ttmetal-g10x10-10t_wide_vs_2t_narrow]",
        "test_shared_operand[ttmetal-g5x5-10t_wide_vs_2t_narrow]",
    )
    # Grid override: auto vs force_compat
    compare(
        "Grid override: auto vs force_compat_5x5",
        "test_shared_operand_grid_override[ttmetal-auto]",
        "test_shared_operand_grid_override[ttmetal-force_compat_5x5]",
    )
    # Virtual grid: 18x128 vs 128x18
    compare(
        "VGrid 4x4: 18x128 vs 128x18",
        "test_vgrid_multiply[ttmetal-4x4x18x128]",
        "test_vgrid_multiply[ttmetal-4x4x128x18]",
    )
    compare(
        "VGrid 2x2: 18x128 vs 128x18",
        "test_vgrid_multiply[ttmetal-2x2x18x128]",
        "test_vgrid_multiply[ttmetal-2x2x128x18]",
    )
    compare(
        "VGrid 8x8: 18x128 vs 128x18",
        "test_vgrid_multiply[ttmetal-8x8x18x128]",
        "test_vgrid_multiply[ttmetal-8x8x128x18]",
    )
    # Clean baseline
    compare(
        "VGrid 4x4: 64x128 vs 128x64 (clean)",
        "test_vgrid_multiply[ttmetal-4x4x64x128]",
        "test_vgrid_multiply[ttmetal-4x4x128x64]",
    )


if __name__ == "__main__":
    main()
