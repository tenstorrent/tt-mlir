#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Plot grid sweep profiling results.

Usage:
  python benchmark/sweep/plot_sweep.py [--prof-root benchmark/sweep/prof_per_config]
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def load_results(prof_root: Path) -> list[dict]:
    rows = []
    for config_dir in sorted(prof_root.iterdir()):
        if not config_dir.is_dir():
            continue
        report_csvs = sorted(config_dir.glob("reports/*/ops_perf_results_*.csv"))
        if not report_csvs:
            continue
        report = report_csvs[-1]  # most recent
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
                    if kernel_ns is None:
                        continue

                    # D2M format first (test_d2m_X[ttmetal-...gNxN...])
                    # Add: test_d2m_add[ttmetal-2048x2048-g2x2_bf8x8]
                    # Matmul: test_d2m_matmul[ttmetal-g8x8_bs128x128x64_bf4_M1024N1024K256]
                    m2 = re.match(
                        r"(test_d2m_\w+)\[ttmetal-(?:(.+)-)?g(\d+)x(\d+)(?:_(.+))?\]",
                        config_dir.name,
                    )
                    # TTIR format: test_op[ttmetal-shape-grid] or test_op[ttmetal-shape-gNxN]
                    m = re.match(
                        r"(test_\w+)\[ttmetal-(.+)-g?(\d+x\d+)\]", config_dir.name
                    )
                    if m2:
                        op = m2.group(1)
                        shape = m2.group(2) or m2.group(5) or ""
                        gr, gc = int(m2.group(3)), int(m2.group(4))
                        grid = f"{gr}x{gc}"
                    elif m:
                        op, shape, grid = m.group(1), m.group(2), m.group(3)
                        gr, gc = map(int, grid.split("x"))
                    else:
                        continue

                    rows.append(
                        {
                            "config": config_dir.name,
                            "op": op,
                            "shape": shape,
                            "grid": grid,
                            "grid_rows": gr,
                            "grid_cols": gc,
                            "cores": cores,
                            "kernel_ns": kernel_ns,
                            "dm_ns": dm_ns,
                            "compute_ns": compute_ns,
                        }
                    )
        except Exception as e:
            print(f"Warning: could not read {report}: {e}")
    return rows


def aggregate_total(rows: list[dict]) -> list[dict]:
    """Sum all kernel durations for the same config.

    Each config corresponds to one flatbuffer program (possibly a chain of ops).
    Summing gives total device execution time across all dispatches in the program,
    which is what we care about for comparing parallelism efficiency.
    """
    by_config = defaultdict(list)
    meta = {}
    for r in rows:
        key = r["config"]
        by_config[key].append(r)
        meta[key] = {
            k: r[k] for k in ("op", "shape", "grid", "grid_rows", "grid_cols", "cores")
        }

    agg = []
    for key, group in by_config.items():

        def total(field):
            vals = [r[field] for r in group if r[field] is not None]
            return int(sum(vals)) if vals else None

        kernel_vals = [r["kernel_ns"] for r in group if r["kernel_ns"] is not None]
        max_kernel = max(kernel_vals) if kernel_vals else None

        # Compute-only: sum of dispatches whose core count matches the grid
        grid_cores = meta[key]["grid_rows"] * meta[key]["grid_cols"]
        compute_only = sum(
            r["kernel_ns"]
            for r in group
            if r["kernel_ns"] is not None and r["cores"] == grid_cores
        )
        # DM-only: everything else
        dm_only = sum(
            r["kernel_ns"]
            for r in group
            if r["kernel_ns"] is not None and r["cores"] != grid_cores
        )

        agg.append(
            {
                **meta[key],
                "kernel_ns": total("kernel_ns"),
                "max_kernel_ns": max_kernel,
                "compute_only_ns": compute_only or None,
                "dm_only_ns": dm_only or None,
                "dm_ns": total("dm_ns"),
                "compute_ns": total("compute_ns"),
            }
        )
    return agg


def grid_sort_key(grid: str):
    r, c = map(int, grid.split("x"))
    return (r * c, r, c)


def plot_scaling(rows: list[dict], out_dir: Path):
    """kernel_ns vs grid config per (op, shape). One subplot per op."""
    ops = sorted(set(r["op"] for r in rows))
    fig, axes = plt.subplots(len(ops), 1, figsize=(12, 4 * len(ops)))
    if len(ops) == 1:
        axes = [axes]

    all_grids = sorted(set(r["grid"] for r in rows), key=grid_sort_key)

    for ax, op in zip(axes, ops):
        op_rows = [r for r in rows if r["op"] == op and r["kernel_ns"]]
        shapes = sorted(set(r["shape"] for r in op_rows))
        colors = cm.tab10(np.linspace(0, 1, len(shapes)))

        for shape, color in zip(shapes, colors):
            shape_rows = {r["grid"]: r for r in op_rows if r["shape"] == shape}
            xs = [i for i, g in enumerate(all_grids) if g in shape_rows]
            ys = [shape_rows[all_grids[i]]["kernel_ns"] for i in xs]
            if xs:
                ax.plot(xs, ys, "o-", color=color, label=shape)

        ax.set_title(op)
        ax.set_xticks(range(len(all_grids)))
        ax.set_xticklabels(all_grids, rotation=45, ha="right")
        ax.set_ylabel("Kernel duration [ns]")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Kernel duration vs grid config", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "scaling.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_speedup(rows: list[dict], out_dir: Path):
    """Speedup relative to 1x1 baseline vs grid config per (op, shape)."""
    ops = sorted(set(r["op"] for r in rows))
    fig, axes = plt.subplots(len(ops), 1, figsize=(12, 4 * len(ops)))
    if len(ops) == 1:
        axes = [axes]

    all_grids = sorted(set(r["grid"] for r in rows), key=grid_sort_key)

    for ax, op in zip(axes, ops):
        op_rows = [r for r in rows if r["op"] == op and r["kernel_ns"]]
        shapes = sorted(set(r["shape"] for r in op_rows))
        colors = cm.tab10(np.linspace(0, 1, len(shapes)))

        # Ideal: speedup = number of cores relative to baseline cores
        baseline_grids = [g for g in all_grids if g in set(r["grid"] for r in op_rows)]
        if baseline_grids:
            br, bc = map(int, baseline_grids[0].split("x"))
            base_cores = br * bc
            ideal_xs = list(range(len(all_grids)))
            ideal_ys = [(grid_sort_key(g)[0] / base_cores) for g in all_grids]
            ax.plot(ideal_xs, ideal_ys, "k--", alpha=0.4, label="ideal", linewidth=1.5)

        for shape, color in zip(shapes, colors):
            shape_rows = {r["grid"]: r for r in op_rows if r["shape"] == shape}
            baseline = shape_rows.get("1x1") or shape_rows.get(
                min(shape_rows, key=grid_sort_key)
            )
            if baseline is None or baseline["kernel_ns"] is None:
                continue
            baseline_ns = baseline["kernel_ns"]

            xs = [
                i
                for i, g in enumerate(all_grids)
                if g in shape_rows and shape_rows[g]["kernel_ns"]
            ]
            ys = [baseline_ns / shape_rows[all_grids[i]]["kernel_ns"] for i in xs]
            if xs:
                ax.plot(xs, ys, "o-", color=color, label=shape)

        ax.set_title(op)
        ax.set_xticks(range(len(all_grids)))
        ax.set_xticklabels(all_grids, rotation=45, ha="right")
        ax.set_ylabel("Speedup vs 1x1")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Speedup vs grid config (ideal = dashed)", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "speedup.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_dm_vs_compute(rows: list[dict], out_dir: Path):
    """kernel, compute, and DM duration vs grid config per op (averaged across shapes)."""
    ops = sorted(set(r["op"] for r in rows))
    fig, axes = plt.subplots(len(ops), 1, figsize=(12, 4 * len(ops)))
    if len(ops) == 1:
        axes = [axes]

    all_grids = sorted(set(r["grid"] for r in rows), key=grid_sort_key)

    for ax, op in zip(axes, ops):
        op_rows = [r for r in rows if r["op"] == op and r["kernel_ns"]]
        by_grid = defaultdict(list)
        for r in op_rows:
            by_grid[r["grid"]].append(r)

        xs = [i for i, g in enumerate(all_grids) if g in by_grid]
        grids_present = [all_grids[i] for i in xs]

        def med_field(g, field):
            vals = [r[field] for r in by_grid[g] if r.get(field) is not None]
            return statistics.median(vals) if vals else 0

        kernel_ys = [med_field(g, "kernel_ns") for g in grids_present]
        compute_ys = [med_field(g, "compute_ns") for g in grids_present]
        dm_ys = [med_field(g, "dm_ns") for g in grids_present]

        ax.plot(xs, kernel_ys, "k-o", label="kernel total", linewidth=2)
        ax.plot(xs, compute_ys, "b--s", label="TRISC1 (compute)")
        ax.plot(xs, dm_ys, "r--^", label="NCRISC (DM)")

        ax.set_title(op)
        ax.set_xticks(range(len(all_grids)))
        ax.set_xticklabels(all_grids, rotation=45, ha="right")
        ax.set_ylabel("Duration [ns] (median across shapes)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Kernel / Compute / DM breakdown vs grid config", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "dm_vs_compute.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def _parse_d2m_rows(rows, op_name):
    """Parse D2M sweep rows into labeled data grouped by size."""
    d2m_rows = [r for r in rows if r["op"] == op_name]
    if not d2m_rows:
        return {}, []

    parsed = []
    for r in d2m_rows:
        shape_str = r["shape"] or ""
        gr, gc = r["grid_rows"], r["grid_cols"]
        sm = re.match(r"(\d+)x(\d+)", shape_str)
        mm = re.search(r"M(\d+)N(\d+)K(\d+)", shape_str)
        if sm:
            size_label = f"{sm.group(1)}x{sm.group(2)}"
        elif mm:
            size_label = f"{mm.group(1)}x{mm.group(2)}xK{mm.group(3)}"
        else:
            continue
        parsed.append({**r, "size_label": size_label, "cores": gr * gc})

    seen = set()
    unique = []
    for p in parsed:
        key = (p["size_label"], p["grid"])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    by_size = defaultdict(list)
    for p in unique:
        by_size[p["size_label"]].append(p)

    all_grids = sorted(
        set(p["grid"] for p in unique),
        key=lambda g: int(g.split("x")[0]) * int(g.split("x")[1]),
    )
    return by_size, all_grids


def _plot_d2m_sweep(
    rows: list[dict], out_dir: Path, op_name: str, title: str, filename: str
):
    """Three subplots: total time, compute vs DM breakdown, speedup."""
    by_size, all_grids = _parse_d2m_rows(rows, op_name)
    if not by_size:
        return

    grid_cores = [int(g.split("x")[0]) * int(g.split("x")[1]) for g in all_grids]
    grid_labels = [f"{g}\n({c}c)" for g, c in zip(all_grids, grid_cores)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    colors = cm.tab10(np.linspace(0, 1, max(len(by_size), 1)))

    for (size_label, pts), color in zip(sorted(by_size.items()), colors):
        pts.sort(key=lambda p: p["cores"])
        xs = [all_grids.index(p["grid"]) for p in pts if p["grid"] in all_grids]
        total_ys = [p["kernel_ns"] or 0 for p in pts if p["grid"] in all_grids]
        compute_ys = [
            p.get("compute_only_ns") or 0 for p in pts if p["grid"] in all_grids
        ]
        dm_ys = [p.get("dm_only_ns") or 0 for p in pts if p["grid"] in all_grids]
        if not xs:
            continue

        # Panel 1: Total time
        ax1.plot(
            xs, total_ys, "o-", color=color, label=size_label, linewidth=2, markersize=7
        )

        # Panel 2: Compute vs DM
        ax2.plot(
            xs,
            compute_ys,
            "o-",
            color=color,
            label=f"{size_label} compute",
            linewidth=2,
            markersize=7,
        )
        ax2.plot(
            xs,
            dm_ys,
            "s--",
            color=color,
            label=f"{size_label} DM",
            linewidth=1,
            markersize=5,
            alpha=0.6,
        )

        # Panel 3: Speedup (total and compute-only)
        if total_ys[0] > 0:
            total_speedup = [total_ys[0] / y if y > 0 else 0 for y in total_ys]
            ax3.plot(
                xs,
                total_speedup,
                "o-",
                color=color,
                label=f"{size_label} total",
                linewidth=2,
                markersize=7,
            )
        if compute_ys[0] > 0:
            compute_speedup = [compute_ys[0] / y if y > 0 else 0 for y in compute_ys]
            ax3.plot(
                xs,
                compute_speedup,
                "s--",
                color=color,
                label=f"{size_label} compute",
                linewidth=1.5,
                markersize=5,
            )

    # Ideal speedup reference
    if "1x1" in all_grids:
        base_idx = all_grids.index("1x1")
        base_cores = grid_cores[base_idx]
        ideal = [c / base_cores for c in grid_cores]
        ax3.plot(
            range(len(all_grids)), ideal, "k--", alpha=0.4, linewidth=1.5, label="ideal"
        )

    for ax in (ax1, ax2, ax3):
        ax.set_xticks(range(len(all_grids)))
        ax.set_xticklabels(grid_labels, fontsize=9)
        ax.set_xlabel("Grid")
        ax.grid(True, axis="y", alpha=0.3)

    ax1.set_ylabel("Duration [ns]")
    ax1.set_title(f"{title} — total time")
    ax1.legend(fontsize=8, title="Input size")
    ax2.set_ylabel("Duration [ns]")
    ax2.set_title(f"{title} — compute vs DM")
    ax2.legend(fontsize=7)
    ax3.set_ylabel("Speedup vs smallest grid")
    ax3.set_title(f"{title} — speedup")
    ax3.legend(fontsize=7)

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_d2m_matmul(rows, out_dir):
    _plot_d2m_sweep(
        rows,
        out_dir,
        "test_d2m_matmul",
        "D2M Matmul: compute kernel vs grid (fixed output size)",
        "d2m_matmul.png",
    )


def plot_d2m_add(rows, out_dir):
    _plot_d2m_sweep(
        rows,
        out_dir,
        "test_d2m_add",
        "D2M Add: compute kernel vs grid (fixed size, spatial streaming)",
        "d2m_add.png",
    )


# ---------------------------------------------------------------------------
# K-dim sweep plots
#
# Config name format: test_k_dim_parallelism_sweep[ttmetal-gRxC-kNt]
#                     test_shared_k_dim_sweep[ttmetal-gRxC-kNt]
# ---------------------------------------------------------------------------

_K_SWEEP_RE = re.compile(
    r"(test_k_dim_parallelism_sweep|test_shared_k_dim_sweep)"
    r"\[ttmetal-(g\d+x\d+)-(k\d+t)\]"
)

GRID_LINE_STYLE = {
    "g1x10": ("serial-K (1×10)", "o-", "#e41a1c"),
    "g5x10": ("partial-K (5×10)", "s--", "#377eb8"),
    "g10x10": ("full-K (10×10)", "^:", "#4daf4a"),
}


def _parse_k_sweep(prof_root: Path) -> list[dict]:
    rows = []
    for config_dir in sorted(prof_root.iterdir()):
        if not config_dir.is_dir():
            continue
        m = _K_SWEEP_RE.match(config_dir.name)
        if not m:
            continue
        test_name, grid_str, k_str = m.group(1), m.group(2), m.group(3)
        k_tiles = int(re.search(r"\d+", k_str).group())
        gr, gc = map(int, grid_str[1:].split("x"))

        report_csvs = sorted(config_dir.glob("reports/*/ops_perf_results_*.csv"))
        if not report_csvs:
            continue
        try:
            with open(report_csvs[-1]) as f:
                reader = csv.DictReader(f)
                dispatches = list(reader)

            def safe_int(v):
                try:
                    return int(v)
                except (ValueError, TypeError):
                    return None

            total_kernel = sum(
                safe_int(r.get("DEVICE KERNEL DURATION [ns]")) or 0
                for r in dispatches
            )
            total_dm = sum(
                safe_int(r.get("DEVICE NCRISC KERNEL DURATION [ns]")) or 0
                for r in dispatches
            )
            total_compute = sum(
                safe_int(r.get("DEVICE TRISC1 KERNEL DURATION [ns]")) or 0
                for r in dispatches
            )
            if total_kernel == 0:
                continue

            rows.append(
                {
                    "test": test_name,
                    "grid": grid_str,
                    "grid_rows": gr,
                    "grid_cols": gc,
                    "k_tiles": k_tiles,
                    "kernel_ns": total_kernel,
                    "dm_ns": total_dm,
                    "compute_ns": total_compute,
                }
            )
        except Exception as e:
            print(f"Warning: could not read {report_csvs[-1]}: {e}")
    return rows


def plot_k_dim_sweep(prof_root: Path, out_dir: Path):
    """Two figures:
    1. Kernel time vs K-tiles — isolated vs shared, one line per grid.
    2. DM vs compute breakdown per grid strategy across K sizes.
    """
    rows = _parse_k_sweep(prof_root)
    if not rows:
        print("No K-dim sweep results found — skipping.")
        return

    isolated = [r for r in rows if r["test"] == "test_k_dim_parallelism_sweep"]
    shared = [r for r in rows if r["test"] == "test_shared_k_dim_sweep"]

    all_k = sorted(set(r["k_tiles"] for r in rows))
    all_grids = sorted(set(r["grid"] for r in rows), key=lambda g: (int(g[1:].split("x")[0]), int(g[1:].split("x")[1])))

    # ---- Figure 1: total kernel time ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, subset, title in [
        (axes[0], isolated, "Isolated matmul (M=N=10t, K varies)"),
        (axes[1], shared, "Shared operand (M=N_wide=10t, K varies)"),
    ]:
        for grid in all_grids:
            pts = sorted(
                [r for r in subset if r["grid"] == grid], key=lambda r: r["k_tiles"]
            )
            if not pts:
                continue
            xs = [r["k_tiles"] for r in pts]
            ys = [r["kernel_ns"] / 1e3 for r in pts]  # → µs
            label, style, color = GRID_LINE_STYLE.get(
                grid, (grid, "o-", None)
            )
            ax.plot(xs, ys, style, color=color, label=label, linewidth=2, markersize=8)

        ax.set_title(title)
        ax.set_xlabel("K dimension [tiles]")
        ax.set_ylabel("Total kernel time [µs]")
        ax.set_xticks(all_k)
        ax.set_xticklabels([f"{k}t\n({k*32})" for k in all_k])
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Parallelism vs DM: kernel time across K sizes", fontsize=13)
    fig.tight_layout()
    p = out_dir / "k_dim_sweep_total.png"
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)

    # ---- Figure 2: DM vs compute breakdown ----------------------------------
    fig, axes = plt.subplots(1, len(all_grids), figsize=(7 * len(all_grids), 6), sharey=False)
    if len(all_grids) == 1:
        axes = [axes]

    for ax, grid in zip(axes, all_grids):
        pts_iso = sorted(
            [r for r in isolated if r["grid"] == grid], key=lambda r: r["k_tiles"]
        )
        if not pts_iso:
            continue
        xs = [r["k_tiles"] for r in pts_iso]
        dm_ys = [r["dm_ns"] / 1e3 for r in pts_iso]
        compute_ys = [r["compute_ns"] / 1e3 for r in pts_iso]
        kernel_ys = [r["kernel_ns"] / 1e3 for r in pts_iso]

        ax.plot(xs, kernel_ys, "k-o", label="kernel total", linewidth=2, markersize=8)
        ax.plot(xs, compute_ys, "b--s", label="TRISC1 (compute)", linewidth=1.5, markersize=6)
        ax.plot(xs, dm_ys, "r--^", label="NCRISC (DM)", linewidth=1.5, markersize=6)

        label, _, _ = GRID_LINE_STYLE.get(grid, (grid, "", ""))
        ax.set_title(f"Grid: {label}")
        ax.set_xlabel("K dimension [tiles]")
        ax.set_ylabel("Time [µs]")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{k}t" for k in xs])
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("DM vs compute breakdown (isolated matmul) by grid strategy", fontsize=13)
    fig.tight_layout()
    p = out_dir / "k_dim_sweep_dm_vs_compute.png"
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HW grid asymmetry sweep plots
#
# Config name formats:
#   test_hw_grid_isolated[ttmetal-STRAT-MNt-KNt]
#   test_hw_grid_shared_compat[ttmetal-NNt-MNt-KNt]
#   test_hw_grid_shared_per_op[ttmetal-NNt-MNt-KNt]
# ---------------------------------------------------------------------------

_HW_ISOLATED_RE = re.compile(
    r"test_hw_grid_isolated\[ttmetal-(compat|per_op_rhs|per_op_lhs)-M(\d+)t-K(\d+)t\]"
)
_HW_SHARED_RE = re.compile(
    r"test_hw_grid_shared_(compat|per_op)\[ttmetal-N(\d+)t-M(\d+)t-K(\d+)t\]"
)


def _parse_hw_grid(prof_root: Path) -> list[dict]:
    rows = []
    for config_dir in sorted(prof_root.iterdir()):
        if not config_dir.is_dir():
            continue

        m_iso = _HW_ISOLATED_RE.match(config_dir.name)
        m_shared = _HW_SHARED_RE.match(config_dir.name)
        if not m_iso and not m_shared:
            continue

        report_csvs = sorted(config_dir.glob("reports/*/ops_perf_results_*.csv"))
        if not report_csvs:
            continue
        try:
            with open(report_csvs[-1]) as f:
                dispatches = list(csv.DictReader(f))

            def safe_int(v):
                try:
                    return int(v)
                except (ValueError, TypeError):
                    return None

            total_kernel = sum(safe_int(r.get("DEVICE KERNEL DURATION [ns]")) or 0 for r in dispatches)
            total_dm = sum(safe_int(r.get("DEVICE NCRISC KERNEL DURATION [ns]")) or 0 for r in dispatches)
            total_compute = sum(safe_int(r.get("DEVICE TRISC1 KERNEL DURATION [ns]")) or 0 for r in dispatches)
            if total_kernel == 0:
                continue

            if m_iso:
                rows.append({
                    "test": "isolated",
                    "strategy": m_iso.group(1),
                    "m_tiles": int(m_iso.group(2)),
                    "n_tiles": int(m_iso.group(2)),  # N=M for isolated
                    "k_tiles": int(m_iso.group(3)),
                    "kernel_ns": total_kernel,
                    "dm_ns": total_dm,
                    "compute_ns": total_compute,
                })
            else:
                rows.append({
                    "test": "shared",
                    "strategy": m_shared.group(1),
                    "m_tiles": int(m_shared.group(3)),
                    "n_tiles": int(m_shared.group(2)),
                    "k_tiles": int(m_shared.group(4)),
                    "kernel_ns": total_kernel,
                    "dm_ns": total_dm,
                    "compute_ns": total_compute,
                })
        except Exception as e:
            print(f"Warning: could not read {report_csvs[-1]}: {e}")
    return rows


def plot_hw_grid_sweep(prof_root: Path, out_dir: Path):
    """Key plot: compat/per_op ratio vs K, one line per (M, N) pair.

    Ratio > 1 means compat is slower (per-op optimal + reblock wins).
    Ratio < 1 means compat wins (reblock too expensive).
    Crossover at ratio = 1 is the grid selection decision boundary.
    """
    rows = _parse_hw_grid(prof_root)
    if not rows:
        print("No HW grid sweep results found — skipping.")
        return

    shared = [r for r in rows if r["test"] == "shared"]
    if not shared:
        print("No shared HW grid results — skipping ratio plot.")
        return

    # Build compat vs per_op lookup: key = (m, n, k)
    compat_map = {(r["m_tiles"], r["n_tiles"], r["k_tiles"]): r
                  for r in shared if r["strategy"] == "compat"}
    per_op_map = {(r["m_tiles"], r["n_tiles"], r["k_tiles"]): r
                  for r in shared if r["strategy"] == "per_op"}

    all_k = sorted(set(r["k_tiles"] for r in shared))
    mn_pairs = sorted(set((r["m_tiles"], r["n_tiles"]) for r in shared))

    # ---- Figure 1: Ratio plot (the money plot) ------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = cm.tab10(np.linspace(0, 1, max(len(mn_pairs), 1)))

    for (m, n), color in zip(mn_pairs, colors):
        ratios = []
        ks = []
        for k in all_k:
            key = (m, n, k)
            if key in compat_map and key in per_op_map:
                c_ns = compat_map[key]["kernel_ns"]
                p_ns = per_op_map[key]["kernel_ns"]
                if p_ns > 0:
                    ratios.append(c_ns / p_ns)
                    ks.append(k)
        if ks:
            ax.plot(ks, ratios, "o-", color=color, label=f"M={m}t N={n}t",
                    linewidth=2, markersize=8)

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_xlabel("K dimension [tiles]")
    ax.set_ylabel("compat time / per-op time")
    ax.set_title("Grid selection tradeoff: compat (no reblock) vs per-op optimal (reblock W)\n"
                 "Above 1.0 = per-op wins, below 1.0 = compat wins")
    ax.set_xticks(all_k)
    ax.set_xticklabels([f"{k}t\n({k*32})" for k in all_k])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = out_dir / "hw_grid_compat_vs_per_op.png"
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)

    # ---- Figure 2: Absolute times, compat vs per_op side by side -----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, strategy, title in [
        (axes[0], "compat", "Compat grid (no reblock)"),
        (axes[1], "per_op", "Per-op optimal (reblock W)"),
    ]:
        subset = [r for r in shared if r["strategy"] == strategy]
        for (m, n), color in zip(mn_pairs, colors):
            pts = sorted([r for r in subset if r["m_tiles"] == m and r["n_tiles"] == n],
                         key=lambda r: r["k_tiles"])
            if not pts:
                continue
            xs = [r["k_tiles"] for r in pts]
            ys = [r["kernel_ns"] / 1e3 for r in pts]
            ax.plot(xs, ys, "o-", color=color, label=f"M={m}t N={n}t",
                    linewidth=2, markersize=8)

        ax.set_title(title)
        ax.set_xlabel("K dimension [tiles]")
        ax.set_ylabel("Total kernel time [us]")
        ax.set_xticks(all_k)
        ax.set_xticklabels([f"{k}t" for k in all_k])
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Shared W: absolute kernel time by strategy", fontsize=13)
    fig.tight_layout()
    p = out_dir / "hw_grid_absolute.png"
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)

    # ---- Figure 3: Isolated — compat vs per_op_rhs vs per_op_lhs ----------
    isolated = [r for r in rows if r["test"] == "isolated"]
    if isolated:
        strategies = sorted(set(r["strategy"] for r in isolated))
        m_vals = sorted(set(r["m_tiles"] for r in isolated))
        fig, axes = plt.subplots(1, len(m_vals), figsize=(7 * len(m_vals), 6), sharey=False)
        if len(m_vals) == 1:
            axes = [axes]

        strat_style = {
            "compat": ("compat", "o-", "#e41a1c"),
            "per_op_rhs": ("per-op RHS (10 rows)", "s--", "#377eb8"),
            "per_op_lhs": ("per-op LHS (11 cols)", "^:", "#4daf4a"),
        }

        for ax, m in zip(axes, m_vals):
            for strat in strategies:
                pts = sorted([r for r in isolated if r["m_tiles"] == m and r["strategy"] == strat],
                             key=lambda r: r["k_tiles"])
                if not pts:
                    continue
                xs = [r["k_tiles"] for r in pts]
                ys = [r["kernel_ns"] / 1e3 for r in pts]
                label, style, color = strat_style.get(strat, (strat, "o-", None))
                ax.plot(xs, ys, style, color=color, label=label, linewidth=2, markersize=7)

            ax.set_title(f"M=N={m}t")
            ax.set_xlabel("K dimension [tiles]")
            ax.set_ylabel("Kernel time [us]")
            ax.set_xticks(all_k)
            ax.set_xticklabels([f"{k}t" for k in all_k])
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Isolated matmul: compat vs per-op grid strategies", fontsize=13)
        fig.tight_layout()
        p = out_dir / "hw_grid_isolated.png"
        fig.savefig(p, dpi=150)
        print(f"Saved {p}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--prof-root",
        default="benchmark/sweep/prof_per_config",
        help="Directory containing per-config tracy reports (default: benchmark/sweep/prof_per_config)",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmark/sweep/plots",
        help="Output directory for plots (default: benchmark/sweep/plots)",
    )
    args = parser.parse_args()

    prof_root = ROOT / args.prof_root
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {prof_root} ...")
    raw = load_results(prof_root)
    rows = aggregate_total(raw)
    print(f"  {len(raw)} data points -> {len(rows)} configs after aggregation")

    if rows:
        plot_scaling(rows, out_dir)
        plot_speedup(rows, out_dir)
        plot_d2m_matmul(rows, out_dir)
        plot_d2m_add(rows, out_dir)
        plot_dm_vs_compute(rows, out_dir)
    else:
        print("No legacy sweep results found — skipping scaling/speedup/DM plots.")
    plot_k_dim_sweep(prof_root, out_dir)
    plot_hw_grid_sweep(prof_root, out_dir)

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
