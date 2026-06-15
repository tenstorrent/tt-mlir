# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import math
import pathlib
import re
from collections import defaultdict
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# basically all FW and KERNEL zones that are not of interest
EXCLUDED_ZONES: frozenset[str] = frozenset(
    {
        "BRISC-FW",
        "NCRISC-FW",
        "TRISC_0-FW",
        "TRISC_1-FW",
        "TRISC_2-FW",
        "ERISC-FW",
        "IDLE_ERISC-FW",
        "SUBORDINATE_IDLE_ERISC-FW",
        "NCRISC-KERNEL",
        "BRISC-KERNEL",
    }
)

WAIT_ZONES: frozenset[str] = frozenset(
    {"cb_wait_front", "cb_reserve_back", "tile_regs_acquire", "tile_regs_wait"}
)


def cycles_to_ns(cycles: int, freq_mhz: float) -> float:
    return cycles / freq_mhz * 1e3


def dur(cycles: int, freq_mhz: float) -> dict:
    """a duration as both raw cycles and derived nanoseconds, for JSON output"""

    return {"cycles": int(cycles), "ns": cycles_to_ns(cycles, freq_mhz)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_perf_dir",
        type=pathlib.Path,
        help="path to the directory containing the profile_log_device.csv file",
    )
    parser.add_argument(
        "--by_kernel",
        action="store_true",
        help="split ops up by kernel for detailed view (applies to all other options)",
    )
    parser.add_argument(
        "--op_times",
        action="store_true",
        help="displays a table of all captured ops along with their associated data",
    )
    parser.add_argument(
        "--op_graph",
        metavar="PATH",
        default=None,
        help="plots a bar graph of each op's total duration to PATH",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="prints the timeline of every op call in chronological order",
    )
    parser.add_argument(
        "--runtimes", action="store_true", help="shows how long the program took to run"
    )
    parser.add_argument(
        "--waits",
        action="store_true",
        help="prints information about waits in the program",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="writes a structured JSON snapshot of every section to PATH (machine/LLM consumption); composes with the text flags",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="parse hardware counter stats and print as a table",
    )
    parser.add_argument(
        "--stat_graph",
        metavar="PATH",
        default=None,
        help="plots a heatmap graph of each stat collected",
    )
    parser.add_argument(
        "--compare_to",
        type=pathlib.Path,
        metavar="PATH",
        default=None,
        help="compare this report to another report, highlight differences",
    )
    args = parser.parse_args()

    if not args.path_to_perf_dir.is_dir():
        parser.error(f"{args.path_to_perf_dir} is not a valid directory")

    return args


def read_chip_freq_mhz(profile_log: pathlib.Path) -> float:
    with profile_log.open() as f:
        header = f.readline()
    m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header)
    if not m:
        raise ValueError(f"Could not find CHIP_FREQ in {profile_log}")
    return float(m.group(1))


def collect_device_timeline(profile_log: pathlib.Path) -> tuple[list[dict], int]:
    """returns a list of all captured zones (excluding firmware and kernel setup zones)
    as well as the total runtime of the program as a tuple"""

    with profile_log.open() as f:
        f.readline()  # skip arch header
        reader = csv.DictReader(f)

        result: list[dict] = []

        open_zones: dict[tuple, int] = {}
        kernel_name: str = ""
        # Track the timestamp span per core (core_x/core_y). The computed wall time
        # below is the longest per-core span observed in the trace (it may still
        # include host-side gaps between dispatches depending on instrumentation).
        minmax = defaultdict(lambda: (float("inf"), 0))

        for row in reader:
            row = {k.strip().lower(): v for k, v in row.items()}

            cycles = int(row["time[cycles since reset]"])
            key = (int(row["core_x"]), int(row["core_y"]))
            minmax[key] = (min(minmax[key][0], cycles), max(minmax[key][1], cycles))

            if row["zone name"].startswith("kernel_outer"):
                kernel_name = row["zone name"][13:]
                continue

            if row["zone name"] in EXCLUDED_ZONES:
                continue

            key = (
                row["core_x"],
                row["core_y"],
                row["risc processor type"],
                row["timer_id"],
                row["run host id"],
                row["zone name"],
            )

            if row["type"] == "ZONE_START":
                open_zones[key] = cycles
            elif row["type"] == "ZONE_END":
                start_cycles = open_zones.pop(key, None)
                if start_cycles is None:
                    continue

                zone = {
                    "name": row["zone name"],
                    "kernel": kernel_name,
                    "host_id": row["run host id"],
                    "core": (int(row["core_x"]), int(row["core_y"])),
                    "risc": row["risc processor type"],
                    "duration": cycles - start_cycles,
                    "start": start_cycles,
                    "end": cycles,
                }

                result.append(zone)

        # Sum per-dispatch spans so inter-dispatch host idle is excluded; for a
        # single-dispatch trace this equals the old global max-min.
        wall_cycles = max((hi - lo for lo, hi in minmax.values()), default=0)
        return result, wall_cycles


def aggregate_by_zone(
    rows: list[dict], split_by_kernel: bool
) -> dict[str, tuple[int, int, int]]:
    """returns a dict mapping op name -> (total duration in cycles, number of calls)"""

    result: dict[str, tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))

    for row in rows:
        if row["name"] in ("TRISC-KERNEL", "TRISC-FW"):
            continue

        if split_by_kernel:
            key = f'{row["name"]}[{row["risc"]}:{row["kernel"]}]'
        else:
            key = f'{row["name"]}'
        total, count, maximum = result[key]
        result[key] = (
            total + row["duration"],
            count + 1,
            max(maximum, row["duration"]),
        )

    return result


def get_wait_share(rows: list[dict]) -> float:
    """fraction of inner-zone cycles spent in CB / tile-reg waits"""

    inner = [
        r
        for r in rows
        if not r["name"].endswith("-KERNEL") and not r["name"].endswith("-FW")
    ]
    total = sum(r["duration"] for r in inner)
    wait = sum(r["duration"] for r in inner if r["name"] in WAIT_ZONES)
    return wait / total if total else 0.0


def get_runtimes(rows: list[dict], wall_cycles: int) -> dict[str, float]:
    '''returns a dict with fields "device wall time", "device kernel time", "compute share", "wait share"'''

    trisc_fw_cycles = sum(r["duration"] for r in rows if r["name"] == "TRISC-FW")
    trisc_kernels: dict[tuple, list] = defaultdict(list)

    for row in rows:
        if row["name"] == "TRISC-KERNEL":
            key = (row["core"], row["host_id"])
            trisc_kernels[key].append(row["start"])
            trisc_kernels[key].append(row["end"])

    kernel_cycles = sum(max(trisc) - min(trisc) for trisc in trisc_kernels.values())

    return {
        "device wall time": wall_cycles,
        "device kernel time": kernel_cycles,
        "compute share": kernel_cycles / trisc_fw_cycles if trisc_fw_cycles else 0.0,
        "wait share": get_wait_share(rows),
    }


def time_formatter(cycles: int, freq_mhz: float) -> str:
    """formats time as {cycles} ({time in ns} ns)"""

    return f"{int(cycles):} ({cycles_to_ns(cycles, freq_mhz):.3f} ns)"


def print_runtimes(runtimes: dict, freq_mhz: float) -> None:
    formatted = {
        k: f"{v * 100:.5f}%" if "share" in k else time_formatter(v, freq_mhz)
        for k, v in runtimes.items()
    }
    label_w = max(len(k) for k in formatted)
    val_w = max(len(v) for v in formatted.values())

    print("\n===== RUNTIME =====\n")
    for k, v in formatted.items():
        print(f"{k:<{label_w}}  {v:>{val_w}}")


def plot_histogram(rows: dict[str, tuple[int, int, int]], output_file: str) -> None:
    labels = list(rows.keys())
    totals = [t for t, _, _ in rows.values()]
    counts = [c for _, c, _ in rows.values()]
    maxes = [m for _, _, m in rows.values()]

    fig, (ax_dur, ax_cnt, ax_max) = plt.subplots(
        1,
        3,
        sharey=True,
        figsize=(12, max(4, 0.3 * len(labels))),
    )

    ax_dur.barh(labels, totals)
    ax_dur.set_xlabel("Total duration (cycles)")
    ax_dur.set_ylabel("TTKernel Op")

    ax_cnt.barh(labels, counts)
    ax_cnt.set_xlabel("Number of calls")

    ax_max.barh(labels, maxes)
    ax_max.set_xlabel("Longest call (cycles)")

    fig.suptitle("Kernel runtime breakdown")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def plot_stats(perf_stats: dict, output_file: str) -> None:
    """plots each H/W counter stat as a per-core heatmap laid out on the chip
    grid (core_x across, core_y down). perf_stats is
    {stat: {"(core_x, core_y)": value, ...}, ...}; each stat gets its own color
    scale since the metrics span very different ranges (e.g. fpu utilization vs.
    noc vs compute)."""

    def parse_core(core: str) -> tuple[int, int]:
        x, y = core.strip("()").split(",")
        return int(x), int(y)

    # collect the physical grid axes shared across all stats; cores are sparse
    # (non-compute rows/cols are absent) so we index by sorted position, not by
    # a contiguous 0..N range.
    all_cores = {core for by_core in perf_stats.values() for core in by_core}

    if not all_cores:
        print("No hardware counter stats found; skipping stat graph.")
        return

    xs = sorted({parse_core(c)[0] for c in all_cores})
    ys = sorted({parse_core(c)[1] for c in all_cores})
    x_idx = {x: i for i, x in enumerate(xs)}
    y_idx = {y: i for i, y in enumerate(ys)}

    ncols = min(4, len(perf_stats))
    nrows = math.ceil(len(perf_stats) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.2 * nrows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, (stat, by_core) in zip(axes.flat, perf_stats.items()):
        grid = np.full((len(ys), len(xs)), np.nan)
        for core, value in by_core.items():
            x, y = parse_core(core)
            grid[y_idx[y], x_idx[x]] = value

        im = ax.imshow(grid, cmap="inferno", aspect="equal")
        ax.set_title(stat, fontsize=9)
        ax.axis("on")
        ax.set_xlabel("core_x")
        ax.set_ylabel("core_y")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=6)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Per-core hardware counter stats (chip grid)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def print_op_times(rows: dict[str, tuple[int, int, int]], freq_mhz: float) -> None:
    print("\n===== OP TIMES =====\n")
    total_cycles = sum(v[0] for v in rows.values())

    headers = ("Op", "Calls", "Total time", "Longest Call", "Cycles/call", "%")
    sorted_rows = sorted(rows.items(), key=lambda kv: kv[1][0], reverse=True)
    table = [
        (
            k,
            str(count),
            time_formatter(total, freq_mhz),
            time_formatter(maximum, freq_mhz),
            f"{total / count:.3f}",
            f"{total / total_cycles * 100:.2f}%",
        )
        for k, (total, count, maximum) in sorted_rows
    ]
    widths = [max(len(row[i]) for row in (*table, headers)) for i in range(6)]

    def fmt(row: tuple[str, str, str, str, str]) -> str:
        return (
            f"{row[0]:<{widths[0]}}  {row[1]:>{widths[1]}}  "
            f"{row[2]:>{widths[2]}}  {row[3]:>{widths[3]}}  {row[4]:>{widths[4]}}  {row[5]:>{widths[5]}}"
        )

    print(fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in table:
        print(fmt(row))


def print_timeline(rows: list[dict], freq_mhz: float) -> None:
    current_kernel: str = None

    max_width = 0
    for row in rows:
        max_width = max(max_width, len(row["name"]))

    print("\n===== TIMELINE =====")
    for row in rows:
        if row["kernel"] != current_kernel:
            current_kernel = row["kernel"]
            print(current_kernel)

        print(
            f'\t{row["name"]:<{max_width + 3}}\t[core:{row["core"]}, RISC: {row["risc"]}]\t{time_formatter(row["duration"], freq_mhz)}'
        )


def aggregate_waits(rows: list[dict]) -> dict:
    """returns per-kernel wait totals and a summary of the longest waiter:
    {"per_kernel": {kernel: wait_cycles}, "longest": {...} | None}"""

    total_waits: dict[str, int] = defaultdict(int)
    total_compute: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["name"] == "TRISC-KERNEL":
            total_compute[row["kernel"]] += row["duration"]

        if row["name"] in WAIT_ZONES:
            total_waits[row["kernel"]] += row["duration"]

    sum_waits = sum(total_waits.values())
    longest_kernel = max(total_waits, key=total_waits.get, default=None)

    longest = None
    if longest_kernel is not None:
        longest_wait = total_waits[longest_kernel]
        envelope = total_compute[longest_kernel]
        longest = {
            "kernel": longest_kernel,
            "wait_cycles": longest_wait,
            "share_of_total_wait": longest_wait / sum_waits if sum_waits else 0.0,
            "share_of_kernel_envelope": longest_wait / envelope if envelope else 0.0,
        }

    return {"per_kernel": dict(total_waits), "longest": longest}


def print_waits(rows: list[dict], freq_mhz: float) -> None:
    waits = aggregate_waits(rows)
    per_kernel = waits["per_kernel"]

    print("\n===== WAITS ACROSS KERNELS =====\n")
    kernel_width = max((len(k) for k in per_kernel), default=0)
    for kernel, wait in per_kernel.items():
        print(f"{kernel:<{kernel_width + 3}}\t{time_formatter(wait, freq_mhz)}")

    longest = waits["longest"]
    if longest is None:
        print("\nLongest wait: none")
        return
    print(
        f"\nLongest wait: {longest['kernel']:<{kernel_width + 3}}\t"
        f"{time_formatter(longest['wait_cycles'], freq_mhz)}\t"
        f"{longest['share_of_total_wait'] * 100:.2f}% of total wait, "
        f"{longest['share_of_kernel_envelope'] * 100:.2f}% of kernel envelope"
    )


def collect_device_runtimes(profile_log: pathlib.Path, freq_mhz: float) -> dict:
    """Per-op "DEVICE KERNEL DURATION" computed straight from the device log,
    independent of ops_perf_results.csv.

    Mirrors tt-metal's `device_kernel_duration` analysis (device_post_proc_config.py):
    `across: ops`, `type: op_first_last` over every `*-KERNEL` zone. For each op
    (keyed by "run host id") it is the span from the FIRST `*-KERNEL` ZONE_START to
    the LAST `*-KERNEL` ZONE_END, across ALL cores and ALL RISCs
    (BRISC/NCRISC/TRISC/ERISC) — not just TRISC. Returned in cycles; the rest of the
    tool works in cycles and converts to ns at print time.
    """
    # per op (run host id) -> [min *-KERNEL start, max *-KERNEL end]
    op_kernel_span: dict[str, list] = defaultdict(lambda: [float("inf"), 0])

    with profile_log.open() as f:
        f.readline()  # skip arch header
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip().lower(): v for k, v in row.items()}
            if not row["zone name"].endswith("-KERNEL"):
                continue
            cycles = int(row["time[cycles since reset]"])
            span = op_kernel_span[row["run host id"]]
            if row["type"] == "ZONE_START":
                span[0] = min(span[0], cycles)
            elif row["type"] == "ZONE_END":
                span[1] = max(span[1], cycles)

    # one value per op, matching a DEVICE KERNEL DURATION row in ops_perf_results.csv
    kernel_durations = [
        hi - lo for lo, hi in op_kernel_span.values() if lo != float("inf") and hi >= lo
    ]
    if not kernel_durations:
        print(
            f"\n***** {profile_log} has no kernel zones. Unable to collect runtime information."
        )
        return {"kernel duration": 0}

    # reduce to the longest single op, as before (swap for sum() for total kernel time)
    return {"kernel duration": max(kernel_durations)}


def collect_perf_counters(profile_log: pathlib.Path) -> pd.DataFrame:
    PERF_COUNTER_TIMER_ID: str = "9090"
    perf_counter_events = []

    with profile_log.open() as f:
        f.readline()
        reader = csv.DictReader(f)

        for row in reader:
            row = {k.strip().lower(): v for k, v in row.items()}
            if row["timer_id"] == PERF_COUNTER_TIMER_ID:
                raw_md = row["meta data"]
                meta_data = json.loads(raw_md.replace(";", ",").replace("'", '"'))
                perf_counter_events.append(
                    {
                        "run_host_id": row["run host id"],
                        "record time": row["time[cycles since reset]"],
                        "core_x": int(row["core_x"]),
                        "core_y": int(row["core_y"]),
                        "risc_type": row["risc processor type"],
                        "counter type": meta_data.get("counter type", ""),
                        "value": meta_data.get("value", 0),
                        "ref cnt": meta_data.get("ref cnt", 0),
                    }
                )

    return pd.DataFrame(perf_counter_events)


def get_perf_counter_stats(perf_counters: pd.DataFrame) -> dict:
    """Calculate meaningful metrics from raw h/w performance counter values (from tt-metal)."""
    if perf_counters.empty:
        return {}

    def get_counter_values(*counter_names: str):
        mask = perf_counters["counter type"].isin(counter_names)
        return perf_counters[mask].set_index(["run_host_id", "core_x", "core_y"])[
            "value"
        ]

    def get_counter_ref_cnt(*counter_names: str):
        mask = perf_counters["counter type"].isin(counter_names)
        return perf_counters[mask].set_index(["run_host_id", "core_x", "core_y"])[
            "ref cnt"
        ]

    def get_value_ref_ratio(*counter_names: str):
        idx = ["run_host_id", "core_x", "core_y"]
        vals = get_counter_values(*counter_names).groupby(level=idx).sum()
        refs = get_counter_ref_cnt(*counter_names).groupby(level=idx).sum()
        return (vals / refs).replace([float("inf"), -float("inf")], float("nan"))

    def get_stats(series: pd.Series):
        return {"min": series.min(), "max": series.max(), "mean": series.mean()}

    def get_stats_by_core(series: pd.Series):
        series = series.groupby(level=("core_x", "core_y"))
        return {f"({name[0]},{name[1]})": group.mean() for name, group in series}

    sfpu_util = get_value_ref_ratio("SFPU_COUNTER")
    fpu_util = get_value_ref_ratio("FPU_COUNTER")
    mmio_idle_t0 = get_value_ref_ratio("WAITING_FOR_MMIO_IDLE_0")
    sfpu_idle_t1 = get_value_ref_ratio("WAITING_FOR_SFPU_IDLE_1")
    thcon_idle_t0 = get_value_ref_ratio("WAITING_FOR_THCON_IDLE_0")
    move_idle_t0 = get_value_ref_ratio("WAITING_FOR_MOVE_IDLE_0")
    semaphore_zero_wait_0 = get_value_ref_ratio("WAITING_FOR_NONZERO_SEM_0")
    semaphore_zero_wait_1 = get_value_ref_ratio("WAITING_FOR_NONZERO_SEM_1")
    semaphore_zero_wait_2 = get_value_ref_ratio("WAITING_FOR_NONZERO_SEM_2")
    semaphore_full_wait_0 = get_value_ref_ratio("WAITING_FOR_NONFULL_SEM_0")
    semaphore_full_wait_1 = get_value_ref_ratio("WAITING_FOR_NONFULL_SEM_1")
    semaphore_full_wait_2 = get_value_ref_ratio("WAITING_FOR_NONFULL_SEM_2")

    noc_out = (
        get_counter_values("L1_0_NOC_RING0_OUTGOING_0")
        + get_counter_values("L1_0_NOC_RING0_OUTGOING_1")
    ) / 2
    noc_in = (
        get_counter_values("L1_0_NOC_RING0_INCOMING_0")
        + get_counter_values("L1_0_NOC_RING0_INCOMING_1")
    ) / 2
    fpu_counter = get_counter_values("FPU_COUNTER")
    noc_vs_compute = ((noc_out + noc_in) / (fpu_counter + noc_out + noc_in)).replace(
        float("nan"), 0
    )

    return {
        "sfpu utilization": get_stats_by_core(sfpu_util),
        "fpu utilization": get_stats_by_core(fpu_util),
        "mmio idle t0": get_stats_by_core(mmio_idle_t0),
        "sfpu idle t1": get_stats_by_core(sfpu_idle_t1),
        "thcon idle t0": get_stats_by_core(thcon_idle_t0),
        "move idle t0": get_stats_by_core(move_idle_t0),
        "semaphore zero wait t0": get_stats_by_core(semaphore_zero_wait_0),
        "semaphore zero wait t1": get_stats_by_core(semaphore_zero_wait_1),
        "semaphore zero wait t2": get_stats_by_core(semaphore_zero_wait_2),
        "semaphore full wait t0": get_stats_by_core(semaphore_full_wait_0),
        "semaphore full wait t1": get_stats_by_core(semaphore_full_wait_1),
        "semaphore full wait t2": get_stats_by_core(semaphore_full_wait_2),
        "noc vs compute": get_stats_by_core(noc_vs_compute),
    }


def print_perf_counter_stats(perf_stats: dict) -> None:
    print("\n===== H/W COUNTER STATS =====\n")

    # cores live in the inner dicts; collect them in first-seen order across all stats
    cores: list = []
    for by_core in perf_stats.values():
        for core in by_core:
            if core not in cores:
                cores.append(core)

    headers = ("Stat",) + tuple(f"Mean{core}" for core in cores)
    table = [
        (k,)
        + tuple(
            f"{by_core[core] * 100:.3f}%" if core in by_core else "-" for core in cores
        )
        for k, by_core in perf_stats.items()
    ]
    widths = [
        max(len(row[i]) for row in (*table, headers)) for i in range(len(headers))
    ]

    def fmt(row: tuple[str, str, str, str]) -> str:
        return "  ".join(f"{column:>{width}}" for column, width in zip(row, widths))

    print(fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in table:
        print(fmt(row))


def build_report(profile_log: pathlib.Path, by_kernel: bool = False) -> dict:
    """assembles a self-describing JSON snapshot of every analysis section for a
    single profile_log, with durations as raw cycles plus derived ns (never
    pre-formatted strings).

    Self-contained: every section is derived from the profile_log path alone, so
    this can be reused/mapped across multiple profiles, e.g.
    ``{p: build_report(p) for p in profile_logs}``."""

    freq_mhz = read_chip_freq_mhz(profile_log)
    raw_timeline, wall_cycles = collect_device_timeline(profile_log)
    zone_grouped_rows = aggregate_by_zone(raw_timeline, by_kernel)
    perf_stats = get_perf_counter_stats(collect_perf_counters(profile_log))
    runtimes = collect_device_runtimes(profile_log, freq_mhz)

    total_cycles = sum(t for t, _, _ in zone_grouped_rows.values())
    op_times = [
        {
            "op": op,
            "calls": count,
            "total": dur(total, freq_mhz),
            "longest": dur(maximum, freq_mhz),
            "cycles_per_call": total / count if count else 0.0,
            "pct_of_total": total / total_cycles * 100 if total_cycles else 0.0,
        }
        for op, (total, count, maximum) in sorted(
            zone_grouped_rows.items(), key=lambda kv: kv[1][0], reverse=True
        )
    ]

    waits = aggregate_waits(raw_timeline)
    waits_json = {
        "per_kernel": [
            {"kernel": k, "wait": dur(c, freq_mhz)}
            for k, c in sorted(
                waits["per_kernel"].items(), key=lambda kv: kv[1], reverse=True
            )
        ],
        "longest": None
        if waits["longest"] is None
        else {
            "kernel": waits["longest"]["kernel"],
            "wait": dur(waits["longest"]["wait_cycles"], freq_mhz),
            "share_of_total_wait": waits["longest"]["share_of_total_wait"],
            "share_of_kernel_envelope": waits["longest"]["share_of_kernel_envelope"],
        },
    }

    return {
        "metadata": {
            "source": str(profile_log),
            "chip_freq_mhz": freq_mhz,
            "device_wall_time": {
                **dur(wall_cycles, freq_mhz),
                "note": "max-min timestamp across the trace; includes host-side gaps between dispatches",
            },
        },
        "runtimes": {
            "device_kernel_time": dur(runtimes["kernel duration"], freq_mhz),
        },
        "op_times": op_times,
        "waits": waits_json,
        "stats": perf_stats,
    }


def compare_reports(
    profile_log1: pathlib.Path,
    profile_log2: pathlib.Path,
    by_kernel: bool = False,
) -> dict:
    """builds a report for each profile and lays the two side by side.

    Returns a dict whose leaves are ``{"a": <val from profile1>, "b": <val from
    profile2>, "ratio": a/b}`` triples (ratio is None where it can't be computed
    or where b == 0). Sections: ``runtimes``, ``op_times`` (op by op, including
    ops present in only one profile), and ``stats``."""

    report1 = build_report(profile_log1, by_kernel)
    report2 = build_report(profile_log2, by_kernel)

    def ratio(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b

    def cmp(a, b):
        return {"a": a, "b": b, "ratio": ratio(a, b)}

    def cmp_dur(a: dict | None, b: dict | None) -> dict:
        # a/b are {"cycles", "ns"} dicts (or None when the op is missing here)
        ac = None if a is None else a["cycles"]
        bc = None if b is None else b["cycles"]
        return {
            "cycles": cmp(ac, bc),
            "ns": cmp(
                None if a is None else a["ns"],
                None if b is None else b["ns"],
            ),
        }

    # ---- runtimes -------------------------------------------------------
    rt1 = report1["runtimes"]["device_kernel_time"]
    rt2 = report2["runtimes"]["device_kernel_time"]
    runtimes = {"device_kernel_time": cmp_dur(rt1, rt2)}

    # ---- op_times (op by op) -------------------------------------------
    ops1 = {o["op"]: o for o in report1["op_times"]}
    ops2 = {o["op"]: o for o in report2["op_times"]}
    # preserve profile1's ordering (already sorted by total cycles desc), then
    # append any ops that only appear in profile2
    op_order = list(ops1) + [op for op in ops2 if op not in ops1]

    op_times = {}
    for op in op_order:
        o1 = ops1.get(op)
        o2 = ops2.get(op)
        op_times[op] = {
            "present_in": {"a": o1 is not None, "b": o2 is not None},
            "calls": cmp(
                None if o1 is None else o1["calls"],
                None if o2 is None else o2["calls"],
            ),
            "total": cmp_dur(
                None if o1 is None else o1["total"],
                None if o2 is None else o2["total"],
            ),
            "longest": cmp_dur(
                None if o1 is None else o1["longest"],
                None if o2 is None else o2["longest"],
            ),
            "cycles_per_call": cmp(
                None if o1 is None else o1["cycles_per_call"],
                None if o2 is None else o2["cycles_per_call"],
            ),
        }

    # ---- stats (h/w counters, per stat per core) -----------------------
    stats1 = report1["stats"]
    stats2 = report2["stats"]
    stat_order = list(stats1) + [s for s in stats2 if s not in stats1]

    stats = {}
    for stat in stat_order:
        by_core1 = stats1.get(stat, {})
        by_core2 = stats2.get(stat, {})
        cores = list(by_core1) + [c for c in by_core2 if c not in by_core1]
        stats[stat] = {
            str(core): cmp(by_core1.get(core), by_core2.get(core)) for core in cores
        }

    return {
        "a": str(profile_log1),
        "b": str(profile_log2),
        "runtimes": runtimes,
        "op_times": op_times,
        "stats": stats,
    }


def print_comparison(comparison: dict) -> None:
    """pretty-prints the dict produced by compare_reports as side-by-side tables
    (A | B | ratio) for the runtimes, op_times, and stats sections."""

    def num(v, fmt: str = "{:.3f}") -> str:
        # render a comparison leaf value (may be None when an op/stat is absent)
        return "-" if v is None else fmt.format(v) if isinstance(v, float) else str(v)

    def ratio_str(r) -> str:
        return "-" if r is None else f"{r:.3f}x"

    def print_table(headers: tuple, table: list[tuple]) -> None:
        widths = [
            max(len(row[i]) for row in (*table, headers)) for i in range(len(headers))
        ]

        def fmt(row: tuple) -> str:
            cells = [f"{row[0]:<{widths[0]}}"] + [
                f"{row[i]:>{widths[i]}}" for i in range(1, len(row))
            ]
            return "  ".join(cells)

        print(fmt(headers))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for row in table:
            print(fmt(row))

    a_label = comparison["a"]
    b_label = comparison["b"]
    print("\n===== COMPARISON =====\n")
    print(f"  A: {a_label}")
    print(f"  B: {b_label}")
    print("  ratio = A / B")

    # ---- runtimes -------------------------------------------------------
    print("\n----- RUNTIMES -----\n")
    headers = ("Metric", "A [ns]", "B [ns]", "ratio")
    table = []
    for name, leaf in comparison["runtimes"].items():
        ns = leaf["ns"]
        table.append((name, num(ns["a"]), num(ns["b"]), ratio_str(ns["ratio"])))
    print_table(headers, table)

    # ---- op_times -------------------------------------------------------
    print("\n----- OP TIMES (cycles) -----\n")
    headers = (
        "Op",
        "Calls A",
        "Calls B",
        "Total A",
        "Total B",
        "ratio",
        "Longest A",
        "Longest B",
        "Cyc/call A",
        "Cyc/call B",
    )
    table = []
    for op, leaf in comparison["op_times"].items():
        calls = leaf["calls"]
        total = leaf["total"]["cycles"]
        longest = leaf["longest"]["cycles"]
        cpc = leaf["cycles_per_call"]
        table.append(
            (
                op,
                num(calls["a"]),
                num(calls["b"]),
                num(total["a"]),
                num(total["b"]),
                ratio_str(total["ratio"]),
                num(longest["a"]),
                num(longest["b"]),
                num(cpc["a"]),
                num(cpc["b"]),
            )
        )
    print_table(headers, table)

    # ---- stats ----------------------------------------------------------
    if comparison["stats"]:
        print("\n----- H/W COUNTER STATS (A% | B% | ratio, per core) -----\n")
        # one row per (stat, core); percentages match print_perf_counter_stats
        headers = ("Stat", "Core", "A", "B", "ratio")
        table = []
        for stat, by_core in comparison["stats"].items():
            for core, leaf in by_core.items():
                a = leaf["a"]
                b = leaf["b"]
                table.append(
                    (
                        stat,
                        core,
                        "-" if a is None else f"{a * 100:.3f}%",
                        "-" if b is None else f"{b * 100:.3f}%",
                        ratio_str(leaf["ratio"]),
                    )
                )
        print_table(headers, table)


def main() -> None:
    args = parse_args()
    perf_dir: pathlib.Path = args.path_to_perf_dir
    profile_log = perf_dir / "profile_log_device.csv"

    if not profile_log.exists():
        raise FileNotFoundError("profile_log_device.csv not found")

    print(f"Reading from {profile_log}...")

    freq = read_chip_freq_mhz(profile_log)
    raw_timeline, wall_time = collect_device_timeline(profile_log)
    zone_grouped_rows = aggregate_by_zone(raw_timeline, args.by_kernel)
    runtimes = collect_device_runtimes(profile_log, freq)
    perf_counters = collect_perf_counters(profile_log)
    perf_stats = get_perf_counter_stats(perf_counters)
    if args.op_graph:
        plot_histogram(zone_grouped_rows, args.op_graph)
        print()
    if args.op_times:
        print_op_times(zone_grouped_rows, freq)
        print()
    if args.timeline:
        print_timeline(raw_timeline, freq)
        print()
    if args.runtimes:
        print_runtimes(runtimes, freq)
        print()
    if args.waits:
        print_waits(raw_timeline, freq)
        print()
    if args.stats:
        print_perf_counter_stats(perf_stats)
        print()
    if args.stat_graph:
        plot_stats(perf_stats, args.stat_graph)
        print()
    if args.compare_to:
        prof2 = args.compare_to / "profile_log_device.csv"
        print(f"Reading from {prof2}...")
        print("***** Make sure that both reports contain the same instrumentation")
        comp = compare_reports(profile_log, prof2, args.by_kernel)
        print_comparison(comp)
        print()
    if args.json:
        report = build_report(profile_log, args.by_kernel)
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report -> {args.json}")


if __name__ == "__main__":
    main()
