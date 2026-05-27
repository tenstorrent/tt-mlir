import argparse
import csv
import pathlib
import re
import statistics
import typing
from collections import defaultdict


GENERIC_COMPUTE_ZONES = {
    "binary_op_init_common",
    "init_sfpu",
    "copy_tile_init",
    "compute_kernel_hw_startup",
}

FIRMWARE_ZONES = {
    "BRISC-FW", "NCRISC-FW", "TRISC-FW",
    "BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_perf_dir", type=pathlib.Path)
    parser.add_argument("--opt_times", action="store_true")
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


def parse_profile_log(profile_log: pathlib.Path):
    """Yield (host_id, zone_name, core_x, core_y, risc, duration_cycles)."""
    with profile_log.open() as f:
        f.readline()  # arch/freq header
        reader = csv.reader(f)
        next(reader)  # column header

        open_zones: dict[tuple, int] = {}
        for row in reader:
            row = [c.strip() for c in row]
            core_x, core_y = row[1], row[2]
            risc = row[3]
            timer_id = row[4]
            cycles = int(row[5])
            host_id = row[7]
            zone_name = row[10]
            zone_type = row[11]

            key = (core_x, core_y, risc, timer_id, host_id, zone_name)
            if zone_type == "ZONE_START":
                open_zones[key] = cycles
            elif zone_type == "ZONE_END":
                start = open_zones.pop(key, None)
                if start is None:
                    continue
                yield host_id, zone_name, core_x, core_y, risc, cycles - start


def derive_op_label(compute_zones: set[str]) -> str:
    primitives = set()
    for z in compute_zones:
        if z in GENERIC_COMPUTE_ZONES or z.startswith("kernel_outer_"):
            continue
        if z in FIRMWARE_ZONES:
            continue
        primitives.add(z[:-5] if z.endswith("_init") else z)
    return "+".join(sorted(primitives)) if primitives else "<unknown>"


def aggregate_by_host_id(profile_log: pathlib.Path, freq_mhz: float):
    """Returns dict[host_id] -> {op_label, total_compute_ns, zone_stats}."""
    zones_per_host: dict[str, set[str]] = defaultdict(set)
    zone_cycles: dict[tuple[str, str], list[int]] = defaultdict(list)

    for host_id, zone_name, *_core, dur in parse_profile_log(profile_log):
        if zone_name in FIRMWARE_ZONES or zone_name.startswith("kernel_outer_"):
            continue
        zones_per_host[host_id].add(zone_name)
        zone_cycles[(host_id, zone_name)].append(dur)

    result = {}
    cycles_to_ns = 1000.0 / freq_mhz
    for host_id, zones in zones_per_host.items():
        zone_stats = {}
        total_compute_cycles = 0
        for z in zones:
            durs = zone_cycles[(host_id, z)]
            total_compute_cycles += sum(durs)
            zone_stats[z] = {
                "count": len(durs),
                "total_ns": sum(durs) * cycles_to_ns,
                "mean_ns": statistics.mean(durs) * cycles_to_ns,
            }
        result[host_id] = {
            "op_label": derive_op_label(zones),
            "total_compute_ns": total_compute_cycles * cycles_to_ns,
            "zone_stats": zone_stats,
        }
    return result


def join_with_ops_perf(ops_perf_csv: pathlib.Path, per_host: dict):
    """Yield merged dict rows: ops_perf row + derived op label/compute time."""
    with ops_perf_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            host_id = row["GLOBAL CALL COUNT"].strip()
            derived = per_host.get(host_id, {})
            yield {
                "host_id": host_id,
                "op_label": derived.get("op_label", "<no-profile-data>"),
                "loc": row["OP CODE"],
                "device_kernel_ns": int(row["DEVICE KERNEL DURATION [ns]"] or 0),
                "host_ns": int(row["HOST DURATION [ns]"] or 0),
                "op_to_op_ns": int(row["OP TO OP LATENCY [ns]"] or 0),
                "core_count": int(row["CORE COUNT"] or 0),
                "compute_ns_from_zones": derived.get("total_compute_ns", 0.0),
            }


def print_per_op_table(rows: list[dict]) -> None:
    header = ["host_id", "op_label", "kernel_ns", "host_ns", "op2op_ns", "cores", "loc"]
    widths = [12, 28, 12, 12, 20, 6, 40]
    print("\n\n===== PER-OP TABLE ===== \n")
    print("  ".join(f"{h:<{w}}" for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        cells = [
            r["host_id"],
            r["op_label"][:widths[1]],
            f"{r['device_kernel_ns']:,}",
            f"{r['host_ns']:,}",
            f"{r['op_to_op_ns']:,}",
            str(r["core_count"]),
            r["loc"][:widths[6]],
        ]
        print("  ".join(f"{c:<{w}}" for c, w in zip(cells, widths)))


def print_aggregate_by_op(rows: list[dict]) -> None:
    by_op: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_op[r["op_label"]].append(r["device_kernel_ns"])

    total = sum(r["device_kernel_ns"] for r in rows) or 1

    header = ['op_label', 'calls', 'total_ns', '%', 'mean_ns', 'max_ns']
    widths = [28, 6, 14, 8, 12, 12]

    print("\n\n===== AGGREGATE TABLE ===== \n")
    print("  ".join(f'{h:<{w}}' for h, w in zip(header, widths)))
    print("  ".join('-' * w for w in widths))
    for label, durs in sorted(by_op.items(), key=lambda kv: -sum(kv[1])):
        s = sum(durs)
        cells = [
            label,
            len(durs),
            s,
            f'{100 * s / total:.2f}%',
            int(statistics.mean(durs)),
            max(durs)
        ]
        print("  ".join(f'{str(c):<{w}}' for c, w in zip(cells, widths)))


def main() -> None:
    args = parse_args()
    perf_dir: pathlib.Path = args.path_to_perf_dir
    profile_log = perf_dir / "profile_log_device.csv"
    ops_perf = perf_dir / "ops_perf_results.csv"

    for p in (profile_log, ops_perf):
        if not p.is_file():
            raise SystemExit(f"missing: {p}")

    freq = read_chip_freq_mhz(profile_log)
    per_host = aggregate_by_host_id(profile_log, freq)
    rows = list(join_with_ops_perf(ops_perf, per_host))

    if args.opt_times:
        print_per_op_table(rows)
    print_aggregate_by_op(rows)


if __name__ == "__main__":
    main()
