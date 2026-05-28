import argparse
import csv
import pathlib
import re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# basically all FW and KERNEL zones that are not of interest
EXCLUDED_ZONES: frozenset[str] = frozenset({
    'BRISC-FW',
    'NCRISC-FW',
    'TRISC-FW',
    'TRISC_0-FW',
    'TRISC_1-FW',
    'TRISC_2-FW',
    'ERISC-FW',
    'IDLE_ERISC-FW',
    'SUBORDINATE_IDLE_ERISC-FW',
    'NCRISC-KERNEL',
    'BRISC-KERNEL'
})


def cycles_to_ns(cycles: int, freq_mhz: float) -> float:
    return cycles / freq_mhz * 1e3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_perf_dir', type=pathlib.Path)
    parser.add_argument('--by_kernel', action='store_true', help='split ops up by kernel for detailed view (applies to all other options)')
    parser.add_argument('--op_times', action='store_true', help='displays a table of all captured ops along with their associated data')
    parser.add_argument('--op_graph', metavar='PATH', default=None, help='plots a bar graph of each op\'s total duration to PATH')
    parser.add_argument('--timeline', action='store_true', help='prints the timeline of every op call in chronological order')
    parser.add_argument('--runtimes', action='store_true', help='shows how long the program took to run')
    # parser.add_argument('--stats', action='store_true', help='performs statistical analysis of the perf output (NOT IMPLEMENTED)')
    args = parser.parse_args()

    if not args.path_to_perf_dir.is_dir():
        parser.error(f'{args.path_to_perf_dir} is not a valid directory')

    return args


def read_chip_freq_mhz(profile_log: pathlib.Path) -> float:
    with profile_log.open() as f:
        header = f.readline()
    m = re.search(r'CHIP_FREQ\[MHz\]:\s*(\d+)', header)
    if not m:
        raise ValueError(f'Could not find CHIP_FREQ in {profile_log}')
    return float(m.group(1))


def collect_device_timeline(profile_log: pathlib.Path) -> tuple[list[dict], int]:
    ''' returns a list of all captured zones (excluding firmware and kernel setup zones)
        as well as the total runtime of the program as a tuple '''

    with profile_log.open() as f:
        f.readline()    # skip arch header
        reader = csv.DictReader(f)
        
        result: list[dict] = []

        open_zones: dict[tuple, int] = {}
        kernel_name: str = ''
        min_cycle, max_cycle = float('inf'), 0
        for row in reader:
            row = {k.strip().lower(): v for k, v in row.items()}

            if row['zone name'].startswith('kernel_outer'):
                kernel_name = row['zone name'][13:]
                continue

            if row['zone name'] in EXCLUDED_ZONES:
                continue

            key = (row['core_x'], row['core_y'], row['risc processor type'], row['timer_id'], row['run host id'], row['zone name'])
            cycles = int(row['time[cycles since reset]'])
            
            min_cycle = min(min_cycle, cycles)
            max_cycle = max(max_cycle, cycles)
            
            if row['type'] == 'ZONE_START':
                open_zones[key] = cycles
            elif row['type'] == 'ZONE_END':
                start_cycles = open_zones.pop(key, None)
                if start_cycles is None:
                    continue

                zone = {
                    'name': row['zone name'],
                    'kernel': kernel_name,
                    'host_id': row['run host id'],
                    'core': (int(row['core_x']), int(row['core_y'])),
                    'risc': row['risc processor type'],
                    'duration': cycles - start_cycles
                }

                result.append(zone)
        
        return result, max_cycle - min_cycle


def aggregate_by_zone(rows: list[dict], split_by_kernel: bool) -> dict[str, tuple[int, int]]:
    ''' returns a dict mapping op name -> (total duration in cycles, number of calls) '''

    result: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))

    for row in rows:
        if row['name'].startswith('TRISC-KERNEL'):
            continue

        if split_by_kernel:
            key = f'{row["name"]}[{row["risc"]}:{row["kernel"]}]'
        else:
            key = f'{row["name"]}'
        total, count = result[key]
        result[key] = (total + row['duration'], count + 1)

    return result


def get_runtimes(rows: list[dict], wall_cycles: int) -> dict[str, float]:
    ''' returns a dict with fields "device wall time", "device kernel time", "compute share" '''

    # count up trisc kernel envelopes
    kernel_cycles = sum(r['duration'] for r in rows if r['name'] == 'TRISC-KERNEL')
    return {
        'device wall time': wall_cycles,
        'device kernel time': kernel_cycles,
        'compute share': kernel_cycles / wall_cycles if wall_cycles else 0.0,
    }


def get_stats(rows: list[dict]) -> dict:
    return {'slowest op': 'slowest op'}


def time_formatter(cycles: int, freq_mhz: float) -> str:
    ''' formats time as {cycles} ({time in ns} ns) '''

    return f'{int(cycles):} ({cycles_to_ns(cycles, freq_mhz):.3f} ns)'


def print_runtimes(rows: list[dict], wall_cycles: int, freq_mhz: float) -> None:
    runtimes = get_runtimes(rows, wall_cycles)
    formatted = {
        k: f'{v * 100:.5f}%' if k == 'compute share' else time_formatter(v, freq_mhz)
        for k, v in runtimes.items()
    }
    label_w = max(len(k) for k in formatted)
    val_w = max(len(v) for v in formatted.values())

    print('\n===== RUNTIME =====\n')
    for k, v in formatted.items():
        print(f'{k:<{label_w}}\t{v:>{val_w}}')


def plot_histogram(rows: dict[str, tuple[int, int]], output_file: str) -> None:
    labels = list(rows.keys())
    totals = [t for t, _ in rows.values()]
    counts = [c for _, c in rows.values()]

    fig, (ax_dur, ax_cnt) = plt.subplots(
        1, 2, sharey=True,
        figsize=(12, max(4, 0.3 * len(labels))),
    )

    ax_dur.barh(labels, totals)
    ax_dur.set_xlabel('Total duration (ns)')
    ax_dur.set_ylabel('TTKernel Op')

    ax_cnt.barh(labels, counts)
    ax_cnt.set_xlabel('Number of calls')

    fig.suptitle('Kernel runtime breakdown')
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def print_op_times(rows: dict[str, tuple[int, int]], freq_mhz: float) -> None:
    print('\n===== OP TIMES =====\n')

    headers = ('Op', 'Calls', 'Total time', 'Cycles/call')
    table = [
        (k, str(count), time_formatter(total, freq_mhz), f'{total / count:.3f}')
        for k, (total, count) in rows.items()
    ]
    widths = [max(len(row[i]) for row in (*table, headers)) for i in range(4)]

    def fmt(row: tuple[str, str, str, str]) -> str:
        return (f'{row[0]:<{widths[0]}}  {row[1]:>{widths[1]}}  '
                f'{row[2]:>{widths[2]}}  {row[3]:>{widths[3]}}')

    print(fmt(headers))
    print('-' * (sum(widths) + 6))
    for row in table:
        print(fmt(row))


def print_timeline(rows: list[dict], freq_mhz: float) -> None:
    current_kernel: str = None

    max_width = 0
    for row in rows:
        max_width = max(max_width, len(row['name']))

    print('\n===== TIMELINE =====')
    for row in rows:
        if row['kernel'] != current_kernel:
            current_kernel = row['kernel']
            print(current_kernel)

        print(f'\t{row['name']:<{max_width + 3}}\t[core:{row["core"]}, RISC: {row["risc"]}]\t{time_formatter(row['duration'], freq_mhz)}')


def main() -> None:
    args = parse_args()
    perf_dir: pathlib.Path = args.path_to_perf_dir
    profile_log = perf_dir / 'profile_log_device.csv'
    ops_perf = perf_dir / 'ops_perf_results.csv'

    for file in (profile_log, ops_perf):
        if not file.is_file():
            raise FileNotFoundError(f'missing: {file}')

    freq = read_chip_freq_mhz(profile_log)
    raw_timeline, wall_time = collect_device_timeline(profile_log)
    zone_grouped_rows = aggregate_by_zone(raw_timeline, args.by_kernel)

    if args.op_graph:
        plot_histogram(zone_grouped_rows, args.op_graph)
    if args.op_times:
        print_op_times(zone_grouped_rows, freq)
    if args.timeline:
        print_timeline(raw_timeline, freq)
    if args.runtimes:
        print_runtimes(raw_timeline, wall_time, freq)

    print()


if __name__ == '__main__':
    main()
