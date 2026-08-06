# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Profile a BF16 Llama 3 8B down projection with two-way tensor parallelism."""

import argparse
import collections
import contextlib
import csv
import os
import pathlib
import re
import shutil
import signal
import socket
import subprocess
import time


def _tracy_tool(name):
    tt_metal_home = pathlib.Path(os.environ["TT_METAL_HOME"])
    candidates = (
        tt_metal_home / name,
        tt_metal_home / "build" / "tools" / "profiler" / "bin" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} was not found under {tt_metal_home}")


def _available_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def _capture_tracy(trace_path, port):
    capture = subprocess.Popen(
        [
            str(_tracy_tool("tracy-capture")),
            "-o",
            str(trace_path),
            "-f",
            "-p",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(0.1)
    if capture.poll() is not None:
        output, _ = capture.communicate()
        raise RuntimeError(f"tracy-capture exited before the workload:\n{output}")
    try:
        yield
    finally:
        capture.send_signal(signal.SIGINT)
        try:
            output, _ = capture.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            capture.kill()
            output, _ = capture.communicate()
        if capture.returncode not in (0, -signal.SIGINT):
            raise RuntimeError(f"tracy-capture failed:\n{output}")


def _export_tracy(trace_path):
    csv_path = trace_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        subprocess.run(
            [str(_tracy_tool("tracy-csvexport")), "-u", str(trace_path)],
            check=True,
            stdout=output,
        )
    return csv_path


def _summarize_tracy(csv_path, measured_iterations):
    selected = {
        "MeshShardCommand": [],
        "EnqueueProgramCommand": [],
        "FinishCommand": [],
        "EnqueueReadBufferCommand": [],
    }
    submits = []
    with csv_path.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            name = row["name"]
            start_ns = int(row["ns_since_start"])
            duration_ns = int(row["exec_time_ns"])
            if name == "submit":
                submits.append((start_ns, duration_ns))
            if name in selected:
                selected[name].append((start_ns, duration_ns))

    submits = sorted(submits)[-measured_iterations:]
    submit_durations = [duration for _, duration in submits]
    if submit_durations:
        print(
            f"tracy submit: count={len(submit_durations)} "
            f"mean_us={sum(submit_durations) / len(submit_durations) / 1_000:.1f} "
            f"min_us={min(submit_durations) / 1_000:.1f} "
            f"max_us={max(submit_durations) / 1_000:.1f}"
        )
    for name, zones in selected.items():
        counts = []
        totals = []
        for submit_start, submit_duration in submits:
            durations = [
                duration
                for start, duration in zones
                if submit_start <= start < submit_start + submit_duration
            ]
            counts.append(len(durations))
            totals.append(sum(durations))
        if totals and any(counts):
            count = (
                str(counts[0])
                if len(set(counts)) == 1
                else f"{min(counts)}-{max(counts)}"
            )
            print(
                f"tracy {name}: count_per_submit={count} "
                f"mean_total_us={sum(totals) / len(totals) / 1_000:.1f} "
                f"min_total_us={min(totals) / 1_000:.1f} "
                f"max_total_us={max(totals) / 1_000:.1f}"
            )


def _device_kernel_ns(profile_csv, warmup, iterations):
    events_by_host_id = collections.defaultdict(list)
    with profile_csv.open(encoding="utf-8", newline="") as input_file:
        header = input_file.readline()
        match = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header)
        if match is None:
            raise ValueError(f"chip frequency is missing from {profile_csv}")
        frequency_mhz = float(match.group(1))
        for raw_row in csv.DictReader(input_file):
            row = {key.strip().lower(): value for key, value in raw_row.items()}
            if row["zone name"].endswith("-KERNEL"):
                events_by_host_id[row["run host id"]].append(
                    (int(row["time[cycles since reset]"]), row["type"])
                )

    captured_runs = warmup + iterations
    measured_durations = []
    for events in events_by_host_id.values():
        events.sort()
        if len(events) % captured_runs:
            raise ValueError("device profiler events do not divide into iterations")
        events_per_run = len(events) // captured_runs
        for run_index in range(warmup, captured_runs):
            run_events = events[
                run_index * events_per_run : (run_index + 1) * events_per_run
            ]
            starts = [cycle for cycle, kind in run_events if kind == "ZONE_START"]
            ends = [cycle for cycle, kind in run_events if kind == "ZONE_END"]
            if starts and ends:
                measured_durations.append(max(ends) - min(starts))
    if not measured_durations:
        return None
    return max(measured_durations) * 1_000 / frequency_mhz


def _pcc(actual, expected):
    import torch

    values = torch.stack((actual.float().flatten(), expected.float().flatten()))
    return torch.corrcoef(values)[0, 1].item()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("single", "two-chip-serialized", "two-chip-overlap"),
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--m", type=int, default=576)
    parser.add_argument("--k", type=int, default=14336)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--grid-y", type=int, default=6)
    parser.add_argument("--grid-x", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tracy-file", type=pathlib.Path)
    parser.add_argument("--device-profile-dir", type=pathlib.Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup must be non-negative")

    trace_context = contextlib.nullcontext()
    if args.tracy_file:
        args.tracy_file.parent.mkdir(parents=True, exist_ok=True)
        port = _available_port()
        os.environ["TRACY_PORT"] = str(port)
        trace_context = _capture_tracy(args.tracy_file, port)

    if args.device_profile_dir:
        args.device_profile_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(args.device_profile_dir / ".logs", ignore_errors=True)
        kernel_cache = args.device_profile_dir / "kernel-cache"
        shutil.rmtree(kernel_cache, ignore_errors=True)
        os.environ["TT_METAL_PROFILER_DIR"] = str(args.device_profile_dir)
        os.environ["TT_METAL_CACHE"] = str(kernel_cache)
        os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
        os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

    with trace_context:
        import llama_down_projection_workload as workload
        from d2m_jit._src.builder import _Builder

        if args.device_profile_dir:
            from d2m_jit._src.config import config as d2m_config

            d2m_config.enable_perf_trace = True

        config = workload.WorkloadConfig(
            m=args.m,
            k=args.k,
            n=args.n,
            grid_y=args.grid_y,
            grid_x=args.grid_x,
            seed=args.seed,
        )
        activations, weight = workload.make_operands(config)
        expected = workload.golden(activations, weight)
        global_flops = 2 * config.m * config.k * config.n
        print(
            f"workload: llama3-8b-down-projection mode={args.mode} dtype=bf16 "
            f"shape={config.m}x{config.k}x{config.n} "
            f"grid={config.grid_y}x{config.grid_x} "
            f"global_flops={global_flops}",
            flush=True,
        )
        if args.mode == "single":
            run = lambda: workload.run_single_chip(config, activations, weight)
        else:
            overlap = args.mode == "two-chip-overlap"
            run = lambda: workload.run_two_chip(
                config,
                activations,
                weight,
                overlap=overlap,
            )
        result = None
        warmup_context = contextlib.nullcontext()
        if args.device_profile_dir:
            from autotuner.autotuner import _silence_native_output

            warmup_context = _silence_native_output(
                str(args.device_profile_dir / "native_compile.log")
            )
        with warmup_context:
            for _ in range(args.warmup):
                _Builder.reset()
                result = run()
        for iteration in range(args.iterations):
            _Builder.reset()
            result = run()
            print(
                f"iteration={iteration + 1} mode={args.mode} "
                f"pcc={_pcc(result, expected):.6f}",
                flush=True,
            )
        _Builder.reset()

    if args.tracy_file:
        tracy_csv = _export_tracy(args.tracy_file)
        _summarize_tracy(tracy_csv, args.iterations)
        print(f"tracy trace: {args.tracy_file}")
        print(f"tracy zones: {tracy_csv}")
    if args.device_profile_dir:
        profile_csv = args.device_profile_dir / ".logs" / "profile_log_device.csv"
        kernel_ns = _device_kernel_ns(profile_csv, args.warmup, args.iterations)
        print(f"device longest_kernel_ns={kernel_ns}")
        print(f"device profile: {profile_csv}")


if __name__ == "__main__":
    main()
