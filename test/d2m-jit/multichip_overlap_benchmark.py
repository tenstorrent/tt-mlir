# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Profile BF16 Llama 3 8B tensor parallelism at realistic host boundaries."""

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
        "EnqueueWriteBufferCommand": [],
        "EnqueueProgramCommand": [],
        "FinishCommand": [],
        "EnqueueReadBufferCommand": [],
    }
    submits = []
    invokes = []
    programs = []
    lifecycle_zones = {
        "D2MPrepareDevice": [],
        "D2MConfigureFabric": [],
        "D2MReleaseDevice": [],
        "D2MWait": [],
        "D2MReadback": [],
        "D2MHostCopy": [],
    }
    with csv_path.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            name = row["name"]
            start_ns = int(row["ns_since_start"])
            duration_ns = int(row["exec_time_ns"])
            if name == "submit":
                submits.append((start_ns, duration_ns))
            if name == "D2MInvoke":
                invokes.append((start_ns, duration_ns))
            if name == "EnqueueProgramCommand":
                programs.append((start_ns, duration_ns, row["zone_text"].strip()))
            if name in selected:
                selected[name].append((start_ns, duration_ns))
            if name in lifecycle_zones:
                lifecycle_zones[name].append(duration_ns)

    anchors = invokes if invokes else submits
    anchor_name = "D2MInvoke" if invokes else "submit"
    anchor_label = "invoke" if invokes else "submit"
    anchors = sorted(anchors)[-measured_iterations:]
    submit_durations = [duration for _, duration in anchors]
    if submit_durations:
        print(
            f"tracy {anchor_name}: count={len(submit_durations)} "
            f"mean_us={sum(submit_durations) / len(submit_durations) / 1_000:.1f} "
            f"min_us={min(submit_durations) / 1_000:.1f} "
            f"max_us={max(submit_durations) / 1_000:.1f}"
        )
    for name, zones in selected.items():
        counts = []
        totals = []
        for submit_start, submit_duration in anchors:
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
                f"tracy {name}: count_per_{anchor_label}={count} "
                f"mean_total_us={sum(totals) / len(totals) / 1_000:.1f} "
                f"min_total_us={min(totals) / 1_000:.1f} "
                f"max_total_us={max(totals) / 1_000:.1f}"
            )
    for label in sorted({label for _, _, label in programs}):
        counts = []
        totals = []
        for invoke_start, invoke_duration in anchors:
            durations = [
                duration
                for start, duration, program_label in programs
                if program_label == label
                and invoke_start <= start < invoke_start + invoke_duration
            ]
            counts.append(len(durations))
            totals.append(sum(durations))
        if any(counts):
            count = (
                str(counts[0])
                if len(set(counts)) == 1
                else f"{min(counts)}-{max(counts)}"
            )
            print(
                f"tracy program {label or '<unattributed>'}: "
                f"count_per_{anchor_label}={count} "
                f"mean_total_us={sum(totals) / len(totals) / 1_000:.1f}"
            )
    for name, durations in lifecycle_zones.items():
        if durations:
            samples = durations[-measured_iterations:]
            print(
                f"tracy {name}: count={len(samples)} "
                f"mean_us={sum(samples) / len(samples) / 1_000:.1f} "
                f"min_us={min(samples) / 1_000:.1f} "
                f"max_us={max(samples) / 1_000:.1f}"
            )


def _device_program_intervals(profile_csv, warmup, iterations):
    events_by_program = collections.defaultdict(list)
    with profile_csv.open(encoding="utf-8", newline="") as input_file:
        header = input_file.readline()
        match = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header)
        if match is None:
            raise ValueError(f"chip frequency is missing from {profile_csv}")
        frequency_mhz = float(match.group(1))
        for raw_row in csv.DictReader(input_file):
            row = {key.strip().lower(): value for key, value in raw_row.items()}
            if row["zone name"].endswith("-KERNEL"):
                device = row.get("device id") or row.get("pcie slot") or "unknown"
                events_by_program[(device, row["run host id"])].append(
                    (int(row["time[cycles since reset]"]), row["type"])
                )

    captured_runs = warmup + iterations
    intervals_by_device_run = collections.defaultdict(list)
    for (device, _), events in events_by_program.items():
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
                intervals_by_device_run[(device, run_index)].append(
                    (min(starts), max(ends))
                )

    return frequency_mhz, intervals_by_device_run


def _device_kernel_ns(profile_csv, warmup, iterations):
    frequency_mhz, intervals_by_device_run = _device_program_intervals(
        profile_csv, warmup, iterations
    )
    measured_durations = [
        end - start
        for intervals in intervals_by_device_run.values()
        for start, end in intervals
    ]
    if not measured_durations:
        return None
    return max(measured_durations) * 1_000 / frequency_mhz


def _device_timing_samples(profile_csv, warmup, iterations):
    frequency_mhz, intervals_by_device_run = _device_program_intervals(
        profile_csv, warmup, iterations
    )
    critical_path_ns = []
    active_program_ns = []
    for run_index in range(warmup, warmup + iterations):
        device_timings = []
        for (device, interval_run), intervals in intervals_by_device_run.items():
            if interval_run != run_index:
                continue
            device_timings.append(
                (
                    max(end for _, end in intervals)
                    - min(start for start, _ in intervals),
                    sum(end - start for start, end in intervals),
                )
            )
        if device_timings:
            critical_cycles, active_cycles = max(
                device_timings, key=lambda timing: timing[0]
            )
            critical_path_ns.append(critical_cycles * 1_000 / frequency_mhz)
            active_program_ns.append(active_cycles * 1_000 / frequency_mhz)
    return {
        "critical_path_ns": critical_path_ns,
        "active_program_ns": active_program_ns,
    }


def _device_critical_path_ns(profile_csv, warmup, iterations):
    return _device_timing_samples(profile_csv, warmup, iterations)["critical_path_ns"]


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
    parser.add_argument(
        "--workload",
        choices=("down-projection", "full-mlp"),
        default="full-mlp",
    )
    parser.add_argument(
        "--lifecycle",
        choices=("rebuilt", "prepared"),
        default="prepared",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--m", type=int, default=576)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
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
        os.environ["TT_METAL_PROFILER_DIR"] = str(args.device_profile_dir)
        os.environ["TT_METAL_CACHE"] = str(kernel_cache)
        os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
        os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

    with trace_context:
        from d2m_jit._src.builder import _Builder

        if args.workload == "full-mlp":
            import llama_mlp_workload as workload
        else:
            import llama_down_projection_workload as workload

        if args.device_profile_dir:
            from d2m_jit._src.config import config as d2m_config

            d2m_config.enable_perf_trace = True

        if args.workload == "full-mlp":
            config = workload.WorkloadConfig(
                m=args.m,
                hidden=args.hidden,
                intermediate=args.intermediate,
                grid_y=args.grid_y,
                grid_x=args.grid_x,
                seed=args.seed,
            )
            operands = workload.make_operands(config)
            expected = workload.golden(*operands)
            global_flops = 6 * config.m * config.hidden * config.intermediate
            shape = f"m={config.m} hidden={config.hidden} intermediate={config.intermediate}"
        else:
            config = workload.WorkloadConfig(
                m=args.m,
                k=args.intermediate,
                n=args.hidden,
                grid_y=args.grid_y,
                grid_x=args.grid_x,
                seed=args.seed,
            )
            operands = workload.make_operands(config)
            expected = workload.golden(*operands)
            global_flops = 2 * config.m * config.k * config.n
            shape = f"m={config.m} k={config.k} n={config.n}"
        print(
            f"workload: llama3-8b-{args.workload} mode={args.mode} "
            f"lifecycle={args.lifecycle} dtype=bf16 {shape} "
            f"grid={config.grid_y}x{config.grid_x} "
            f"global_flops={global_flops}",
            flush=True,
        )
        overlap = args.mode == "two-chip-overlap"
        if args.workload == "full-mlp":
            if args.mode == "single":
                run = lambda: workload.run_single_chip(config, operands)
                prepare = lambda: workload.prepare_single_chip(config, operands)
            else:
                run = lambda: workload.run_two_chip(config, operands, overlap)
                prepare = lambda: workload.prepare_two_chip(config, operands, overlap)
        elif args.mode == "single":
            run = lambda: workload.run_single_chip(config, *operands)
            prepare = lambda: workload.prepare_single_chip(config, *operands)
        else:
            run = lambda: workload.run_two_chip(config, *operands, overlap)
            prepare = lambda: workload.prepare_two_chip(config, *operands, overlap)

        def normalize(result):
            if tuple(result.shape) != tuple(expected.shape):
                raise ValueError(
                    f"result shape {tuple(result.shape)} does not match "
                    f"golden shape {tuple(expected.shape)}"
                )
            return result

        result = None
        warmup_context = contextlib.nullcontext()
        if args.device_profile_dir:
            from autotuner.autotuner import _silence_native_output

            warmup_context = _silence_native_output(
                str(args.device_profile_dir / "native_compile.log")
            )
        if args.lifecycle == "rebuilt":
            with warmup_context:
                for _ in range(args.warmup):
                    _Builder.reset()
                    result = normalize(run())
            for iteration in range(args.iterations):
                _Builder.reset()
                begin = time.perf_counter_ns()
                result = normalize(run())
                total_ms = (time.perf_counter_ns() - begin) / 1_000_000
                print(
                    f"iteration={iteration + 1} mode={args.mode} "
                    f"pcc={_pcc(result, expected):.6f} total_ms={total_ms:.3f}",
                    flush=True,
                )
            _Builder.reset()
        else:
            _Builder.reset()
            begin = time.perf_counter_ns()
            executable = prepare()
            prepare_ms = (time.perf_counter_ns() - begin) / 1_000_000
            print(f"prepare_ms={prepare_ms:.3f}", flush=True)
            try:
                with warmup_context:
                    for _ in range(args.warmup):
                        result = normalize(executable.run()[0])
                for iteration in range(args.iterations):
                    total_begin = time.perf_counter_ns()
                    begin = time.perf_counter_ns()
                    submitted = executable.submit()
                    submit_ms = (time.perf_counter_ns() - begin) / 1_000_000
                    begin = time.perf_counter_ns()
                    executable.wait(submitted)
                    wait_ms = (time.perf_counter_ns() - begin) / 1_000_000
                    begin = time.perf_counter_ns()
                    result = normalize(executable.readback(submitted)[0])
                    readback_ms = (time.perf_counter_ns() - begin) / 1_000_000
                    total_ms = (time.perf_counter_ns() - total_begin) / 1_000_000
                    print(
                        f"iteration={iteration + 1} mode={args.mode} "
                        f"pcc={_pcc(result, expected):.6f} "
                        f"submit_ms={submit_ms:.3f} wait_ms={wait_ms:.3f} "
                        f"readback_ms={readback_ms:.3f} total_ms={total_ms:.3f}",
                        flush=True,
                    )
                print(
                    f"program_cache_entries={executable.program_cache_entries} "
                    f"input_reuse_stats={executable.input_reuse_stats}",
                    flush=True,
                )
            finally:
                executable.close()
            _Builder.reset()

    if args.tracy_file:
        tracy_csv = _export_tracy(args.tracy_file)
        _summarize_tracy(tracy_csv, args.iterations)
        print(f"tracy trace: {args.tracy_file}")
        print(f"tracy zones: {tracy_csv}")
    if args.device_profile_dir:
        profile_csv = args.device_profile_dir / ".logs" / "profile_log_device.csv"
        kernel_ns = _device_kernel_ns(profile_csv, args.warmup, args.iterations)
        timing_samples = _device_timing_samples(
            profile_csv, args.warmup, args.iterations
        )
        print(f"device longest_kernel_ns={kernel_ns}")
        critical_samples = timing_samples["critical_path_ns"]
        active_samples = timing_samples["active_program_ns"]
        if critical_samples:
            print(
                "device critical_path_ns="
                f"{critical_samples} mean_ns={sum(critical_samples) / len(critical_samples)}"
            )
            print(
                "device active_program_ns="
                f"{active_samples} mean_ns={sum(active_samples) / len(active_samples)}"
            )
        print(f"device profile: {profile_csv}")


if __name__ == "__main__":
    main()
