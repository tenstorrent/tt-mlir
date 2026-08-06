# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Measure sustained bidirectional D2M DRAM-to-DRAM fabric bandwidth."""

import argparse
import collections
import csv
import os
import pathlib
import re
import shutil
import time

import torch

import d2m_jit as d2m


@d2m.kernel
def exchange_payload(source, destination, start, done, packet_count):
    dy = mesh_position(0)
    dx = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(
        start,
        start_device=[dy, 0],
        mcast_shape=[1, 2],
        num_receivers=1,
        core_indices=[cy, cx],
    )
    profile_event("d2m.fabric.saturation.begin", "datamovement")
    for _ in range(packet_count):
        payload = remote_load(source, [0, cx])
        remote_store(
            destination,
            [0, cx],
            payload,
            start_device=[dy, 1 - dx],
            device_mcast_shape=[1, 1],
            semaphore=done,
            semaphore_indices=[cy, cx],
        )
    semaphore_wait(done, packet_count)
    profile_event("d2m.fabric.saturation.end", "datamovement")


def _payload_geometry(payload_kib):
    payload_bytes = payload_kib * 1024
    tile_bytes = 32 * 32 * 2
    if payload_bytes % tile_bytes:
        raise ValueError("payload size must contain a whole number of BF16 tiles")
    tile_count = payload_bytes // tile_bytes
    tiles_x = min(tile_count, 32)
    if tile_count % tiles_x:
        raise ValueError("payload tile count must factor into at most 32 columns")
    return tile_count // tiles_x, tiles_x


def _build(payload_kib, packet_count, num_links, num_senders, seed):
    tiles_y, tiles_x = _payload_geometry(payload_kib)
    shape = (tiles_y * 32, num_senders * tiles_x * 32)
    layout = d2m.Layout(
        shape=shape,
        dtype=d2m.bfloat16,
        block_shape=[tiles_y, tiles_x],
        grid_shape=[1, num_senders],
        mem_space="dram",
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    source_host = torch.randn(shape, dtype=torch.bfloat16, generator=generator)

    d2m.mesh((1, 2), topology=("linear", "linear"))
    source = d2m.mesh_shard(
        source_host,
        layout,
        shard_dims=[-1, -1],
        shard_shape=[1, 1],
    )
    destination = d2m.empty(layout)
    exchange_payload(
        source,
        destination,
        d2m.global_semaphore(grid_shape=(8, 8)),
        d2m.global_semaphore(grid_shape=(8, 8), init=0),
        packet_count,
        grid=(1, num_senders),
        fabric=d2m.fabric_config(
            cluster_axis=1,
            topology="linear",
            num_links=num_links,
            router_cores=[(0, x) for x in range(num_senders)],
        ),
        kernel_io_in_dram=True,
    )
    replicated = d2m.mesh_gather(
        destination,
        shard_dims=[-1, -1],
        shard_shape=[1, 1],
    )
    return d2m.prepare(replicated), source_host


def _device_durations_ns(profile_csv, warmup, iterations):
    events = collections.defaultdict(list)
    with profile_csv.open(encoding="utf-8", newline="") as input_file:
        header = input_file.readline()
        match = re.search(r"CHIP_FREQ\[MHz\]:\s*([0-9.]+)", header)
        if match is None:
            raise ValueError(f"chip frequency is missing from {profile_csv}")
        frequency_mhz = float(match.group(1))
        for raw_row in csv.DictReader(input_file):
            row = {key.strip().lower(): value.strip() for key, value in raw_row.items()}
            label = row["zone name"]
            if label not in (
                "d2m.fabric.saturation.begin",
                "d2m.fabric.saturation.end",
            ):
                continue
            device = row.get("device id") or row.get("pcie slot") or "unknown"
            stream = (
                device,
                row["core_x"],
                row["core_y"],
                row["risc processor type"],
            )
            events[stream].append(
                (
                    int(row["time[cycles since reset]"]),
                    label.rpartition(".")[2],
                )
            )

    captured_runs = warmup + iterations
    durations = collections.defaultdict(list)
    for stream, stream_events in events.items():
        stream_events.sort()
        intervals = []
        start = None
        for cycle, boundary in stream_events:
            if boundary == "begin":
                if start is not None:
                    raise ValueError(f"nested profiler events on stream {stream}")
                start = cycle
            else:
                if start is None:
                    raise ValueError(f"unmatched profiler end event on stream {stream}")
                intervals.append((cycle - start) * 1000 / frequency_mhz)
                start = None
        if start is not None or len(intervals) != captured_runs:
            raise ValueError(
                f"expected {captured_runs} intervals on stream {stream}, "
                f"found {len(intervals)}"
            )
        durations[stream] = intervals[warmup:]
    if not durations:
        raise ValueError("fabric saturation events are missing from the device profile")
    return durations


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-kib", type=int, default=128)
    parser.add_argument("--packet-count", type=int, default=512)
    parser.add_argument("--num-links", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-senders", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/d2m-fabric-saturation"),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.payload_kib <= 0 or args.packet_count <= 0:
        raise ValueError("payload size and packet count must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    num_senders = args.num_senders
    if num_senders > args.num_links:
        raise ValueError("number of senders cannot exceed number of links")

    args.profile_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.profile_dir / ".logs", ignore_errors=True)
    os.environ["TT_METAL_PROFILER_DIR"] = str(args.profile_dir)
    os.environ["TT_METAL_CACHE"] = str(args.profile_dir / "kernel-cache")
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
    os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

    from d2m_jit._src.builder import _Builder
    from d2m_jit._src.config import config as d2m_config

    d2m_config.enable_perf_trace = True
    _Builder.reset()
    executable, expected = _build(
        args.payload_kib,
        args.packet_count,
        args.num_links,
        num_senders,
        args.seed,
    )
    result = None
    host_samples_ms = []
    try:
        for _ in range(args.warmup):
            result = executable.run()[0]
        for _ in range(args.iterations):
            begin = time.perf_counter_ns()
            result = executable.run()[0]
            host_samples_ms.append((time.perf_counter_ns() - begin) / 1_000_000)
    finally:
        executable.close()
        _Builder.reset()

    if result is None or not torch.equal(result, expected):
        max_diff = None if result is None else (result - expected).abs().max().item()
        raise ValueError(f"fabric exchange failed correctness, max_diff={max_diff}")

    profile_csv = args.profile_dir / ".logs" / "profile_log_device.csv"
    stream_durations = _device_durations_ns(profile_csv, args.warmup, args.iterations)
    streams_by_device = collections.defaultdict(list)
    for stream, samples in stream_durations.items():
        streams_by_device[stream[0]].append(samples)
    for device, streams in streams_by_device.items():
        if len(streams) != num_senders:
            raise ValueError(
                f"expected {num_senders} profiler streams on device {device}, "
                f"found {len(streams)}"
            )
    durations = {
        device: [
            max(stream[iteration] for stream in streams)
            for iteration in range(args.iterations)
        ]
        for device, streams in streams_by_device.items()
    }
    bytes_per_direction = args.payload_kib * 1024 * args.packet_count * num_senders
    all_samples_ns = [sample for samples in durations.values() for sample in samples]
    mean_ns = sum(all_samples_ns) / len(all_samples_ns)
    per_direction_gbps = bytes_per_direction / mean_ns
    print(
        f"payload_kib={args.payload_kib} packets={args.packet_count} "
        f"num_links={args.num_links} num_senders={num_senders} "
        f"bytes_per_direction={bytes_per_direction}"
    )
    for device, samples in sorted(durations.items()):
        bandwidth = [bytes_per_direction / sample for sample in samples]
        print(
            f"device={device} duration_us={[round(sample / 1000, 3) for sample in samples]} "
            f"directional_GBps={[round(sample, 3) for sample in bandwidth]}"
        )
    print(
        f"mean_directional_GBps={per_direction_gbps:.3f} "
        f"mean_per_sender_GBps={per_direction_gbps / num_senders:.3f} "
        f"mean_full_duplex_GBps={2 * per_direction_gbps:.3f} "
        f"host_ms={[round(sample, 3) for sample in host_samples_ms]} "
        f"profile={profile_csv}"
    )


if __name__ == "__main__":
    main()
