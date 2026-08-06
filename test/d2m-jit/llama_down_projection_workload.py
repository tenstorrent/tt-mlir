# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama 3 8B prefill down projection with two-way tensor parallelism."""

import math
from dataclasses import dataclass

import torch

import d2m_jit as d2m


@dataclass(frozen=True)
class WorkloadConfig:
    m: int = 576
    k: int = 14336
    n: int = 4096
    grid_y: int = 6
    grid_x: int = 8
    seed: int = 0

    m_block_tiles: int = 1
    k_block_tiles: int = 56
    n_block_tiles: int = 4
    k_segments: int = 8

    def __post_init__(self):
        if self.m_block_tiles != 1 or self.n_block_tiles != 4:
            raise ValueError("the kernels require 32x128 output blocks")
        if self.k_segments != 8:
            raise ValueError("the kernels require eight K segments")
        if self.k % self.k_segments:
            raise ValueError("the contraction dimension must split into eight segments")
        dimensions = (
            (self.m, self.m_block_elements * self.grid_y, "m"),
            (self.k_segment_elements, self.k_block_elements, "K segment"),
            (self.n, self.n_block_elements * self.grid_x, "n"),
        )
        for size, divisor, name in dimensions:
            if size % divisor:
                raise ValueError(f"{name}={size} must be divisible by {divisor}")
        if self.k_blocks_per_segment > 8:
            raise ValueError("a K segment must fit across one worker-grid dimension")
        if self.n_blocks > 32:
            raise ValueError("the staged weight conversion supports at most 32 blocks")

    @property
    def m_block_elements(self):
        return self.m_block_tiles * 32

    @property
    def k_block_elements(self):
        return self.k_block_tiles * 32

    @property
    def n_block_elements(self):
        return self.n_block_tiles * 32

    @property
    def m_blocks(self):
        return self.m // self.m_block_elements

    @property
    def n_blocks(self):
        return self.n // self.n_block_elements

    @property
    def m_blocks_per_core(self):
        return self.m_blocks // self.grid_y

    @property
    def n_blocks_per_core(self):
        return self.n_blocks // self.grid_x

    @property
    def output_blocks_per_core(self):
        return self.m_blocks_per_core * self.n_blocks_per_core

    @property
    def k_segment_elements(self):
        return self.k // self.k_segments

    @property
    def k_blocks_per_segment(self):
        return self.k_segment_elements // self.k_block_elements


def make_operands(config):
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    activations = torch.randn(
        config.m,
        config.k,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.25)
    weight = torch.randn(
        config.k,
        config.n,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.015625)
    return activations, weight


def golden(activations, weight):
    return torch.matmul(activations.float(), weight.float()).to(torch.bfloat16)


def _layout(shape, block_shape, grid_shape=None):
    return d2m.Layout(
        shape=shape,
        dtype=d2m.bfloat16,
        block_shape=block_shape,
        grid_shape=grid_shape,
        mem_space="dram",
    )


@d2m.kernel
def single_chip_down_projection(
    activations,
    weight,
    output,
    m_blocks_per_core,
    n_blocks_per_core,
    k_blocks,
):
    cy = core_index(0)
    cx = core_index(1)
    for m_slot in range(m_blocks_per_core):
        m_block = cy * m_blocks_per_core + m_slot
        for n_slot in range(n_blocks_per_core):
            n_block = cx * n_blocks_per_core + n_slot
            acc = zeros([1, 4], dtype="bf16")
            for k_block in range(k_blocks):
                lhs = remote_load(activations, [m_block, k_block])
                rhs = remote_load(weight, [k_block, n_block])
                acc += lhs @ rhs
            remote_store(output, [m_block, n_block], acc)


@d2m.kernel
def place_activation_segment(
    source,
    destination,
    k_block_offset,
    m_blocks_per_core,
):
    cy = core_index(0)
    cx = core_index(1)
    for m_slot in range(m_blocks_per_core):
        m_block = cy * m_blocks_per_core + m_slot
        block = remote_load(source, [m_block, cx])
        remote_store(destination, [m_block, k_block_offset + cx], block)


@d2m.kernel
def place_weight_segment(
    source,
    destination,
    k_block_offset,
    n_blocks_per_core,
):
    cy = core_index(0)
    cx = core_index(1)
    for n_slot in range(n_blocks_per_core):
        n_block = cx * n_blocks_per_core + n_slot
        block = remote_load(source, [cy, n_block])
        remote_store(destination, [k_block_offset + cy, n_block], block)


@d2m.kernel
def serialized_two_chip_down_projection(
    activations,
    weight,
    partials,
    start_sem,
    ready,
    consumed,
    fabric_done,
    transfer_done,
    m_blocks,
    m_blocks_per_core,
    n_blocks_per_core,
    k_blocks,
    grid_y,
    grid_x,
    worker_count,
    output_blocks_per_core,
):
    cy = core_index(0)
    cx = core_index(1)
    if is_router_core():
        dy = mesh_position(0)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
    for item in range(output_blocks_per_core):
        m_slot = item // n_blocks_per_core
        n_slot = item % n_blocks_per_core
        m_block = cy * m_blocks_per_core + m_slot
        n_block = cx * n_blocks_per_core + n_slot
        profile_event("d2m.block.compute.begin", "compute")
        acc = zeros([1, 4], dtype="bf16")
        for k_block in range(k_blocks):
            lhs = remote_load(activations, [m_block, k_block])
            rhs = remote_load(weight, [k_block, n_block])
            acc += lhs @ rhs
        profile_event("d2m.block.compute.end", "compute")
        semaphore_inc(ready, 1, core=[0, 0], compute=True)

        if is_router_core():
            dy = mesh_position(0)
            profile_event("d2m.router.ready_wait.begin", "datamovement")
            semaphore_wait(ready, (item + 1) * worker_count)
            profile_event("d2m.router.ready_wait.end", "datamovement")
            dx = mesh_position(1)
            # Per-worker timestamp pairs overflow the profiler's per-RISC buffer.
            profile_event("d2m.router.transfer.begin", "datamovement")
            for ty in range(grid_y):
                target_m = ty * m_blocks_per_core + m_slot
                for tx in range(grid_x):
                    target_n = tx * n_blocks_per_core + n_slot
                    partial = empty_like(acc)
                    partial = core_read(partial, acc, core=[ty, tx])
                    semaphore_inc(consumed, 1, core=[ty, tx])
                    remote_store(
                        partials,
                        [dx * m_blocks + target_m, target_n],
                        partial,
                        start_device=[dy, 0],
                        device_mcast_shape=[1, 2],
                        semaphore=fabric_done,
                        semaphore_indices=[cy, cx],
                    )
            profile_event("d2m.router.transfer.end", "datamovement")
            profile_event("d2m.router.fabric_wait.begin", "datamovement")
            semaphore_wait(fabric_done, (item + 1) * 2 * worker_count)
            profile_event("d2m.router.fabric_wait.end", "datamovement")
            for ty in range(grid_y):
                for tx in range(grid_x):
                    semaphore_inc(transfer_done, 1, core=[ty, tx])
        semaphore_wait(consumed, item + 1, compute=True)
        semaphore_wait(transfer_done, item + 1)


@d2m.kernel
def overlapped_two_chip_down_projection(
    activations,
    weight,
    partials,
    start_sem,
    ready,
    consumed,
    fabric_done,
    m_blocks,
    m_blocks_per_core,
    n_blocks_per_core,
    k_blocks,
    grid_y,
    grid_x,
    worker_count,
    output_blocks_per_core,
):
    cy = core_index(0)
    cx = core_index(1)
    if is_router_core():
        dy = mesh_position(0)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
    for item in range(output_blocks_per_core):
        m_slot = item // n_blocks_per_core
        n_slot = item % n_blocks_per_core
        m_block = cy * m_blocks_per_core + m_slot
        n_block = cx * n_blocks_per_core + n_slot
        profile_event("d2m.block.compute.begin", "compute")
        acc = zeros([1, 4], dtype="bf16")
        for k_block in range(k_blocks):
            lhs = remote_load(activations, [m_block, k_block])
            rhs = remote_load(weight, [k_block, n_block])
            acc += lhs @ rhs
        profile_event("d2m.block.compute.end", "compute")
        semaphore_inc(ready, 1, core=[0, 0], compute=True)

        if is_router_core():
            dy = mesh_position(0)
            profile_event("d2m.router.ready_wait.begin", "datamovement")
            semaphore_wait(ready, (item + 1) * worker_count)
            profile_event("d2m.router.ready_wait.end", "datamovement")
            dx = mesh_position(1)
            # Per-worker timestamp pairs overflow the profiler's per-RISC buffer.
            profile_event("d2m.router.transfer.begin", "datamovement")
            for ty in range(grid_y):
                target_m = ty * m_blocks_per_core + m_slot
                for tx in range(grid_x):
                    target_n = tx * n_blocks_per_core + n_slot
                    partial = empty_like(acc)
                    partial = core_read(partial, acc, core=[ty, tx])
                    semaphore_inc(consumed, 1, core=[ty, tx])
                    remote_store(
                        partials,
                        [dx * m_blocks + target_m, target_n],
                        partial,
                        start_device=[dy, 0],
                        device_mcast_shape=[1, 2],
                        semaphore=fabric_done,
                        semaphore_indices=[cy, cx],
                    )
            profile_event("d2m.router.transfer.end", "datamovement")
            profile_event("d2m.router.fabric_wait.begin", "datamovement")
            semaphore_wait(fabric_done, (item + 1) * 2 * worker_count)
            profile_event("d2m.router.fabric_wait.end", "datamovement")
        semaphore_wait(consumed, item + 1, compute=True)


@d2m.kernel
def reduce_two_chip_partials(
    partials,
    output,
    m_blocks,
    m_blocks_per_core,
    n_blocks_per_core,
):
    cy = core_index(0)
    cx = core_index(1)
    for m_slot in range(m_blocks_per_core):
        m_block = cy * m_blocks_per_core + m_slot
        for n_slot in range(n_blocks_per_core):
            n_block = cx * n_blocks_per_core + n_slot
            profile_event("d2m.block.reduction.begin", "compute")
            first = remote_load(partials, [m_block, n_block])
            second = remote_load(partials, [m_blocks + m_block, n_block])
            remote_store(output, [m_block, n_block], first + second)
            profile_event("d2m.block.reduction.end", "compute")


def _stage_k_segments(config, activation_sources, weight_sources, k_elements):
    k_blocks = k_elements // config.k_block_elements
    activation_layout = _layout(
        (config.m, k_elements),
        [config.m_block_tiles, config.k_block_tiles],
        (config.grid_y, math.gcd(k_blocks, config.grid_x)),
    )
    weight_layout = _layout(
        (k_elements, config.n),
        [config.k_block_tiles, config.n_block_tiles],
        (math.gcd(k_blocks, 8), config.grid_x),
    )
    staged_activations = d2m.empty(activation_layout)
    staged_weight = d2m.empty(weight_layout)
    activation_grid = (config.grid_y, config.k_blocks_per_segment)
    weight_grid = (config.k_blocks_per_segment, config.grid_x)
    for segment, (activation, weight) in enumerate(
        zip(activation_sources, weight_sources)
    ):
        k_block_offset = segment * config.k_blocks_per_segment
        place_activation_segment(
            activation,
            staged_activations,
            k_block_offset,
            config.m_blocks_per_core,
            grid=activation_grid,
            kernel_io_in_dram=True,
        )
        place_weight_segment(
            weight,
            staged_weight,
            k_block_offset,
            config.n_blocks_per_core,
            grid=weight_grid,
            kernel_io_in_dram=True,
        )
    return staged_activations, staged_weight, k_blocks


def build_single_chip(config, activations, weight):
    d2m.mesh((1, 1), topology=("linear", "linear"))
    grid = (config.grid_y, config.grid_x)
    segment_activation_layout = _layout(
        (config.m, config.k_segment_elements),
        [config.m_block_tiles, config.k_block_tiles],
    )
    segment_weight_layout = _layout(
        (config.k_segment_elements, config.n),
        [config.k_block_tiles, config.n_block_tiles],
        (2, config.n_blocks),
    )
    output_layout = _layout(
        (config.m, config.n),
        [config.m_block_tiles, config.n_block_tiles],
        grid,
    )
    segment_size = config.k_segment_elements
    activation_sources = []
    weight_sources = []
    reusable_weights = []
    for segment in range(config.k_segments):
        start = segment * segment_size
        stop = start + segment_size
        activation_sources.append(
            d2m.to_layout(
                activations[:, start:stop].contiguous(),
                segment_activation_layout,
            )
        )
        weight_segment = weight[start:stop, :]
        reusable_weights.append(weight_segment)
        weight_sources.append(d2m.to_layout(weight_segment, segment_weight_layout))
    staged_activations, staged_weight, k_blocks = _stage_k_segments(
        config,
        activation_sources,
        weight_sources,
        config.k,
    )
    output = d2m.empty(output_layout)
    single_chip_down_projection(
        staged_activations,
        staged_weight,
        output,
        config.m_blocks_per_core,
        config.n_blocks_per_core,
        k_blocks,
        grid=grid,
        kernel_io_in_dram=True,
    )
    return output, tuple(reusable_weights)


def run_single_chip(config, activations, weight):
    output, _ = build_single_chip(config, activations, weight)
    return output.to_host()


def prepare_single_chip(config, activations, weight):
    output, reusable_weights = build_single_chip(config, activations, weight)
    return d2m.prepare(output, reusable_inputs=reusable_weights)


def build_two_chip(config, activations, weight, overlap):
    d2m.mesh((1, 2), topology=("linear", "linear"))
    grid = (config.grid_y, config.grid_x)
    segment_activation_layout = _layout(
        (config.m, config.k_segment_elements),
        [config.m_block_tiles, config.k_block_tiles],
    )
    segment_weight_layout = _layout(
        (config.k_segment_elements, config.n),
        [config.k_block_tiles, config.n_block_tiles],
        (2, config.n_blocks),
    )
    partial_layout = _layout(
        (2 * config.m, config.n),
        [config.m_block_tiles, config.n_block_tiles],
        grid,
    )
    output_layout = _layout(
        (config.m, config.n),
        [config.m_block_tiles, config.n_block_tiles],
        grid,
    )
    segment_size = config.k_segment_elements
    activation_sources = []
    weight_sources = []
    reusable_weights = []
    for local_segment in range(config.k_segments // 2):
        low_start = local_segment * segment_size
        high_start = low_start + config.k // 2
        packed_activations = torch.cat(
            (
                activations[:, low_start : low_start + segment_size],
                activations[:, high_start : high_start + segment_size],
            ),
            dim=1,
        )
        packed_weight = torch.cat(
            (
                weight[low_start : low_start + segment_size, :],
                weight[high_start : high_start + segment_size, :],
            ),
            dim=0,
        )
        reusable_weights.append(packed_weight)
        activation_sources.append(
            d2m.mesh_shard(
                packed_activations,
                segment_activation_layout,
                shard_dims=[-1, 1],
                shard_shape=[1, 2],
            )
        )
        weight_sources.append(
            d2m.mesh_shard(
                packed_weight,
                segment_weight_layout,
                shard_dims=[-1, 0],
                shard_shape=[2, 1],
            )
        )
    local_activations, local_weight, local_k_blocks = _stage_k_segments(
        config,
        activation_sources,
        weight_sources,
        config.k // 2,
    )
    partials = d2m.empty(partial_layout)
    output = d2m.empty(output_layout)
    start_sem = d2m.global_semaphore(grid_shape=(8, 8))
    ready = d2m.global_semaphore(grid_shape=(8, 8), init=0)
    consumed = d2m.global_semaphore(grid_shape=(8, 8), init=0)
    fabric_done = d2m.global_semaphore(grid_shape=(8, 8), init=0)
    common_args = (
        local_activations,
        local_weight,
        partials,
        start_sem,
        ready,
        consumed,
        fabric_done,
    )
    shape_args = (
        config.m_blocks,
        config.m_blocks_per_core,
        config.n_blocks_per_core,
        local_k_blocks,
        config.grid_y,
        config.grid_x,
        config.grid_y * config.grid_x,
        config.output_blocks_per_core,
    )
    kernel_options = {
        "grid": grid,
        "fabric": d2m.fabric_config(
            cluster_axis=1,
            topology="linear",
            router_cores=[(0, 0)],
        ),
        "kernel_io_in_dram": True,
    }
    if overlap:
        overlapped_two_chip_down_projection(
            *common_args,
            *shape_args,
            **kernel_options,
        )
    else:
        serialized_two_chip_down_projection(
            *common_args,
            d2m.global_semaphore(grid_shape=(8, 8), init=0),
            *shape_args,
            **kernel_options,
        )
    reduce_two_chip_partials(
        partials,
        output,
        config.m_blocks,
        config.m_blocks_per_core,
        config.n_blocks_per_core,
        grid=grid,
        kernel_io_in_dram=True,
    )
    replicated = d2m.mesh_gather(
        output,
        shard_dims=[-1, -1],
        shard_shape=[1, 1],
    )
    return replicated, tuple(reusable_weights)


def run_two_chip(config, activations, weight, overlap):
    output, _ = build_two_chip(config, activations, weight, overlap)
    return output.to_host()


def prepare_two_chip(config, activations, weight, overlap):
    output, reusable_weights = build_two_chip(config, activations, weight, overlap)
    return d2m.prepare(output, reusable_inputs=reusable_weights)
