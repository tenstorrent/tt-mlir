# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BF16 Llama 3 8B prefill MLP with two-way tensor parallelism."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import d2m_jit as d2m

from llama_down_projection_workload import (
    overlapped_two_chip_down_projection,
    reduce_two_chip_partials,
    serialized_two_chip_down_projection,
    single_chip_down_projection,
)


@dataclass(frozen=True)
class WorkloadConfig:
    m: int = 576
    hidden: int = 4096
    intermediate: int = 14336
    grid_y: int = 6
    grid_x: int = 8
    seed: int = 0

    m_block_tiles: int = 1
    projection_k_block_tiles: int = 16
    projection_n_block_tiles: int = 4
    down_k_block_tiles: int = 14

    def __post_init__(self):
        if self.m_block_tiles != 1 or self.projection_n_block_tiles != 4:
            raise ValueError("the kernels require 32x128 output blocks")
        if self.intermediate % 2:
            raise ValueError("intermediate must be divisible across two chips")
        dimensions = (
            (self.m, self.m_block_elements * self.grid_y, "m"),
            (self.hidden, self.projection_k_block_elements, "hidden"),
            (
                self.hidden,
                self.projection_k_block_elements * self.grid_x,
                "hidden",
            ),
            (self.hidden, self.hidden_block_elements * self.grid_x, "hidden"),
            (
                self.local_intermediate,
                self.projection_n_block_elements * self.grid_x,
                "local intermediate",
            ),
            (
                self.local_intermediate,
                self.down_k_block_elements * self.grid_x,
                "local intermediate",
            ),
        )
        for size, divisor, name in dimensions:
            if size % divisor:
                raise ValueError(f"{name}={size} must be divisible by {divisor}")

    @property
    def local_intermediate(self):
        return self.intermediate // 2

    @property
    def m_block_elements(self):
        return self.m_block_tiles * 32

    @property
    def projection_k_block_elements(self):
        return self.projection_k_block_tiles * 32

    @property
    def projection_n_block_elements(self):
        return self.projection_n_block_tiles * 32

    @property
    def down_k_block_elements(self):
        return self.down_k_block_tiles * 32

    @property
    def hidden_block_elements(self):
        return self.projection_n_block_elements

    @property
    def m_blocks(self):
        return self.m // self.m_block_elements

    @property
    def n_blocks(self):
        return self.hidden // self.hidden_block_elements

    @property
    def m_blocks_per_core(self):
        return self.m_blocks // self.grid_y

    @property
    def n_blocks_per_core(self):
        return self.n_blocks // self.grid_x

    @property
    def output_blocks_per_core(self):
        return self.m_blocks_per_core * self.n_blocks_per_core


def make_operands(config):
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    hidden_states = torch.randn(
        config.m, config.hidden, dtype=torch.bfloat16, generator=generator
    ).mul_(0.125)
    gate_weight = torch.randn(
        config.hidden,
        config.intermediate,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.0625)
    up_weight = torch.randn(
        config.hidden,
        config.intermediate,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.0625)
    down_weight = torch.randn(
        config.intermediate,
        config.hidden,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.015625)
    residual = torch.randn(
        config.m, config.hidden, dtype=torch.bfloat16, generator=generator
    ).mul_(0.125)
    return hidden_states, gate_weight, up_weight, down_weight, residual


def golden(hidden_states, gate_weight, up_weight, down_weight, residual):
    gate = torch.matmul(hidden_states.float(), gate_weight.float()).to(torch.bfloat16)
    up = torch.matmul(hidden_states.float(), up_weight.float()).to(torch.bfloat16)
    activated = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    down = torch.matmul(activated.float(), down_weight.float()).to(torch.bfloat16)
    return (down + residual).to(torch.bfloat16)


def _layout(shape, block_shape, grid_shape, tiled=True):
    return d2m.Layout(
        shape=shape,
        dtype=d2m.bfloat16,
        block_shape=block_shape,
        grid_shape=grid_shape,
        tiled=tiled,
        mem_space="dram",
    )


def _weight_grid(block_rows, block_columns):
    def grid_extent(blocks):
        return max(extent for extent in range(1, 9) if blocks % extent == 0)

    return grid_extent(block_rows), grid_extent(block_columns)


@d2m.kernel
def tilize_weight(
    source,
    destination,
    row_blocks_per_core,
    column_blocks_per_core,
):
    cy = core_index(0)
    cx = core_index(1)
    for row_slot in range(row_blocks_per_core):
        row_block = cy * row_blocks_per_core + row_slot
        for column_slot in range(column_blocks_per_core):
            column_block = cx * column_blocks_per_core + column_slot
            block = remote_load(source, [row_block, column_block])
            remote_store(
                destination,
                [row_block, column_block],
                tilize_block(block),
            )


@d2m.kernel
def local_projection(
    hidden_states,
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
                lhs = remote_load(hidden_states, [m_block, k_block])
                rhs = remote_load(weight, [k_block, n_block])
                acc += lhs @ rhs
            remote_store(output, [m_block, n_block], acc)


@d2m.kernel
def gated_activation(gate, up, output, m_blocks_per_core, n_blocks_per_core):
    cy = core_index(0)
    cx = core_index(1)
    for m_slot in range(m_blocks_per_core):
        m_block = cy * m_blocks_per_core + m_slot
        for n_slot in range(n_blocks_per_core):
            n_block = cx * n_blocks_per_core + n_slot
            gate_block = remote_load(gate, [m_block, n_block])
            up_block = remote_load(up, [m_block, n_block])
            remote_store(output, [m_block, n_block], silu(gate_block) * up_block)


@d2m.kernel
def add_residual(
    projected,
    residual,
    output,
    m_blocks_per_core,
    n_blocks_per_core,
):
    cy = core_index(0)
    cx = core_index(1)
    for m_slot in range(m_blocks_per_core):
        m_block = cy * m_blocks_per_core + m_slot
        for n_slot in range(n_blocks_per_core):
            n_block = cx * n_blocks_per_core + n_slot
            projected_block = remote_load(projected, [m_block, n_block])
            residual_block = remote_load(residual, [m_block, n_block])
            remote_store(output, [m_block, n_block], projected_block + residual_block)


def _layouts(config, intermediate):
    grid = (config.grid_y, config.grid_x)
    hidden_layout = _layout(
        (config.m, config.hidden),
        [config.m_block_tiles, config.projection_k_block_tiles],
        grid,
    )
    projection_weight_grid = _weight_grid(
        config.hidden // config.projection_k_block_elements,
        intermediate // config.projection_n_block_elements,
    )
    projection_weight_host_layout = _layout(
        (config.hidden, intermediate),
        [
            config.projection_k_block_elements,
            config.projection_n_block_elements,
        ],
        projection_weight_grid,
        tiled=False,
    )
    projection_weight_layout = _layout(
        (config.hidden, intermediate),
        [config.projection_k_block_tiles, config.projection_n_block_tiles],
        projection_weight_grid,
    )
    projection_output_layout = _layout(
        (config.m, intermediate),
        [config.m_block_tiles, config.projection_n_block_tiles],
        grid,
    )
    down_activation_layout = _layout(
        (config.m, intermediate),
        [config.m_block_tiles, config.down_k_block_tiles],
        grid,
    )
    down_weight_grid = _weight_grid(
        intermediate // config.down_k_block_elements,
        config.hidden // config.projection_n_block_elements,
    )
    down_weight_host_layout = _layout(
        (intermediate, config.hidden),
        [config.down_k_block_elements, config.projection_n_block_elements],
        down_weight_grid,
        tiled=False,
    )
    down_weight_layout = _layout(
        (intermediate, config.hidden),
        [config.down_k_block_tiles, config.projection_n_block_tiles],
        down_weight_grid,
    )
    output_layout = _layout(
        (config.m, config.hidden),
        [config.m_block_tiles, config.projection_n_block_tiles],
        grid,
    )
    return (
        hidden_layout,
        projection_weight_host_layout,
        projection_weight_layout,
        projection_output_layout,
        down_activation_layout,
        down_weight_host_layout,
        down_weight_layout,
        output_layout,
    )


def _tilize_weight(source, layout):
    output = d2m.empty(layout)
    grid_y, grid_x = layout.grid_shape
    block_rows, block_columns = layout.blocked_grid_shape
    tilize_weight(
        source,
        output,
        block_rows // grid_y,
        block_columns // grid_x,
        grid=(grid_y, grid_x),
        kernel_io_in_dram=True,
    )
    return output


def _build_local_mlp(
    config,
    hidden_states,
    gate_weight,
    up_weight,
    down_weight,
    residual,
    two_chip,
    overlap,
):
    grid = (config.grid_y, config.grid_x)
    intermediate = config.local_intermediate if two_chip else config.intermediate
    (
        hidden_layout,
        projection_weight_host_layout,
        projection_weight_layout,
        projection_output_layout,
        down_activation_layout,
        down_weight_host_layout,
        down_weight_layout,
        output_layout,
    ) = _layouts(config, intermediate)

    if two_chip:
        hidden_device = d2m.mesh_shard(
            hidden_states,
            hidden_layout,
            shard_dims=[-1, -1],
            shard_shape=[1, 1],
        )
        gate_source = d2m.mesh_shard(
            gate_weight,
            projection_weight_host_layout,
            shard_dims=[-1, 1],
            shard_shape=[1, 2],
        )
        up_source = d2m.mesh_shard(
            up_weight,
            projection_weight_host_layout,
            shard_dims=[-1, 1],
            shard_shape=[1, 2],
        )
        down_source = d2m.mesh_shard(
            down_weight,
            down_weight_host_layout,
            shard_dims=[-1, 0],
            shard_shape=[2, 1],
        )
        residual_device = d2m.mesh_shard(
            residual,
            output_layout,
            shard_dims=[-1, -1],
            shard_shape=[1, 1],
        )
    else:
        hidden_device = d2m.to_layout(hidden_states, hidden_layout)
        gate_source = d2m.to_layout(gate_weight, projection_weight_host_layout)
        up_source = d2m.to_layout(up_weight, projection_weight_host_layout)
        down_source = d2m.to_layout(down_weight, down_weight_host_layout)
        residual_device = d2m.to_layout(residual, output_layout)

    gate_device = _tilize_weight(gate_source, projection_weight_layout)
    up_device = _tilize_weight(up_source, projection_weight_layout)
    down_device = _tilize_weight(down_source, down_weight_layout)

    projection_n_blocks_per_core = (
        intermediate // config.projection_n_block_elements // config.grid_x
    )
    projection_k_blocks = config.hidden // config.projection_k_block_elements
    gate_output = d2m.empty(projection_output_layout)
    up_output = d2m.empty(projection_output_layout)
    for weight_device, projection_output in (
        (gate_device, gate_output),
        (up_device, up_output),
    ):
        local_projection(
            hidden_device,
            weight_device,
            projection_output,
            config.m_blocks_per_core,
            projection_n_blocks_per_core,
            projection_k_blocks,
            grid=grid,
            kernel_io_in_dram=True,
        )

    activated = d2m.empty(projection_output_layout)
    gated_activation(
        gate_output,
        up_output,
        activated,
        config.m_blocks_per_core,
        projection_n_blocks_per_core,
        grid=grid,
        kernel_io_in_dram=True,
    )
    down_activations = (
        activated
        if config.projection_n_block_tiles == config.down_k_block_tiles
        else d2m.to_layout(activated, down_activation_layout)
    )
    down_k_blocks = intermediate // config.down_k_block_elements

    if not two_chip:
        projected = d2m.empty(output_layout)
        single_chip_down_projection(
            down_activations,
            down_device,
            projected,
            config.m_blocks_per_core,
            config.n_blocks_per_core,
            down_k_blocks,
            grid=grid,
            kernel_io_in_dram=True,
        )
    else:
        partial_layout = _layout(
            (2 * config.m, config.hidden),
            [config.m_block_tiles, config.projection_n_block_tiles],
            grid,
        )
        partials = d2m.empty(partial_layout)
        start_sem = d2m.global_semaphore(grid_shape=(8, 8))
        ready = d2m.global_semaphore(grid_shape=(8, 8), init=0)
        consumed = d2m.global_semaphore(grid_shape=(8, 8), init=0)
        fabric_done = d2m.global_semaphore(grid_shape=(8, 8), init=0)
        common_args = (
            down_activations,
            down_device,
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
            down_k_blocks,
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
        projected = d2m.empty(output_layout)
        reduce_two_chip_partials(
            partials,
            projected,
            config.m_blocks,
            config.m_blocks_per_core,
            config.n_blocks_per_core,
            grid=grid,
            kernel_io_in_dram=True,
        )

    output = d2m.empty(output_layout)
    add_residual(
        projected,
        residual_device,
        output,
        config.m_blocks_per_core,
        config.n_blocks_per_core,
        grid=grid,
        kernel_io_in_dram=True,
    )
    if two_chip:
        output = d2m.mesh_gather(
            output,
            shard_dims=[-1, -1],
            shard_shape=[1, 1],
        )
    return output


def build_single_chip(config, operands):
    d2m.mesh((1, 1), topology=("linear", "linear"))
    output = _build_local_mlp(config, *operands, two_chip=False, overlap=False)
    return output, operands[1:4]


def build_two_chip(config, operands, overlap):
    d2m.mesh((1, 2), topology=("linear", "linear"))
    output = _build_local_mlp(config, *operands, two_chip=True, overlap=overlap)
    return output, operands[1:4]


def run_single_chip(config, operands):
    output, _ = build_single_chip(config, operands)
    return output.to_host()


def run_two_chip(config, operands, overlap):
    output, _ = build_two_chip(config, operands, overlap)
    return output.to_host()


def prepare_single_chip(config, operands):
    output, reusable_weights = build_single_chip(config, operands)
    return d2m.prepare(output, reusable_inputs=reusable_weights)


def prepare_two_chip(config, operands, overlap):
    output, reusable_weights = build_two_chip(config, operands, overlap)
    return d2m.prepare(output, reusable_inputs=reusable_weights)
