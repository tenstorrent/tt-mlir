# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

import d2m_jit as d2m


@dataclass(frozen=True)
class WorkloadConfig:
    grid_y: int = 4
    grid_x: int = 4
    num_chunks: int = 2
    block_tiles: int = 4
    k_tiles: int = 24
    compute_repeats: int = 6
    seed: int = 0

    @property
    def block_elements(self):
        return self.block_tiles * 32

    @property
    def k_elements(self):
        return self.k_tiles * 32


@d2m.kernel
def single_chip_matmuls(lhs, rhs, output, num_items, compute_repeats):
    cy = core_index(0)
    cx = core_index(1)
    for item in range(num_items):
        a = remote_load(lhs, [cy * num_items + item, cx])
        b = remote_load(rhs, [cy * num_items + item, cx])
        acc = a @ b
        for repeat in range(compute_repeats):
            acc += a @ b
        remote_store(output, [cy * num_items + item, cx], acc)


@d2m.kernel
def two_chip_matmul_all_gather(
    lhs,
    rhs,
    output,
    start_sem,
    ready,
    consumed,
    end_sem,
    num_chunks,
    grid_y,
    grid_x,
    worker_count,
    compute_repeats,
):
    cy = core_index(0)
    cx = core_index(1)
    for chunk in range(num_chunks):
        a = remote_load(lhs, [cy * num_chunks + chunk, cx])
        b = remote_load(rhs, [cy * num_chunks + chunk, cx])
        acc = a @ b
        for repeat in range(compute_repeats):
            acc += a @ b
        semaphore_inc(ready, 1, core=[0, 0], compute=True)
        if is_router_core():
            dy = mesh_position(0)
            device_synchronize(
                start_sem,
                start_device=[dy, 0],
                mcast_shape=[1, 2],
                num_receivers=1,
                core_indices=[cy, cx],
            )
            semaphore_wait(ready, (chunk + 1) * worker_count)
            dx = mesh_position(1)
            for ty in range(grid_y):
                for tx in range(grid_x):
                    gathered = empty_like(acc)
                    gathered = core_read(gathered, acc, core=[ty, tx])
                    semaphore_inc(consumed, 1, core=[ty, tx])
                    remote_store(
                        output,
                        [
                            dx * grid_y * num_chunks + ty * num_chunks + chunk,
                            tx,
                        ],
                        gathered,
                        start_device=[dy, 0],
                        device_mcast_shape=[1, 2],
                        semaphore=end_sem,
                        semaphore_indices=[cy, 0],
                    )
            semaphore_wait(end_sem, (chunk + 1) * 2 * worker_count)
        semaphore_wait(consumed, chunk + 1, compute=True)


def _layout(rows, columns, block_shape):
    return d2m.Layout(
        shape=(rows * block_shape[0] * 32, columns * block_shape[1] * 32),
        dtype=d2m.bfloat16,
        block_shape=block_shape,
        grid_shape=[rows, columns],
        mem_space="dram",
    )


def _blocked_matmuls(lhs, rhs, config, block_rows, block_columns):
    output = torch.empty(
        block_rows * config.block_elements,
        block_columns * config.block_elements,
    )
    for by in range(block_rows):
        row = slice(
            by * config.block_elements,
            (by + 1) * config.block_elements,
        )
        rhs_row = slice(by * config.k_elements, (by + 1) * config.k_elements)
        for bx in range(block_columns):
            column = slice(
                bx * config.block_elements,
                (bx + 1) * config.block_elements,
            )
            lhs_column = slice(
                bx * config.k_elements,
                (bx + 1) * config.k_elements,
            )
            output[row, column] = (config.compute_repeats + 1) * (
                lhs[row, lhs_column] @ rhs[rhs_row, column]
            )
    return output


def run_single_chip(config):
    torch.manual_seed(config.seed)
    d2m.mesh((1, 1), topology=("linear", "linear"))
    num_items = 2 * config.num_chunks
    block_rows = config.grid_y * num_items
    lhs_layout = _layout(
        block_rows,
        config.grid_x,
        [config.block_tiles, config.k_tiles],
    )
    rhs_layout = _layout(
        block_rows,
        config.grid_x,
        [config.k_tiles, config.block_tiles],
    )
    output_layout = _layout(
        block_rows,
        config.grid_x,
        [config.block_tiles, config.block_tiles],
    )
    lhs_host = torch.randn(lhs_layout.logical_shape, dtype=torch.bfloat16) * 0.125
    rhs_host = torch.randn(rhs_layout.logical_shape, dtype=torch.bfloat16) * 0.125
    output = d2m.empty(output_layout)
    single_chip_matmuls(
        d2m.to_layout(lhs_host, lhs_layout),
        d2m.to_layout(rhs_host, rhs_layout),
        output,
        num_items,
        config.compute_repeats,
        grid=(config.grid_y, config.grid_x),
        kernel_io_in_dram=True,
    )
    result = output.to_host()
    expected = _blocked_matmuls(
        lhs_host,
        rhs_host,
        config,
        block_rows,
        config.grid_x,
    )
    return result, expected


def run_two_chip(config):
    torch.manual_seed(config.seed)
    d2m.mesh((1, 2), topology=("linear", "linear"))
    input_rows = config.grid_y * config.num_chunks
    lhs_layout = _layout(
        input_rows,
        config.grid_x,
        [config.block_tiles, config.k_tiles],
    )
    rhs_layout = _layout(
        input_rows,
        config.grid_x,
        [config.k_tiles, config.block_tiles],
    )
    output_layout = _layout(
        2 * input_rows,
        config.grid_x,
        [config.block_tiles, config.block_tiles],
    )
    lhs_host = (
        torch.randn(
            lhs_layout.logical_shape[0],
            2 * lhs_layout.logical_shape[1],
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    rhs_host = (
        torch.randn(
            rhs_layout.logical_shape[0],
            2 * rhs_layout.logical_shape[1],
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    lhs = d2m.mesh_shard(
        lhs_host,
        lhs_layout,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    )
    rhs = d2m.mesh_shard(
        rhs_host,
        rhs_layout,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    )
    output = d2m.empty(output_layout)
    two_chip_matmul_all_gather(
        lhs,
        rhs,
        output,
        d2m.global_semaphore(grid_shape=(8, 8)),
        d2m.global_semaphore(grid_shape=(8, 8), init=0),
        d2m.global_semaphore(grid_shape=(8, 8), init=0),
        d2m.global_semaphore(grid_shape=(8, 8)),
        config.num_chunks,
        config.grid_y,
        config.grid_x,
        config.grid_y * config.grid_x,
        config.compute_repeats,
        grid=(config.grid_y, config.grid_x),
        fabric=d2m.fabric_config(
            cluster_axis=1,
            topology="linear",
            router_cores=[(0, 0)],
        ),
        kernel_io_in_dram=True,
    )
    result = d2m.mesh_gather(
        output,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    ).to_host()

    gathered_blocks = []
    for device in range(2):
        for cy in range(config.grid_y):
            for chunk in range(config.num_chunks):
                row_index = cy * config.num_chunks + chunk
                row = slice(
                    row_index * config.block_elements,
                    (row_index + 1) * config.block_elements,
                )
                row_blocks = []
                for cx in range(config.grid_x):
                    column_index = device * config.grid_x + cx
                    column = slice(
                        column_index * config.block_elements,
                        (column_index + 1) * config.block_elements,
                    )
                    lhs_column = slice(
                        column_index * config.k_elements,
                        (column_index + 1) * config.k_elements,
                    )
                    rhs_row = slice(
                        row_index * config.k_elements,
                        (row_index + 1) * config.k_elements,
                    )
                    row_blocks.append(
                        (config.compute_repeats + 1)
                        * (lhs_host[row, lhs_column] @ rhs_host[rhs_row, column])
                    )
                gathered_blocks.append(torch.cat(row_blocks, dim=1))
    gathered = torch.cat(gathered_blocks, dim=0)
    expected = torch.cat([gathered, gathered], dim=1)
    return result, expected
