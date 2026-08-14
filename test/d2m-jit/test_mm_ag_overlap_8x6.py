# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""8x6 MM + 2x1 AG overlap example.

Fixed shape: M=2048, K=10752, N=6144, C=4 M-chunks.
LHS/RHS in DRAM, partial C in L1, gathered in DRAM.
Chunk 0 matmul on 8x8; later chunks overlap 8x6 matmul with 2x1 all-gather.
"""

import functools
import json
import os
import re

import pytest
import torch

import d2m_jit as d2m
from utils import assert_pcc

try:
    from _ttmlir_runtime import binary
except (ImportError, ModuleNotFoundError):
    binary = None

NUM_DEVICES = 8
NUM_M_CHUNKS = 4
M, K, N = 2048, 10752, 6144
M_CHUNK = M // NUM_M_CHUNKS
LOCAL_N = N // NUM_DEVICES
LOCAL_N_TILES = LOCAL_N // 32
M_TILES = M // 32
K_TILES = K // 32

MM_GRID = 8
SPATIAL_MM_GRID_N = 6
AG_GRID_M, AG_GRID_N = 2, 1
MM_GRID_RANGE = ((0, 1), (7, 6))
AG_GRID_RANGE = ((0, 0), (1, 0))

AG_M_BLOCK = M_TILES // MM_GRID
AG_STORAGE_GRID_M = M_TILES // AG_M_BLOCK
ROWS_PER_CHUNK = AG_STORAGE_GRID_M // NUM_M_CHUNKS
AG_WORKERS = AG_GRID_M * AG_GRID_N

MM_M_BLOCK = (M_TILES // NUM_M_CHUNKS) // MM_GRID
MM_K_BLOCK = K_TILES // MM_GRID
MM_N_BLOCK = LOCAL_N_TILES // MM_GRID
SPATIAL_K_BLOCK = K_TILES // SPATIAL_MM_GRID_N
SPATIAL_N_BLOCK = LOCAL_N_TILES // SPATIAL_MM_GRID_N
BASELINE_M_BLOCK = M_TILES // MM_GRID
LHS_ROW_GRID = M_TILES // MM_M_BLOCK


def _L(shape, block, grid, mem="dram"):
    return d2m.Layout(
        shape=shape,
        dtype=d2m.bfloat16,
        block_shape=list(block),
        grid_shape=list(grid),
        mem_space=mem,
    )


LHS_FULL = _L((M, K), [M_TILES // MM_GRID, MM_K_BLOCK], [MM_GRID, MM_GRID])
RHS = _L((K, LOCAL_N), [MM_K_BLOCK, MM_N_BLOCK], [MM_GRID, MM_GRID])
PARTIAL = _L((M, LOCAL_N), [BASELINE_M_BLOCK, MM_N_BLOCK], [MM_GRID, MM_GRID], "l1")
GATHERED = _L((M, N), [AG_M_BLOCK, LOCAL_N_TILES], [AG_STORAGE_GRID_M, NUM_DEVICES])
CHUNK_LHS = _L((M_CHUNK, K), [MM_M_BLOCK, MM_K_BLOCK], [MM_GRID, MM_GRID])
CHUNK_PARTIAL = _L(
    (M_CHUNK, LOCAL_N), [MM_M_BLOCK, MM_N_BLOCK], [MM_GRID, MM_GRID], "l1"
)
SPATIAL_LHS = _L(
    (M_CHUNK, K), [MM_M_BLOCK, SPATIAL_K_BLOCK], [MM_GRID, SPATIAL_MM_GRID_N]
)
SPATIAL_PARTIAL = _L(
    (M_CHUNK, LOCAL_N),
    [MM_M_BLOCK, SPATIAL_N_BLOCK],
    [MM_GRID, SPATIAL_MM_GRID_N],
    "l1",
)


@functools.lru_cache(maxsize=1)
def _num_devices():
    system_desc = os.environ.get("SYSTEM_DESC_PATH")
    if binary is None or not system_desc:
        return 0
    try:
        desc = binary.load_system_desc_from_path(system_desc).as_json()
        desc = re.sub(r"\bnan\b", "NaN", desc)
        desc = re.sub(r"\binf\b", "Infinity", desc)
        return len(json.loads(desc)["system_desc"]["chip_desc_indices"])
    except Exception:
        return 0


requires_mesh = pytest.mark.skipif(
    _num_devices() < NUM_DEVICES,
    reason=f"requires a >={NUM_DEVICES}-device mesh",
)


@pytest.fixture
def _stream_buffers(monkeypatch):
    monkeypatch.setattr(d2m.config, "num_stream_buffers", 1)


def _make_slice(chunk_idx, m_block, k_block):
    row_offset = chunk_idx * MM_GRID

    def slice_m(in0, out0):
        cy = core_index(0)
        cx = core_index(1)
        block = empty([m_block, k_block], dtype="bf16")
        block = remote_load(block, in0, [row_offset + cy, cx])
        remote_store(out0, [cy, cx], block)

    slice_m.__name__ = f"slice_m_{chunk_idx}"
    return d2m.kernel(slice_m)


def _make_matmul(k_partitions, m_block, k_block, n_block, gy, gx, name):
    def matmul(lhs, rhs, out):
        cy = core_index(0)
        cx = core_index(1)
        acc = zeros([m_block, n_block], dtype="bf16")
        for k in range(k_partitions):
            a = empty([m_block, k_block], dtype="bf16")
            a = remote_load(
                a, lhs, [cy, k], mcast_start_index=[cy, 0], mcast_shape=[1, gx]
            )
            b = empty([k_block, n_block], dtype="bf16")
            b = remote_load(
                b, rhs, [k, cx], mcast_start_index=[0, cx], mcast_shape=[gy, 1]
            )
            acc += a @ b
        remote_store(out, [cy, cx], acc)

    matmul.__name__ = name
    return d2m.kernel(matmul)


def _make_all_gather(num_devices, row_start, num_rows, m_block, n_block, name):
    rows_per_worker = num_rows // AG_WORKERS
    ag_grid_n = AG_GRID_N

    def all_gather(in0, out0, start_sem, end_sem):
        dy = mesh_position(0)
        dx = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        worker = cy * ag_grid_n + cx
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, num_devices],
            num_receivers=num_devices - 1,
            core_indices=[cy, cx],
        )
        for local_i in range(rows_per_worker):
            in_row = worker * rows_per_worker + local_i
            out_row = row_start + in_row
            block = empty([m_block, n_block], dtype="bf16")
            block = remote_load(block, in0, [in_row, 0])
            remote_store(
                out0,
                [out_row, dx],
                block,
                start_device=[dy, 0],
                device_mcast_shape=[1, num_devices],
                semaphore=end_sem,
                semaphore_indices=[cy, cx],
            )
        semaphore_wait(end_sem, rows_per_worker * num_devices)

    all_gather.__name__ = name
    return d2m.kernel(all_gather)


def _shard_lhs(lhs):
    return d2m.mesh_shard(
        lhs,
        LHS_FULL,
        shard_dims=[-1],
        shard_shape=[1, NUM_DEVICES],
        shard_type="replicate",
    )


def _shard_rhs(rhs):
    return d2m.mesh_shard(rhs, RHS, shard_dims=[-1, 1], shard_shape=[1, NUM_DEVICES])


def _slice_chunk(lhs_full, chunk_idx, out_layout, k_grid):
    src = d2m.reblock(lhs_full, [LHS_ROW_GRID, k_grid])
    out = d2m.empty(out_layout)
    _make_slice(chunk_idx, MM_M_BLOCK, out_layout.block_shape[1])(
        src, out, grid=(MM_GRID, k_grid)
    )
    return out


def _run_ag(partial_d, name, row_start, num_rows, gathered):
    start_sem = d2m.global_semaphore()
    end_sem = d2m.global_semaphore()
    _make_all_gather(NUM_DEVICES, row_start, num_rows, AG_M_BLOCK, LOCAL_N_TILES, name)(
        d2m.reblock(partial_d, [num_rows, 1]),
        gathered,
        start_sem,
        end_sem,
        grid=(AG_GRID_M, AG_GRID_N),
        fabric=d2m.fabric_config(cluster_axis=1),
    )


def _check(gathered, expected):
    result = d2m.mesh_gather(
        gathered,
        shard_dims=[-1],
        shard_shape=[1, NUM_DEVICES],
        shard_type="replicate",
    ).to_host()
    assert tuple(result.shape) == (M, N), result.shape
    assert_pcc(expected, result, threshold=0.90)


@requires_mesh
def test_baseline(_stream_buffers):
    torch.manual_seed(0)
    lhs = (torch.randn(M, K, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    rhs = (torch.randn(K, N, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    expected = lhs.float() @ rhs.float()

    d2m.mesh((1, NUM_DEVICES), topology=("linear", "ring"))
    lhs_d = d2m.reblock(_shard_lhs(lhs), [MM_GRID, MM_GRID])
    rhs_d = d2m.reblock(_shard_rhs(rhs), [MM_GRID, MM_GRID])
    partial_d = d2m.empty(PARTIAL)
    _make_matmul(
        MM_GRID,
        BASELINE_M_BLOCK,
        MM_K_BLOCK,
        MM_N_BLOCK,
        MM_GRID,
        MM_GRID,
        "matmul",
    )(lhs_d, rhs_d, partial_d, grid=(MM_GRID, MM_GRID))

    gathered = d2m.empty(GATHERED)
    _run_ag(partial_d, "all_gather", 0, AG_STORAGE_GRID_M, gathered)
    _check(gathered, expected)


@requires_mesh
def test_overlap(_stream_buffers):
    torch.manual_seed(0)
    lhs = (torch.randn(M, K, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    rhs = (torch.randn(K, N, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    expected = lhs.float() @ rhs.float()

    d2m.mesh((1, NUM_DEVICES), topology=("linear", "ring"))
    lhs_full = _shard_lhs(lhs)
    rhs_d = _shard_rhs(rhs)
    spatial_rhs = d2m.reblock(rhs_d, [SPATIAL_MM_GRID_N, SPATIAL_MM_GRID_N])

    lhs_chunk = _slice_chunk(lhs_full, 0, CHUNK_LHS, MM_GRID)
    partial = d2m.empty(CHUNK_PARTIAL)
    _make_matmul(
        MM_GRID,
        MM_M_BLOCK,
        MM_K_BLOCK,
        MM_N_BLOCK,
        MM_GRID,
        MM_GRID,
        "matmul_chunk_0",
    )(lhs_chunk, rhs_d, partial, grid=(MM_GRID, MM_GRID))

    gathered = d2m.empty(GATHERED)
    for step in range(NUM_M_CHUNKS - 1):
        prev = partial
        lhs_d = _slice_chunk(lhs_full, step + 1, SPATIAL_LHS, SPATIAL_MM_GRID_N)
        partial_spatial = d2m.empty(SPATIAL_PARTIAL)
        matmul = _make_matmul(
            SPATIAL_MM_GRID_N,
            MM_M_BLOCK,
            SPATIAL_K_BLOCK,
            SPATIAL_N_BLOCK,
            MM_GRID,
            SPATIAL_MM_GRID_N,
            f"matmul_spatial_{step}",
        )
        ag = _make_all_gather(
            NUM_DEVICES,
            step * ROWS_PER_CHUNK,
            ROWS_PER_CHUNK,
            AG_M_BLOCK,
            LOCAL_N_TILES,
            f"all_gather_spatial_{step}",
        )
        start_sem = d2m.global_semaphore()
        end_sem = d2m.global_semaphore()
        ag_in = d2m.reblock(prev, [ROWS_PER_CHUNK, 1])
        d2m.spatial(
            inputs=[lhs_d, spatial_rhs, ag_in],
            outputs=[partial_spatial, gathered],
            grid_ranges=[MM_GRID_RANGE, AG_GRID_RANGE],
            region_builders=[
                lambda: matmul(
                    lhs_d,
                    spatial_rhs,
                    partial_spatial,
                    grid=(MM_GRID, SPATIAL_MM_GRID_N),
                ),
                lambda: ag(
                    ag_in,
                    gathered,
                    start_sem,
                    end_sem,
                    grid=(AG_GRID_M, AG_GRID_N),
                    fabric=d2m.fabric_config(cluster_axis=1),
                ),
            ],
        )
        partial = partial_spatial

    _run_ag(
        partial,
        "all_gather_final",
        (NUM_M_CHUNKS - 1) * ROWS_PER_CHUNK,
        ROWS_PER_CHUNK,
        gathered,
    )
    _check(gathered, expected)
