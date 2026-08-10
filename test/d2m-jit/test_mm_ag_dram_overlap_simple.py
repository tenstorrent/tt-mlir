# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DRAM-spill MM+AG Baseline vs Overlap (M=1024, C=4). SKIP_PCC=1 skips checks."""

import functools
import json
import os
import re

import pytest
import torch

import d2m_jit as d2m
from d2m_jit._src.builder import _get_system_desc_path
from utils import assert_pcc

try:
    from _ttmlir_runtime import binary
except (ImportError, ModuleNotFoundError):
    binary = None


NUM_DEVICES = 8
NUM_M_CHUNKS = 4
M, K, N = 1024, 10752, 6144
M_CHUNK = M // NUM_M_CHUNKS
LOCAL_N = N // NUM_DEVICES
LOCAL_N_TILES = LOCAL_N // 32
M_TILES = M // 32
K_TILES = K // 32
MM_GRID = 8
MM_K_PARTITIONS = 8
BASELINE_K_PARTITIONS = 168
SPATIAL_MM_GRID_N = 6
SPATIAL_MM_K_PARTITIONS = 6
BUNDLED_MM_GRID_RANGE = ((0, 1), (7, 6))
AG_GRID_M = 2
AG_GRID_RANGE = ((0, 0), (1, 0))
AG_STORAGE_GRID_M = 8
AG_M_BLOCK = 4
ROWS_PER_CHUNK = 2
LHS_ROW_GRID = M_TILES

MM_M_BLOCK = 1
MM_K_BLOCK = K_TILES // MM_K_PARTITIONS
MM_N_BLOCK = LOCAL_N_TILES // MM_GRID
SPATIAL_K_BLOCK = K_TILES // SPATIAL_MM_K_PARTITIONS
SPATIAL_N_BLOCK = LOCAL_N_TILES // SPATIAL_MM_GRID_N
BASELINE_M_BLOCK = M_TILES // MM_GRID
BASELINE_K_BLOCK = K_TILES // BASELINE_K_PARTITIONS

SKIP_PCC = os.environ.get("SKIP_PCC", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PCC_THRESHOLD = 0.90


def _L(shape, block, grid, mem="dram"):
    return d2m.Layout(
        shape=shape,
        dtype=d2m.bfloat16,
        block_shape=list(block),
        grid_shape=list(grid),
        mem_space=mem,
    )


BASELINE_LHS = _L((M, K), [M_TILES // MM_GRID, MM_K_BLOCK], [MM_GRID, MM_GRID])
BASELINE_RHS = _L((K, LOCAL_N), [MM_K_BLOCK, MM_N_BLOCK], [MM_GRID, MM_GRID])
BASELINE_PARTIAL = _L((M, LOCAL_N), [BASELINE_M_BLOCK, MM_N_BLOCK], [MM_GRID, MM_GRID])
GATHERED_FULL = _L(
    (M, N), [AG_M_BLOCK, LOCAL_N_TILES], [AG_STORAGE_GRID_M, NUM_DEVICES]
)

CHUNK_LHS = _L((M_CHUNK, K), [MM_M_BLOCK, MM_K_BLOCK], [MM_GRID, MM_K_PARTITIONS])
CHUNK_RHS = _L((K, LOCAL_N), [MM_K_BLOCK, MM_N_BLOCK], [MM_K_PARTITIONS, MM_GRID])
CHUNK_PARTIAL = _L(
    (M_CHUNK, LOCAL_N), [MM_M_BLOCK, MM_N_BLOCK], [MM_GRID, MM_GRID], "l1"
)
SPATIAL_LHS = _L(
    (M_CHUNK, K),
    [MM_M_BLOCK, SPATIAL_K_BLOCK],
    [MM_GRID, SPATIAL_MM_K_PARTITIONS],
    "l1",
)
SPATIAL_RHS = _L(
    (K, LOCAL_N),
    [SPATIAL_K_BLOCK, SPATIAL_N_BLOCK],
    [SPATIAL_MM_K_PARTITIONS, SPATIAL_MM_GRID_N],
    "l1",
)
SPATIAL_PARTIAL = _L(
    (M_CHUNK, LOCAL_N),
    [MM_M_BLOCK, SPATIAL_N_BLOCK],
    [MM_GRID, SPATIAL_MM_GRID_N],
    "l1",
)


@functools.lru_cache(maxsize=1)
def _num_devices():
    if binary is None:
        return 0
    system_desc_path = _get_system_desc_path()
    if not system_desc_path:
        return 0
    try:
        system_desc_json = binary.load_system_desc_from_path(system_desc_path).as_json()
        system_desc_json = re.sub(r"\bnan\b", "NaN", system_desc_json)
        system_desc_json = re.sub(r"\binf\b", "Infinity", system_desc_json)
        system_desc = json.loads(system_desc_json)["system_desc"]
        return len(system_desc["chip_desc_indices"])
    except Exception:
        return 0


def _make_slice(chunk_idx, m_block_tiles, k_block_tiles):
    row_offset = chunk_idx * MM_GRID

    def slice_m(in0, out0):
        cy = core_index(0)
        cx = core_index(1)
        block = empty([m_block_tiles, k_block_tiles], dtype="bf16")
        block = remote_load(block, in0, [row_offset + cy, cx])
        remote_store(out0, [cy, cx], block)

    slice_m.__name__ = f"slice_m_{chunk_idx}"
    return d2m.kernel(slice_m)


def _make_matmul(k_partitions, m_block, k_block, n_block, name="matmul"):
    # Reduction GenericOp is required here: high-level mcast_dims on
    # remote_load only lower correctly with block_index/yield + reduction
    # iterators. Explicit core_index+K-loop leaves mcast in explicit DM form
    # and fails verification.
    def matmul(lhs, rhs, out):
        mbi = block_index(0)
        nbi = block_index(1)
        kbi = block_index(2)
        lhs_block = empty([m_block, k_block], dtype="bf16")
        lhs_block = remote_load(lhs_block, lhs, [mbi, kbi], mcast_dims=[0])
        rhs_block = empty([k_block, n_block], dtype="bf16")
        rhs_block = remote_load(rhs_block, rhs, [kbi, nbi], mcast_dims=[1])
        result = remote_store(out, [mbi, nbi], lhs_block @ rhs_block)
        yield result

    matmul.__name__ = name
    kern = d2m.kernel(matmul)

    def _call(*args, grid, **kwargs):
        return kern(
            *args,
            grid=grid,
            block_factors=[1, 1, k_partitions],
            indexing_maps=[
                lambda m, n, k: (m, k),
                lambda m, n, k: (k, n),
                lambda m, n, k: (m, n),
            ],
            iterator_types=["parallel", "parallel", "reduction"],
            **kwargs,
        )

    _call.__name__ = name
    return _call


def _make_all_gather_into_range(
    num_devices, row_start, num_rows, m_block, n_block, name="all_gather"
):
    assert num_rows % AG_GRID_M == 0
    rows_per_worker = num_rows // AG_GRID_M

    def all_gather(in0, out0, start_sem, end_sem):
        dy = mesh_position(0)
        dx = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, num_devices],
            num_receivers=num_devices - 1,
            core_indices=[cy, cx],
        )
        for local_i in range(rows_per_worker):
            in_row = cy * rows_per_worker + local_i
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


@pytest.fixture
def _stream_buffers(monkeypatch):
    # Single stream buffer is required for this problem size: default 2
    # overflows L1 on the overlap graph (~1.74MB needed vs ~1.39MB usable).
    monkeypatch.setattr(d2m.config, "num_stream_buffers", 1)
    monkeypatch.setattr(d2m.config, "kernel_io_in_dram", False)


def _make_inputs():
    torch.manual_seed(0)
    lhs = (torch.randn(M, K, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    rhs = (torch.randn(K, N, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return lhs, rhs, lhs.float() @ rhs.float()


def _begin():
    d2m.mesh((1, NUM_DEVICES), topology=("linear", "ring"))


def _shard_full_lhs(lhs):
    return d2m.mesh_shard(
        lhs,
        BASELINE_LHS,
        shard_dims=[-1],
        shard_shape=[1, NUM_DEVICES],
        shard_type="replicate",
    )


def _run_ag(partial_d, name, *, row_start, num_rows, gathered):
    start_sem = d2m.global_semaphore()
    end_sem = d2m.global_semaphore()
    ag = _make_all_gather_into_range(
        NUM_DEVICES, row_start, num_rows, AG_M_BLOCK, LOCAL_N_TILES, name=name
    )
    ag(
        d2m.reblock(partial_d, [num_rows, 1]),
        gathered,
        start_sem,
        end_sem,
        grid=(AG_GRID_M, 1),
        fabric=d2m.fabric_config(cluster_axis=1),
    )


def _slice_chunk(lhs_full, chunk_idx, out_layout, k_grid):
    src = d2m.reblock(lhs_full, [LHS_ROW_GRID, k_grid])
    out = d2m.empty(out_layout)
    _make_slice(chunk_idx, MM_M_BLOCK, out_layout.block_shape[1])(
        src, out, grid=(MM_GRID, k_grid)
    )
    return out


def _finish(gathered, expected):
    result = d2m.mesh_gather(
        gathered,
        shard_dims=[-1],
        shard_shape=[1, NUM_DEVICES],
        shard_type="replicate",
    ).to_host()
    assert tuple(result.shape) == (M, N), result.shape
    if not SKIP_PCC:
        assert_pcc(expected, result, threshold=PCC_THRESHOLD)


requires_mesh = pytest.mark.skipif(
    _num_devices() < NUM_DEVICES,
    reason=f"requires a >={NUM_DEVICES}-device mesh",
)


@requires_mesh
def test_baseline(_stream_buffers):
    lhs, rhs, expected = _make_inputs()
    _begin()

    lhs_base = _shard_full_lhs(lhs)
    lhs_d = d2m.reblock(lhs_base, [MM_GRID, BASELINE_K_PARTITIONS])
    rhs_base = d2m.mesh_shard(
        rhs, BASELINE_RHS, shard_dims=[-1, 1], shard_shape=[1, NUM_DEVICES]
    )
    rhs_d = d2m.reblock(rhs_base, [BASELINE_K_PARTITIONS, MM_GRID])
    partial_d = d2m.empty(BASELINE_PARTIAL)

    _make_matmul(
        BASELINE_K_PARTITIONS,
        BASELINE_M_BLOCK,
        BASELINE_K_BLOCK,
        MM_N_BLOCK,
        name="matmul",
    )(lhs_d, rhs_d, partial_d, grid=(MM_GRID, MM_GRID))

    gathered = d2m.empty(GATHERED_FULL)
    _run_ag(
        partial_d,
        "all_gather",
        row_start=0,
        num_rows=AG_STORAGE_GRID_M,
        gathered=gathered,
    )
    _finish(gathered, expected)


@requires_mesh
def test_overlap(_stream_buffers):
    lhs, rhs, expected = _make_inputs()
    _begin()

    lhs_full = _shard_full_lhs(lhs)
    lhs_chunk = _slice_chunk(lhs_full, 0, CHUNK_LHS, MM_K_PARTITIONS)
    rhs_d = d2m.mesh_shard(
        rhs, CHUNK_RHS, shard_dims=[-1, 1], shard_shape=[1, NUM_DEVICES]
    )
    partial = d2m.empty(CHUNK_PARTIAL)
    _make_matmul(
        MM_K_PARTITIONS, MM_M_BLOCK, MM_K_BLOCK, MM_N_BLOCK, name="matmul_chunk_0"
    )(lhs_chunk, rhs_d, partial, grid=(MM_GRID, MM_GRID))

    oy, ox = BUNDLED_MM_GRID_RANGE[0]
    spatial_rhs = d2m.mesh_shard(
        rhs,
        SPATIAL_RHS,
        shard_dims=[-1, 1],
        shard_shape=[1, NUM_DEVICES],
        virtual_grid_offset=(oy, ox),
    )
    gathered = d2m.empty(GATHERED_FULL)

    for step in range(NUM_M_CHUNKS - 1):
        prev = partial
        # Origin-placed region-sized lhs (8x6): nested MM generic inherits the
        # region's virt_to_physical map, so same-grid VGM to_layout is neither
        # required nor safe (corrupts). rhs is pre-placed via mesh_shard VGM.
        lhs_d = _slice_chunk(lhs_full, step + 1, SPATIAL_LHS, SPATIAL_MM_K_PARTITIONS)
        partial_spatial = d2m.empty(SPATIAL_PARTIAL)
        matmul = _make_matmul(
            SPATIAL_MM_K_PARTITIONS,
            MM_M_BLOCK,
            SPATIAL_K_BLOCK,
            SPATIAL_N_BLOCK,
            name=f"matmul_spatial_{step}",
        )
        start_sem = d2m.global_semaphore()
        end_sem = d2m.global_semaphore()
        ag = _make_all_gather_into_range(
            NUM_DEVICES,
            step * ROWS_PER_CHUNK,
            ROWS_PER_CHUNK,
            AG_M_BLOCK,
            LOCAL_N_TILES,
            name=f"all_gather_spatial_{step}",
        )
        ag_in = d2m.reblock(prev, [ROWS_PER_CHUNK, 1])
        d2m.spatial(
            inputs=[lhs_d, spatial_rhs, ag_in],
            outputs=[partial_spatial, gathered],
            grid_ranges=[BUNDLED_MM_GRID_RANGE, AG_GRID_RANGE],
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
                    grid=(AG_GRID_M, 1),
                    fabric=d2m.fabric_config(cluster_axis=1),
                ),
            ],
        )
        partial = partial_spatial

    _run_ag(
        partial,
        "all_gather_final",
        row_start=(NUM_M_CHUNKS - 1) * ROWS_PER_CHUNK,
        num_rows=ROWS_PER_CHUNK,
        gathered=gathered,
    )
    _finish(gathered, expected)
