# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DRAM-spill MM+AG Baseline vs Overlap. SKIP_PCC=1 skips checks.

Fairness knobs (env, applied to both graphs):
  M, K, N, NUM_M_CHUNKS
  LHS_MEM / RHS_MEM / PARTIAL_MEM: l1 or dram
  K_PART_RULE: grid (k_partitions = grid_k) or legacy168
  OCCUPANCY: 2x1 (AG 2x1, MM 8x6), 8x7 (AG 2x1, MM 8x7), or 1x2 (AG 1x2, MM 8x6)
  KERNEL_IO_IN_DRAM: 0/1 (default 0; use 1 for large-N CB pressure)
"""

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


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_mem(name, default):
    raw = os.environ.get(name, default).strip().lower()
    if raw not in ("l1", "dram"):
        raise ValueError(f"{name}={raw!r}; expected l1 or dram")
    return raw


NUM_DEVICES = 8
NUM_M_CHUNKS = _env_int("NUM_M_CHUNKS", 4)
M = _env_int("M", 1024)
K = _env_int("K", 10752)
N = _env_int("N", 6144)
LHS_MEM = _env_mem("LHS_MEM", "l1")
RHS_MEM = _env_mem("RHS_MEM", "dram")
PARTIAL_MEM = _env_mem("PARTIAL_MEM", "l1")
K_PART_RULE = os.environ.get("K_PART_RULE", "grid").strip().lower()
OCCUPANCY = os.environ.get("OCCUPANCY", "2x1").strip().lower()

M_CHUNK = M // NUM_M_CHUNKS
LOCAL_N = N // NUM_DEVICES
LOCAL_N_TILES = LOCAL_N // 32
M_TILES = M // 32
K_TILES = K // 32
MM_GRID = 8
MM_K_PARTITIONS = MM_GRID
if K_PART_RULE == "legacy168":
    BASELINE_K_PARTITIONS = 168
elif K_PART_RULE == "grid":
    BASELINE_K_PARTITIONS = MM_GRID
else:
    raise ValueError(f"K_PART_RULE={K_PART_RULE!r}; expected grid or legacy168")

# AG uses 2 exclusive cores; MM takes a rectangular leftover.
# 8x7 uses the column that 2x1/8x6 left idle (col 7).
SPATIAL_MM_GRID_M = 8
if OCCUPANCY in ("2x1", "ag_2x1"):
    SPATIAL_MM_GRID_N = 6
    AG_GRID_M, AG_GRID_N = 2, 1
    BUNDLED_MM_GRID_RANGE = ((0, 1), (7, 6))
    AG_GRID_RANGE = ((0, 0), (1, 0))
    SPATIAL_VIRTUAL_OFFSET = (0, 1)
elif OCCUPANCY in ("8x7", "ag_2x1_8x7"):
    SPATIAL_MM_GRID_N = 7
    AG_GRID_M, AG_GRID_N = 2, 1
    BUNDLED_MM_GRID_RANGE = ((0, 1), (7, 7))
    AG_GRID_RANGE = ((0, 0), (1, 0))
    SPATIAL_VIRTUAL_OFFSET = (0, 1)
elif OCCUPANCY in ("1x2", "ag_1x2"):
    SPATIAL_MM_GRID_N = 6
    AG_GRID_M, AG_GRID_N = 1, 2
    BUNDLED_MM_GRID_RANGE = ((0, 0), (7, 5))
    AG_GRID_RANGE = ((0, 6), (0, 7))
    SPATIAL_VIRTUAL_OFFSET = (0, 0)
else:
    raise ValueError(f"OCCUPANCY={OCCUPANCY!r}; expected 2x1, 8x7, or 1x2")
SPATIAL_MM_K_PARTITIONS = SPATIAL_MM_GRID_N

AG_WORKERS = AG_GRID_M * AG_GRID_N
# Keep gathered storage on an 8x8 worker grid. AG_M_BLOCK=4 at M=1024
# (M_TILES=32) is unchanged; larger M would otherwise use a 16x8 virtual
# grid and hit "volume 128 has no valid 2D factorization within [8, 8]".
AG_M_BLOCK = M_TILES // MM_GRID
AG_STORAGE_GRID_M = M_TILES // AG_M_BLOCK
ROWS_PER_CHUNK = AG_STORAGE_GRID_M // NUM_M_CHUNKS

MM_M_BLOCK = (M_TILES // NUM_M_CHUNKS) // MM_GRID
MM_K_BLOCK = K_TILES // MM_K_PARTITIONS
MM_N_BLOCK = LOCAL_N_TILES // MM_GRID
SPATIAL_K_BLOCK = K_TILES // SPATIAL_MM_K_PARTITIONS
SPATIAL_N_BLOCK = LOCAL_N_TILES // SPATIAL_MM_GRID_N
BASELINE_M_BLOCK = M_TILES // MM_GRID
BASELINE_K_BLOCK = K_TILES // BASELINE_K_PARTITIONS
LHS_ROW_GRID = M_TILES // MM_M_BLOCK

_TILE = 32
_errs = []
if M % _TILE or K % _TILE or N % _TILE:
    _errs.append("M,K,N must be multiples of 32")
if M_TILES % MM_GRID:
    _errs.append("M_TILES must divide MM_GRID")
if (M_TILES // NUM_M_CHUNKS) % MM_GRID:
    _errs.append("chunk M tiles must divide MM_GRID")
if K_TILES % MM_K_PARTITIONS or K_TILES % SPATIAL_MM_K_PARTITIONS:
    _errs.append("K_TILES must divide 8x8 and spatial grid_k")
if K_TILES % BASELINE_K_PARTITIONS:
    _errs.append("K_TILES must divide BASELINE_K_PARTITIONS")
if LOCAL_N_TILES % MM_GRID or LOCAL_N_TILES % SPATIAL_MM_GRID_N:
    _errs.append("LOCAL_N_TILES must divide MM grids")
if M_TILES % AG_M_BLOCK:
    _errs.append("M_TILES must divide AG_M_BLOCK")
if AG_STORAGE_GRID_M % NUM_M_CHUNKS:
    _errs.append("AG_STORAGE_GRID_M must divide NUM_M_CHUNKS")
if ROWS_PER_CHUNK % AG_WORKERS:
    _errs.append("ROWS_PER_CHUNK must divide AG_WORKERS")
if MM_M_BLOCK < 1:
    _errs.append("MM_M_BLOCK must be >= 1")
if _errs:
    raise ValueError(
        f"illegal tiling M={M} K={K} N={N} C={NUM_M_CHUNKS} "
        f"K_PART_RULE={K_PART_RULE} OCCUPANCY={OCCUPANCY}: " + "; ".join(_errs)
    )

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


# Same mem_space on analogous tensors for baseline and overlap.
# Overlap still shards the full LHS into DRAM for slicing (strategy cost).
BASELINE_LHS_FULL = _L((M, K), [M_TILES // MM_GRID, MM_K_BLOCK], [MM_GRID, MM_GRID])
BASELINE_LHS = _L((M, K), [M_TILES // MM_GRID, MM_K_BLOCK], [MM_GRID, MM_GRID], LHS_MEM)
BASELINE_RHS = _L((K, LOCAL_N), [MM_K_BLOCK, MM_N_BLOCK], [MM_GRID, MM_GRID], RHS_MEM)
BASELINE_PARTIAL = _L(
    (M, LOCAL_N), [BASELINE_M_BLOCK, MM_N_BLOCK], [MM_GRID, MM_GRID], PARTIAL_MEM
)
GATHERED_FULL = _L(
    (M, N), [AG_M_BLOCK, LOCAL_N_TILES], [AG_STORAGE_GRID_M, NUM_DEVICES]
)
CHUNK_LHS = _L(
    (M_CHUNK, K), [MM_M_BLOCK, MM_K_BLOCK], [MM_GRID, MM_K_PARTITIONS], LHS_MEM
)
CHUNK_RHS = _L(
    (K, LOCAL_N), [MM_K_BLOCK, MM_N_BLOCK], [MM_K_PARTITIONS, MM_GRID], RHS_MEM
)
CHUNK_PARTIAL = _L(
    (M_CHUNK, LOCAL_N), [MM_M_BLOCK, MM_N_BLOCK], [MM_GRID, MM_GRID], PARTIAL_MEM
)
SPATIAL_LHS = _L(
    (M_CHUNK, K),
    [MM_M_BLOCK, SPATIAL_K_BLOCK],
    [MM_GRID, SPATIAL_MM_K_PARTITIONS],
    LHS_MEM,
)
SPATIAL_RHS = _L(
    (K, LOCAL_N),
    [SPATIAL_K_BLOCK, SPATIAL_N_BLOCK],
    [SPATIAL_MM_K_PARTITIONS, SPATIAL_MM_GRID_N],
    RHS_MEM,
)
SPATIAL_PARTIAL = _L(
    (M_CHUNK, LOCAL_N),
    [MM_M_BLOCK, SPATIAL_N_BLOCK],
    [MM_GRID, SPATIAL_MM_GRID_N],
    PARTIAL_MEM,
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


def _make_matmul(k_partitions, m_block, k_block, n_block, gy, gx, name="matmul"):
    # Explicit DM + low-level mcast. gy/gx are int captures (not runtime
    # scalar args) so saved flatbuffers do not need program-arg scalars.
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
    kern = d2m.kernel(matmul)

    def _call(*args, grid, **kwargs):
        assert (int(grid[0]), int(grid[1])) == (gy, gx), (grid, gy, gx)
        return kern(*args, grid=grid, **kwargs)

    _call.__name__ = name
    return _call


def _make_all_gather_into_range(
    num_devices, row_start, num_rows, m_block, n_block, name="all_gather"
):
    assert num_rows % AG_WORKERS == 0
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


KERNEL_IO_IN_DRAM = os.environ.get("KERNEL_IO_IN_DRAM", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


@pytest.fixture
def _stream_buffers(monkeypatch):
    # Default 2 stream buffers overflows L1 on this overlap graph.
    monkeypatch.setattr(d2m.config, "num_stream_buffers", 1)
    monkeypatch.setattr(d2m.config, "kernel_io_in_dram", KERNEL_IO_IN_DRAM)


def _make_inputs():
    torch.manual_seed(0)
    lhs = (torch.randn(M, K, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    rhs = (torch.randn(K, N, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    # Skip host GEMM when PCC is off; N=14336 would be a large CPU matmul.
    expected = None if SKIP_PCC else lhs.float() @ rhs.float()
    return lhs, rhs, expected


def _begin():
    d2m.mesh((1, NUM_DEVICES), topology=("linear", "ring"))


def _shard_full_lhs(lhs, layout=None):
    if layout is None:
        layout = BASELINE_LHS_FULL
    return d2m.mesh_shard(
        lhs,
        layout,
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
        grid=(AG_GRID_M, AG_GRID_N),
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

    lhs_d = d2m.reblock(
        _shard_full_lhs(lhs, BASELINE_LHS), [MM_GRID, BASELINE_K_PARTITIONS]
    )
    rhs_d = d2m.reblock(
        d2m.mesh_shard(
            rhs, BASELINE_RHS, shard_dims=[-1, 1], shard_shape=[1, NUM_DEVICES]
        ),
        [BASELINE_K_PARTITIONS, MM_GRID],
    )
    partial_d = d2m.empty(BASELINE_PARTIAL)
    _make_matmul(
        BASELINE_K_PARTITIONS,
        BASELINE_M_BLOCK,
        BASELINE_K_BLOCK,
        MM_N_BLOCK,
        MM_GRID,
        MM_GRID,
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
        MM_K_PARTITIONS,
        MM_M_BLOCK,
        MM_K_BLOCK,
        MM_N_BLOCK,
        MM_GRID,
        MM_GRID,
        name="matmul_chunk_0",
    )(lhs_chunk, rhs_d, partial, grid=(MM_GRID, MM_GRID))

    # Always view the already-tilized 8x8 RHS. A second host tilize of a
    # 6x6/7x7 tensor either misses L1 (large N) or, with virtual_grid_offset,
    # corrupts spatial-chunk matmuls (8x6 PCC ~ 1/C). 8x7 already used this
    # path and passed PCC.
    spatial_rhs = d2m.reblock(rhs_d, [SPATIAL_MM_K_PARTITIONS, SPATIAL_MM_GRID_N])
    gathered = d2m.empty(GATHERED_FULL)

    for step in range(NUM_M_CHUNKS - 1):
        prev = partial
        lhs_d = _slice_chunk(lhs_full, step + 1, SPATIAL_LHS, SPATIAL_MM_K_PARTITIONS)
        partial_spatial = d2m.empty(SPATIAL_PARTIAL)
        matmul = _make_matmul(
            SPATIAL_MM_K_PARTITIONS,
            MM_M_BLOCK,
            SPATIAL_K_BLOCK,
            SPATIAL_N_BLOCK,
            MM_GRID,
            SPATIAL_MM_GRID_N,
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
                    grid=(AG_GRID_M, AG_GRID_N),
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
