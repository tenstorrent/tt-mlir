# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import os
import re

import pytest
import torch

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder
from utils import assert_pcc

try:
    from _ttmlir_runtime import binary, runtime
except (ImportError, ModuleNotFoundError):
    binary = None
    runtime = None


@functools.lru_cache(maxsize=1)
def _num_devices():
    """Read the chip count without opening a runtime device."""
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
    runtime is None or _num_devices() < 2,
    reason="requires SYSTEM_DESC_PATH for a system with at least two devices",
)

requires_fabric_mesh = pytest.mark.skipif(
    runtime is None
    or _num_devices() < 2
    or os.environ.get("D2M_JIT_RUN_FABRIC_TESTS") != "1",
    reason=(
        "requires a two-device system and D2M_JIT_RUN_FABRIC_TESTS=1 "
        "to opt into fabric execution"
    ),
)


def test_mesh_configuration():
    d2m.mesh((1, 2), topology=("linear", "ring"))
    builder = _Builder.get()

    assert '#ttcore.meshes<[<"mesh" = 1x2>]>' in str(builder.module.operation)
    assert builder._mesh_shape == [1, 2]
    assert builder._mesh_topology == ["linear", "ring"]


def test_fabric_configuration_matches_mesh():
    d2m.mesh((1, 2), topology=("linear", "ring"))
    builder = _Builder.get()
    fabric = d2m.fabric_config(cluster_axis=1)

    builder.enable_fabric(fabric)

    assert fabric.topology == "ring"
    assert fabric.routing == "unidir_ring_torus"
    assert builder._fabric_runtime_mode == "FABRIC_1D_RING"

    with pytest.raises(ValueError, match="does not match mesh topology"):
        builder.enable_fabric(d2m.fabric_config(cluster_axis=1, topology="linear"))


def test_mesh_gather_derives_full_shape():
    d2m.mesh((2, 2), topology=("linear", "linear"))
    layout = d2m.Layout(
        shape=(64, 32),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 1],
    )

    gathered = d2m.mesh_gather(
        d2m.empty(layout),
        shard_dims=[1, 1],
        shard_shape=[1, 4],
    )

    assert gathered.mesh.full_shape == [64, 128]


@requires_mesh
def test_mesh_shard_round_trip_1x2():
    d2m.mesh((1, 2), topology=("linear", "ring"))
    layout = d2m.Layout(
        shape=(512, 512),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
    )
    full = torch.randn((512, 1024), dtype=torch.float32)

    shard = d2m.mesh_shard(
        full,
        layout,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    )
    result = shard.to_host()

    assert result.shape == full.shape
    assert torch.allclose(result, full, atol=1e-2)


@requires_mesh
def test_mesh_compute_round_trip_1x2():
    @d2m.kernel
    def sigmoid_kernel(input_, output, m_blocks, n_blocks):
        m_offset = core_index(0) * m_blocks
        n_offset = core_index(1) * n_blocks
        for m in range(m_blocks):
            for n in range(n_blocks):
                value = remote_load(input_, [m_offset + m, n_offset + n])
                remote_store(
                    output,
                    [m_offset + m, n_offset + n],
                    sigmoid(value),
                )

    d2m.mesh((1, 2), topology=("linear", "ring"))
    layout = d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
    )
    full = torch.randn((64, 128), dtype=torch.float32) * 0.5
    input_ = d2m.mesh_shard(
        full,
        layout,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    )
    output = d2m.empty(layout)

    sigmoid_kernel(input_, output, 1, 1, grid=(2, 2))
    result = d2m.mesh_gather(
        output,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    ).to_host()

    assert result.shape == full.shape
    assert (torch.sigmoid(full) - result).abs().max().item() < 0.05


@d2m.kernel
def _matmul_core_read_gather(lhs, rhs, output, ready):
    cy = core_index(0)
    cx = core_index(1)
    a = remote_load(lhs, [cy, cx])
    b = remote_load(rhs, [cy, cx])
    c = a @ b
    semaphore_inc(ready, 1, core=[0, 0], compute=True)
    if cy == 0:
        semaphore_wait(ready, 2)
        own = empty([4, 4])
        own = core_read(own, c, core=[0, 0])
        peer = empty([4, 4])
        peer = core_read(peer, c, core=[1, 0])
        remote_store(output, [0, 0], own)
        remote_store(output, [1, 0], peer)


@requires_mesh
def test_matmul_core_read_gather():
    """Isolate a multi-tile compute-to-DM producer fence without fabric."""
    layout = d2m.Layout(
        shape=(256, 128),
        dtype=d2m.float32,
        block_shape=[4, 4],
        grid_shape=[2, 1],
    )
    lhs = torch.randn(256, 128, dtype=torch.float32) * 0.125
    rhs = torch.randn(256, 128, dtype=torch.float32) * 0.125
    output = d2m.empty(layout)
    _matmul_core_read_gather(
        d2m.to_layout(lhs, layout),
        d2m.to_layout(rhs, layout),
        output,
        d2m.global_semaphore(grid_shape=(8, 8), init=0),
        grid=(2, 1),
    )
    result = output.to_host()
    expected = torch.cat([lhs[:128] @ rhs[:128], lhs[128:] @ rhs[128:]], dim=0)
    assert result.shape == expected.shape
    assert_pcc(expected, result, threshold=0.99)


@d2m.kernel
def _chunked_matmul_core_read_gather(lhs, rhs, output, ready, consumed):
    for chunk in range(2):
        a = remote_load(lhs, [chunk, 0])
        b = remote_load(rhs, [chunk, 0])
        c = a @ b
        semaphore_inc(ready, 1, core=[0, 0], compute=True)
        semaphore_wait(ready, chunk + 1)
        gathered = empty([4, 4])
        gathered = core_read(gathered, c, core=[0, 0])
        semaphore_inc(consumed, 1, core=[0, 0])
        remote_store(output, [chunk, 0], gathered)
        semaphore_wait(consumed, chunk + 1, compute=True)


@requires_mesh
def test_chunked_matmul_core_read_gather():
    """Isolate producer-CB release and reuse across two chunks."""
    layout = d2m.Layout(
        shape=(256, 128),
        dtype=d2m.float32,
        block_shape=[4, 4],
        grid_shape=[2, 1],
    )
    lhs = torch.randn(256, 128, dtype=torch.float32) * 0.125
    rhs = torch.randn(256, 128, dtype=torch.float32) * 0.125
    output = d2m.empty(layout)
    _chunked_matmul_core_read_gather(
        d2m.to_layout(lhs, layout),
        d2m.to_layout(rhs, layout),
        output,
        d2m.global_semaphore(grid_shape=(8, 8), init=0),
        d2m.global_semaphore(grid_shape=(8, 8), init=0),
        grid=(1, 1),
    )
    result = output.to_host()
    expected = torch.cat([lhs[:128] @ rhs[:128], lhs[128:] @ rhs[128:]], dim=0)
    assert result.shape == expected.shape
    assert_pcc(expected, result, threshold=0.99)


@requires_fabric_mesh
def test_all_gather_round_trip_1x2():
    @d2m.kernel
    def all_gather(input_, output, start_sem, end_sem):
        dy = mesh_position(0)
        dx = mesh_position(1)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
        scratch = empty([2, 2])
        scratch = remote_load(scratch, input_, [0, 0])
        remote_store(
            output,
            [dx, 0],
            scratch,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, cx],
        )
        semaphore_wait(end_sem, 2)

    d2m.mesh((1, 2), topology=("linear", "linear"))
    input_layout = d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[2, 2],
        grid_shape=[1, 1],
    )
    output_layout = d2m.Layout(
        shape=(128, 64),
        dtype=d2m.float32,
        block_shape=[2, 2],
        grid_shape=[2, 1],
    )
    full = torch.randn(64, 128, dtype=torch.float32)
    input_ = d2m.mesh_shard(
        full,
        input_layout,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    )
    output = d2m.empty(output_layout)
    start_sem = d2m.global_semaphore()
    end_sem = d2m.global_semaphore()
    all_gather(
        input_,
        output,
        start_sem,
        end_sem,
        grid=(1, 1),
        fabric=d2m.fabric_config(cluster_axis=1, topology="linear"),
    )
    result = d2m.mesh_gather(
        output,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    ).to_host()

    shard0 = full[:, :64]
    shard1 = full[:, 64:]
    gathered = torch.cat([shard0, shard1], dim=0)
    expected = torch.cat([gathered, gathered], dim=1)
    assert result.shape == expected.shape
    assert (expected - result).abs().max().item() < 0.05


@d2m.kernel
def _multicore_all_gather_1x2(
    input_, output, start_sem, ready, end_sem, grid_y, grid_x, worker_count
):
    cy = core_index(0)
    cx = core_index(1)
    value = remote_load(input_, [cy, cx])
    semaphore_inc(ready, 1, core=[0, 0])
    if is_router_core():
        dy = mesh_position(0)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
        semaphore_wait(ready, worker_count)
        dx = mesh_position(1)
        for ty in range(grid_y):
            for tx in range(grid_x):
                gathered = empty([2, 2])
                gathered = core_read(gathered, value, core=[ty, tx])
                remote_store(
                    output,
                    [dx * grid_y + ty, tx],
                    gathered,
                    start_device=[dy, 0],
                    device_mcast_shape=[1, 2],
                    semaphore=end_sem,
                    semaphore_indices=[cy, 0],
                )
        semaphore_wait(end_sem, 2 * worker_count)


@requires_fabric_mesh
def test_multicore_all_gather_round_trip_1x2():
    """Gather worker blocks through one router core and one fabric link."""
    torch.manual_seed(0)
    d2m.mesh((1, 2), topology=("linear", "linear"))
    grid_y, grid_x = (2, 1)
    block_tiles = 2
    block_elements = block_tiles * 32
    input_layout = d2m.Layout(
        shape=(grid_y * block_elements, grid_x * block_elements),
        dtype=d2m.float32,
        block_shape=[block_tiles, block_tiles],
        grid_shape=[grid_y, grid_x],
    )
    output_layout = d2m.Layout(
        shape=(grid_y * 2 * block_elements, grid_x * block_elements),
        dtype=d2m.float32,
        block_shape=[block_tiles, block_tiles],
        grid_shape=[2 * grid_y, grid_x],
    )
    full = torch.randn(
        grid_y * block_elements, 2 * grid_x * block_elements, dtype=torch.float32
    )
    input_ = d2m.mesh_shard(
        full,
        input_layout,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    )
    output = d2m.empty(output_layout)
    _multicore_all_gather_1x2(
        input_,
        output,
        d2m.global_semaphore(),
        d2m.global_semaphore(init=0),
        d2m.global_semaphore(),
        grid_y,
        grid_x,
        grid_y * grid_x,
        grid=(grid_y, grid_x),
        fabric=d2m.fabric_config(
            cluster_axis=1,
            topology="linear",
            router_cores=[(0, 0)],
        ),
    )
    result = d2m.mesh_gather(
        output,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    ).to_host()

    gathered_blocks = []
    for device in range(2):
        for cy in range(grid_y):
            row = slice(cy * block_elements, (cy + 1) * block_elements)
            col = slice(
                device * grid_x * block_elements,
                (device + 1) * grid_x * block_elements,
            )
            gathered_blocks.append(full[row, col])
    gathered = torch.cat(gathered_blocks, dim=0)
    expected = torch.cat([gathered, gathered], dim=1)

    assert result.shape == expected.shape
    assert_pcc(expected, result, threshold=0.99)


@d2m.kernel
def _chunked_matmul_all_gather_1x2(
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
):
    cy = core_index(0)
    cx = core_index(1)
    for chunk in range(num_chunks):
        a = remote_load(lhs, [cy * num_chunks + chunk, cx])
        b = remote_load(rhs, [cy * num_chunks + chunk, cx])
        c = a @ b
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
                    gathered = empty_like(c)
                    gathered = core_read(gathered, c, core=[ty, tx])
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


@requires_fabric_mesh
@pytest.mark.parametrize("num_chunks", [1, 2])
@pytest.mark.parametrize(
    "worker_grid",
    [(1, 1), (2, 1), (2, 2), (4, 4)],
    ids=[
        "single-core",
        "multicore-2x1",
        "multicore-2x2",
        "saturation-4x4",
    ],
)
def test_chunked_matmul_all_gather_round_trip_1x2(num_chunks, worker_grid):
    """Overlap chunk t's fabric send with chunk t+1's compute across devices."""
    torch.manual_seed(0)
    d2m.mesh((1, 2), topology=("linear", "linear"))
    grid_y, grid_x = worker_grid
    saturation = worker_grid == (4, 4)
    layout_dtype = d2m.bfloat16 if saturation else d2m.float32
    torch_dtype = torch.bfloat16 if saturation else torch.float32
    block_tiles = 4
    block_elements = block_tiles * 32
    input_layout = d2m.Layout(
        shape=(grid_y * num_chunks * block_elements, grid_x * block_elements),
        dtype=layout_dtype,
        block_shape=[block_tiles, block_tiles],
        grid_shape=[grid_y * num_chunks, grid_x],
    )
    output_layout = d2m.Layout(
        shape=(grid_y * 2 * num_chunks * block_elements, grid_x * block_elements),
        dtype=layout_dtype,
        block_shape=[block_tiles, block_tiles],
        grid_shape=[2 * grid_y * num_chunks, grid_x],
    )
    full_lhs = (
        torch.randn(
            grid_y * num_chunks * block_elements,
            2 * grid_x * block_elements,
            dtype=torch_dtype,
        )
        * 0.125
    )
    full_rhs = torch.randn(
        grid_y * num_chunks * block_elements,
        2 * grid_x * block_elements,
        dtype=torch_dtype,
    )
    full_rhs *= 0.125
    lhs = d2m.mesh_shard(
        full_lhs,
        input_layout,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    )
    rhs = d2m.mesh_shard(
        full_rhs,
        input_layout,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    )
    output = d2m.empty(output_layout)
    _chunked_matmul_all_gather_1x2(
        lhs,
        rhs,
        output,
        d2m.global_semaphore(),
        d2m.global_semaphore(init=0),
        d2m.global_semaphore(init=0),
        d2m.global_semaphore(),
        num_chunks,
        grid_y,
        grid_x,
        grid_y * grid_x,
        grid=worker_grid,
        fabric=d2m.fabric_config(
            cluster_axis=1,
            topology="linear",
            router_cores=[(0, 0)],
        ),
    )
    result = d2m.mesh_gather(
        output,
        shard_dims=[0, 1],
        shard_shape=[1, 2],
    ).to_host()

    gathered_blocks = []
    for device in range(2):
        for cy in range(grid_y):
            for chunk in range(num_chunks):
                row_index = cy * num_chunks + chunk
                row = slice(
                    row_index * block_elements, (row_index + 1) * block_elements
                )
                row_blocks = []
                for cx in range(grid_x):
                    col_index = device * grid_x + cx
                    col = slice(
                        col_index * block_elements,
                        (col_index + 1) * block_elements,
                    )
                    row_blocks.append(full_lhs[row, col] @ full_rhs[row, col])
                gathered_blocks.append(torch.cat(row_blocks, dim=1))
    gathered = torch.cat(gathered_blocks, dim=0)
    expected = torch.cat([gathered, gathered], dim=1)

    assert result.shape == expected.shape
    assert_pcc(expected, result, threshold=0.99)
