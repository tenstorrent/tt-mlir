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
from runner import KernelBench, TensorSpec, compute_pcc, layout_from_spec, run_bench

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


def test_mesh_configuration():
    d2m.mesh((1, 2), topology=("linear", "ring"))
    builder = _Builder.get()

    assert '#ttcore.meshes<[<"mesh" = 1x2>]>' in str(builder.module.operation)
    assert builder._mesh_shape == [1, 2]
    assert builder._mesh_topology == ["linear", "ring"]


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


# ----------------------------------------------------------------------
# Autotunable mesh kernel bench.
#
# The mesh mapping (mesh shape, topology, shard dims/factors) is fixed on the
# materializer -- it is not a swept knob.  The autotuner's grid/block/mem
# knobs apply PER SHARD: every device runs the same kernel with the same
# config on its own shard of the full tensor.
#
# Autotuner caveats for mesh benches:
# * `TensorSpec.shape` is the FULL tensor shape (what `make_inputs` generates
#   and `mesh_shard` consumes), but the kernel executes on per-device shards,
#   so auto-generated grid/block candidates are derived from the wrong tile
#   counts.  Pass explicit `AutotuneKnobs.grid_shapes` valid for the shard;
#   infeasible configs fail loudly on the divisibility asserts below.
# * `kernel_ns` from a multi-chip profiler trace is the max over devices
#   (perf-analyzer groups spans per device before reducing, since each chip's
#   cycle counter is independent); the per-device breakdown is on
#   `AutotuneResult.device_kernel_ns`.
# ----------------------------------------------------------------------

_MESH_SHAPE = (1, 2)
_MESH_TOPOLOGY = ("linear", "ring")
_SHARD_DIMS = [0, 1]
_SHARD_SHAPE = [1, 2]


@d2m.kernel
def mesh_sigmoid_kernel(input_, output, m_blocks, n_blocks):
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


def mesh_sigmoid_run(kernel, inputs, tensors, grid_shape):
    """Mesh materializer: shard the full input across the device mesh, run the
    kernel on every device's shard, gather back to the full tensor.

    The per-device shard layout is derived from the spec's full shape via
    ``_SHARD_SHAPE``; swept ``block_shape``/``mem_space`` reach the shard
    Layout through ``layout_from_spec`` and ``grid_shape`` through the kernel
    call, mirroring ``eltwise_block_run``'s contract.
    """
    assert len(tensors) == 1, "mesh_sigmoid_run materializes exactly one input"
    ts = tensors[0]
    full = inputs[0]
    gy, gx = grid_shape

    for dim, factor in zip(ts.shape, _SHARD_SHAPE):
        assert (
            dim % factor == 0
        ), f"full shape {ts.shape} not divisible by shard factors {_SHARD_SHAPE}"
    shard_h = ts.shape[-2] // _SHARD_SHAPE[-2]
    shard_w = ts.shape[-1] // _SHARD_SHAPE[-1]
    assert shard_h % 32 == 0, f"shard height {shard_h} is not tile-aligned"
    assert shard_w % 32 == 0, f"shard width {shard_w} is not tile-aligned"
    bm, bn = ts.block_shape[0], ts.block_shape[1]
    tiles_m, tiles_n = shard_h // 32, shard_w // 32
    assert (
        tiles_m % bm == 0
    ), f"shard M tiles ({tiles_m}) not divisible by block_shape[0]={bm}"
    assert (
        tiles_n % bn == 0
    ), f"shard N tiles ({tiles_n}) not divisible by block_shape[1]={bn}"
    blocks_m, blocks_n = tiles_m // bm, tiles_n // bn
    assert (
        blocks_m % gy == 0
    ), f"shard M blocks ({blocks_m}) not divisible by grid_shape[0]={gy}"
    assert (
        blocks_n % gx == 0
    ), f"shard N blocks ({blocks_n}) not divisible by grid_shape[1]={gx}"

    d2m.mesh(_MESH_SHAPE, topology=_MESH_TOPOLOGY)
    layout = layout_from_spec(ts, grid_shape=[gy, gx], shape=(shard_h, shard_w))
    sharded = d2m.mesh_shard(
        full,
        layout,
        shard_dims=_SHARD_DIMS,
        shard_shape=_SHARD_SHAPE,
    )
    output = d2m.empty(layout)
    kernel(sharded, output, blocks_m // gy, blocks_n // gx, grid=(gy, gx))
    gathered = d2m.mesh_gather(
        output,
        shard_dims=_SHARD_DIMS,
        shard_shape=_SHARD_SHAPE,
    )
    return gathered.to_host()


# Full (128, 256) f32 -> per-device shard (128, 128) = 4x4 tiles, so valid
# per-shard grids with block [1, 1] are all divisor pairs of (4, 4).
KERNEL_BENCHES = {
    "mesh_sigmoid": KernelBench(
        kernel=mesh_sigmoid_kernel,
        golden=torch.sigmoid,
        run=mesh_sigmoid_run,
        tensors=[
            TensorSpec(
                shape=(128, 256),
                block_shape=[1, 1],
                dtype=torch.float32,
                dist="uniform(-2,2)",
            )
        ],
        grid_shape=(2, 2),
    )
}


@requires_mesh
def test_mesh_sigmoid_bench_round_trip():
    bench = KERNEL_BENCHES["mesh_sigmoid"]
    actual, expected = run_bench(bench)
    assert actual.shape == expected.shape
    assert compute_pcc(expected, actual) >= bench.pcc
