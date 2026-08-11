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
from runner import (
    KernelBench,
    MeshSpec,
    TensorSpec,
    compute_pcc,
    mesh_block_run,
    run_bench,
)

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
# The bench declares its mesh (`MeshSpec`) and the joint sharding strategies
# its semantics allow (`shard_strategies`); the stock `mesh_block_run`
# materializer shards each input per its spec's `shard_dims`, runs the same
# per-shard grid/block/mem config on every device, and gathers the result.
# The autotuner sweeps (strategy × grid × block × mem), deriving grid/block
# candidates from each strategy's per-device shard shapes.
#
# `kernel_ns` from a multi-chip profiler trace is the max over devices
# (perf-analyzer groups spans per device before reducing, since each chip's
# cycle counter is independent); the per-device breakdown is on
# `AutotuneResult.device_kernel_ns`.
# ----------------------------------------------------------------------


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


# Full (128, 256) f32.  Default strategy shards columns: per-device shard
# (128, 128) = 4x4 tiles, so valid per-shard grids with block [1, 1] are all
# divisor pairs of (4, 4).  Sigmoid is elementwise, so any partition is
# semantically legal — declared below for the autotuner.
#
# Runtime caveat: the gather path (`tensorShardToFull` -> `concat_ndim` in
# runtime/lib/ttmetal/meshshard_utils.cpp) does not skip `-1` (replicate)
# entries the way the shard path does — `-1` wraps to the last tensor dim and
# collides with it ("dims must be unique").  So a tensor whose mapping is
# gathered must avoid `-1`: the extent-1 mesh axis is mapped to a real dim
# (factor 1, a no-op) instead, and a fully-replicated strategy is not
# declared (full replication is a separate shard_type the d2m-jit builder
# does not emit yet).
KERNEL_BENCHES = {
    "mesh_sigmoid": KernelBench(
        kernel=mesh_sigmoid_kernel,
        golden=torch.sigmoid,
        run=mesh_block_run,
        tensors=[
            TensorSpec(
                shape=(128, 256),
                block_shape=[1, 1],
                dtype=torch.float32,
                dist="uniform(-2,2)",
                shard_dims=[0, 1],
            )
        ],
        grid_shape=(2, 2),
        mesh=MeshSpec(shape=(1, 2), topology=("linear", "ring")),
        shard_strategies={
            "cols": [[0, 1]],
            "rows": [[1, 0]],
        },
    )
}


@requires_mesh
def test_mesh_sigmoid_bench_round_trip():
    bench = KERNEL_BENCHES["mesh_sigmoid"]
    actual, expected = run_bench(bench)
    assert actual.shape == expected.shape
    assert compute_pcc(expected, actual) >= bench.pcc
