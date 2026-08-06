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
