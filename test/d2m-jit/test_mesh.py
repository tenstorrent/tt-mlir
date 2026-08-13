# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder


pytestmark = pytest.mark.machines("n300")


@pytest.mark.device_only(
    reason="asserts device-builder MLIR state (the ttcore.meshes module attr); "
    "the sim mesh declaration only records the layout_math mirror "
    "(SIMULATOR_SPEC.md §14.2)"
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


@pytest.mark.device_only(
    reason="mesh_shard has no sim backing yet (SIMULATOR_SPEC.md §14.3, #9202)"
)
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


@pytest.mark.device_only(
    reason="mesh_shard has no sim backing yet (SIMULATOR_SPEC.md §14.3, #9202)"
)
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
