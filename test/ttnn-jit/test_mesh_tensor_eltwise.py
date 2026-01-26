# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    all_close_check,
    memory_configs_equal,
    create_dram_tensor,
)
from op_definitions import exp, cosh, add

DRAM_INTERLEAVED_SHAPES = [
    (32, 32),
    (2048, 2048),
    (1024, 32),
    (32, 1024),
]


@pytest.mark.parametrize("shape", DRAM_INTERLEAVED_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "op, num_inputs, check_interop",
    [(exp, 1, False), (add, 2, False), (cosh, 1, False), (cosh, 1, True)],
)
@pytest.mark.parametrize(
    "device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}], indirect=True
)  # col dispatch axis fails
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "mesh_mapper_func, dim_arg",
    [
        (ttnn.ReplicateTensorToMesh, None),
        (ttnn.ShardTensorToMesh, 0),
        (ttnn.ShardTensorToMesh, 1),
        (ttnn.ShardTensor2dMesh, (None, 0)),
        (ttnn.ShardTensor2dMesh, (None, 1)),
        (ttnn.ShardTensor2dMesh, (0, None)),
        (ttnn.ShardTensor2dMesh, (1, None)),
        (ttnn.ShardTensor2dMesh, (0, 1)),
        (ttnn.ShardTensor2dMesh, (1, 0)),
    ],
)
def test_mesh_tensor_eltwise(
    shape,
    dtype,
    op,
    num_inputs,
    check_interop,
    mesh_device,
    mesh_mapper_func,
    dim_arg,
):

    if mesh_mapper_func == ttnn.ReplicateTensorToMesh:
        mesh_mapper = mesh_mapper_func(mesh_device=mesh_device)
    elif mesh_mapper_func == ttnn.ShardTensorToMesh:
        mesh_mapper = mesh_mapper_func(mesh_device=mesh_device, dim=dim_arg)
    else:
        mesh_mapper = mesh_mapper_func(
            mesh_device=mesh_device, mesh_shape=mesh_device.shape, dims=dim_arg
        )

    inputs = [
        create_dram_tensor(mesh_device, shape, dtype, mesh_mapper=mesh_mapper)
        for i in range(num_inputs)
    ]

    # JIT path
    enable_cache = False
    op_jit = ttnn_jit.jit(
        debug=True,
        enable_cache=enable_cache,
    )(op)
    interop_result = op_jit(*inputs)

    # Golden path (regular ttnn)
    golden_result = op(*inputs)

    # Run a regular ttnn op (ttnn.sum) using jit output to check interop between ttnn jit and ttnn
    if check_interop:
        interop_result = ttnn.sum(interop_result, dim=0)
        golden_result = ttnn.sum(golden_result, dim=0)

    assert memory_configs_equal(
        interop_result.memory_config(), golden_result.memory_config()
    )

    # compare each device shard
    interop_result_shards = ttnn.get_device_tensors(interop_result.cpu())
    golden_result_shards = ttnn.get_device_tensors(golden_result.cpu())
    assert len(interop_result_shards) == len(golden_result_shards)
    for interop_result_shard, golden_result_shard in zip(
        interop_result_shards, golden_result_shards
    ):
        assert interop_result_shard.shape == golden_result_shard.shape
        assert all_close_check(interop_result_shard, golden_result_shard)
