# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only, get_request_kwargs
import _ttmlir_runtime as tt_runtime

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir, build_module
from builder.base.builder_enums import *
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
    make_shard_shape,
    shard_wrap_factory,
)

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "test_shape",
    [
        (128, 128),
        (256, 256),
        # (1, 32, 32, 32),
        # (1, 32, 32, 1),
        # (32, 32, 1, 1),
        # (1, 32, 32),
        # (32, 32),
        # (32, 40),
        # (40, 32),
        # pytest.param((1, 1, 32, 32, 32), marks=pytest.mark.xfail(reason="run error")),
        # pytest.param(
        #    (1, 1, 1, 1, 1, 1, 32, 32, 32), marks=pytest.mark.xfail(reason="run error")
        # ),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        (1, 8),
        # (1, 2),
        # (1, 32),
        # (8, 4)
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("all_gather_dim", range(4))
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING]
)
def test_all_gather(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    all_gather_dim: int,
    cluster_axis: int,
    dtype: torch.dtype,
    target: str,
    fabric_config: tt_runtime.runtime.FabricConfig,
    request,
    device,
):
    if all_gather_dim >= len(test_shape):
        pytest.skip("all_gather_dim is out of range")
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("all_gather across 1 device is meaningless")

    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_gather(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            all_gather0 = builder.all_gather(
                in_shard,
                all_gather_dim=all_gather_dim,
                cluster_axis=cluster_axis,
            )

            return builder.mesh_shard(
                all_gather0,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    pipeline_options = [
        f"mesh-topology=linear,ring",
    ]

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=pipeline_options,
        **get_request_kwargs(request),
    )

@pytest.mark.parametrize(
    "test_shape",
    [
        (256, 256),
        #(1, 1, 256, 256),
        #(1, 1, 256, 257),
        #(1, 1, 256, 255),
        #(1, 256, 256, 1),
        #(256, 256, 1, 1),
        #(1, 1, 32, 64),
        #(1, 128, 128),
        #(128, 128),
        #(128, 129),
        #(64, 128),
        #(64, 24),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [
        #(2, 4), 
        (1, 8), 
        #(1, 2), 
        #(1, 32), 
        #(8, 4)
    ], 
    ids=shape_str
)
@pytest.mark.parametrize("scatter_dim", range(4))
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING]
)
def test_reduce_scatter(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    scatter_dim: int,
    cluster_axis: int,
    dtype: torch.dtype,
    target: str,
    fabric_config: tt_runtime.runtime.FabricConfig,
    request,
    device,
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("CCL across 1 device is meaningless")
    if scatter_dim >= len(test_shape):
        pytest.skip("scatter_dim is out of range")
    if test_shape[scatter_dim] % mesh_shape[cluster_axis] != 0:
        pytest.skip("scatter_dim is not divisible by mesh_shape[cluster_axis]")

    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    # test 'sum' only for now. Other reduce types are not supported yet.
    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def reduce_scatter(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            reduce_scatter0 = builder.reduce_scatter(
                in_shard,
                reduce_type=ReduceType.Sum.value,
                scatter_dim=scatter_dim,
                cluster_axis=cluster_axis,
            )

            return builder.mesh_shard(
                reduce_scatter0,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    pipeline_options = [
        f"mesh-topology=linear,ring",
    ]

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=pipeline_options,
        **get_request_kwargs(request),
        print_ir=True,
        save_artifacts=True,
    )
