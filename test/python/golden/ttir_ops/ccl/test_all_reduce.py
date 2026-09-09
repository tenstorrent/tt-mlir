# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from collections import OrderedDict
from typing import Tuple

from builder.base.builder_utils import Operand, Shape
from builder.base.builder_enums import (
    MeshShardDirection,
    MeshShardType,
    ReduceType,
)
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs
from test_utils import shape_str, make_shard_shape

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 32, 32, 32),
        (1, 32, 32, 1),
        (32, 32, 1, 1),
        (1, 32, 32),
        (32, 32),
        (32, 40),
        (40, 32),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (1, 32), (8, 4)], ids=shape_str
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize(
    "reduce_type",
    # TTIR currently only supports Sum for all_reduce.
    [ReduceType.Sum],
    ids=["sum"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_reduce(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    reduce_type: ReduceType,
    dtype: torch.dtype,
    request,
    device,
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("all_reduce across 1 device is meaningless")

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
        def all_reduce(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            all_reduce0 = builder.all_reduce(
                in_shard,
                cluster_axis=cluster_axis,
                reduce_type=reduce_type.value,
            )

            return builder.mesh_shard(
                all_reduce0,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("test_size", [32, 64, 128])
@pytest.mark.parametrize(
    "mesh_shape",
    [(1, 2), (1, 4), (1, 8), (2, 1), (4, 1), (8, 1)],
    ids=shape_str,
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize(
    "reduce_type",
    # TTIR currently only supports Sum for all_reduce.
    [ReduceType.Sum],
    ids=["sum"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_reduce_1d(
    test_size: int,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    reduce_type: ReduceType,
    dtype: torch.dtype,
    request,
    device,
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("all_reduce across 1 device is meaningless")

    # The runtime mesh_shard cannot handle 1D tensors with a 2D mesh (the
    # ShardToFull composer requires unique dims, but a 1D tensor only has dim 0).
    # Work around this by sharding a 2D (1, N) tensor, reshaping to 1D before
    # all_reduce (which exercises the compiler's 1D reshape workaround), then
    # reshaping back to 2D to unshard.
    shard_dims = [0, 1]
    shard_shape_2d = make_shard_shape(2, shard_dims, mesh_shape)

    full_input_shape = [1 * mesh_shape[0], test_size * mesh_shape[1]]
    shard_test_shape_1d = [test_size]
    # all_reduce keeps the per-device shape (unlike all_gather which grows it),
    # so reshape back to the shard shape, not the full shape.
    reduced_shape_2d = [1, test_size]

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_reduce(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape_2d,
                shard_dims=shard_dims,
            )

            # Reshape to 1D to exercise the all_reduce 1D workaround.
            in_1d = builder.reshape(in_shard, shape=shard_test_shape_1d)

            all_reduce0 = builder.all_reduce(
                in_1d,
                cluster_axis=cluster_axis,
                reduce_type=reduce_type.value,
            )

            # Reshape back to 2D so mesh_shard can unshard.
            out_2d = builder.reshape(all_reduce0, shape=reduced_shape_2d)

            return builder.mesh_shard(
                out_2d,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape_2d,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
    )
