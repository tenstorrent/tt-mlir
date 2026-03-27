# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from collections import OrderedDict
from typing import Tuple

from builder.base.builder_utils import Operand, Shape
from builder.base.builder_enums import MeshShardDirection, MeshShardType
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs
from test_utils import shape_str, make_shard_shape

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "test_shape",
    [
        (2, 32, 32, 32),
        (32, 32, 32, 32),
        (4, 32, 32),
        (32, 32, 32),
        (8, 32),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("mesh_shape", [(1, 2), (1, 8), (2, 4)], ids=shape_str)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_mesh_partition(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    dim: int,
    cluster_axis: int,
    dtype: torch.dtype,
    request,
    device,
):
    if dim >= len(test_shape):
        pytest.skip("dim is out of range")
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("mesh_partition across 1 device is meaningless")

    num_partitions = mesh_shape[cluster_axis]
    if test_shape[dim] % num_partitions != 0:
        pytest.skip("dim size not divisible by number of partitions")

    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        pytest.skip("mesh rank exceeds tensor rank")

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))

    if dim in shard_dims:
        pytest.skip("partition dim overlaps with shard dims")
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def mesh_partition(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            partitioned = builder.mesh_partition(
                in_shard,
                dim=dim,
                cluster_axis=cluster_axis,
            )

            return builder.mesh_shard(
                partitioned,
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
