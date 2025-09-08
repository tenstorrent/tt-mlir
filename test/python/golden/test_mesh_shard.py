# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Tuple, OrderedDict, Callable

import itertools

from builder.base.builder import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")

MAX_TEST_RANK = 5
MIN_TEST_RANK = 2


def devices_devices(
    in0: Operand,
    builder: TTIRBuilder,
    shard_shape: List[int],
    shard_dims: List[int],
):
    sharded_in0 = builder.mesh_shard(
        in0,
        shard_direction="#ttcore.shard_direction<full_to_shard>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape,
        shard_dims=shard_dims,
    )
    neg_output = builder.neg(sharded_in0)
    output = builder.mesh_shard(
        neg_output,
        shard_direction="#ttcore.shard_direction<shard_to_full>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape,
        shard_dims=shard_dims,
    )
    input_tensor = builder._get_golden_tensor(in0)
    golden_output_tensor = torch.neg(input_tensor)
    builder.set_graph_input_output([input_tensor], [golden_output_tensor])

    return output


def devices_replicate(
    in0: Operand,
    builder: TTIRBuilder,
    shard_shape: List[int],
    shard_dims: List[int],
):
    sharded_in0 = builder.mesh_shard(
        in0,
        shard_direction="#ttcore.shard_direction<full_to_shard>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape,
        shard_dims=shard_dims,
    )
    neg_output = builder.neg(sharded_in0)
    output = builder.mesh_shard(
        neg_output,
        shard_direction="#ttcore.shard_direction<shard_to_full>",
        shard_type="#ttcore.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
    )

    input_tensor = builder._get_golden_tensor(in0)
    golden_output_tensor = torch.neg(input_tensor)
    # extracting the first block of the golden output tensor
    for idx, shard_size in enumerate(shard_shape):
        if shard_size > 1:
            golden_output_tensor = torch.chunk(
                golden_output_tensor, shard_size, dim=idx
            )[0]
    builder.set_graph_input_output([input_tensor], [golden_output_tensor])

    return output


def replicate_devices(
    in0: Operand,
    builder: TTIRBuilder,
    shard_shape: List[int],
    shard_dims: List[int],
):
    sharded_in0 = builder.mesh_shard(
        in0,
        shard_direction="#ttcore.shard_direction<full_to_shard>",
        shard_type="#ttcore.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
    )
    neg_output = builder.neg(sharded_in0)
    output = builder.mesh_shard(
        neg_output,
        shard_direction="#ttcore.shard_direction<shard_to_full>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape,
        shard_dims=shard_dims,
    )

    input_tensor = builder._get_golden_tensor(in0)
    golden_output_tensor = torch.neg(input_tensor)
    # Replicate the golden output tensor block to match shard_shape
    for idx, shard_size in enumerate(shard_shape):
        if shard_size > 1:
            golden_output_tensor = torch.cat(
                [golden_output_tensor.clone() for _ in range(shard_size)],
                dim=idx,
            )
    builder.set_graph_input_output([input_tensor], [golden_output_tensor])

    return output


@pytest.mark.parametrize(
    "input_rank",
    range(MIN_TEST_RANK, MAX_TEST_RANK + 1),
)
@pytest.mark.parametrize(
    "shard_dim_0",
    range(-1, MAX_TEST_RANK),
)
@pytest.mark.parametrize(
    "shard_dim_1",
    range(-1, MAX_TEST_RANK),
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (4, 2), (1, 8), (8, 1), (1, 2), (2, 1)], ids=shape_str
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize("tile_size", [1, 32])
@pytest.mark.parametrize(
    "test_case",
    [devices_devices, devices_replicate, replicate_devices],
    ids=lambda f: f.__name__,
)
def test_mesh_shard_devices(
    input_rank: int,
    shard_dim_0: int,
    shard_dim_1: int,
    mesh_shape: Tuple[int, int],
    target: str,
    tile_size: int,
    test_case: Callable,
    request,
):
    if shard_dim_0 >= input_rank or shard_dim_1 >= input_rank:
        pytest.skip("shard_dim is out of range, skipping test.")
    if shard_dim_0 == shard_dim_1:
        pytest.skip("shard_dims need to be unique, skipping test.")

    # Generate shard_shape from shard_dims.
    shard_dims = (shard_dim_0, shard_dim_1)
    shard_shape = [1] * input_rank
    for mesh_axis, tensor_dim in enumerate(shard_dims):
        if tensor_dim >= 0:
            shard_shape[tensor_dim] = mesh_shape[mesh_axis]
    if all(x == 1 for x in shard_shape):
        pytest.skip("sharding is meaningless, skipping test.")

    # Generate input_shape according to shard_shape.
    # For the last two dimensions, ensure they are multiples of tile_size.
    input_shape = [
        n_shards if idx < input_rank - 2 else tile_size * n_shards
        for idx, n_shards in enumerate(shard_shape)
    ]
    # In the case of the replicate_devices test_case,
    # the input is replicated so there is no need to increase the input according to shard_shape,
    # but for code simplicity, we just test it this way.

    test_fn = lambda inputs, builder: test_case(
        inputs, builder, shard_shape, shard_dims
    )
    test_fn.__name__ = test_case.__name__

    compile_ttir_to_flatbuffer(
        test_fn,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@pytest.mark.parametrize(
    "input_rank",
    range(MIN_TEST_RANK + 1, MAX_TEST_RANK + 1),
)
@pytest.mark.parametrize(
    "shard_dim_0, shard_dim_1, shard_dim_2",
    list(itertools.permutations(range(-1, MAX_TEST_RANK), 3)),
)
@pytest.mark.parametrize("mesh_shape", [(2, 2, 2), (1, 1, 8), (1, 1, 2), (1, 2, 4)])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize("tile_size", [1, 32])
@pytest.mark.parametrize(
    "test_case",
    [devices_devices, devices_replicate, replicate_devices],
    ids=lambda f: f.__name__,
)
def test_mesh_shard_3D(
    input_rank: int,
    shard_dim_0: int,
    shard_dim_1: int,
    shard_dim_2: int,
    mesh_shape: Tuple[int, ...],
    target: str,
    tile_size: int,
    test_case: Callable,
    request,
):
    if (
        shard_dim_0 >= input_rank
        or shard_dim_1 >= input_rank
        or shard_dim_2 >= input_rank
    ):
        pytest.skip("shard_dim is out of range, skipping test.")
    if (
        shard_dim_0 == shard_dim_1
        or shard_dim_0 == shard_dim_2
        or shard_dim_1 == shard_dim_2
    ):
        pytest.skip("shard_dims need to be unique, skipping test.")

    # Generate shard_shape from shard_dims.
    shard_dims = (shard_dim_0, shard_dim_1, shard_dim_2)
    shard_shape = [1] * input_rank
    for mesh_axis, tensor_dim in enumerate(shard_dims):
        if tensor_dim >= 0:
            shard_shape[tensor_dim] = mesh_shape[mesh_axis]
    if all(x == 1 for x in shard_shape):
        pytest.skip("sharding is meaningless, skipping test.")

    # Generate input_shape according to shard_shape.
    # For the last two dimensions, ensure they are multiples of tile_size.
    input_shape = [
        n_shards if idx < input_rank - 2 else tile_size * n_shards
        for idx, n_shards in enumerate(shard_shape)
    ]
    # In the case of the replicate_devices test_case,
    # the input is replicated so there is no need to increase the input according to shard_shape,
    # but for code simplicity, we just test it this way.

    test_fn = lambda inputs, builder: test_case(
        inputs, builder, shard_shape, shard_dims
    )
    test_fn.__name__ = test_case.__name__

    compile_ttir_to_flatbuffer(
        test_fn,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict(
            [("x", mesh_shape[0]), ("y", mesh_shape[1]), ("z", mesh_shape[2])]
        ),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )
