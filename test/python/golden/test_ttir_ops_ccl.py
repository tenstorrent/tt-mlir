# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from typing import List, Tuple
from collections import OrderedDict
from functools import reduce
import operator

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from test_utils import make_shard_shape



@pytest.mark.parametrize(
    "input_rank, shard_dims",
    [
        (5, (1, 4)),
        (5, (4, 1)),
        (5, (2, 4)),
        (5, (1, 4)),
        (5, (-1, 3)),
        (5, (4, -1)),
        (5, (-1, 4)),
        (5, (-1, 0)),
        (4, (1, 3)),
        (4, (3, 1)),
        (4, (2, 3)),
        (4, (3, 2)),
        (4, (0, 2)),
        (4, (1, 0)),
        (4, (-1, 3)),
        (4, (3, -1)),
        (4, (-1, 1)),
        (4, (1, -1)),
        (3, (1, 2)),
        (3, (2, 1)),
        (3, (0, 1)),
        (3, (1, 0)),
        (3, (-1, 2)),
        (3, (2, -1)),
        (3, (-1, 1)),
        (3, (0, -1)),
        (2, (0, 1)),
        (2, (1, 0)),
        (2, (-1, 1)),
        (2, (1, -1)),
        (2, (-1, 0)),
        (2, (0, -1)),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (4, 2), (1, 8), (8, 1), (1, 2), (2, 1)])
def test_mesh_shard_devices(
    input_rank: int, shard_dims: Tuple[int, int], mesh_shape: Tuple[int, int], request
):
    shard_shape = make_shard_shape(input_rank, shard_dims, mesh_shape)
    if all(x == 1 for x in shard_shape):
        pytest.skip("sharding is meaningless, skipping test.")
    input_shape = [n_shards for idx, n_shards in enumerate(shard_shape)]

    def mesh_shard_devices(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )

    compile_ttir_to_flatbuffer(
        mesh_shard_devices,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 32, 32, 32),
        (1, 32, 32),
        (32, 32),
        (1, 32, 32, 34),
        (1, 32, 32, 30),
        (1, 32, 34),
        (1, 32, 32),
        (32, 34),
        (32, 32),
        (1, 32, 32, 1),
        (32, 32, 1, 1),
        (128, 256),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)])
@pytest.mark.parametrize("all_gather_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_all_gather(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    all_gather_dim: int,
    cluster_axis: int,
    request,
    shard_wrap_factory,
):
    if all_gather_dim >= len(test_shape):
        pytest.skip("all_gather_dim is out of range")
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("all_gather across 1 device is meaningless")

    def all_gather(sharded_in: Operand, builder: TTIRBuilder):
        return builder.all_gather(
            sharded_in,
            all_gather_dim=all_gather_dim,
            cluster_axis=cluster_axis,
        )

    input_shape, test_fn = shard_wrap_factory(all_gather)

    compile_ttir_to_flatbuffer(
        test_fn,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 1, 256, 256),
        (1, 256, 256),
        (256, 256),
        (1, 1, 256, 257),
        (1, 1, 256, 255),
        (256, 257),
        (256, 255),
        (1, 256, 256, 1),
        (256, 256, 1, 1),
        (1024, 256),
        (256, 1024),
        pytest.param(
            (1, 1, 32, 64), marks=pytest.mark.run_error
        ),  # https://github.com/tenstorrent/tt-metal/issues/21987
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)])
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_all_reduce(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    request,
    shard_wrap_factory,
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("CCL across 1 device is meaningless")

    # test 'sum' only for now. Other reduce types are not supported yet.
    def all_reduce(sharded_in: Operand, builder: TTIRBuilder):
        return builder.all_reduce(
            sharded_in,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=cluster_axis,
        )

    input_shape, test_fn = shard_wrap_factory(all_reduce)

    compile_ttir_to_flatbuffer(
        test_fn,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 1, 256, 256),
        (1, 256, 256),
        (256, 256),
        pytest.param(
            (256, 248), marks=pytest.mark.run_error
        ),  # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param((248, 256), marks=pytest.mark.run_error),
        pytest.param((256, 264), marks=pytest.mark.run_error),
        pytest.param((264, 256), marks=pytest.mark.run_error),
        pytest.param((1, 1, 128, 256), marks=pytest.mark.run_error),
        pytest.param((1, 1, 256, 128), marks=pytest.mark.run_error),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)])
@pytest.mark.parametrize("scatter_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_reduce_scatter(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    scatter_dim: int,
    cluster_axis: int,
    request,
    shard_wrap_factory,
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("CCL across 1 device is meaningless")
    if scatter_dim >= len(test_shape):
        pytest.skip("scatter_dim is out of range")
    if scatter_dim != len(test_shape) - 1:
        pytest.skip("Known issue : Reduce Scater produces incorrect output")
        # https://github.com/tenstorrent/tt-metal/issues/19433

    # test 'sum' only for now. Other reduce types are not supported yet.
    def reduce_scatter(sharded_in: Operand, builder: TTIRBuilder):
        return builder.reduce_scatter(
            sharded_in,
            reduce_type="#ttcore.reduce_type<sum>",
            scatter_dim=scatter_dim,
            cluster_axis=cluster_axis,
        )

    input_shape, test_fn = shard_wrap_factory(reduce_scatter)

    compile_ttir_to_flatbuffer(
        test_fn,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 1, 256, 512),
        (1, 256, 512),
        (256, 512),
        (256, 512, 1, 1),
        (1, 256, 512, 1),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)])
@pytest.mark.parametrize(
    "source_target_pairs",
    [
        pytest.param(
            [(0, 1)], marks=pytest.mark.fails_golden
        ),  # https://github.com/tenstorrent/tt-mlir/issues/4323
        [(0, 1), (1, 0)],
        [(0, 1), (1, 2), (2, 3), (3, 0)],
        [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)],
        [(0, 4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3)],
        [(0, 2), (1, 3), (4, 6), (5, 7), (2, 0), (3, 1), (6, 4), (7, 5)],
        [(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],
    ],
)
def test_collective_permute(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    source_target_pairs: List[Tuple[int, int]],
    request,
    shard_wrap_factory,
):
    max_id = reduce(operator.mul, mesh_shape, 1)
    if not all(pair[0] < max_id and pair[1] < max_id for pair in source_target_pairs):
        pytest.skip("Source and target pairs are out of range")

    def collective_permute(sharded_in: Operand, builder: TTIRBuilder):
        return builder.collective_permute(
            sharded_in,
            source_target_pairs=source_target_pairs,
        )

    input_shape, test_fn = shard_wrap_factory(collective_permute)

    compile_ttir_to_flatbuffer(
        test_fn,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (256, 128),
        (32, 64, 128),
        (8, 8, 64, 64),
    ],
)
@pytest.mark.parametrize("split_dim", range(4))
@pytest.mark.parametrize("concat_dim", range(4))
@pytest.mark.parametrize(
    "mesh_shape, replica_groups",
    [
        ((1, 8), ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ((2, 4), ((0, 4), (1, 5), (2, 6), (3, 7))),
        ((2, 4), ((0, 1, 2, 3), (4, 5, 6, 7))),
        ((4, 2), ((0, 2, 4, 6), (1, 3, 5, 7))),
        ((4, 2), ((0, 1), (2, 3), (4, 5), (6, 7))),
        ((1, 2), ((0, 1),)),
        ((2, 1), ((0, 1),)),
    ],
)
def test_all_to_all(
    test_shape: Shape,
    split_dim,
    concat_dim,
    mesh_shape,
    replica_groups,
    request,
    shard_wrap_factory,
):
    split_count = len(replica_groups[0])
    if split_dim >= len(test_shape):
        pytest.skip("Split dimension is out of range")
    if concat_dim >= len(test_shape):
        pytest.skip("Concat dimension is out of range")

    def all_to_all(sharded_in: Operand, builder: TTIRBuilder):
        return builder.all_to_all(
            sharded_in,
            split_dim=split_dim,
            concat_dim=concat_dim,
            split_count=split_count,
            replica_groups=replica_groups,
        )

    input_shape, test_fn = shard_wrap_factory(all_to_all)

    compile_ttir_to_flatbuffer(
        test_fn,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (256, 128),
        (32, 128, 64),
        (8, 8, 32, 64),
        (10, 10, 30, 60),
    ],
)
@pytest.mark.parametrize(
    "mesh_shape, replica_groups",
    [
        ((2, 4), [(0, 1, 2, 3), (4, 5, 6, 7)]),
        ((2, 4), [(0, 4), (1, 5), (2, 6), (3, 7)]),
        ((4, 2), [(0, 1), (2, 3), (4, 5), (6, 7)]),
        ((4, 2), [(0, 2, 4, 6), (1, 3, 5, 7)]),
        ((1, 8), [(0, 1, 2, 3, 4, 5, 6, 7)]),
        ((1, 2), ((0, 1),)),
        ((2, 1), ((0, 1),)),
    ],
)
def test_collective_broadcast(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    replica_groups,
    request,
    shard_wrap_factory,
):
    def collective_broadcast(sharded_in: Operand, builder: TTIRBuilder):
        return builder.collective_broadcast(
            sharded_in,
            replica_groups=replica_groups,
        )

    input_shape, test_fn = shard_wrap_factory(collective_broadcast)

    compile_ttir_to_flatbuffer(
        test_fn,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
