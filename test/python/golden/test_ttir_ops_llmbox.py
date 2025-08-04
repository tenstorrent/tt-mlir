# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from typing import List, Tuple
from collections import OrderedDict

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer

pytestmark = [pytest.mark.llmbox, pytest.mark.frontend("ttir")]


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
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8)])
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
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8)])
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
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8)])
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
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8)])
@pytest.mark.parametrize(
    "pairs",
    [
        [(0, 1)],
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
    pairs: List[Tuple[int, int]],
    request,
    shard_wrap_factory,
):
    def collective_permute(sharded_in: Operand, builder: TTIRBuilder):
        return builder.collective_permute(
            sharded_in,
            source_target_pairs=pairs,
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


# TODO: many of these tests can be combined with some logic around `mesh_shape`
@pytest.mark.parametrize(
    "shapes",
    [
        [(8192, 784), (784, 16384)],
        [(1024, 32), (32, 512)],
        [(512, 32), (32, 128)],
        [(1024, 16), (16, 512)],
        [(1024, 8), (8, 512)],
        [(1024, 8), (8, 512)],
        [(256, 128), (128, 128)],
        [(256, 128), (128, 124)],
        [(256, 128), (128, 120)],
        # TODO (#3662), re-enable once tensor spec check
        # accounts for non-uniform shapes due to non-divisibility
        # [(256, 130), (130, 128)],
        [(254, 128), (128, 128)],
        [(252, 128), (128, 128)],
        pytest.param(
            [(258, 128), (128, 128)], marks=pytest.mark.fails_golden
        ),  # https://github.com/tenstorrent/tt-metal/issues/21964
        pytest.param([(260, 128), (128, 128)], marks=pytest.mark.fails_golden),
        pytest.param(
            [(256, 128), (128, 132)], marks=pytest.mark.run_error
        ),  # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param([(256, 128), (128, 136)], marks=pytest.mark.run_error),
        pytest.param([(256, 32), (32, 64)], marks=pytest.mark.run_error),
        pytest.param([(128, 32), (32, 32)], marks=pytest.mark.run_error),
        pytest.param([(64, 32), (32, 16)], marks=pytest.mark.run_error),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_matmul_2x4(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def matmul_2x4(in0: Operand, in1: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(sharded_in0, sharded_in1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )

    compile_ttir_to_flatbuffer(
        matmul_2x4,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # [(8192, 784), (784, 16384)],
        [(1024, 32), (32, 512)],
        [(1024, 16), (16, 512)],
        [(1024, 8), (8, 512)],
        [(256, 128), (128, 124)],
        pytest.param(
            [(256, 128), (128, 132)], marks=pytest.mark.run_error
        ),  # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param([(1024, 8), (8, 512)], marks=pytest.mark.run_error),
        pytest.param([(512, 32), (32, 128)], marks=pytest.mark.run_error),
        pytest.param([(256, 128), (128, 128)], marks=pytest.mark.run_error),
        pytest.param([(256, 128), (128, 120)], marks=pytest.mark.run_error),
        pytest.param([(256, 130), (130, 128)], marks=pytest.mark.run_error),
        pytest.param([(254, 128), (128, 128)], marks=pytest.mark.run_error),
        pytest.param([(252, 128), (128, 128)], marks=pytest.mark.run_error),
        pytest.param([(258, 128), (128, 128)], marks=pytest.mark.run_error),
        pytest.param([(260, 128), (128, 128)], marks=pytest.mark.run_error),
        pytest.param([(256, 128), (128, 136)], marks=pytest.mark.run_error),
        pytest.param([(256, 32), (32, 64)], marks=pytest.mark.run_error),
        pytest.param([(128, 32), (32, 32)], marks=pytest.mark.run_error),
        pytest.param([(64, 32), (32, 16)], marks=pytest.mark.run_error),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 8)])
def test_matmul_1x8(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def matmul_1x8(in0: Operand, in1: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 8),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(8, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(sharded_in0, sharded_in1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    compile_ttir_to_flatbuffer(
        matmul_1x8,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 16, 128),
        (1, 64, 16, 124),
        (1, 64, 16, 120),
        (1, 64, 16, 132),
        (1, 64, 16, 136),
        (1, 62, 16, 128),
        (1, 60, 16, 128),
        (1, 66, 16, 128),
        (1, 68, 16, 128),
        (1, 64, 7, 128),
        (1, 18, 3, 36),
        (1, 2, 1, 4),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_neg_2x4(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_2x4(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2, 1, 4),
            shard_dims=(1, 3),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2, 1, 4),
            shard_dims=(1, 3),
        )

    compile_ttir_to_flatbuffer(
        neg_2x4,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 16, 128),
        (1, 64, 16, 124),
        (1, 64, 16, 120),
        (1, 64, 16, 132),
        (1, 64, 16, 136),
        (1, 62, 16, 128),
        (1, 60, 16, 128),
        (1, 66, 16, 128),
        (1, 68, 16, 128),
        (1, 64, 7, 128),
        (1, 18, 3, 36),
        (1, 2, 1, 4),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_neg_2x4_cluster_0(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_2x4_cluster_0(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(1, -1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(1, -1),
        )

    compile_ttir_to_flatbuffer(
        neg_2x4_cluster_0,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 16, 128),
        (1, 64, 16, 124),
        (1, 64, 16, 120),
        (1, 64, 16, 132),
        (1, 64, 16, 136),
        (1, 62, 16, 128),
        (1, 60, 16, 128),
        (1, 66, 16, 128),
        (1, 68, 16, 128),
        (1, 64, 7, 128),
        (1, 18, 3, 36),
        (1, 2, 1, 4),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_neg_2x4_cluster_1(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_2x4_cluster_1(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 4),
            shard_dims=(1, 3),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 4),
            shard_dims=(1, 3),
        )

    compile_ttir_to_flatbuffer(
        neg_2x4_cluster_1,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 16, 64),
        (1, 124, 16, 64),
        (1, 120, 16, 64),
        (1, 132, 16, 64),
        (1, 136, 16, 64),
        (1, 128, 16, 62),
        (1, 128, 16, 60),
        (1, 128, 16, 66),
        (1, 128, 16, 68),
        (1, 128, 7, 64),
        (1, 36, 3, 18),
        (1, 4, 1, 2),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_neg_2x4_reversed_cluster(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_2x4_reversed_cluster(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 4, 1, 2),
            shard_dims=(3, 1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 4, 1, 2),
            shard_dims=(3, 1),
        )

    compile_ttir_to_flatbuffer(
        neg_2x4_reversed_cluster,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 16, 64),
        (1, 124, 16, 64),
        (1, 120, 16, 64),
        (1, 132, 16, 64),
        (1, 136, 16, 64),
        (1, 128, 16, 62),
        (1, 128, 16, 60),
        (1, 128, 16, 66),
        (1, 128, 16, 68),
        (1, 128, 7, 64),
        (1, 36, 3, 18),
        (1, 4, 1, 2),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_neg_2x4_reversed_cluster_0(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_2x4_reversed_cluster_0(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(3, -1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(3, -1),
        )

    compile_ttir_to_flatbuffer(
        neg_2x4_reversed_cluster_0,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 16, 64),
        (1, 16, 16, 64),
        (1, 15, 16, 64),
        (1, 7, 16, 64),
        (1, 16, 16, 136),
        (1, 16, 16, 128),
        (1, 16, 16, 40),
        (1, 16, 7, 64),
        (1, 7, 7, 64),
        (1, 10, 3, 64),
        (1, 1, 1, 64),
        (1, 1, 1, 24),
        (1, 1, 1, 8),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 8)])
def test_neg_1x8_dim_3(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_1x8_dim_3(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 8),
            shard_dims=(-1, 3),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 8),
            shard_dims=(-1, 3),
        )

    compile_ttir_to_flatbuffer(
        neg_1x8_dim_3,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 16, 128),
        (1, 64, 16, 16),
        (1, 64, 16, 15),
        (1, 64, 16, 7),
        (1, 136, 16, 16),
        (1, 128, 16, 16),
        (1, 40, 16, 16),
        (1, 64, 7, 16),
        (1, 64, 7, 7),
        (1, 64, 3, 10),
        (1, 64, 1, 1),
        (1, 24, 1, 1),
        (1, 8, 1, 1),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 8)])
def test_neg_1x8_dim_1(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_1x8_dim_1(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 8, 1, 1),
            shard_dims=(-1, 1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 8, 1, 1),
            shard_dims=(-1, 1),
        )

    compile_ttir_to_flatbuffer(
        neg_1x8_dim_1,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [(512, 1024), (512, 1024)],
        [(64, 128), (64, 128)],
        [(62, 128), (62, 128)],
        [(60, 128), (60, 128)],
        [(66, 128), (66, 128)],
        [(68, 128), (68, 128)],
        [(64, 124), (64, 124)],
        [(64, 120), (64, 120)],
        [(64, 132), (64, 132)],
        [(64, 136), (64, 136)],
        [(14, 44), (14, 44)],
        [(6, 12), (6, 12)],
        [(2, 4), (2, 4)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_eltwise_multidevice(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def eltwise_multidevice(in0: Operand, in1: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        partial_sum = builder.add(sharded_in0, sharded_in1)
        return builder.mesh_shard(
            partial_sum,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )

    compile_ttir_to_flatbuffer(
        eltwise_multidevice,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512), (1024, 512)],
        [(256, 128), (128, 128), (256, 128)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_matmul_and_binary_op(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request
):
    def matmul_test(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(sharded_in0, sharded_in1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        unsharded = builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )
        output = builder.add(unsharded, in2)
        return output

    compile_ttir_to_flatbuffer(
        matmul_test,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512)],
        [(256, 128), (128, 128)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_matmul_and_unary_op(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def matmul_test(in0: Operand, in1: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(sharded_in0, sharded_in1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        unsharded = builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )
        output = builder.neg(unsharded)
        return output

    compile_ttir_to_flatbuffer(
        matmul_test,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512), (1024, 32), (32, 512)],
        [(256, 128), (128, 128), (256, 128), (128, 128)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_matmul_and_binary_op_2(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request
):
    def matmul_test(
        in0: Operand, in1: Operand, in2: Operand, in3: Operand, builder: TTIRBuilder
    ):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul_0 = builder.matmul(sharded_in0, sharded_in1)
        reduced_0 = builder.all_reduce(
            partial_matmul_0,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        matmul_0 = builder.mesh_shard(
            reduced_0,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )

        sharded_in2 = builder.mesh_shard(
            in2,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in3 = builder.mesh_shard(
            in3,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul_2 = builder.matmul(sharded_in2, sharded_in3)
        reduced_2 = builder.all_reduce(
            partial_matmul_2,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        matmul_2 = builder.mesh_shard(
            reduced_2,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )
        output = builder.add(matmul_0, matmul_2)
        return output

    compile_ttir_to_flatbuffer(
        matmul_test,
        shapes,
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
