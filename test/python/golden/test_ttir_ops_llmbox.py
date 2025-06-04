# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from typing import List, Tuple
from ttir_builder.utils import compile_to_flatbuffer
from ttir_builder import Operand, TTIRBuilder, Shape

pytestmark = pytest.mark.llmbox


def pseudo_golden_all_gather(
    input_tensor: torch.Tensor,
):
    output_tensor = input_tensor.clone()
    return output_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 256, 512),
        (1, 1, 64, 128),
        (1, 1, 66, 128),
        (1, 1, 62, 128),
        pytest.param(
            (1, 1, 64, 132), marks=pytest.mark.fails_golden
        ),  # https://github.com/tenstorrent/tt-metal/issues/21964
        pytest.param((1, 1, 66, 132), marks=pytest.mark.fails_golden),
        pytest.param((1, 1, 64, 124), marks=pytest.mark.fails_golden),
        pytest.param((1, 1, 62, 124), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 258, 516), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 260, 520), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 254, 508), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 252, 504), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 32, 64), marks=pytest.mark.fails_golden),
        pytest.param((1, 1, 2, 4), marks=pytest.mark.fails_golden),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_all_gather(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_gather(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_gather(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )
        gathered = builder.all_gather(
            sharded,
            all_gather_dim=3,
            cluster_axis=1,
        )
        return builder.mesh_shard(
            gathered,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 2, 1),
            shard_dims=(2, -1),
        )

    compile_to_flatbuffer(
        all_gather, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


def pseudo_golden_all_reduce(input_tensor: torch.Tensor):
    shards = torch.chunk(input_tensor, 4, dim=3)
    return sum(shards)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 256, 512),
        (1, 1, 2, 4),
        pytest.param(
            (1, 1, 64, 128), marks=pytest.mark.run_error
        ),  # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param((1, 1, 64, 256), marks=pytest.mark.run_error),
        pytest.param((1, 1, 128, 256), marks=pytest.mark.run_error),
        pytest.param((1, 1, 256, 256), marks=pytest.mark.run_error),
        pytest.param(
            (1, 1, 128, 512), marks=pytest.mark.run_error
        ),  # hangs # https://github.com/tenstorrent/tt-metal/issues/21987
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_all_reduce(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_reduce(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_reduce(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )
        reduced = builder.all_reduce(
            sharded,
            reduce_type="#tt.reduce_type<sum>",
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 2, 1),
            shard_dims=(2, -1),
        )

    compile_to_flatbuffer(
        all_reduce, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


def pseudo_golden_reduce_scatter(
    input_tensor: torch.Tensor,
    scatter_dim: int,
):
    shards = torch.chunk(input_tensor, 4, dim=scatter_dim)
    return sum(shards)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 512, 512),
        (1, 1, 256, 1024),
        (1, 1, 256, 512),
        (1, 1, 254, 1024),
        (1, 1, 256, 1024),
        (1, 1, 128, 1024),
        pytest.param(
            (1, 1, 256, 1008), marks=pytest.mark.run_error
        ),  # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param((1, 1, 256, 1040), marks=pytest.mark.run_error),
        pytest.param((1, 1, 128, 256), marks=pytest.mark.run_error),
        pytest.param((1, 1, 128, 128), marks=pytest.mark.run_error),
        pytest.param((1, 1, 128, 64), marks=pytest.mark.run_error),
        pytest.param((1, 1, 64, 64), marks=pytest.mark.run_error),
        pytest.param((1, 1, 64, 128), marks=pytest.mark.run_error),
        pytest.param((1, 1, 2, 16), marks=pytest.mark.run_error),
        pytest.param(
            (1, 1, 128, 512), marks=pytest.mark.run_error
        ),  # hangs # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param((1, 1, 64, 512), marks=pytest.mark.run_error),  # hangs
        pytest.param((1, 1, 32, 512), marks=pytest.mark.run_error),  # hangs
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_reduce_scatter(shape: Shape, mesh_shape: Tuple[int, int], request):
    def reduce_scatter(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_reduce_scatter(input, 3)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )
        reduced = builder.reduce_scatter(
            sharded,
            reduce_type="#tt.reduce_type<sum>",
            scatter_dim=3,
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )

    compile_to_flatbuffer(
        reduce_scatter,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
    )


def pseudo_golden_collective_permute(
    input_tensor: torch.Tensor,
    source_target_pairs: List[Tuple[int, int]],
):
    # sharding
    shards = [
        chunk
        for shard in torch.chunk(input_tensor, 2, dim=2)
        for chunk in torch.chunk(shard, 4, dim=3)
    ]

    # permute
    permuted = [torch.zeros_like(shard) for shard in shards]
    for src, tgt in source_target_pairs:
        permuted[tgt] = shards[src]

    # unsharding
    return torch.cat(
        [torch.cat(permuted[i : i + 4], dim=3) for i in range(0, len(permuted), 4)],
        dim=2,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 256, 4096),
        (1, 1, 258, 4096),
        (1, 1, 260, 4096),
        (1, 1, 254, 4096),
        (1, 1, 252, 4096),
        (1, 1, 256, 4100),
        (1, 1, 256, 4104),
        (1, 1, 256, 4092),
        (1, 1, 256, 4088),
        (1, 1, 30, 32),
        (1, 1, 2, 4),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_collective_permute(shape: Shape, mesh_shape: Tuple[int, int], request):
    def collective_permute(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
        golden_output = pseudo_golden_collective_permute(input, pairs)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )
        reduced = builder.collective_permute(
            sharded,
            source_target_pairs=pairs,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )

    compile_to_flatbuffer(
        collective_permute,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        weight = builder._get_golden_tensor(in1)
        golden_output = torch.matmul(input, weight)
        builder.set_graph_input_output([input, weight], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(sharded_in0, sharded_in1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#tt.reduce_type<sum>",
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )

    compile_to_flatbuffer(
        matmul_2x4,
        shapes,
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        weight = builder._get_golden_tensor(in1)
        golden_output = torch.matmul(input, weight)
        builder.set_graph_input_output([input, weight], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 8),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(8, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(sharded_in0, sharded_in1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#tt.reduce_type<sum>",
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    compile_to_flatbuffer(
        matmul_1x8,
        shapes,
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2, 1, 4),
            shard_dims=(1, 3),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2, 1, 4),
            shard_dims=(1, 3),
        )

    compile_to_flatbuffer(
        neg_2x4,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(1, -1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(1, -1),
        )

    compile_to_flatbuffer(
        neg_2x4_cluster_0,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 4),
            shard_dims=(1, 3),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 4),
            shard_dims=(1, 3),
        )

    compile_to_flatbuffer(
        neg_2x4_cluster_1,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 4, 1, 2),
            shard_dims=(3, 1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 4, 1, 2),
            shard_dims=(3, 1),
        )

    compile_to_flatbuffer(
        neg_2x4_reversed_cluster,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(3, -1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(3, -1),
        )

    compile_to_flatbuffer(
        neg_2x4_reversed_cluster_0,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 8),
            shard_dims=(-1, 3),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 8),
            shard_dims=(-1, 3),
        )

    compile_to_flatbuffer(
        neg_1x8_dim_3,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 8, 1, 1),
            shard_dims=(-1, 1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 8, 1, 1),
            shard_dims=(-1, 1),
        )

    compile_to_flatbuffer(
        neg_1x8_dim_1,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        weight = builder._get_golden_tensor(in1)
        golden_output = torch.add(input, weight)
        builder.set_graph_input_output([input, weight], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        partial_sum = builder.add(sharded_in0, sharded_in1)
        return builder.mesh_shard(
            partial_sum,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )

    compile_to_flatbuffer(
        eltwise_multidevice,
        shapes,
        mesh_shape=mesh_shape,
        test_base=request.node.name,
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
        input = builder._get_golden_tensor(in0)
        weight = builder._get_golden_tensor(in1)
        bias = builder._get_golden_tensor(in2)
        golden_output = torch.add(torch.matmul(input, weight), bias)
        builder.set_graph_input_output([input, weight], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(sharded_in0, sharded_in1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#tt.reduce_type<sum>",
            cluster_axis=1,
        )
        unsharded = builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )
        output = builder.add(unsharded, in2)
        return output

    compile_to_flatbuffer(
        matmul_test,
        shapes,
        mesh_shape=mesh_shape,
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
        input = builder._get_golden_tensor(in0)
        weight = builder._get_golden_tensor(in1)
        golden_output = torch.neg(torch.matmul(input, weight))
        builder.set_graph_input_output([input, weight], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(sharded_in0, sharded_in1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#tt.reduce_type<sum>",
            cluster_axis=1,
        )
        unsharded = builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )
        output = builder.neg(unsharded)
        return output

    compile_to_flatbuffer(
        matmul_test,
        shapes,
        mesh_shape=mesh_shape,
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
        input = builder._get_golden_tensor(in0)
        weight = builder._get_golden_tensor(in1)
        input_2 = builder._get_golden_tensor(in2)
        weight_2 = builder._get_golden_tensor(in3)
        golden_output = torch.add(
            torch.matmul(input, weight), torch.matmul(input_2, weight_2)
        )
        builder.set_graph_input_output(
            [input, weight, input_2, weight_2], [golden_output]
        )

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul_0 = builder.matmul(sharded_in0, sharded_in1)
        reduced_0 = builder.all_reduce(
            partial_matmul_0,
            reduce_type="#tt.reduce_type<sum>",
            cluster_axis=1,
        )
        matmul_0 = builder.mesh_shard(
            reduced_0,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )

        sharded_in2 = builder.mesh_shard(
            in2,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        sharded_in3 = builder.mesh_shard(
            in3,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul_2 = builder.matmul(sharded_in2, sharded_in3)
        reduced_2 = builder.all_reduce(
            partial_matmul_2,
            reduce_type="#tt.reduce_type<sum>",
            cluster_axis=1,
        )
        matmul_2 = builder.mesh_shard(
            reduced_2,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )
        output = builder.add(matmul_0, matmul_2)
        return output

    compile_to_flatbuffer(
        matmul_test,
        shapes,
        mesh_shape=mesh_shape,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
