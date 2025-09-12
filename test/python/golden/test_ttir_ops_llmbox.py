# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from typing import List, Tuple
from collections import OrderedDict

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_golden import BuilderGoldenTensor
from builder.base.builder_utils import compile_ttir_to_flatbuffer

pytestmark = [pytest.mark.llmbox, pytest.mark.frontend("ttir")]


def get_input_tensors_from_builder(args: List, builder: TTIRBuilder):
    input_tensors = []
    for arg in args:
        input_tensors.append(builder._get_golden_tensor(arg))
    return input_tensors


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 256, 512),
        (1, 1, 64, 128),
        (1, 1, 66, 128),
        (1, 1, 62, 128),
        (1, 1, 64, 132),
        (1, 1, 66, 132),
        (1, 1, 64, 124),
        (1, 1, 62, 124),
        (1, 32, 258, 516),
        (1, 32, 260, 520),
        (1, 32, 254, 508),
        (1, 32, 252, 504),
        (1, 32, 32, 64),
        (1, 1, 2, 4),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_all_gather(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_gather(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
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
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 2, 1),
            shard_dims=(2, -1),
        )

    compile_ttir_to_flatbuffer(
        all_gather,
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
        (1, 1, 256, 512),
        pytest.param((1, 1, 2, 4), marks=pytest.mark.fails_golden),
        pytest.param((1, 1, 64, 128), marks=pytest.mark.run_error),
        pytest.param((1, 1, 64, 256), marks=pytest.mark.run_error),
        pytest.param((1, 1, 128, 256), marks=pytest.mark.run_error),
        (1, 1, 256, 256),
        pytest.param((1, 1, 128, 512), marks=pytest.mark.run_error),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_all_reduce(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_reduce(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )
        reduced = builder.all_reduce(
            sharded,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 2, 1),
            shard_dims=(2, -1),
        )

    compile_ttir_to_flatbuffer(
        all_reduce,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 512, 512),
        (1, 1, 256, 1024),
        (1, 1, 256, 1024),
        (1, 1, 256, 512),
        (1, 1, 254, 1024),
        (1, 1, 256, 1024),
        (1, 1, 128, 1024),
        (1, 1, 256, 1008),
        pytest.param((1, 1, 256, 1040), marks=pytest.mark.run_error),
        pytest.param((1, 1, 128, 256), marks=pytest.mark.run_error),
        pytest.param((1, 1, 128, 128), marks=pytest.mark.run_error),
        (1, 1, 128, 64),
        pytest.param((1, 1, 64, 64), marks=pytest.mark.run_error),
        pytest.param((1, 1, 64, 128), marks=pytest.mark.run_error),
        (1, 1, 2, 16),
        (1, 1, 128, 512),
        (1, 1, 64, 512),
        pytest.param((1, 1, 32, 512), marks=pytest.mark.run_error),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_reduce_scatter(shape: Shape, mesh_shape: Tuple[int, int], request):
    def reduce_scatter(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )
        reduced = builder.reduce_scatter(
            sharded,
            reduce_type="#ttcore.reduce_type<sum>",
            scatter_dim=3,
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )

    compile_ttir_to_flatbuffer(
        reduce_scatter,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
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
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )
        reduced = builder.collective_permute(
            sharded,
            source_target_pairs=pairs,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 2, 4),
            shard_dims=(2, 3),
        )

    compile_ttir_to_flatbuffer(
        collective_permute,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
    )


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
        [(254, 128), (128, 128)],
        [(252, 128), (128, 128)],
        [(258, 128), (128, 128)],
        [(260, 128), (128, 128)],
        [(256, 128), (128, 132)],
        [(256, 128), (128, 136)],
        pytest.param([(256, 32), (32, 64)], marks=pytest.mark.run_error),
        pytest.param([(128, 32), (32, 32)], marks=pytest.mark.run_error),
        [(64, 32), (32, 16)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_matmul_2x4(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def matmul_2x4(in0: Operand, in1: Operand, builder: TTIRBuilder):
        mesh_shard_0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 4),
            shard_dims=(0, 1),
        )
        mesh_shard_1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(4, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(mesh_shard_0, mesh_shard_1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        mesh_shard_2 = builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(0, -1),
        )
        operands = [in0, in1]
        input_tensors = get_input_tensors_from_builder(operands, builder)
        builder.set_goldens_from_builder_tensor(
            {operands[0]: input_tensors[0], operands[1]: input_tensors[1]},
            {mesh_shard_2: torch.matmul(input_tensors[0], input_tensors[1])},
        )
        return mesh_shard_2

    compile_ttir_to_flatbuffer(
        matmul_2x4,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
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
        [(256, 128), (128, 132)],
        [(1024, 8), (8, 512)],
        pytest.param([(512, 32), (32, 128)], marks=pytest.mark.run_error),
        pytest.param([(256, 128), (128, 128)], marks=pytest.mark.run_error),
        [(256, 128), (128, 120)],
        [(254, 128), (128, 128)],
        [(252, 128), (128, 128)],
        [(258, 128), (128, 128)],
        [(260, 128), (128, 128)],
        [(256, 128), (128, 136)],
        pytest.param([(256, 32), (32, 64)], marks=pytest.mark.run_error),
        pytest.param([(128, 32), (32, 32)], marks=pytest.mark.run_error),
        [(64, 32), (32, 16)],
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
        pytest.param(
            [(256, 128), (128, 128), (256, 128), (128, 128)],
            marks=pytest.mark.run_error,
        ),
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


def all_to_all_test(
    input_shape: Shape,
    split_dim,
    concat_dim,
    mesh_shape,
    shard_dims,
    shard_shape,
    cluster_axis,
    replica_groups,
    request,
):
    def all_to_all(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        gathered = builder.all_to_all(
            sharded,
            split_dim=split_dim,
            concat_dim=concat_dim,
            split_count=mesh_shape[cluster_axis],
            replica_groups=replica_groups,
        )
        return builder.mesh_shard(
            gathered,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )

    compile_ttir_to_flatbuffer(
        all_to_all,
        [input_shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("input_shape", [(2048, 2048), (512, 128), (256, 1024)])
@pytest.mark.parametrize("split_dim", range(2))
@pytest.mark.parametrize("concat_dim", range(2))
@pytest.mark.parametrize(
    "mesh_shape, shard_dims, shard_shape, cluster_axis, replica_groups",
    [
        ((1, 8), (-1, 0), (8, 1), 1, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ((1, 8), (-1, 1), (1, 8), 1, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ((2, 4), (0, 1), (2, 4), 0, ((0, 4), (1, 5), (2, 6), (3, 7))),
        ((2, 4), (0, 1), (2, 4), 1, ((0, 1, 2, 3), (4, 5, 6, 7))),
        ((4, 2), (0, 1), (4, 2), 0, ((0, 2, 4, 6), (1, 3, 5, 7))),
        ((4, 2), (0, 1), (4, 2), 1, ((0, 1), (2, 3), (4, 5), (6, 7))),
        ((8, 1), (0, -1), (8, 1), 0, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ((8, 1), (1, -1), (1, 8), 0, ((0, 1, 2, 3, 4, 5, 6, 7),)),
    ],
)
def test_all_to_all_2d(
    input_shape: Shape,
    split_dim,
    concat_dim,
    mesh_shape,
    shard_dims,
    shard_shape,
    cluster_axis,
    replica_groups,
    request,
):
    all_to_all_test(
        input_shape=input_shape,
        split_dim=split_dim,
        concat_dim=concat_dim,
        mesh_shape=mesh_shape,
        shard_dims=shard_dims,
        shard_shape=shard_shape,
        cluster_axis=cluster_axis,
        replica_groups=replica_groups,
        request=request,
    )


@pytest.mark.parametrize(
    "input_shape", [(1, 1, 2048, 2048), (1, 1, 512, 128), (1, 1, 256, 1024)]
)
@pytest.mark.parametrize("split_dim", range(2, 4))
@pytest.mark.parametrize("concat_dim", range(2, 4))
@pytest.mark.parametrize(
    "mesh_shape, shard_dims, shard_shape, cluster_axis, replica_groups",
    [
        ((1, 8), (-1, 2), (1, 1, 8, 1), 1, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ((1, 8), (-1, 3), (1, 1, 1, 8), 1, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ((2, 4), (2, 3), (1, 1, 2, 4), 0, ((0, 4), (1, 5), (2, 6), (3, 7))),
        ((2, 4), (2, 3), (1, 1, 2, 4), 1, ((0, 1, 2, 3), (4, 5, 6, 7))),
        ((4, 2), (2, 3), (1, 1, 4, 2), 0, ((0, 2, 4, 6), (1, 3, 5, 7))),
        ((4, 2), (2, 3), (1, 1, 4, 2), 1, ((0, 1), (2, 3), (4, 5), (6, 7))),
        ((8, 1), (2, -1), (1, 1, 8, 1), 0, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ((8, 1), (3, -1), (1, 1, 1, 8), 0, ((0, 1, 2, 3, 4, 5, 6, 7),)),
    ],
)
def test_all_to_all_4d(
    input_shape: Shape,
    split_dim,
    concat_dim,
    mesh_shape,
    shard_dims,
    shard_shape,
    cluster_axis,
    replica_groups,
    request,
):
    all_to_all_test(
        input_shape=input_shape,
        split_dim=split_dim,
        concat_dim=concat_dim,
        mesh_shape=mesh_shape,
        shard_dims=shard_dims,
        shard_shape=shard_shape,
        cluster_axis=cluster_axis,
        replica_groups=replica_groups,
        request=request,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 256, 4096),
        (1, 1, 128, 256),
        (1, 1, 64, 128),
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
    shape: Shape, mesh_shape: Tuple[int, int], replica_groups, request
):
    shard_shape = (1, 1) + mesh_shape

    def collective_broadcast(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=(2, 3),
        )
        reduced = builder.collective_broadcast(
            sharded,
            replica_groups=replica_groups,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=(2, 3),
        )

    compile_ttir_to_flatbuffer(
        collective_broadcast,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
