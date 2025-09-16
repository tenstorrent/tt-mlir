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
from test_utils import Marks, shape_str

pytestmark = [pytest.mark.n300, pytest.mark.frontend("ttir")]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 128, 128),
        (1, 32, 120, 128),
        (1, 32, 60, 128),
        (1, 32, 30, 128),
        (1, 32, 2, 128),
        (1, 32, 128, 120),
        (1, 32, 120, 120),
        (1, 32, 128, 60),
        (1, 32, 60, 60),
        (1, 32, 128, 30),
        (1, 32, 30, 30),
        (1, 32, 128, 2),
        (1, 32, 2, 2),
        (1, 1, 1, 2),
        (1, 1, 10, 10),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_all_gather(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_gather(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        gathered = builder.all_gather(
            sharded,
            all_gather_dim=3,
            cluster_axis=1,
        )
        return builder.mesh_shard(
            gathered,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    compile_ttir_to_flatbuffer(
        all_gather,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target="emitpy",
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 128, 512),
        (1, 1, 130, 512),
        (1, 1, 126, 512),
        (1, 1, 128, 508),
        (1, 1, 126, 508),
        (1, 1, 130, 508),
        (1, 1, 32, 2),
        pytest.param((1, 1, 1, 2), marks=pytest.mark.fails_golden),
        (1, 1, 128, 516),
        (1, 1, 128, 516),
        (1, 1, 126, 516),
        (1, 1, 130, 516),
        (1, 1, 32, 4),
        (1, 1, 32, 8),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_all_reduce(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_reduce(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        reduced = builder.all_reduce(
            sharded,
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
        all_reduce,
        [shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 128, 512),
        (1, 1, 128, 256),
        (1, 1, 128, 128),
        (1, 1, 127, 512),
        (1, 1, 126, 512),
        (1, 1, 129, 512),
        (1, 1, 130, 512),
        (1, 1, 128, 508),
        pytest.param((1, 1, 128, 64), marks=pytest.mark.run_error),
        (1, 1, 128, 516),
        (1, 1, 64, 128),
        (1, 1, 32, 128),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_reduce_scatter(shape: Shape, mesh_shape: Tuple[int, int], request):
    def reduce_scatter(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
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
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_ttir_to_flatbuffer(
        reduce_scatter,
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
        (1, 1, 128, 1024),
        (1, 1, 128, 512),
        (1, 1, 64, 512),
        (1, 1, 32, 64),
        (1, 1, 30, 60),
        (1, 1, 1, 2),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_collective_permute(shape: Shape, mesh_shape: Tuple[int, int], request):
    def collective_permute(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        reduced = builder.collective_permute(
            sharded,
            source_target_pairs=[(0, 1), (1, 0)],
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_ttir_to_flatbuffer(
        collective_permute,
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
        [(2048, 196), (196, 4096)],
        [(2046, 196), (196, 4094)],
        [(100, 196), (196, 320)],
        [(100, 194), (194, 320)],
        [(98, 196), (196, 318)],
        [(2050, 196), (196, 4098)],
        [(10, 4), (4, 20)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_matmul_1x2(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def matmul_1x2(in0: Operand, in1: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
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
        matmul_1x2,
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
        (1, 256, 64, 256),
        (1, 32, 32, 64),
        (1, 32, 32, 62),
        (1, 32, 32, 66),
        (1, 32, 32, 32),
        (1, 32, 32, 30),
        (1, 32, 32, 34),
        (1, 32, 31, 32),
        (1, 32, 30, 32),
        (1, 1, 1, 2),
        (1, 1, 1, 4),
        (1, 1, 1, 6),
        (1, 1, 1, 8),
        (1, 1, 3, 2),
        (1, 1, 3, 4),
        (1, 1, 3, 6),
        (1, 1, 3, 8),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_neg_1x2_dim_3(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_1x2_dim_3(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_ttir_to_flatbuffer(
        neg_1x2_dim_3,
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
        (1, 256, 64, 256),
        (1, 64, 32, 32),
        (1, 62, 32, 32),
        (1, 66, 32, 32),
        (1, 32, 32, 32),
        (1, 30, 32, 32),
        (1, 34, 32, 32),
        (1, 32, 31, 32),
        (1, 32, 30, 32),
        (1, 2, 1, 1),
        (1, 4, 1, 1),
        (1, 6, 1, 1),
        (1, 8, 1, 1),
        (1, 2, 3, 1),
        (1, 4, 3, 1),
        (1, 6, 3, 1),
        (1, 8, 3, 1),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_neg_1x2_dim_1(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_1x2_dim_1(in0: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(-1, 1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(-1, 1),
        )

    compile_ttir_to_flatbuffer(
        neg_1x2_dim_1,
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
        [(512, 1022), (512, 1022)],
        [(512, 1020), (512, 1020)],
        [(512, 1026), (512, 1026)],
        [(512, 1028), (512, 1028)],
        [(511, 1024), (511, 1024)],
        [(510, 1024), (510, 1024)],
        [(513, 1024), (513, 1024)],
        [(514, 1024), (514, 1024)],
        [(1, 2), (1, 2)],
        [(2, 2), (2, 2)],
        [(3, 6), (3, 6)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_eltwise_multidevice(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def eltwise_multidevice(in0: Operand, in1: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        partial_sum = builder.add(sharded_in0, sharded_in1)
        return builder.mesh_shard(
            partial_sum,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
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
        [(256, 128), (128, 256), (256, 256)],
        [(256, 128), (128, 256), (1, 256)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_matmul_and_binary_op(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request
):
    def matmul_test(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
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
            shard_type="#ttcore.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
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
        [(256, 128), (128, 256)],
        [(256, 256), (256, 256)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_matmul_and_unary_op(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def matmul_test(in0: Operand, in1: Operand, builder: TTIRBuilder):
        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
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
            shard_type="#ttcore.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
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
        [(256, 256), (256, 256), (256, 128), (128, 256)],
        [(256, 128), (128, 256), (256, 256), (256, 256)],
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
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
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
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
            shard_type="#ttcore.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

        sharded_in2 = builder.mesh_shard(
            in2,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in3 = builder.mesh_shard(
            in3,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(2, 1),
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
            shard_type="#ttcore.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
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


@pytest.mark.parametrize("input_shape", [(512, 512), (128, 32), (64, 256)])
@pytest.mark.parametrize("split_dim", range(2))
@pytest.mark.parametrize("concat_dim", range(2))
@pytest.mark.parametrize(
    "mesh_shape, shard_dims, shard_shape, cluster_axis, replica_groups",
    [
        ((1, 2), (-1, 0), (2, 1), 1, ((0, 1),)),
        ((1, 2), (-1, 1), (1, 2), 1, ((0, 1),)),
        ((2, 1), (0, -1), (2, 1), 0, ((0, 1),)),
        ((2, 1), (1, -1), (1, 2), 0, ((0, 1),)),
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
    "input_shape", [(1, 1, 512, 512), (1, 1, 128, 32), (1, 1, 64, 256)]
)
@pytest.mark.parametrize("split_dim", range(2, 4))
@pytest.mark.parametrize("concat_dim", range(2, 4))
@pytest.mark.parametrize(
    "mesh_shape, shard_dims, shard_shape, cluster_axis, replica_groups",
    [
        ((1, 2), (-1, 2), (1, 1, 2, 1), 1, ((0, 1),)),
        ((1, 2), (-1, 3), (1, 1, 1, 2), 1, ((0, 1),)),
        ((2, 1), (2, -1), (1, 1, 2, 1), 0, ((0, 1),)),
        ((2, 1), (3, -1), (1, 1, 1, 2), 0, ((0, 1),)),
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


def golden_collective_broadcast(
    input_tensor: torch.Tensor,
    replica_groups: List[Tuple[int, int]],
):
    shards = list(torch.chunk(input_tensor, 2, dim=3))
    for group in replica_groups:
        for device in group:
            shards[device] = shards[group[0]]
    result_tensor = torch.cat(shards, dim=3)
    return result_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 128, 1024),
        (1, 1, 32, 64),
        (1, 1, 30, 60),
        (1, 1, 1, 2),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
@pytest.mark.parametrize("replica_groups", [[(0, 1)]])
def test_collective_broadcast(
    shape: Shape, mesh_shape: Tuple[int, int], replica_groups, request
):
    def collective_broadcast(in0: Operand, builder: TTIRBuilder):
        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        reduced = builder.collective_broadcast(
            sharded,
            replica_groups=replica_groups,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
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
