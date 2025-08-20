# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from typing import List, Tuple
from collections import OrderedDict

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from test_utils import make_shard_shape


def build_matmul_multi(
    mesh_shape: Tuple[int, int],
    input_rank: int,
    cluster_axis: int,
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    do_unshard: bool = False,
):
    # Select dims based on axis
    if cluster_axis == 0:
        shard_dims_in = [1, 0]
        shard_dims_wt = [0, -1]
    else:
        shard_dims_in = [0, 1]
        shard_dims_wt = [-1, 0]

    shard_shape_in = make_shard_shape(input_rank, shard_dims_in, mesh_shape)
    shard_shape_wt = make_shard_shape(input_rank, shard_dims_wt, mesh_shape)

    sharded_in0 = builder.mesh_shard(
        in0,
        shard_direction="#ttcore.shard_direction<full_to_shard>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape_in,
        shard_dims=shard_dims_in,
    )
    sharded_in1 = builder.mesh_shard(
        in1,
        shard_direction="#ttcore.shard_direction<full_to_shard>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape_wt,
        shard_dims=shard_dims_wt,
    )
    partial_matmul = builder.matmul(sharded_in0, sharded_in1)
    sharded_result = builder.all_reduce(
        partial_matmul,
        reduce_type="#ttcore.reduce_type<sum>",
        cluster_axis=cluster_axis,
    )
    if not do_unshard:
        return sharded_result
    else:
        if mesh_shape[1 - cluster_axis] == 1:
            # Fully replicated result
            shard_dims_out = [-1]
            shard_shape_out = [1]
            unshard_type = "#ttcore.shard_type<replicate>"
        else:
            if cluster_axis == 0:
                shard_dims_out = [-1, 0]
            else:
                shard_dims_out = [0, -1]
            shard_shape_out = make_shard_shape(input_rank, shard_dims_out, mesh_shape)
            unshard_type = "#ttcore.shard_type<devices>"
        return builder.mesh_shard(
            sharded_result,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type=unshard_type,
            shard_shape=shard_shape_out,
            shard_dims=shard_dims_out,
        )


@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512)],
        [(512, 32), (32, 256)],
        [(1024, 16), (16, 512)],
        [(1024, 8), (8, 512)],
        [(1024, 8), (8, 512)],
        [(256, 128), (128, 256)],
    ],
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        (2, 4),
        (4, 2),
        (1, 8),
        (8, 1),
        (1, 2),
        (2, 1),
    ],
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_matmul_multi_2d(
    shapes: List[Shape], mesh_shape: Tuple[int, int], cluster_axis: int, request
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("parallelism across 1 device is meaningless")

    def matmul_multi(in0: Operand, in1: Operand, builder: TTIRBuilder):
        return build_matmul_multi(
            mesh_shape,
            len(shapes[0]),
            cluster_axis,
            in0,
            in1,
            builder,
            do_unshard=True,
        )

    compile_ttir_to_flatbuffer(
        matmul_multi,
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
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)])
def test_matmul_and_binary_op(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request
):
    def matmul_test(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        unsharded = build_matmul_multi(
            mesh_shape,
            len(shapes[0]),
            1,
            in0,
            in1,
            builder,
            do_unshard=True,
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
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)])
def test_matmul_and_unary_op(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def matmul_test(in0: Operand, in1: Operand, builder: TTIRBuilder):
        unsharded = build_matmul_multi(
            mesh_shape,
            len(shapes[0]),
            1,
            in0,
            in1,
            builder,
            do_unshard=True,
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
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)])
def test_matmul_and_binary_op_2(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request
):
    def matmul_test(
        in0: Operand, in1: Operand, in2: Operand, in3: Operand, builder: TTIRBuilder
    ):
        matmul_0 = build_matmul_multi(
            mesh_shape,
            len(shapes[0]),
            1,
            in0,
            in1,
            builder,
            do_unshard=True,
        )
        matmul_1 = build_matmul_multi(
            mesh_shape,
            len(shapes[2]),
            1,
            in2,
            in3,
            builder,
            do_unshard=True,
        )
        output = builder.add(matmul_0, matmul_1)
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
