# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from typing import List, Tuple
from collections import OrderedDict

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir
from test_utils import shape_str, make_shard_shape

pytestmark = pytest.mark.frontend("ttir")


# Helper function to build a matmul graph with parallelism over the contraction (reduction) dimension.
# Supports only 2D tensors and mesh shapes.
#    e.g. [M, K] x [K, N] -> [M, N], where K is the contraction dimension.
#    Under this parallelism, the K dimension is split across devices in the mesh.
#    For example, with mesh_shape=(2, 4):
#      - The input [M, K] is sharded along K (columns) across devices.
#      - The weight [K, N] is sharded along K (rows) across devices.
#      - Each device computes a partial matmul on its shard, then reduce_scatter is used to sum partial results and distribute the output [M, N] across devices.
#    The sharding looks like:
#      input:   [M, K/num_shards]  (each device gets a slice of K)
#      weight:  [K/num_shards, N]  (each device gets a slice of K)
#      output:  [M, N/num_shards]  (after reduce_scatter, each device gets a slice of N)
# Set do_unshard=True to have this function return a full tensor;
# otherwise, the output will remain sharded.
def _build_matmul_parallel(
    input: Operand,
    weight: Operand,
    builder: TTIRBuilder,
    mesh_shape: Tuple[int, int],
    parallelize_axis: int = 1,
    do_unshard: bool = False,
):
    # Shard the contraction dimension (dimension 1 for input, dimension 0 for weight) along the specified parallelization axis.
    if parallelize_axis == 0:
        shard_dims_inout = [1, 0]
        shard_dims_wt = [0, -1]
    else:
        shard_dims_inout = [0, 1]
        shard_dims_wt = [-1, 0]

    shard_shape_inout = make_shard_shape(2, shard_dims_inout, mesh_shape)
    shard_shape_wt = make_shard_shape(2, shard_dims_wt, mesh_shape)

    mesh_shard_in = builder.mesh_shard(
        input,
        shard_direction="#ttcore.shard_direction<full_to_shard>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape_inout,
        shard_dims=shard_dims_inout,
    )
    mesh_shard_wt = builder.mesh_shard(
        weight,
        shard_direction="#ttcore.shard_direction<full_to_shard>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape_wt,
        shard_dims=shard_dims_wt,
    )
    partial_matmul = builder.matmul(mesh_shard_in, mesh_shard_wt)
    reduced = builder.reduce_scatter(
        partial_matmul,
        reduce_type="#ttcore.reduce_type<sum>",
        scatter_dim=1,
        cluster_axis=parallelize_axis,
    )
    if not do_unshard:
        return reduced
    else:
        return builder.mesh_shard(
            reduced,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_inout,
            shard_dims=shard_dims_inout,
        )


# Test matmul with k-split parallelism across different mesh configurations.
# Verifies that matrix multiplication can be parallelized by splitting the
# contraction dimension across multiple devices.
@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512)],
        [(512, 32), (32, 256)],
        [(1024, 16), (16, 512)],
        [(1024, 8), (8, 512)],
        [(256, 128), (128, 256)],
    ],
    ids=lambda s: shape_str(s[0]) + "_" + shape_str(s[1]),
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (4, 2), (8, 1), (2, 1)], ids=shape_str
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
def test_matmul_k_split_parallelism(
    shapes: List[Shape], mesh_shape: Tuple[int, int], cluster_axis: int, request, device
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("parallelism across 1 device is meaningless")

    def matmul_multi(in0: Operand, in1: Operand, builder: TTIRBuilder):
        output = _build_matmul_parallel(
            in0,
            in1,
            builder,
            mesh_shape,
            cluster_axis,
            do_unshard=True,
        )

        input_0 = builder._get_golden_tensor(in0)
        input_1 = builder._get_golden_tensor(in1)
        golden = torch.matmul(input_0, input_1)
        builder.set_goldens_from_builder_tensor(
            {in0: input_0, in1: input_1}, {output: golden}
        )
        return output

    compile_and_execute_ttir(
        matmul_multi,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        skip_exec=True,
    )


# Test parallelized matmul with unary operation chaining.
# Performs matmul with k-split parallelism followed by a parallelized unary operation.
@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512)],
        [(256, 128), (128, 256)],
    ],
    ids=lambda s: shape_str(s[0]) + "_" + shape_str(s[1]),
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)], ids=shape_str)
@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
def test_parallelized_matmul_with_unary_chaining(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request, device
):
    def matmul_test(in0: Operand, in1: Operand, builder: TTIRBuilder):
        matmul_shard = _build_matmul_parallel(
            in0,
            in1,
            builder,
            mesh_shape,
            do_unshard=False,
        )
        output = builder.neg(matmul_shard)

        shard_dims_out = [0, 1]
        shard_shape_out = make_shard_shape(2, shard_dims_out, mesh_shape)
        output = builder.mesh_shard(
            output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_out,
            shard_dims=shard_dims_out,
        )

        input_0 = builder._get_golden_tensor(in0)
        input_1 = builder._get_golden_tensor(in1)
        golden = torch.neg(torch.matmul(input_0, input_1))
        builder.set_goldens_from_builder_tensor(
            {in0: input_0, in1: input_1}, {output: golden}
        )

        return output

    compile_and_execute_ttir(
        matmul_test,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        skip_exec=True,
    )


# Test parallelized matmul with binary operation chaining.
# Performs matmul with k-split parallelism followed by adding a third tensor.
@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512), (1024, 512)],
        [(256, 128), (128, 256), (256, 256)],
    ],
    ids=lambda s: shape_str(s[0]) + "_" + shape_str(s[1]),
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)], ids=shape_str)
@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
def test_parallelized_matmul_with_binary_chaining(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request, device
):
    def matmul_test(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        matmul_shard = _build_matmul_parallel(
            in0,
            in1,
            builder,
            mesh_shape,
            do_unshard=False,
        )

        shard_dims = [0, 1]
        shard_shape = make_shard_shape(2, shard_dims, mesh_shape)
        mesh_shard_in2 = builder.mesh_shard(
            in2,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        add = builder.add(matmul_shard, mesh_shard_in2)
        output = builder.mesh_shard(
            add,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        input_0 = builder._get_golden_tensor(in0)
        input_1 = builder._get_golden_tensor(in1)
        input_2 = builder._get_golden_tensor(in2)
        golden = torch.add(torch.matmul(input_0, input_1), input_2)
        builder.set_goldens_from_builder_tensor(
            {in0: input_0, in1: input_1, in2: input_2}, {output: golden}
        )
        return output

    compile_and_execute_ttir(
        matmul_test,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        skip_exec=True,
    )


# Test parallelized matmul fusion with binary operation chaining.
# Performs two independent parallelized matmul operations and combines their results
# using a parallelized add operation.
@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512), (1024, 64), (64, 512)],
        [(256, 128), (128, 256), (256, 64), (64, 256)],
    ],
    ids=lambda s: shape_str(s[0]) + "_" + shape_str(s[1]),
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)], ids=shape_str)
@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
def test_parallelized_matmul_fusion_with_binary_chaining(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request, device
):
    def matmul_test(
        in0: Operand, in1: Operand, in2: Operand, in3: Operand, builder: TTIRBuilder
    ):
        matmul_0 = _build_matmul_parallel(
            in0,
            in1,
            builder,
            mesh_shape,
            do_unshard=False,
        )
        matmul_1 = _build_matmul_parallel(
            in2,
            in3,
            builder,
            mesh_shape,
            do_unshard=False,
        )
        output = builder.add(matmul_0, matmul_1)

        shard_dims_out = [0, 1]
        shard_shape_out = make_shard_shape(2, shard_dims_out, mesh_shape)
        output = builder.mesh_shard(
            output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_out,
            shard_dims=shard_dims_out,
        )
        input_0 = builder._get_golden_tensor(in0)
        input_1 = builder._get_golden_tensor(in1)
        input_2 = builder._get_golden_tensor(in2)
        input_3 = builder._get_golden_tensor(in3)
        golden = torch.add(
            torch.matmul(input_0, input_1), torch.matmul(input_2, input_3)
        )
        builder.set_goldens_from_builder_tensor(
            {in0: input_0, in1: input_1, in2: input_2, in3: input_3}, {output: golden}
        )
        return output

    compile_and_execute_ttir(
        matmul_test,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        skip_exec=True,
    )


# Test parallelized element-wise operations across different tensor dimensions.
# Verifies that element-wise operations can be parallelized by sharding tensors
# across different dimensions and mesh configurations.
@pytest.mark.parametrize(
    "shape, shard_dims",
    [
        [(64, 128), (0, 1)],
        [(64, 128), (1, 0)],
        [(32, 32, 32), (0, 1)],
        [(32, 32, 32), (0, 2)],
        [(32, 32, 32), (1, 2)],
        [(32, 32, 32), (2, 0)],
        [(32, 32, 32, 32), (0, 1)],
        [(32, 32, 32, 32), (0, 3)],
        [(32, 32, 32, 32), (1, 3)],
        [(32, 32, 32, 32), (2, 3)],
        [(32, 32, 32, 32), (2, 1)],
        [(32, 32, 32, 32), (3, 2)],
        [(32, 32, 32, 32), (3, 0)],
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (4, 2), (8, 1), (2, 1)], ids=shape_str
)
@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
def test_parallelized_elementwise_operations(
    shape: Shape, shard_dims: List[int], mesh_shape: Tuple[int, int], request, device
):
    def eltwise_parallel(in0: Operand, in1: Operand, builder: TTIRBuilder):
        shard_shape = make_shard_shape(len(shape), shard_dims, mesh_shape)
        mesh_shard_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        mesh_shard_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        partial_sum = builder.add(mesh_shard_in0, mesh_shard_in1)
        output = builder.mesh_shard(
            partial_sum,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        input_0 = builder._get_golden_tensor(in0)
        input_1 = builder._get_golden_tensor(in1)
        golden = torch.add(input_0, input_1)
        builder.set_goldens_from_builder_tensor(
            {in0: input_0, in1: input_1}, {output: golden}
        )
        return output

    compile_and_execute_ttir(
        eltwise_parallel,
        [shape, shape],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        skip_exec=True,
    )


# Mixed device parallelism tests verify regressions in mixed device parallelism.
# These tests run graphs with multi-device operations followed by single-device operations.
# Refer to https://github.com/tenstorrent/tt-mlir/issues/3058 for more details.


# Test mixed device parallelism with unary operation.
# Performs parallelized matmul followed by single-device unary operation.
@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512)],
        [(256, 128), (128, 256)],
    ],
    ids=lambda s: shape_str(s[0]) + "_" + shape_str(s[1]),
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)], ids=shape_str)
@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
def test_mixed_device_parallelism_with_unary(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request, device
):
    def matmul_test(in0: Operand, in1: Operand, builder: TTIRBuilder):
        matmul_result = _build_matmul_parallel(
            in0,
            in1,
            builder,
            mesh_shape,
            do_unshard=True,
        )

        output = builder.neg(matmul_result)

        # Golden is computed on a single device
        input_0 = builder._get_golden_tensor(in0)
        input_1 = builder._get_golden_tensor(in1)
        golden = torch.neg(torch.matmul(input_0, input_1))
        builder.set_goldens_from_builder_tensor(
            {in0: input_0, in1: input_1}, {output: golden}
        )

        return output

    compile_and_execute_ttir(
        matmul_test,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        skip_exec=True,
    )


# Test mixed device parallelism with binary operation.
# Performs parallelized matmul followed by single-device binary operation.
@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512), (1024, 512)],
        [(256, 128), (128, 256), (256, 256)],
    ],
    ids=lambda s: shape_str(s[0]) + "_" + shape_str(s[1]),
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)], ids=shape_str)
@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
def test_mixed_device_parallelism_with_binary(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request, device
):
    def matmul_test(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        matmul_result = _build_matmul_parallel(
            in0,
            in1,
            builder,
            mesh_shape,
            do_unshard=True,
        )
        output = builder.add(matmul_result, in2)

        # Golden is computed on a single device
        input_0 = builder._get_golden_tensor(in0)
        input_1 = builder._get_golden_tensor(in1)
        input_2 = builder._get_golden_tensor(in2)
        golden = torch.add(torch.matmul(input_0, input_1), input_2)
        builder.set_goldens_from_builder_tensor(
            {in0: input_0, in1: input_1, in2: input_2}, {output: golden}
        )
        return output

    compile_and_execute_ttir(
        matmul_test,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        skip_exec=True,
    )


# Test mixed device parallelism with dual matmul and binary operation.
# Performs two parallelized matmuls followed by single-device binary operation.
@pytest.mark.parametrize(
    "shapes",
    [
        [(1024, 32), (32, 512), (1024, 64), (64, 512)],
        [(256, 128), (128, 256), (256, 64), (64, 256)],
    ],
    ids=lambda s: shape_str(s[0]) + "_" + shape_str(s[1]),
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (1, 8), (1, 2)], ids=shape_str)
@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
def test_mixed_device_parallelism_with_dual_matmul(
    shapes: List[Shape], mesh_shape: Tuple[int, int], request, device
):
    def matmul_test(
        in0: Operand, in1: Operand, in2: Operand, in3: Operand, builder: TTIRBuilder
    ):
        matmul_0 = _build_matmul_parallel(
            in0,
            in1,
            builder,
            mesh_shape,
            do_unshard=True,
        )
        matmul_1 = _build_matmul_parallel(
            in2,
            in3,
            builder,
            mesh_shape,
            do_unshard=True,
        )
        output = builder.add(matmul_0, matmul_1)

        # Golden is computed on a single device
        input_0 = builder._get_golden_tensor(in0)
        input_1 = builder._get_golden_tensor(in1)
        input_2 = builder._get_golden_tensor(in2)
        input_3 = builder._get_golden_tensor(in3)
        golden = torch.add(
            torch.matmul(input_0, input_1), torch.matmul(input_2, input_3)
        )
        builder.set_goldens_from_builder_tensor(
            {in0: input_0, in1: input_1, in2: input_2, in3: input_3}, {output: golden}
        )

        return output

    compile_and_execute_ttir(
        matmul_test,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        skip_exec=True,
    )


# The test cases below are from the device_parallel_*.py files.
# These graphs are generated by JAX @jit for matmul with various types of parallelism.
JIT_PARALLELISM_TESTS_INPUT_SHAPES = [(64, 1, 128, 2048), (1, 1, 2048, 128)]


@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
@pytest.mark.parametrize("mesh_shape", [(1, 2), (1, 8), (1, 32)], ids=shape_str)
def test_jit_tensor_parallel(mesh_shape: Tuple[int, int], request, device):
    shapes = JIT_PARALLELISM_TESTS_INPUT_SHAPES

    def jit_tensor_parallel(in0: Operand, in1: Operand, builder: TTIRBuilder):
        shard_dims_in0 = [-1, 3]
        shard_shape_in0 = make_shard_shape(4, shard_dims_in0, mesh_shape)
        mesh_shard_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_in0,
            shard_dims=shard_dims_in0,
        )
        shard_dims_in1 = [-1, 2]
        shard_shape_in1 = make_shard_shape(4, shard_dims_in1, mesh_shape)
        mesh_shard_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_in1,
            shard_dims=shard_dims_in1,
        )
        reshape_in0 = builder.squeeze(mesh_shard_in0, dim=1)
        reshape_in1 = builder.squeeze(mesh_shard_in1, dim=1)
        dot_general = builder.dot_general(
            reshape_in0,
            reshape_in1,
            batch_dims_lhs=[],
            batch_dims_rhs=[],
            contract_dims_lhs=[2],
            contract_dims_rhs=[1],
        )
        permute = builder.permute(dot_general, [0, 2, 1, 3])
        reduce_scatter = builder.reduce_scatter(
            permute,
            scatter_dim=2,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        mesh_shard_out = builder.mesh_shard(
            reduce_scatter,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_in1,
            shard_dims=shard_dims_in1,
        )
        return mesh_shard_out

    compile_and_execute_ttir(
        jit_tensor_parallel,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        skip_exec=True,
    )


@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
@pytest.mark.parametrize("mesh_shape", [(1, 2), (1, 8), (1, 32)], ids=shape_str)
def test_jit_data_parallel(mesh_shape: Tuple[int, int], request, device):
    shapes = JIT_PARALLELISM_TESTS_INPUT_SHAPES

    def jit_data_parallel(in0: Operand, in1: Operand, builder: TTIRBuilder):
        shard_dims_in0 = [-1, 0]
        shard_shape_in0 = make_shard_shape(4, shard_dims_in0, mesh_shape)
        mesh_shard_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_in0,
            shard_dims=shard_dims_in0,
        )
        mesh_shard_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<replicate>",
            shard_shape=[1],
            shard_dims=[-1],
        )
        reshape_in0 = builder.squeeze(mesh_shard_in0, dim=1)
        reshape_in1 = builder.squeeze(mesh_shard_in1, dim=1)
        dot_general = builder.dot_general(
            reshape_in0,
            reshape_in1,
            batch_dims_lhs=[],
            batch_dims_rhs=[],
            contract_dims_lhs=[2],
            contract_dims_rhs=[1],
        )
        permute = builder.permute(dot_general, [0, 2, 1, 3])
        shard_dims_out = [-1, 0]
        shard_shape_out = make_shard_shape(4, shard_dims_out, mesh_shape)
        mesh_shard_out = builder.mesh_shard(
            permute,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_out,
            shard_dims=shard_dims_out,
        )
        return mesh_shard_out

    compile_and_execute_ttir(
        jit_data_parallel,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        skip_exec=True,
    )


@pytest.mark.xfail(
    reason="Runtime hang, see https://github.com/tenstorrent/tt-mlir/issues/5262"
)
@pytest.mark.parametrize("mesh_shape", [(2, 4), (8, 4)], ids=shape_str)
def test_jit_data_tensor_parallel(mesh_shape: Tuple[int, int], request, device):
    shapes = JIT_PARALLELISM_TESTS_INPUT_SHAPES

    def jit_data_tensor_parallel(in0: Operand, in1: Operand, builder: TTIRBuilder):
        shard_dims_in0 = [0, 3]
        shard_shape_in0 = make_shard_shape(4, shard_dims_in0, mesh_shape)
        mesh_shard_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_in0,
            shard_dims=shard_dims_in0,
        )
        shard_dims_in1 = [-1, 2]
        shard_shape_in1 = make_shard_shape(4, shard_dims_in1, mesh_shape)
        mesh_shard_in1 = builder.mesh_shard(
            in1,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_in1,
            shard_dims=shard_dims_in1,
        )
        reshape_in0 = builder.squeeze(mesh_shard_in0, dim=1)
        reshape_in1 = builder.squeeze(mesh_shard_in1, dim=1)
        dot_general = builder.dot_general(
            reshape_in0,
            reshape_in1,
            batch_dims_lhs=[],
            batch_dims_rhs=[],
            contract_dims_lhs=[2],
            contract_dims_rhs=[1],
        )
        permute = builder.permute(dot_general, [0, 2, 1, 3])
        reduce_scatter = builder.reduce_scatter(
            permute,
            scatter_dim=2,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=1,
        )
        shard_dims_out = [0, 2]
        shard_shape_out = make_shard_shape(4, shard_dims_out, mesh_shape)
        mesh_shard_out = builder.mesh_shard(
            reduce_scatter,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape_out,
            shard_dims=shard_dims_out,
        )
        return mesh_shard_out

    compile_and_execute_ttir(
        jit_data_tensor_parallel,
        shapes,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        skip_exec=True,
    )
