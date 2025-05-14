# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from typing import List, Tuple
from ttir_builder.test_utils import compile_to_flatbuffer
from ttir_builder.ttir_builder import Operand, TTIRBuilder, Shape

pytestmark = pytest.mark.n300


def pseudo_golden_all_gather(
    input_tensor: torch.Tensor,
):
    output_tensor = input_tensor.clone()
    return output_tensor


@pytest.mark.parametrize("shape", [(1, 32, 128, 128)])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_all_gather(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_gather(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_gather(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
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
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    compile_to_flatbuffer(
        all_gather, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


def pseudo_golden_all_reduce(input_tensor: torch.Tensor):
    shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=3)
    output_tensor = shard_1 + shard_2
    return output_tensor


@pytest.mark.parametrize("shape", [(1, 1, 128, 512)])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_all_reduce(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_reduce(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_reduce(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        reduced = builder.all_reduce(
            sharded,
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
        all_reduce, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


def pseudo_golden_reduce_scatter(
    input_tensor: torch.Tensor,
    scatter_dim: int,
):
    shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=scatter_dim)
    output_tensor = shard_1 + shard_2
    return output_tensor


@pytest.mark.parametrize("shape", [(1, 1, 8192, 512)])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_reduce_scatter(shape: Shape, mesh_shape: Tuple[int, int], request):
    def reduce_scatter(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_reduce_scatter(input, 3)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
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
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_to_flatbuffer(
        reduce_scatter, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


def pseudo_golden_collective_permute(
    input_tensor: torch.Tensor,
    source_target_pairs: List[Tuple[int, int]],
):
    shards = list(torch.chunk(input_tensor, 2, dim=3))
    permuted_tensor = shards.copy()
    for source, target in source_target_pairs:
        permuted_tensor[target] = shards[source]
    result_tensor = torch.cat(permuted_tensor, dim=3)
    return result_tensor


@pytest.mark.parametrize("shape", [(1, 1, 128, 1024)])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_collective_permute(shape: Shape, mesh_shape: Tuple[int, int], request):
    def collective_permute(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_collective_permute(input, [(0, 1), (1, 0)])
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        reduced = builder.collective_permute(
            sharded,
            source_target_pairs=[(0, 1), (1, 0)],
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_to_flatbuffer(
        collective_permute, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


@pytest.mark.parametrize("shapes", [[(2048, 196), (196, 4096)]])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_matmul_1x2(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def matmul_1x2(in0: Operand, in1: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        weight = builder._get_golden_tensor(in1)
        golden_output = torch.matmul(input, weight)
        builder.set_graph_input_output([input, weight], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 1),
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
        matmul_1x2, shapes, mesh_shape=mesh_shape, test_base=request.node.name
    )


@pytest.mark.parametrize("shape", [(1, 256, 64, 256)])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_neg_1x2_dim_3(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_1x2_dim_3(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_to_flatbuffer(
        neg_1x2_dim_3,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
    )


@pytest.mark.parametrize("shape", [(1, 256, 64, 256)])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_neg_1x2_dim_1(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_1x2_dim_1(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(-1, 1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(-1, 1),
        )

    compile_to_flatbuffer(
        neg_1x2_dim_1,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
    )


@pytest.mark.parametrize("shapes", [[(512, 1024), (512, 1024)]])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
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
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        partial_sum = builder.add(sharded_in0, sharded_in1)
        return builder.mesh_shard(
            partial_sum,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )

    compile_to_flatbuffer(
        eltwise_multidevice, shapes, mesh_shape=mesh_shape, test_base=request.node.name
    )
