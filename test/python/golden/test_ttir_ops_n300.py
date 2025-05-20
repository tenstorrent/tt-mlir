# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from typing import List, Tuple, Union
from ttir_builder.utils import compile_to_flatbuffer
from ttir_builder import Operand, TTIRBuilder, Shape

pytestmark = pytest.mark.n300


def pseudo_golden_all_gather(
    input_tensor: torch.Tensor,
):
    output_tensor = input_tensor.clone()
    return output_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 128, 128),
        (1, 32, 120, 128),
        (1, 32, 60, 128),
        (1, 32, 30, 128),
        (1, 32, 2, 128),
        pytest.param(
            (1, 32, 128, 120), marks=pytest.mark.fails_golden
        ),  # https://github.com/tenstorrent/tt-metal/issues/21964
        pytest.param((1, 32, 120, 120), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 128, 60), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 60, 60), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 128, 30), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 30, 30), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 128, 2), marks=pytest.mark.fails_golden),
        pytest.param((1, 32, 2, 2), marks=pytest.mark.fails_golden),
        pytest.param((1, 1, 1, 2), marks=pytest.mark.fails_golden),
        pytest.param((1, 1, 10, 10), marks=pytest.mark.fails_golden),
    ],
)
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
        all_gather,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def pseudo_golden_all_reduce(input_tensor: torch.Tensor):
    shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=3)
    output_tensor = shard_1 + shard_2
    return output_tensor


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
        pytest.param(
            (1, 1, 1, 2), marks=pytest.mark.fails_golden
        ),  # https://github.com/tenstorrent/tt-metal/issues/21964
        pytest.param(
            (1, 1, 128, 516), marks=pytest.mark.run_error
        ),  # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param((1, 1, 128, 516), marks=pytest.mark.run_error),
        pytest.param((1, 1, 126, 516), marks=pytest.mark.run_error),
        pytest.param((1, 1, 130, 516), marks=pytest.mark.run_error),
        pytest.param((1, 1, 32, 4), marks=pytest.mark.run_error),
        pytest.param((1, 1, 32, 8), marks=pytest.mark.run_error),
    ],
)
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
        all_reduce,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def pseudo_golden_reduce_scatter(
    input_tensor: torch.Tensor,
    scatter_dim: int,
):
    shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=scatter_dim)
    output_tensor = shard_1 + shard_2
    return output_tensor


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
        pytest.param(
            (1, 1, 128, 508), marks=pytest.mark.fails_golden
        ),  # ToDo: Analyze why this fails
        pytest.param(
            (1, 1, 128, 64), marks=pytest.mark.run_error
        ),  # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param((1, 1, 128, 516), marks=pytest.mark.run_error),
        pytest.param(
            (1, 1, 64, 128), marks=pytest.mark.run_error
        ),  # hangs # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param(
            (1, 1, 32, 128), marks=pytest.mark.run_error
        ),  # hangs # https://github.com/tenstorrent/tt-metal/issues/21987
    ],
)
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
        reduce_scatter,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
        collective_permute,
        [shape],
        mesh_shape=mesh_shape,
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
        pytest.param(
            [(2050, 196), (196, 4098)], marks=pytest.mark.run_error
        ),  # https://github.com/tenstorrent/tt-metal/issues/21987
        pytest.param([(10, 4), (4, 20)], marks=pytest.mark.run_error),
    ],
)
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
        matmul_1x2,
        shapes,
        mesh_shape=mesh_shape,
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
        eltwise_multidevice,
        shapes,
        mesh_shape=mesh_shape,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def pseudo_golden_all_to_all(
    input: torch.Tensor,
    split_dim: int,
    concat_dim: int,
    mesh_shape: Tuple[int, int],
    shard_dims: Tuple[int, int],
    cluster_axis: int,
):
    # sharding
    num_of_clusters = mesh_shape[1 - cluster_axis]
    devices_in_cluster = mesh_shape[cluster_axis]
    clusters = []

    def sharding(
        input: torch.Tensor, shard_dim: Union[None, int], number_of_shards: int
    ):
        output = []
        if shard_dim == None or shard_dim == -1:
            output = [input]
        else:
            output = torch.chunk(input, number_of_shards, dim=shard_dim)
        return output

    for cluster in sharding(input, shard_dims[1 - cluster_axis], num_of_clusters):
        clusters.append(sharding(cluster, shard_dims[cluster_axis], devices_in_cluster))
    output_per_cluster = []
    # all to all
    for cluster in clusters:
        assert devices_in_cluster == len(cluster)
        # prepare for all to all
        scattered_tensors_per_device = []
        for i in range(devices_in_cluster):
            scattered_tensors_per_device.append([])
        # slice and scatter/gather
        for device_tensor in cluster:
            sliced = torch.chunk(device_tensor, devices_in_cluster, dim=split_dim)
            for i in range(devices_in_cluster):
                scattered_tensors_per_device[i].append(sliced[i])
        # concat
        output_per_device = []
        for i in range(devices_in_cluster):
            output_per_device.append(
                torch.cat(scattered_tensors_per_device[i], dim=concat_dim)
            )
        output_per_cluster.append(output_per_device)

    def unsharding(inputs: List[torch.Tensor], shard_dim: Union[None, int]):
        if shard_dim == None or shard_dim == -1:
            assert len(inputs) == 1
            return inputs[0]
        else:
            return torch.cat(inputs, dim=shard_dim)

    # unsharding
    outputs = []
    for output in output_per_cluster:
        outputs.append(unsharding(output, shard_dims[cluster_axis]))
    output = unsharding(outputs, shard_dims[1 - cluster_axis])
    return output


def generateShardShape(
    input_rank: int, mesh_shape: Tuple[int, int], shard_dims: Tuple[int, int]
):
    shard_shape = [1] * input_rank
    if shard_dims[0] != -1:
        shard_shape[shard_dims[0]] = mesh_shape[0]
    if shard_dims[1] != -1:
        shard_shape[shard_dims[1]] = mesh_shape[1]
    return shard_shape


def isValidDeviceSharding(input_shape: Shape, mesh_shape: Tuple[int, int], shard_dims):
    if shard_dims[0] == shard_dims[1]:
        return False
    shard_shape = generateShardShape(len(input_shape), mesh_shape, shard_dims)
    if all(x == 1 for x in shard_shape):
        return False
    if len(input_shape) != len(shard_shape):
        return False
    if any(i % s != 0 for i, s in zip(input_shape, shard_shape)):
        return False
    return True


@pytest.mark.parametrize("input_shape", [(128, 128), (4, 4), (128, 64), (192, 12)])
@pytest.mark.parametrize("mesh_shape", [(1, 2), (2, 1)])
@pytest.mark.parametrize("split_dim", [0, 1])
@pytest.mark.parametrize("concat_dim", [0, 1])
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("shard_dim_0", [-1, 0, 1])
@pytest.mark.parametrize("shard_dim_1", [-1, 0, 1])
def test_all_to_all_2d(
    input_shape: Shape,
    mesh_shape: Tuple[int, int],
    split_dim,
    concat_dim,
    cluster_axis,
    shard_dim_0,
    shard_dim_1,
    request,
):
    shard_dims = (shard_dim_0, shard_dim_1)
    if isValidDeviceSharding(input_shape, mesh_shape, shard_dims) == False:
        pytest.skip("Sharding is not possible")
    shard_shape = generateShardShape(len(input_shape), mesh_shape, shard_dims)
    if shard_dims[cluster_axis] == -1 or shard_shape[shard_dims[cluster_axis]] == 1:
        pytest.skip("all to all across 1 device")
    if (input_shape[split_dim] / shard_shape[split_dim]) % mesh_shape[
        cluster_axis
    ] != 0:
        pytest.skip("Cannot split tensor evenly")

    def all_to_all(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_to_all(
            input,
            split_dim=split_dim,
            concat_dim=concat_dim,
            mesh_shape=mesh_shape,
            shard_dims=shard_dims,
            cluster_axis=cluster_axis,
        )
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        gathered = builder.all_to_all(
            sharded,
            split_dim=split_dim,
            concat_dim=concat_dim,
            split_count=mesh_shape[cluster_axis],
            cluster_axis=cluster_axis,
        )
        return builder.mesh_shard(
            gathered,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )

    def seq2str(seq, delim="x"):
        return delim.join(str(num) for num in seq)

    def generate_test_base():
        return f"test-all-to-all_input_{seq2str(input_shape)}_mesh_{seq2str(mesh_shape)}_split_{split_dim}_concat_{concat_dim}_cluster_{cluster_axis}_shard-dims_{seq2str(shard_dims)}_shard-shape_{seq2str(shard_shape)}"

    compile_to_flatbuffer(
        all_to_all,
        [input_shape],
        mesh_shape=mesh_shape,
        test_base=generate_test_base(),
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "input_shape",
    [(1, 1, 128, 128), (1, 1, 4, 4), (1, 1, 128, 64), (1, 16, 64, 1), (16, 4, 4, 128)],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2), (2, 1)])
@pytest.mark.parametrize("split_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("concat_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("shard_dim_0", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("shard_dim_1", [-1, 0, 1, 2, 3])
def test_all_to_all_4d(
    input_shape: Shape,
    mesh_shape: Tuple[int, int],
    split_dim,
    concat_dim,
    cluster_axis,
    shard_dim_0,
    shard_dim_1,
    request,
):
    shard_dims = (shard_dim_0, shard_dim_1)
    if isValidDeviceSharding(input_shape, mesh_shape, shard_dims) == False:
        pytest.skip("Sharding is not possible")
    shard_shape = generateShardShape(len(input_shape), mesh_shape, shard_dims)
    if shard_dims[cluster_axis] == -1 or shard_shape[shard_dims[cluster_axis]] == 1:
        pytest.skip("all to all across 1 device")
    if (input_shape[split_dim] / shard_shape[split_dim]) % mesh_shape[
        cluster_axis
    ] != 0:
        pytest.skip("Cannot split tensor evenly")

    def all_to_all(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_to_all(
            input,
            split_dim=split_dim,
            concat_dim=concat_dim,
            mesh_shape=mesh_shape,
            shard_dims=shard_dims,
            cluster_axis=cluster_axis,
        )
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        gathered = builder.all_to_all(
            sharded,
            split_dim=split_dim,
            concat_dim=concat_dim,
            split_count=mesh_shape[cluster_axis],
            cluster_axis=cluster_axis,
        )
        return builder.mesh_shard(
            gathered,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )

    def seq2str(seq, delim="x"):
        return delim.join(str(num) for num in seq)

    def generate_test_base():
        return f"test-all-to-all_input_{seq2str(input_shape)}_mesh_{seq2str(mesh_shape)}_split_{split_dim}_concat_{concat_dim}_cluster_{cluster_axis}_shard-dims_{seq2str(shard_dims)}_shard-shape_{seq2str(shard_shape)}"

    compile_to_flatbuffer(
        all_to_all,
        [input_shape],
        mesh_shape=mesh_shape,
        test_base=generate_test_base(),
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
