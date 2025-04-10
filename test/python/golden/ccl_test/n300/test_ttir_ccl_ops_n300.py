# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect
import torch

from typing import Callable, List, Tuple
from ttmlir.test_utils import compile_to_flatbuffer, set_output_path
from ttmlir.ttir_builder import Operand, TTIRBuilder


def pseudo_golden_all_gather(
    input_tensor: torch.Tensor,
):
    output_tensor = input_tensor.clone()
    return output_tensor


@compile_to_flatbuffer(
    [
        (1, 32, 128, 128),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
)
def test_all_gather(in0: Operand, builder: TTIRBuilder):
    input = builder._get_golden_tensor(in0)
    golden_output = pseudo_golden_all_gather(input)
    builder.set_graph_input_output([input], [golden_output])

    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
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
        shard_shape=[1],
        shard_dims=[-1],
    )


def pseudo_golden_all_reduce(input_tensor: torch.Tensor):
    shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=3)
    output_tensor = shard_1 + shard_2
    return output_tensor


@compile_to_flatbuffer(
    [
        (1, 1, 128, 512),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
)
def test_all_reduce(in0: Operand, builder: TTIRBuilder):
    input = builder._get_golden_tensor(in0)
    golden_output = pseudo_golden_all_reduce(input)
    builder.set_graph_input_output([input], [golden_output])

    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
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
        shard_shape=[1],
        shard_dims=[-1],
    )


def pseudo_golden_reduce_scatter(
    input_tensor: torch.Tensor,
    scatter_dim: int,
):
    shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=scatter_dim)
    output_tensor = shard_1 + shard_2
    return output_tensor


@compile_to_flatbuffer(
    [
        (1, 1, 32, 1024),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
)
def test_reduce_scatter(in0: Operand, builder: TTIRBuilder):
    input = builder._get_golden_tensor(in0)
    golden_output = pseudo_golden_reduce_scatter(input, 3)
    builder.set_graph_input_output([input], [golden_output])

    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
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
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
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


@compile_to_flatbuffer(
    [
        (1, 1, 128, 1024),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
)
def test_collective_permute(in0: Operand, builder: TTIRBuilder):
    input = builder._get_golden_tensor(in0)
    golden_output = pseudo_golden_collective_permute(input, [(0, 1), (1, 0)])
    builder.set_graph_input_output([input], [golden_output])

    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
    )
    reduced = builder.collective_permute(
        sharded,
        source_target_pairs=[(0, 1), (1, 0)],
    )
    return builder.mesh_shard(
        reduced,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
    )


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Run TTIR Builder Op tests")
    parser.add_argument(
        "--path",
        type=str,
        help="Optional output path for the flatbuffer. Creates path if supplied path doesn't exist",
    )
    args = parser.parse_args()

    if args.path and os.path.exists(args.path):
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        set_output_path(args.path)

    test_functions = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isfunction
    )
    for function_name, func in test_functions:
        if function_name.startswith("test_"):
            func()
