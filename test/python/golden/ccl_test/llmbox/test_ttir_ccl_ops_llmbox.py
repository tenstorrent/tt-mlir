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
        (1, 1, 256, 512),
    ],
    targets=["ttnn"],
    mesh_shape=[2, 4],
)
def test_all_gather(in0: Operand, builder: TTIRBuilder):
    input = builder._get_golden_tensor(in0)
    golden_output = pseudo_golden_all_gather(input)
    builder.set_graph_input_output([input], [golden_output])

    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 2, 4],
        shard_dims=[2, 3],
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
        shard_shape=[1, 1, 2, 1],
        shard_dims=[2, -1],
    )


def pseudo_golden_all_reduce(input_tensor: torch.Tensor):
    shards = torch.chunk(input_tensor, 4, dim=3)
    return sum(shards)


@compile_to_flatbuffer(
    [
        (1, 1, 256, 512),
    ],
    targets=["ttnn"],
    mesh_shape=[2, 4],
)
def test_all_reduce(in0: Operand, builder: TTIRBuilder):
    input = builder._get_golden_tensor(in0)
    golden_output = pseudo_golden_all_reduce(input)
    builder.set_graph_input_output([input], [golden_output])

    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 2, 4],
        shard_dims=[2, 3],
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
        shard_shape=[1, 1, 2, 1],
        shard_dims=[2, -1],
    )


def pseudo_golden_reduce_scatter(
    input_tensor: torch.Tensor,
    scatter_dim: int,
):
    shards = torch.chunk(input_tensor, 4, dim=scatter_dim)
    return sum(shards)


@compile_to_flatbuffer(
    [
        (1, 1, 8192, 512),
    ],
    targets=["ttnn"],
    mesh_shape=[2, 4],
)
def test_reduce_scatter(in0: Operand, builder: TTIRBuilder):
    input = builder._get_golden_tensor(in0)
    golden_output = pseudo_golden_reduce_scatter(input, 3)
    builder.set_graph_input_output([input], [golden_output])

    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 2, 4],
        shard_dims=[2, 3],
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
        shard_shape=[1, 1, 2, 4],
        shard_dims=[2, 3],
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


@compile_to_flatbuffer(
    [
        (1, 1, 256, 4096),
    ],
    targets=["ttnn"],
    mesh_shape=[2, 4],
)
def test_collective_permute(in0: Operand, builder: TTIRBuilder):
    input = builder._get_golden_tensor(in0)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
    golden_output = pseudo_golden_collective_permute(input, pairs)
    builder.set_graph_input_output([input], [golden_output])

    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 2, 4],
        shard_dims=[2, 3],
    )
    reduced = builder.collective_permute(
        sharded,
        source_target_pairs=pairs,
    )
    return builder.mesh_shard(
        reduced,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 2, 4],
        shard_dims=[2, 3],
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
