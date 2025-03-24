# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect
import torch

from ttmlir.test_utils import compile_to_flatbuffer, set_output_path
from ttmlir.ttir_builder import Operand, TTIRBuilder


@compile_to_flatbuffer(
    [
        (1, 32, 128, 128),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
)
def test_all_gather(in0: Operand, builder: TTIRBuilder):
    def pseudo_golden_all_gather(in0):
        input_tensor = builder._get_golden_tensor(in0)
        output_tensor = input_tensor.clone()
        return output_tensor

    input_shape = builder._get_golden_tensor(in0).shape
    half_shape = input_shape[:-1] + (input_shape[-1] // 2,)
    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
        output_shape=half_shape,
    )
    result = builder.all_gather(
        sharded,
        all_gather_dim=3,
        cluster_axis=1,
        output_shape=half_shape,
    )
    return builder.mesh_shard(
        result,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
        provided_golden=pseudo_golden_all_gather(in0),
    )


@compile_to_flatbuffer(
    [
        (1, 64, 512, 1024),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
)
def test_all_reduce(in0: Operand, builder: TTIRBuilder):
    def pseudo_golden_all_reduce(in0):
        input_tensor = builder._get_golden_tensor(in0)
        shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=3)
        output_tensor = shard_1 + shard_2
        return output_tensor

    input_shape = builder._get_golden_tensor(in0).shape
    half_shape = input_shape[:-1] + (input_shape[-1] // 2,)

    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
        output_shape=half_shape,
    )
    result = builder.all_reduce(
        sharded,
        reduce_type="#tt.reduce_type<sum>",
        cluster_axis=1,
        output_shape=half_shape,
    )
    return builder.mesh_shard(
        result,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
        provided_golden=pseudo_golden_all_reduce(in0),
    )


@compile_to_flatbuffer(
    [
        (1, 1, 32, 1024),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
)
def test_reduce_scatter(in0: Operand, builder: TTIRBuilder):
    def pseudo_golden_reduce_scatter(in0):
        input_tensor = builder._get_golden_tensor(in0)
        shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=3)
        output_tensor = shard_1 + shard_2
        # reduce with sum
        # scatter -> 'shard to full' is not needed
        return output_tensor

    input_shape = builder._get_golden_tensor(in0).shape
    input_shape = builder._get_golden_tensor(in0).shape
    half_shape = input_shape[:-1] + (input_shape[-1] // 2,)
    quarter_shape = input_shape[:-1] + (half_shape[-1] // 2,)
    sharded = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
        output_shape=half_shape,
    )
    result = builder.reduce_scatter(
        sharded,
        reduce_type="#tt.reduce_type<sum>",
        scatter_dim=3,
        cluster_axis=1,
        output_shape=quarter_shape,
    )
    return builder.mesh_shard(
        result,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
        provided_golden=pseudo_golden_reduce_scatter(in0),
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
