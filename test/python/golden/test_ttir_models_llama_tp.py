# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from typing import List, Tuple
from collections import OrderedDict

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_golden import BuilderGoldenTensor
from builder.base.builder_utils import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


# utility functions to increase readability
def get_input_tensors_from_builder(args: List, builder: TTIRBuilder):
    input_tensors = []
    for arg in args:
        input_tensors.append(builder._get_golden_tensor(arg))
    return input_tensors


def full_to_shard_device(input, builder, dim):
    rank = len(builder._get_golden_tensor(input).shape)
    num_devices = builder.mesh_shape[1]
    shard_shape = [1] * rank
    shard_shape[dim] = num_devices
    return builder.mesh_shard(
        input,
        shard_direction="#ttcore.shard_direction<full_to_shard>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape,
        shard_dims=[-1, dim],
    )


def full_to_shard_replicate(input, builder):
    return builder.mesh_shard(
        input,
        shard_direction="#ttcore.shard_direction<full_to_shard>",
        shard_type="#ttcore.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
    )


def shard_to_full_device(input, builder, dim):
    rank = len(builder._get_golden_tensor(input).shape)
    num_devices = builder.mesh_shape[1]
    shard_shape = [1] * rank
    shard_shape[dim] = num_devices
    return builder.mesh_shard(
        input,
        shard_direction="#ttcore.shard_direction<shard_to_full>",
        shard_type="#ttcore.shard_type<devices>",
        shard_shape=shard_shape,
        shard_dims=[-1, dim],
    )


def shard_to_full_replicate(input, builder):
    return builder.mesh_shard(
        input,
        shard_direction="#ttcore.shard_direction<shard_to_full>",
        shard_type="#ttcore.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
    )


def golden_llama(
    arg0: torch.Tensor,
    arg1: torch.Tensor,
    arg2: torch.Tensor,
    arg3: torch.Tensor,
    arg4: torch.Tensor,
    arg5: torch.Tensor,
    arg6: torch.Tensor,
    arg7: torch.Tensor,
    arg8: torch.Tensor,
    arg9: torch.Tensor,
    arg10: torch.Tensor,
    arg11: torch.Tensor,
    arg12: torch.Tensor,
    arg13: torch.Tensor,
    arg14: torch.Tensor,
):
    output1 = torch.squeeze(arg0, dim=0)
    output3 = torch.matmul(output1, arg11)
    output5 = torch.reshape(output3, shape=(1, 128, 32, 128))
    output7 = torch.transpose(output5, dim0=-3, dim1=-2)
    output9 = torch.unsqueeze(arg2, dim=1)
    output11 = torch.matmul(arg3, output9)  # [1, 64, 128]
    output13 = torch.transpose(output11, dim0=-2, dim1=-1)  # [1, 128, 64]
    output15 = torch.concat((output13, output13), dim=-1)  # [1, 128, 128]
    output17 = torch.cos(output15)  # [1, 128, 128]
    output19 = torch.unsqueeze(output17, dim=1)  # [1, 1, 128, 128]
    output21 = torch.multiply(output7, output19)  # [1, 32, 128, 128]
    output23 = torch.transpose(output7, dim0=-2, dim1=-1)  # [1, 32, 128, 128]
    output25 = torch.matmul(arg4, output23)  # [1, 32, 64, 128]
    output27 = torch.transpose(output25, dim0=-2, dim1=-1)  # [1, 32, 128, 64]
    output29 = torch.multiply(output27, arg5)  # [1, 32, 128, 64]
    output31 = torch.transpose(output7, dim0=-2, dim1=-1)  # [1, 32, 128, 128]
    output33 = torch.matmul(arg6, output31)  # [1, 32, 64, 128]
    output35 = torch.transpose(output33, dim0=-2, dim1=-1)  # [1, 32, 128, 64]
    output37 = torch.concat((output29, output35), dim=-1)  # [1, 32, 128, 128]
    output39 = torch.sin(output15)  # [1, 128, 128]
    output41 = torch.unsqueeze(output39, dim=1)  # [1, 1, 128, 128]
    output43 = torch.multiply(output37, output41)  # [1, 32, 128, 128]
    output45 = torch.add(output21, output43)  # [1, 32, 128, 128]
    output47 = torch.squeeze(output45, dim=0)  # [32, 128, 128]

    output49 = torch.matmul(output1, arg12)  # [128, 4096]
    output51 = torch.reshape(output49, shape=(1, 128, 32, 128))  # [1, 128, 32, 128]
    output53 = torch.transpose(output51, dim0=-3, dim1=-2)  # [1, 32, 128, 128]
    output55 = torch.multiply(output53, output19)  # [1, 32, 128, 128]
    output57 = torch.transpose(output53, dim0=-2, dim1=-1)  # [1, 32, 128, 128]
    output59 = torch.matmul(arg7, output57)  # [1, 32, 64, 128]
    output61 = torch.transpose(output59, dim0=-2, dim1=-1)  # [1, 32, 128, 64]
    output63 = torch.multiply(output61, arg8)  # [1, 32, 128, 64]
    output65 = torch.transpose(output53, dim0=-2, dim1=-1)  # [1, 32, 128, 128]
    output67 = torch.matmul(arg9, output65)  # [1, 32, 64, 128]
    output69 = torch.transpose(output67, dim0=-2, dim1=-1)  # [1, 32, 128, 64]
    output71 = torch.concat([output63, output69], -1)  # [1, 32, 128, 128]
    output73 = torch.multiply(output71, output41)  # [1, 32, 128, 128]
    output75 = torch.add(output55, output73)  # [1, 32, 128, 128]
    output77 = torch.squeeze(output75, dim=0)  # [32, 128, 128]

    output79 = torch.transpose(output77, dim0=-2, dim1=-1)  # [32, 128, 128]
    output81 = torch.matmul(output47, output79)  # [32, 128, 128]
    output83 = torch.unsqueeze(output81, dim=0)  # [1, 32, 128, 128]
    output1 = torch.squeeze(arg0, dim=0)
    output85 = torch.multiply(output83, arg10)  # [1, 32, 128, 128]
    output87 = torch.add(output85, arg1)  # [1, 32, 128, 128]
    output89 = torch.softmax(output87, dim=-1)  # [1, 32, 128, 128]
    output91 = torch.squeeze(output89, dim=0)  # [32, 128, 128]
    output93 = torch.matmul(output1, arg13)  # [128, 4096]
    output95 = torch.reshape(output93, shape=(1, 128, 32, 128))  # [1, 128, 32, 128]
    output97 = torch.transpose(output95, dim0=-3, dim1=-2)  # [1, 32, 128, 128]
    output99 = torch.transpose(output97, dim0=-2, dim1=-1)  # [1, 32, 128, 128]
    output101 = torch.squeeze(output99, dim=0)  # [32, 128, 128]
    output103 = torch.transpose(output101, dim0=-2, dim1=-1)  # [32, 128, 128]
    output105 = torch.matmul(output91, output103)  # [32, 128, 128]
    output107 = torch.unsqueeze(output105, dim=0)  # [1, 32, 128, 128]
    output109 = torch.transpose(output107, dim0=-3, dim1=-2)  # [1, 128, 32, 128]
    output111 = torch.reshape(output109, shape=(128, 4096))  # [128, 4096]
    output113 = torch.matmul(output111, arg14)  # [128, 4096]
    output115 = torch.unsqueeze(output113, dim=0)  # [1, 128, 4096]
    return output115


# llama attention with 1xN Tensor Parallelism
@pytest.mark.skip(reason="PCC fail")
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 128, 4096),  # arg0
            (1, 1, 128, 128),  # arg1
            (1, 128),  # arg2
            (1, 64, 1),  # arg3
            (1, 32, 64, 128),  # arg4
            (1, 1),  # arg5
            (1, 32, 64, 128),  # arg6
            (1, 32, 64, 128),  # arg7
            (1, 1),  # arg8
            (1, 32, 64, 128),  # arg9
            (1, 1),  # arg10
            (4096, 4096),  # arg11
            (4096, 4096),  # arg12
            (4096, 4096),  # arg13
            (4096, 4096),  # arg14
        ],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 15], ids=["fp32"])
@pytest.mark.parametrize(
    "target",
    [
        "ttnn",
        pytest.param("ttmetal", marks=pytest.mark.skip("TTMetal not supported yet")),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2), (1, 8), (2, 4)])
def test_llama_attention_1xn_tp(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    target: str,
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    def model(
        arg0: Operand,
        arg1: Operand,
        arg2: Operand,
        arg3: Operand,
        arg4: Operand,
        arg5: Operand,
        arg6: Operand,
        arg7: Operand,
        arg8: Operand,
        arg9: Operand,
        arg10: Operand,
        arg11: Operand,
        arg12: Operand,
        arg13: Operand,
        arg14: Operand,
        builder: TTIRBuilder,
    ):
        local_vars = locals()
        operands = [local_vars[f"arg{i}"] for i in range(15)]

        output1 = full_to_shard_device(arg0, builder, 2)
        output1 = builder.squeeze(output1, 0)
        arg11_mesh_shard = full_to_shard_device(arg11, builder, 0)
        output3 = builder.matmul(output1, arg11_mesh_shard)
        output3 = builder.all_reduce(
            output3, reduce_type="#ttcore.reduce_type<sum>", cluster_axis=1
        )
        output5 = builder.reshape(output3, (1, 128, 32, 128))
        output7 = builder.transpose(output5, -3, -2)
        output7 = shard_to_full_replicate(output7, builder)
        output7 = full_to_shard_device(output7, builder, 3)
        output9 = full_to_shard_device(arg2, builder, 1)
        output9 = builder.unsqueeze(output9, 1)
        arg3_mesh_shard = full_to_shard_replicate(arg3, builder)
        output11 = builder.matmul(arg3_mesh_shard, output9)
        output11 = builder.all_gather(
            output11,
            all_gather_dim=2,
            cluster_axis=1,
        )
        output13 = builder.transpose(output11, -2, -1)
        output15 = builder.concat([output13, output13], -1)
        output15 = shard_to_full_replicate(output15, builder)
        output15 = full_to_shard_device(output15, builder, 2)
        output17 = builder.cos(output15)
        output19 = builder.unsqueeze(output17, 1)
        output21 = builder.multiply(output7, output19)
        output23 = builder.transpose(output7, -2, -1)
        arg4_mesh_shard = full_to_shard_device(arg4, builder, 3)
        output25 = builder.matmul(arg4_mesh_shard, output23)
        output25 = builder.reduce_scatter(
            output25,
            reduce_type="#ttcore.reduce_type<sum>",
            scatter_dim=3,
            cluster_axis=1,
        )
        output27 = builder.transpose(output25, -2, -1)
        arg5_mesh_shard = full_to_shard_replicate(arg5, builder)
        output29 = builder.multiply(output27, arg5_mesh_shard)
        output31 = builder.transpose(output7, -2, -1)
        arg6_mesh_shard = full_to_shard_device(arg6, builder, 3)
        output33 = builder.matmul(arg6_mesh_shard, output31)
        output33 = builder.reduce_scatter(
            output33,
            reduce_type="#ttcore.reduce_type<sum>",
            scatter_dim=3,
            cluster_axis=1,
        )
        output35 = builder.transpose(output33, -2, -1)
        output35 = builder.all_gather(output35, all_gather_dim=2, cluster_axis=1)
        output29 = builder.all_gather(output29, all_gather_dim=2, cluster_axis=1)
        output37 = builder.concat([output29, output35], -1)
        output37 = shard_to_full_replicate(output37, builder)
        output37 = full_to_shard_device(output37, builder, 3)
        output39 = builder.sin(output15)
        output41 = builder.unsqueeze(output39, 1)
        output43 = builder.multiply(output37, output41)
        output45 = builder.add(output21, output43)
        output47 = builder.squeeze(output45, 0)
        arg12_mesh_shard = full_to_shard_device(arg12, builder, 0)
        output49 = builder.matmul(output1, arg12_mesh_shard)
        output49 = builder.all_reduce(
            output49, reduce_type="#ttcore.reduce_type<sum>", cluster_axis=1
        )
        output51 = builder.reshape(output49, (1, 128, 32, 128))
        output53 = builder.transpose(output51, -3, -2)
        output53 = shard_to_full_replicate(output53, builder)
        output53 = full_to_shard_device(output53, builder, 3)
        output55 = builder.multiply(output53, output19)
        output57 = builder.transpose(output53, -2, -1)
        arg7_mesh_shard = full_to_shard_device(arg7, builder, 3)
        output59 = builder.matmul(arg7_mesh_shard, output57)
        output59 = builder.reduce_scatter(
            output59,
            reduce_type="#ttcore.reduce_type<sum>",
            scatter_dim=3,
            cluster_axis=1,
        )
        output61 = builder.transpose(output59, -2, -1)
        arg8_mesh_shard = full_to_shard_replicate(arg8, builder)
        output63 = builder.multiply(output61, arg8_mesh_shard)
        output65 = builder.transpose(output53, -2, -1)
        arg9_mesh_shard = full_to_shard_device(arg9, builder, 3)
        output67 = builder.matmul(arg9_mesh_shard, output65)
        output67 = builder.reduce_scatter(
            output67,
            reduce_type="#ttcore.reduce_type<sum>",
            scatter_dim=3,
            cluster_axis=1,
        )
        output69 = builder.transpose(output67, -2, -1)
        output63 = builder.all_gather(output63, all_gather_dim=2, cluster_axis=1)
        output69 = builder.all_gather(output69, all_gather_dim=2, cluster_axis=1)
        output71 = builder.concat([output63, output69], -1)
        output71 = shard_to_full_replicate(output71, builder)
        output71 = full_to_shard_device(output71, builder, 3)
        output73 = builder.multiply(output71, output41)
        output75 = builder.add(output55, output73)
        output77 = builder.squeeze(output75, 0)
        output79 = builder.transpose(output77, -2, -1)

        # Use weight sharding to avoid PCC drop
        output47 = shard_to_full_device(output47, builder, 2)
        output79 = shard_to_full_device(output79, builder, 1)
        output47 = full_to_shard_replicate(output47, builder)
        output79 = full_to_shard_device(output79, builder, 2)
        output81 = builder.matmul(output47, output79)
        output83 = builder.unsqueeze(output81, 0)
        output83 = shard_to_full_device(output83, builder, 3)

        output83 = full_to_shard_device(output83, builder, 3)
        arg10_mesh_shard = full_to_shard_replicate(arg10, builder)
        output85 = builder.multiply(output83, arg10_mesh_shard)
        arg1_mesh_shard = full_to_shard_device(arg1, builder, 3)
        output87 = builder.add(output85, arg1_mesh_shard)
        output87 = shard_to_full_device(output87, builder, 3)
        output87 = full_to_shard_device(output87, builder, 2)
        output89 = builder.softmax(output87, -1, numeric_stable=True)
        output89 = shard_to_full_device(output89, builder, 2)
        output89 = full_to_shard_device(output89, builder, 3)
        output91 = builder.squeeze(output89, 0)
        arg13_mesh_shard = full_to_shard_device(arg13, builder, 0)
        output93 = builder.matmul(output1, arg13_mesh_shard)
        output93 = builder.all_reduce(
            output93, reduce_type="#ttcore.reduce_type<sum>", cluster_axis=1
        )
        output95 = builder.reshape(output93, (1, 128, 32, 128))
        output95 = shard_to_full_replicate(output95, builder)
        output95 = full_to_shard_device(output95, builder, 1)
        output97 = builder.transpose(output95, -3, -2)
        output99 = builder.transpose(output97, -2, -1)
        output101 = builder.squeeze(output99, 0)
        output103 = builder.transpose(output101, -2, -1)
        output105 = builder.matmul(output91, output103)
        output105 = builder.all_reduce(
            output105, reduce_type="#ttcore.reduce_type<sum>", cluster_axis=1
        )
        output107 = builder.unsqueeze(output105, 0)
        output109 = builder.transpose(output107, -3, -2)
        output111 = builder.reshape(output109, (128, 4096))
        arg14_mesh_shard = full_to_shard_device(arg14, builder, 1)
        output113 = builder.matmul(output111, arg14_mesh_shard)
        output115 = builder.unsqueeze(output113, 0)
        output116 = shard_to_full_device(output115, builder, dim=2)

        # Retrieve the input tensors from the builder.
        input_tensors_torch = [
            input_tensor.shard_map[0]
            for input_tensor in get_input_tensors_from_builder(
                operands,
                builder,
            )
        ]

        # Generate the golden output using a single CPU device
        golden = golden_llama(*input_tensors_torch)

        # Set the input and output tensors for the program-level golden check.
        builder.set_goldens(
            {
                operand: input_tensor
                for operand, input_tensor in zip(operands, input_tensors_torch)
            },
            {
                output116: golden,
            },
        )
        # Clear the golden check list for the op-level golden check.
        # This is a bit hacky, but it's the only way to get both the op-level and program-level golden checks.
        builder.set_goldens_to_check([], override=True)

        return output116

    compile_and_execute_ttir(
        model,
        shapes,
        dtypes,
        target=target,
        device=device,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
