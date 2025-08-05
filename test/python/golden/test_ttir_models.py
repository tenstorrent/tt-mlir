# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from typing import List

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.ttir.ttir_utils import compile_ttir_to_flatbuffer


@pytest.mark.parametrize("shapes", [[(32, 32), (32, 32), (32, 32)]], ids=["32x32"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3], ids=["f32"])
def test_arbitrary_model(shapes: List[Shape], dtypes: List[torch.dtype], request):
    def model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
        add = builder.add(in0, in1)
        exp = builder.exp(in2)
        return builder.multiply(add, exp)

    compile_ttir_to_flatbuffer(
        model,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("dtypes", [[torch.float32] * 5], ids=["f32"])
@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 784), (784, 256), (1, 256), (256, 10), (1, 10)],
        [(1, 1024), (1024, 256), (1, 256), (256, 10), (1, 10)],
        [
            (1, 1024),
            (1024, 256),
            (1, 256),
            (256, 26),
            (1, 26),
        ],
    ],
    ids=["28x28_digits", "32x32_digits", "32x32_letters"],
)
@pytest.mark.parametrize(
    "target",
    [
        "ttnn",
        pytest.param("ttmetal", marks=pytest.mark.skip("TTMetal not supported yet")),
    ],
)
def test_mnist(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    target: str,
    request,
):
    def model(
        in0: Operand,  # Input 28x28 image
        in1: Operand,  # Weight 1
        in2: Operand,  # Bias 1
        in3: Operand,  # Weight 2
        in4: Operand,  # Bias 2
        builder: TTIRBuilder,
    ):
        matmul_1 = builder.matmul(in0, in1)
        add_2 = builder.add(matmul_1, in2)
        relu_3 = builder.relu(add_2)
        matmul_5 = builder.matmul(relu_3, in3)
        add_6 = builder.add(matmul_5, in4)
        return builder.softmax(add_6, dimension=1)

    # TODO: figure out a better way to name these tests for filename purposes
    compile_ttir_to_flatbuffer(
        model,
        shapes,
        dtypes,
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 12, 3200),  # arg0
            (1, 1, 12, 12),  # arg1
            (1, 12),  # arg2
            (1, 50, 1),  # arg3
            (1, 32, 50, 100),  # arg4
            (1, 1),  # arg5
            (1, 32, 50, 100),  # arg6
            (1, 32, 50, 100),  # arg7
            (1, 1),  # arg8
            (1, 32, 50, 100),  # arg9
            (1, 1),  # arg10
            (3200, 3200),  # arg11
            (3200, 3200),  # arg12
            (3200, 3200),  # arg13
            (3200, 3200),  # arg14
        ],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 15], ids=["f32"])
@pytest.mark.parametrize(
    "target",
    [
        "ttnn",
        pytest.param("ttmetal", marks=pytest.mark.skip("TTMetal not supported yet")),
    ],
)
def test_llama_attention(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    target: str,
    request,
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
        output1 = builder.squeeze(arg0, 0)
        output3 = builder.matmul(output1, arg11)
        output5 = builder.reshape(output3, (1, 12, 32, 100))
        output7 = builder.transpose(output5, -3, -2)
        output9 = builder.unsqueeze(arg2, 1)
        output11 = builder.matmul(arg3, output9)
        output13 = builder.transpose(output11, -2, -1)
        output15 = builder.concat([output13, output13], -1)
        output17 = builder.cos(output15)
        output19 = builder.unsqueeze(output17, 1)
        output21 = builder.multiply(output7, output19)
        output23 = builder.transpose(output7, -2, -1)
        output25 = builder.matmul(arg4, output23)
        output27 = builder.transpose(output25, -2, -1)
        output29 = builder.multiply(output27, arg5)
        output31 = builder.transpose(output7, -2, -1)
        output33 = builder.matmul(arg6, output31)
        output35 = builder.transpose(output33, -2, -1)
        output37 = builder.concat([output29, output35], -1)
        output39 = builder.sin(output15)
        output41 = builder.unsqueeze(output39, 1)
        output43 = builder.multiply(output37, output41)
        output45 = builder.add(output21, output43)
        output47 = builder.squeeze(output45, 0)
        output49 = builder.matmul(output1, arg12)
        output51 = builder.reshape(output49, (1, 12, 32, 100))
        output53 = builder.transpose(output51, -3, -2)
        output55 = builder.multiply(output53, output19)
        output57 = builder.transpose(output53, -2, -1)
        output59 = builder.matmul(arg7, output57)
        output61 = builder.transpose(output59, -2, -1)
        output63 = builder.multiply(output61, arg8)
        output65 = builder.transpose(output53, -2, -1)
        output67 = builder.matmul(arg9, output65)
        output69 = builder.transpose(output67, -2, -1)
        output71 = builder.concat([output63, output69], -1)
        output73 = builder.multiply(output71, output41)
        output75 = builder.add(output55, output73)
        output77 = builder.squeeze(output75, 0)
        output79 = builder.transpose(output77, -2, -1)
        output81 = builder.matmul(output47, output79)
        output83 = builder.unsqueeze(output81, 0)
        output85 = builder.multiply(output83, arg10)
        output87 = builder.add(output85, arg1)
        output89 = builder.softmax(output87, -1)
        output91 = builder.squeeze(output89, 0)
        output93 = builder.matmul(output1, arg13)
        output95 = builder.reshape(output93, (1, 12, 32, 100))
        output97 = builder.transpose(output95, -3, -2)
        output99 = builder.transpose(output97, -2, -1)
        output101 = builder.squeeze(output99, 0)
        output103 = builder.transpose(output101, -2, -1)
        output105 = builder.matmul(output91, output103)
        output107 = builder.unsqueeze(output105, 0)
        output109 = builder.transpose(output107, -3, -2)
        output111 = builder.reshape(output109, (12, 3200))
        output113 = builder.matmul(output111, arg14)
        output115 = builder.unsqueeze(output113, 0)

        return output115

    compile_ttir_to_flatbuffer(
        model,
        shapes,
        dtypes,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


conv2d_config = {
    "weights_dtype": "f32",
    "activation": "none",
    "deallocate_activation": "false",
    "reallocate_halo_output": "false",
    "act_block_h_override": "0",
    "act_block_w_div": "1",
    "reshard_if_not_optimal": "false",
    "override_sharding_config": "false",
    "shard_layout": "height_sharded",
    "core_grid": "#ttnn.core_range_set<>",
    "transpose_shards": "true",
    "output_layout": "tile",
    "enable_act_double_buffer": "false",
    "enable_weights_double_buffer": "false",
    "enable_split_reader": "false",
    "enable_subblock_padding": "false",
}


output_layout_overrides = {
    "data_type": "f32",
    "memory_layout": "tile",
    "buffer_type": "l1",
    "tensor_memory_layout": "interleaved",
    "grid_shape": "[1x1]",
}


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 32, 3, 3),
            (1, 1, 1, 64),
            (1, 16, 28, 64),
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 4], ids=["f32"])
@pytest.mark.parametrize(
    "stride,padding,dilation,groups", [([2, 1], [2, 1], [2, 1], 2)]
)
def test_overrides_model(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
):
    def model(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        in1: Operand,
        builder: TTIRBuilder,
    ):
        add_0 = builder.add(in0, in0)
        builder.set_output_layout_override(output_layout_overrides, add_0)
        conv2d_0 = builder.conv2d(
            add_0,
            weight,
            bias,
            in1,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        builder.set_conv2d_config_override(conv2d_config, conv2d_0)
        return conv2d_0

    compile_ttir_to_flatbuffer(
        model,
        shapes,
        dtypes,
        optimization_policy="DF Sharding",
        test_base=request.node.name,
        output_root=request.config.getoption("path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
